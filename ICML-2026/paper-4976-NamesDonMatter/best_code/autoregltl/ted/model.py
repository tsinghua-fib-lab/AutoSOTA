"""
Transformer encoder-decoder implementation
Based on https://github.com/tensorflow/models/tree/master/official/nlp/transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from autoregltl.ltl.vocab import EncDecVocab, CharVocab, MergedLTLVocab
from autoregltl.dataset import EncDecLTLCollator, EncDecLTLDataset
from autoregltl.embedding import EmbedderConfig, DynamicEmbedder
from autoregltl import positional_encoding as pe
from autoregltl.ltl.parser import ParseError, ltl_formula, ltl_trace
from .layers import attention
from .beam_search import BeamSearch

import json
import os
from tqdm.auto import tqdm
from dataclasses import dataclass, field, asdict
from collections import namedtuple
from typing import Dict, Union, Any, Optional, Tuple, List


@dataclass
class TransformerConfig:
    vocab: EncDecVocab | MergedLTLVocab

    d_embed_enc: int  # dimension of encoder embedding
    d_embed_dec: int  # dimension of decoder embedding
    d_ff: int  # hidden dimension of feed-forward networks
    ff_activation: str  # activation function used in feed-forward networks
    dropout: float  # percentage of droped out units
    num_heads: int  # number of attention heads
    num_layers: int  # number of encoder / decoder layers
    layer_norm_eps: float

    # Cross attention configuration: supports 'per', 'agg', 'per-agg', 'agg-per', or ''
    # 'per': cross attention with each AP pipeline
    # 'agg': cross attention with aggregated APs
    # 'per-agg': per-pipeline first, then aggregated
    # 'agg-per': aggregated first, then per-pipeline
    # '': no cross attention
    cross_attn: str = ""

    no_enc_agg: bool = False  # whether to disable aggregation attention in encoder
    no_dec_agg: bool = False  # whether to disable aggregation attention in decoder
    no_enc_per: bool = False  # whether to disable per-pipeline attention in encoder
    no_dec_per: bool = False  # whether to disable per-pipeline attention in decoder

    merged_embedder: Optional[EmbedderConfig] = None

    # Used for constructing the positional encoding buffers
    max_encode_length: int = 1024  # maximum length of input sequence
    max_decode_length: int = 1024  # maximum length of target sequence

    tree_pos_enc: bool = False

    datatype: str = 'float32'  # datatype for floating point computations

    enc_pe: str = 'sinusoid'  # type of positional encoding for encoder
    dec_pe: str = 'sinusoid'  # type of positional encoding for decoder
    no_pe_cross_keys: bool = False  # whether to use positional encoding for cross-attention keys

    # The methods needed by Trainer
    def to_json_string(self):
        return json.dumps(asdict(self))
    def to_dict(self):
        return asdict(self)
    
    def __post_init__(self):
        self.dtype = getattr(torch, self.datatype)
        # For loading from dictionary
        if isinstance(self.merged_embedder, dict):
            self.merged_embedder = EmbedderConfig(**self.merged_embedder)

        if isinstance(self.vocab, dict):
            if self.merged_embedder:
                self.vocab = MergedLTLVocab(**self.vocab)
            else:
                self.vocab = EncDecVocab(CharVocab(**self.vocab["inp"]), CharVocab(**self.vocab["out"]))

        if self.d_embed_dec is None:
            self.d_embed_dec = self.d_embed_enc
        self.d_embed_enc -= self.d_embed_enc % self.num_heads  # round down
        self.d_embed_dec -= self.d_embed_dec % self.num_heads  # round down


def get_activation(activation):
    """
    Args:
        activation: str, name of the activation function
    """
    if activation =='relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Unknown activation function {activation}')


def create_padding_mask(input, pad_id, dtype=torch.float32):
    """
    Args:
        input: int tensor with shape (batch_size, input_length)
        pad_id: int, encodes the padding token
        dtype: data type of look ahead mask
    Returns:
        padding mask with shape (batch_size, 1, 1, 1, input_length) that indicates padding with 1 and 0 everywhere else
    """
    mask = (input == pad_id).to(dtype)
    return mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)


def create_look_ahead_mask(size, device, dtype=torch.float32):
    """
    Creates a look ahead mask that masks future positions in a sequence, e.g., [[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]] for size 3
    Args:
        size: int, specifies the size of the look ahead mask
        device: torch.device, device where the tensors reside
        dtype: data type of look ahead mask
    Returns:
        look ahead mask with shape (1, 1, 1, size, size) that indicates masking with 1 and 0 everywhere else
    """
    # NOTE: 0 means keep, 1 means ignore
    mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
    mask = mask.unsqueeze(0) # broadcasted over num_heads
    mask = mask.unsqueeze(0) # broadcasted over ap_count
    mask = mask.unsqueeze(0) # broadcasted over batch_size
    return mask


def aggregate_aps(input, ap_mask, seq_ap_mask):
    """
    Args:
        input: float tensor with shape (batch_size, ap_count, seq_length, d_embed)
        ap_mask: int tensor with shape (batch_size, ap_count, seq_length)
        seq_ap_mask: int tensor with shape (batch_size, ap_count)
    Returns:
        all_aps: float tensor with shape (batch_size, 1, seq_length, d_embed)
    """
    # A tensor that combines all AP embeddings, replacing the placeholder embeddings
    # We need to take the mean across ap_count dimension but ignore if seq_ap_mask is 0
    all_aps = torch.sum(input * seq_ap_mask.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)
    divisor = seq_ap_mask.sum(dim=1, keepdim=True)
    divisor[divisor == 0] = 1  # to avoid division by zero
    all_aps = all_aps / divisor.unsqueeze(-1).unsqueeze(-1)
    ap_count = ap_mask.size(1)
    for i in range(ap_count):
        ap_positions = ap_mask[:, i, :].unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, seq_length, 1)
        ap_embedding = input[:, i, :, :].unsqueeze(1)  # (batch_size, 1, seq_length, d_embed)
        all_aps = all_aps * (1 - ap_positions) + ap_embedding * ap_positions
    return all_aps


class TransformerEncoderLayer(nn.Module):
    """A single encoder layer of the Transformer that consists of two sub-layers: a multi-head
    self-attention mechanism followed by a fully-connected feed-forward network. Both sub-layers
    employ a residual connection followed by a layer normalization."""

    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerEncoderLayer, self).__init__()
        if not config.no_enc_per:
            self.self_attn_enabled = True
            self.multi_head_attn = attention.MultiHeadAttention(config.d_embed_enc, config.num_heads, config.enc_pe)
            self.norm_attn = nn.LayerNorm(config.d_embed_enc, eps=config.layer_norm_eps)
            self.dropout_attn = nn.Dropout(config.dropout)
        else:
            self.self_attn_enabled = False
        if not config.no_enc_agg:
            self.ap_attn_enabled = True
            self.multi_head_ap_attn = attention.MultiHeadAttention(config.d_embed_enc, config.num_heads, config.enc_pe)
            self.norm_ap_attn = nn.LayerNorm(config.d_embed_enc, eps=config.layer_norm_eps)
            self.dropout_ap_attn = nn.Dropout(config.dropout)
        else:
            self.ap_attn_enabled = False

        self.ff = nn.Sequential(
            nn.Linear(config.d_embed_enc, config.d_ff),
            get_activation(config.ff_activation),
            nn.Linear(config.d_ff, config.d_embed_enc)
        )
        self.norm_ff = nn.LayerNorm(config.d_embed_enc, eps=config.layer_norm_eps)
        self.dropout_ff = nn.Dropout(config.dropout)

    def forward(self, input, mask, ap_mask, seq_ap_mask):
        """
        Args:
            input: float tensor with shape (batch_size, ap_count, input_length, d_embed_dec)
            mask: float tensor with shape (batch_size, 1, 1, 1, input_length)
            seq_ap_mask: int tensor with shape (batch_size, ap_count)
        """
        attn_weights = {}
        # Default self-attention work independently in each AP pipeline
        # Each pipeline attends to itself only
        if self.self_attn_enabled:
            attn, self_attn_weights = self.multi_head_attn(input, input, input, mask)
            attn_weights['self_attn'] = self_attn_weights
            attn = self.dropout_attn(attn)
            norm_attn = self.norm_attn(attn + input)
        else:
            norm_attn = input

		# New in proposed method: attend over all APs (interchangeable tokens)
        if self.ap_attn_enabled:
            all_aps = aggregate_aps(norm_attn, ap_mask, seq_ap_mask)
            ap_attn, ap_attn_weights = self.multi_head_ap_attn(
                norm_attn, all_aps, all_aps,
                mask
            )
            attn_weights['ap_attn'] = ap_attn_weights
            ap_attn = self.dropout_ap_attn(ap_attn)
            norm_ap_attn = self.norm_ap_attn(ap_attn + norm_attn)
        else:
            norm_ap_attn = norm_attn

        ff_out = self.ff(norm_ap_attn)
        ff_out = self.dropout_ff(ff_out)
        norm_ff_out = self.norm_ff(ff_out + norm_ap_attn)

        return norm_ff_out, attn_weights


class CrossAttn(nn.Module):
    def __init__(self, config: TransformerConfig, aggregate: bool = False):
        super().__init__()
        self.attn = attention.MultiHeadAttention(config.d_embed_dec, config.num_heads, config.dec_pe)
        self.norm = nn.LayerNorm(config.d_embed_dec, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.aggregate = aggregate
        self.no_pe_cross_keys = config.no_pe_cross_keys

    def forward(self, input, kv, padding_mask, past_queries=0):
        output, _ = self.attn(
            input, kv, kv,
            padding_mask,
            past_queries = past_queries,
            no_pe_keys = self.no_pe_cross_keys,
        )
        output = self.dropout(output)
        return self.norm(output + input)


class TransformerDecoderLayer(nn.Module):
    """A single decoder layer of the Transformer that consists of three sub-layers: a multi-head
    self-attention mechanism followed by a multi-head encoder-decoder-attention mechanism followed
    by a fully-connected feed-forward network. All three sub-layers employ a residual connection
    followed by a layer normalization."""

    def __init__(self, config):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of dropped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                cross_attn: string specifying cross attention configuration
        """
        super(TransformerDecoderLayer, self).__init__()
        if not config.no_dec_per:
            self.self_attn_enabled = True
            self.multi_head_self_attn = attention.MultiHeadAttention(config.d_embed_dec, config.num_heads, config.dec_pe)
            self.norm_self_attn = nn.LayerNorm(config.d_embed_dec, eps=config.layer_norm_eps)
            self.dropout_self_attn = nn.Dropout(config.dropout)
        else:
            self.self_attn_enabled = False
        if not config.no_dec_agg:
            self.ap_attn_enabled = True
            self.multi_head_ap_attn = attention.MultiHeadAttention(config.d_embed_dec, config.num_heads, config.dec_pe)
            self.norm_ap_attn = nn.LayerNorm(config.d_embed_dec, eps=config.layer_norm_eps)
            self.dropout_ap_attn = nn.Dropout(config.dropout)
        else:
            self.ap_attn_enabled = False
        # Parse cross_attn configuration
        cross_attn_config = config.cross_attn.split('-') if config.cross_attn else []
        self.cross_attns = nn.ModuleList([CrossAttn(config, aggregate=(mode=='agg')) for mode in cross_attn_config])
        self.no_pe_cross_keys = config.no_pe_cross_keys

        self.ff = nn.Sequential(
            nn.Linear(config.d_embed_dec, config.d_ff),
            get_activation(config.ff_activation),
            nn.Linear(config.d_ff, config.d_embed_dec)
        )
        self.norm_ff = nn.LayerNorm(config.d_embed_dec, eps=config.layer_norm_eps)
        self.dropout_ff = nn.Dropout(config.dropout)

    def forward(self, input, look_ahead_mask, ap_mask, seq_ap_mask, enc_output, encoder_ap_mask, padding_mask, cache=None, ap_cache=None):
        """
        Args:
            input: float tensor with shape (batch_size, ap_count, seq_length, d_embed)
            look_ahead_mask: float tensor with shape (1, 1, seq_length, seq_length)
            ap_mask: int tensor with shape (batch_size, ap_count, seq_length)
            seq_ap_mask: int tensor with shape (batch_size, ap_count)
            enc_output: float tensor with shape (batch_size, ap_count, input_length, d_embed_enc)
            padding_mask: float tensor with shape (batch_size, ap_count, 1, 1, input_length)
            cache: dict with 'keys' and 'values' for KV caching during generation
            ap_cache: dict with 'keys' and 'values' for KV caching during generation for AP attention
        """
        # keys: (batch_size, ap_count, num_heads, num_keys, d_heads)
        past_queries = cache['keys'].size(-2) if cache is not None else 0
        attn_weights = {}

		# Default self-attention work independently in each AP pipeline
        # Each pipeline attends to itself only
        if self.self_attn_enabled:
            self_attn, self_attn_weights = self.multi_head_self_attn(
                input, input, input,
                look_ahead_mask,
                cache,
                past_queries = past_queries,
            )
            attn_weights['self_attn'] = self_attn_weights
            self_attn = self.dropout_self_attn(self_attn)
            norm_self_attn = self.norm_self_attn(self_attn + input)
        else:
            norm_self_attn = input

		# New in proposed method: attend over all APs (interchangeable tokens)
        if self.ap_attn_enabled:
            all_aps = aggregate_aps(norm_self_attn, ap_mask, seq_ap_mask)
            ap_attn, ap_attn_weights = self.multi_head_ap_attn(
                norm_self_attn, all_aps, all_aps,
                look_ahead_mask,
                ap_cache,
                past_queries=past_queries,
            )
            attn_weights['ap_attn'] = ap_attn_weights
            ap_attn = self.dropout_ap_attn(ap_attn)
            last_output = self.norm_ap_attn(ap_attn + norm_self_attn)
        else:
            last_output = norm_self_attn

        for cross_attn in self.cross_attns:
            # Updated cross-attention to support both per-pipeline and aggregated AP attention
            kv = aggregate_aps(enc_output, encoder_ap_mask, seq_ap_mask) if cross_attn.aggregate else enc_output
            last_output = cross_attn(last_output, kv, padding_mask, past_queries=past_queries)

        ff_out = self.ff(last_output)
        ff_out = self.dropout_ff(ff_out)
        norm_ff_out = self.norm_ff(ff_out + last_output)

        return norm_ff_out, attn_weights


class TransformerEncoder(nn.Module):
    """The encoder of the Transformer that is composed of num_layers identical layers."""

    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
        """
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, x, padding_mask, ap_mask, seq_ap_mask):
        attn_weights = {}
        for i, layer in enumerate(self.enc_layers):
            x, layer_attn_weights = layer(x, padding_mask, ap_mask, seq_ap_mask)
            attn_weights[f'layer_{i+1}'] = layer_attn_weights
        return x, attn_weights


class TransformerDecoder(nn.Module):
    """The decoder of the Transformer that is composed of num_layers identical layers."""

    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
        """
        super(TransformerDecoder, self).__init__()
        self.dec_layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, x, look_ahead_mask, ap_mask, seq_ap_mask, enc_output, encoder_ap_mask, padding_mask, cache=None):
        attn_weights = {}
        for i, layer in enumerate(self.dec_layers):
            layer_cache = cache[f'layer_{i}'] if cache is not None else None
            layer_ap_cache = cache[f'layer_{i}_ap'] if cache is not None else None
            x, layer_attn_weights = layer(x, look_ahead_mask, ap_mask, seq_ap_mask, enc_output, encoder_ap_mask, padding_mask, layer_cache, layer_ap_cache)
            attn_weights[f'layer_{i+1}'] = layer_attn_weights
        return x, attn_weights


class Transformer(nn.Module):
    """The Transformer that consists of an encoder and a decoder. The encoder maps the input
    sequence to a sequence of continuous representations. The decoder then generates an output
    sequence in an auto - regressive way."""

    def __init__(self, config, dtype=torch.float32):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layer
                max_encode_length: int, maximum length of input sequence
                max_decode_length: int, maximum lenght of target sequence
                dropout: float, percentage of droped out units
            dtype: datatype for floating point computations
        """
        super(Transformer, self).__init__()
        self.config = config

        if config.d_embed_enc != config.d_embed_dec:
            raise ValueError("Cannot merge vocabularies: embedding dimensions don't match")
        merged_embedder = config.merged_embedder.build(config.d_embed_enc, config.vocab, dtype=dtype)
        embedding_func = lambda x: merged_embedder.embed(x)
        self.encoder_embedding = embedding_func
        self.decoder_embedding = embedding_func
        self.final_projection = lambda x, y: merged_embedder.project(x, y)
        self.merged_embedder = merged_embedder
        self.start_id = config.vocab.start_id
        self.pad_id = config.vocab.pad_id

        self.register_buffer(
            'encoder_positional_encoding',
            pe.positional_encoding(config.max_encode_length, config.d_embed_enc),
            persistent=False,
        )
        self.encoder_dropout = nn.Dropout(config.dropout)

        self.encoder_stack = TransformerEncoder(config)

        self.register_buffer(
            'decoder_positional_encoding',
            pe.positional_encoding(config.max_decode_length, config.d_embed_dec),
            persistent=False,
        )
        self.decoder_dropout = nn.Dropout(config.dropout)

        self.decoder_stack = TransformerDecoder(config)

        self.softmax = nn.Softmax(dim=-1)

        self.dtype = dtype

    def encode(self, inputs, padding_mask, positional_encoding):
        """
        Args:
            inputs: int tensor with shape (batch_size, input_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc)
        """
        input_embedding, ap_mask = self.encoder_embedding(inputs)

		# (batch_size, ap_count)
        # 1 if any position in the sequence is an AP token, 0 otherwise
        seq_ap_mask = ap_mask.any(dim=2).int()

        if positional_encoding is not None:
            input_embedding += positional_encoding.unsqueeze(1)  # Add ap_count dimension
        input_embedding = self.encoder_dropout(input_embedding)

        encoder_output, attn_weights = self.encoder_stack(input_embedding, padding_mask, ap_mask, seq_ap_mask)

        return encoder_output, ap_mask, seq_ap_mask, attn_weights

    def decode(self, target, encoder_output, encoder_ap_mask, seq_ap_mask, input_padding_mask):
        """
        Args:
            target: int tensor with shape (batch_size, target_length)
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            seq_ap_mask: int tensor with shape (batch_size, ap_count)
            input_padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
        
        Returns:
            logits: float tensor with shape (batch_size, target_length, out_vocab_size)
            attn_weights: dictionary with keys 'layer_i' where i is the layer number and values are float tensors with shape (batch_size, num_heads, target_length, input_length)
        """
        target_length = target.size(1)
        look_ahead_mask = create_look_ahead_mask(target_length, target.device, self.dtype)
        target_padding_mask = create_padding_mask(target, self.pad_id, self.dtype)
        look_ahead_mask = torch.maximum(look_ahead_mask, target_padding_mask)

        # shift targets to the right, insert start_id at first postion, and remove last element
        target = F.pad(target, (1, 0), value=self.start_id)[:, :-1]

        target_embedding, ap_mask = self.decoder_embedding(target)  # (batch_size, target_length, d_embedding)
        if self.config.dec_pe == 'sinusoid':
            target_embedding += self.decoder_positional_encoding[:, :target_length, :]
        decoder_embedding = self.decoder_dropout(target_embedding)

        decoder_output, attn_weights = self.decoder_stack(
            decoder_embedding,
            look_ahead_mask,
            ap_mask,
            seq_ap_mask,
            encoder_output,
            encoder_ap_mask,
            input_padding_mask,
        )
        output = self.final_projection(decoder_output, seq_ap_mask)
        return output, attn_weights

    def forward(self, input, target, positional_encoding=None):
        """
        Args:
            input: int tensor with shape (batch_size, input_length)
            (optional) target: int tensor with shape (batch_size, target_length)
            padding mask with shape (batch_size, 1, 1, input_length) that indicates padding with 1 and 0 everywhere else
            (optional) positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc), custom postional encoding
        """
        if self.training and self.merged_embedder:
            self.merged_embedder.prepare()

        input_padding_mask = create_padding_mask(input, self.pad_id, self.dtype)

        if positional_encoding is None and self.config.enc_pe == 'sinusoid':
            assert not self.config.tree_pos_enc
            seq_len = input.size(1)
            positional_encoding = self.encoder_positional_encoding[:, :seq_len, :]
        encoder_output, encoder_ap_mask, seq_ap_mask, encoder_attn_weights = self.encode(input, input_padding_mask, positional_encoding)

        logits, _ = self.decode(target, encoder_output, encoder_ap_mask, seq_ap_mask, input_padding_mask)
        return logits

    def generate(
            self,
            input,
            max_decode_length,
            positional_encoding=None,
            alpha=1.0,
            beam_size=1,
            syntax_enforcer=None,
        ):
        """
        Args:
            input_padding_mask: flaot tensor with shape (batch_size, 1, 1, input_length)
            alpha: float, strength of normalization in beam search algorithm
            beam_size: int, number of beams kept by beam search algorithm
        """
        batch_size = input.size(0)

        input_padding_mask = create_padding_mask(input, self.pad_id, self.dtype)

        if positional_encoding is None and self.config.enc_pe == 'sinusoid':
            seq_len = input.size(1)
            positional_encoding = self.encoder_positional_encoding[:, :seq_len, :]
        encoder_output, encoder_ap_mask, seq_ap_mask, encoder_attn_weights = self.encode(input, input_padding_mask, positional_encoding)

        num_heads = self.config.num_heads
        d_heads = self.config.d_embed_dec // num_heads
        # Create an empty KV cache structure for decoder attention
        ap_count = max(1, self.merged_embedder.ap_count)
        cache = {
            f'layer_{layer}': {
                'keys': torch.zeros(batch_size, ap_count, num_heads, 0, d_heads, device=encoder_output.device, dtype=self.dtype),
                'values': torch.zeros(batch_size, ap_count, num_heads, 0, d_heads, device=encoder_output.device, dtype=self.dtype)
            } for layer in range(self.config.num_layers)
        }
        cache |= {
            f'layer_{layer}_ap': {
                'keys': torch.zeros(batch_size, 1, num_heads, 0, d_heads, device=encoder_output.device, dtype=self.dtype),
                'values': torch.zeros(batch_size, 1, num_heads, 0, d_heads, device=encoder_output.device, dtype=self.dtype)
            } for layer in range(self.config.num_layers)
        }
        # add encoder output to cache
        cache['seq_ap_mask'] = seq_ap_mask
        cache['encoder_output'] = encoder_output
        cache['encoder_ap_mask'] = encoder_ap_mask
        cache['input_padding_mask'] = input_padding_mask

        look_ahead_mask = create_look_ahead_mask(max_decode_length, input.device, self.dtype)

        def logits_fn(ids, i, cache):
            """
            Args:
                ids: int tensor with shape (batch_size * beam_size, index + 1)
                index: int, current index
                cache: dictionary storing encoder output, previous decoder attention values
            Returns:
                logits with shape (batch_size * beam_size, vocab_size) and updated cache
            """
            nonlocal look_ahead_mask
            # set input to last generated id
            decoder_input = ids[:, -1:]
            decoder_input, decoder_ap_mask = self.decoder_embedding(decoder_input)
            if self.config.dec_pe == 'sinusoid':
                decoder_input += self.decoder_positional_encoding[:, i:i + 1, :]

            self_attention_mask = look_ahead_mask[:, :, :, i:i + 1, :i + 1]
            decoder_output, _ = self.decoder_stack(
                decoder_input,
                self_attention_mask,
                decoder_ap_mask,
                cache['seq_ap_mask'],
                cache['encoder_output'],
                cache['encoder_ap_mask'],
                cache['input_padding_mask'],
                cache,
            )
            output = self.final_projection(decoder_output, cache['seq_ap_mask'])
            output = output.squeeze(1)
            return output, cache

        initial_ids = torch.ones(batch_size, dtype=torch.int32, device=encoder_output.device) * self.start_id

        beam_search = BeamSearch(
            logits_fn,
            batch_size,
            encoder_output.device,
            syntax_enforcer,
            max_decode_length,
            self.start_id,
            self.config.vocab.eos_id if self.merged_embedder else self.config.vocab.out.eos_id,
            self.merged_embedder.output_vocab_size if self.merged_embedder else self.config.vocab.out.size(),
            alpha,
            beam_size,
            self.dtype,
        )
        decoded_ids, scores = beam_search.search(initial_ids, cache)

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        # compute attention weights
        _, decoder_attn_weights = self.decode(top_decoded_ids, cache['encoder_output'], cache['encoder_ap_mask'], cache['seq_ap_mask'], cache['input_padding_mask'])

        return {'outputs': top_decoded_ids, 'scores': top_scores, 'enc_attn_weights': encoder_attn_weights, 'dec_attn_weights': decoder_attn_weights}
    
    @torch.inference_mode()
    def generate_predictions(self, dataset, max_length, gen_args, leave_tqdm=True, prepare_embedder=True):
        self.eval()
        if prepare_embedder:
            self.merged_embedder.prepare()
        for param in self.parameters():
            model_device = param.device
            break

        vocab = self.config.vocab
        if isinstance(vocab, EncDecVocab):
            input_encode = lambda x: vocab.inp.encode(x, prepend_start_token=False)
            output_decode = lambda x: vocab.out.decode(x)
        elif isinstance(vocab, MergedLTLVocab):
            input_encode = lambda x: vocab.encode_ltl(x, eos=True)
            output_decode = lambda x: vocab.decode(x)
        else:
            raise ValueError(f"Unsupported vocab type: {type(vocab)}")

        predictions = []
        if "gen_batch_size" in gen_args:
            gen_args = gen_args.copy()
            batch_size = gen_args.pop("gen_batch_size")
        else:
            batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size)
        with tqdm(total=len(dataset), desc="Predict", leave=leave_tqdm) as pbar:
            for (traces, formulas) in dataloader:
                # Pad by adding pad tokens to the right (end)
                input_ids = [torch.tensor(input_encode(formula), dtype=torch.long) for formula in formulas]
                input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
                positional_encoding = None
                if self.config.tree_pos_enc:
                    positional_encoding = []
                    max_seq_len = input_ids.size(1)
                    for formula in formulas:
                        position_list = ltl_formula(formula, 'network-polish').binary_position_list(format='lbt', add_first=True)
                        padded_position_list = [
                            (l + [0] * (self.config.d_embed_enc - len(l)))[:self.config.d_embed_enc]
                            for l in position_list
                        ]
                        pe = torch.tensor(padded_position_list, dtype=torch.float32)
                        pe = F.pad(pe, (0, self.config.d_embed_enc - pe.size(-1), 0, max_seq_len - pe.size(-2)))
                        positional_encoding.append(pe)
                    positional_encoding = torch.stack(positional_encoding, dim=0)
                out = self.generate(
                    input=input_ids.to(model_device),
                    # +1 for start token
                    max_decode_length = max_length + 1,
                    positional_encoding = positional_encoding.to(model_device) if positional_encoding is not None else None,
                    **gen_args,
                )['outputs']
                for prediction, trace, formula in zip(out.tolist(), traces, formulas):
                    prediction = output_decode(prediction)
                    # formula trace target
                    predictions.append((prediction, trace, formula))
                pbar.update(len(formulas))
        return predictions

    @classmethod
    def load_pretrained(cls, directory, dtype=torch.float32, device=None, **kwargs):
        if not os.path.exists(directory):
            raise FileNotFoundError("Model directory is not found")

        with open(os.path.join(directory, 'config.json'), 'r') as f:
            config_data = json.load(f)
        config = TransformerConfig(**config_data)

        model = cls(config, dtype=dtype, **kwargs)
        if device is not None:
            model = model.to(device)
        state_dict = torch.load(os.path.join(directory, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict)
        model.merged_embedder.prepare()
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=4)
    
