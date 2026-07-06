import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional
from transformers.models.opt.modeling_opt import (
    OPTLearnedPositionalEmbedding,
    OPTDecoderLayer,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast


def split_opt(config, split_point: int, torch_dtype, device_map):
    # Load base model to copy weights
    base_model = AutoModelForCausalLM.from_pretrained(
        config.name_or_path, torch_dtype=torch_dtype
    )
    config = base_model.config

    # Create client model
    client_model = ClientOPTDecoder(config, split_point)
    # Copy weights
    _copy_opt_client_weights(client_model, base_model, split_point)

    # Create server model
    server_model = ServerOPTDecoder(config, split_point)
    # Copy weights
    _copy_opt_server_weights(server_model, base_model, split_point)

    return client_model.to(device_map), server_model.to(device_map)


def _copy_opt_client_weights(client_model, base_model, split_point):
    # Copy embed_tokens
    client_model.embed_tokens.load_state_dict(
        base_model.model.decoder.embed_tokens.state_dict()
    )
    # Copy embed_positions
    client_model.embed_positions.load_state_dict(
        base_model.model.decoder.embed_positions.state_dict()
    )
    # Copy project_in if exists
    if client_model.project_in is not None:
        client_model.project_in.load_state_dict(
            base_model.model.decoder.project_in.state_dict()
        )
    # Copy first split_point layers
    for i in range(split_point):
        client_model.layers[i].load_state_dict(
            base_model.model.decoder.layers[i].state_dict()
        )


def _copy_opt_server_weights(server_model, base_model, split_point):
    # Copy remaining layers
    for i in range(split_point, base_model.config.num_hidden_layers):
        server_model.layers[i - split_point].load_state_dict(
            base_model.model.decoder.layers[i].state_dict()
        )
    # Copy final_layer_norm if exists
    if server_model.final_layer_norm is not None:
        server_model.final_layer_norm.load_state_dict(
            base_model.model.decoder.final_layer_norm.state_dict()
        )
    # Copy project_out if exists
    if server_model.project_out is not None:
        server_model.project_out.load_state_dict(
            base_model.model.decoder.project_out.state_dict()
        )
    # Copy lm_head
    server_model.lm_head.load_state_dict(base_model.lm_head.state_dict())


class ClientOPTDecoder(nn.Module):
    def __init__(self, config, split_point: int):
        super().__init__()
        self.config = config
        self.split_point = split_point
        # check split_point validity
        if split_point <= 0 or split_point > config.num_hidden_layers:
            raise ValueError(
                f"split_point must be in the range (0, {config.num_hidden_layers}], got {split_point}"
            )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, config.pad_token_id
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        # Only first split_point layers
        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config) for _ in range(split_point)]
        )

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None
    ):
        # Simplified forward, similar to OPTDecoder but only up to split_point
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            raise ValueError("input_ids must be provided")

        inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = input_shape

        # Attention mask handling
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, device=inputs_embeds.device
            )
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, 0
        )

        # Fix: Prevent NaN from all-masked rows in causal attention mask
        # Left-padding causes positions where ALL keys are masked (by both causal and padding masks)
        # softmax(all -inf) = NaN, which propagates through the entire network
        mask_min = torch.finfo(causal_attention_mask.dtype).min
        all_masked = (causal_attention_mask <= mask_min * 0.5).all(dim=-1)  # [B, 1, S]
        diag_mask = torch.eye(seq_length, device=causal_attention_mask.device, dtype=torch.bool)
        fix_mask = all_masked.unsqueeze(-1) & diag_mask.unsqueeze(0).unsqueeze(0)  # [B, 1, S, S]
        causal_attention_mask = causal_attention_mask.masked_fill(fix_mask, 0.0)

        pos_embeds = self.embed_positions(attention_mask, 0)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # Process layers
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

        return hidden_states, causal_attention_mask


class ServerOPTDecoder(nn.Module):
    def __init__(self, config, split_point: int):
        super().__init__()
        self.config = config
        self.split_point = split_point

        # Remaining layers
        self.layers = nn.ModuleList(
            [
                OPTDecoderLayer(config)
                for _ in range(config.num_hidden_layers - split_point)
            ]
        )

        # Final layer norm and project_out
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        # LM head
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # Process remaining layers
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            logits=logits,
        )
