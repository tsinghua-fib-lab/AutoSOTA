# Adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/model/transformer.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    modulate_fused,
)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        #NOTE: Cast t_freq to match t
        t_freq = t_freq.to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        cos, sin = rotary_cos_sin
        qkv = rotary.apply_rotary_pos_emb(
            qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
        )

        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :,2, :, :]
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')

        # input shape of the attention should be [b h s d]
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = rearrange(x, 'b h s d -> b s (h d)')

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Linear(vocab_dim, dim)
        torch.nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding(x)
    

class ArgmaxEmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        argmax = x.argmax(dim=-1)
        return self.embedding[argmax]
    

class TopkEmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim, num_top=0):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        self.num_top = num_top

    def forward(self, x):
        topk_val, topk_idx = x.topk(self.num_top)
        return torch.einsum("...i,...ij->...j", topk_val, self.embedding[topk_idx])


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_size: int,
            n_heads: int,
            cond_dim: int,
            dropout: float,
            n_blocks: int,
            **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.vocab_embed = EmbeddingLayer(hidden_size, input_dim)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = rotary.Rotary(hidden_size // n_heads)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout) 
            for _ in range(n_blocks)
        ])

        self.output_layer = DiTFinalLayer(hidden_size, output_dim, cond_dim)
        self.kwargs = kwargs

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, input_vector, t):
        x = self.vocab_embed(input_vector)
        c = F.silu(self.sigma_map(t))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            x = self.output_layer(x, c) # BxCxD

        return x
    

################################
# MIXED
################################

class MixeDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, disc_out_dim, cont_out_dim, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        
        # Two output heads
        self.linear_disc = nn.Linear(hidden_size, disc_out_dim)
        self.linear_cont = nn.Linear(hidden_size, cont_out_dim)
        
        # Zero init for both
        self.linear_disc.weight.data.zero_()
        self.linear_disc.bias.data.zero_()
        self.linear_cont.weight.data.zero_()
        self.linear_cont.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        # Return tuple of (discrete_logits, continuous_noise_pred)
        return self.linear_disc(x), self.linear_cont(x)


class MixeDiT(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            input_dim: int,           # Discrete vocab size
            continuous_dim: int,      # New: Continuous feature dimension
            hidden_size: int,
            n_heads: int,
            cond_dim: int,
            dropout: float,
            n_blocks: int,
            **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim # Used by get_model_fn
        
        # 1. Embeddings
        self.vocab_embed = EmbeddingLayer(hidden_size, input_dim)
        # New: Continuous embedding
        self.cont_embed = nn.Linear(continuous_dim, hidden_size)
        torch.nn.init.kaiming_uniform_(self.cont_embed.weight, a=math.sqrt(5))

        # 2. Dual Timestep Embedding
        self.sigma_map_disc = TimestepEmbedder(cond_dim)
        self.sigma_map_cont = TimestepEmbedder(cond_dim)
        # Combiner: Project concatenated time embeddings back to cond_dim
        self.time_combiner = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.rotary_emb = rotary.Rotary(hidden_size // n_heads)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout) 
            for _ in range(n_blocks)
        ])

        # 3. Dual Output Layer
        self.output_layer = MixeDiTFinalLayer(hidden_size, input_dim, continuous_dim, cond_dim)
        self.kwargs = kwargs

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    # Modified forward signature
    def forward(self, x_disc, x_cont, t_disc, t_cont):
        # Embed inputs and sum (feature fusion)
        h_disc = self.vocab_embed(x_disc)
        h_cont = self.cont_embed(x_cont)
        x = h_disc + h_cont

        # Embed timesteps and combine
        c_disc = self.sigma_map_disc(t_disc)
        c_cont = self.sigma_map_cont(t_cont)
        c = self.time_combiner(torch.cat([c_disc, c_cont], dim=-1))

        # Apply rotary to the fused sequence
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            
            # Get both outputs
            out_disc, out_cont = self.output_layer(x, c)

        return out_disc, out_cont