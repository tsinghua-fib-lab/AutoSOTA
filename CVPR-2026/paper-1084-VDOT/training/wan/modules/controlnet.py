from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import ModelMixin
from diffusers.models.attention_processor import Attention
from xfuser.core.long_ctx_attention import xFuserLongContextAttention


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class WanAttnProcessorSP:
    def __init__(self, sp_size=1):
        self.sp_size = sp_size
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)  # [b,h,l,c]
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if self.sp_size > 1:
            def half(x):
                return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)

            # convert [batch, nhead, length, channel] -> [batch, length, nhead, channel]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = xFuserLongContextAttention()(None, query=half(query), key=half(key), value=half(value))
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SimpleAttnProcessor2_0:
    def __init__(self):
        self.sp_size = 1
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SimpleAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_emb: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)  # [b,head,l,c]

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        if self.sp_size > 1:
            def half(x):
                return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = xFuserLongContextAttention()(None, query=half(query), key=half(key), value=half(value))
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SimpleCogVideoXLayerNormZero(nn.Module):
    def __init__(
            self,
            conditioning_dim: int,
            embedding_dim: int,
            elementwise_affine: bool = True,
            eps: float = 1e-5,
            bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor):
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states, gate[:, None, :]


class SingleAttentionBlock(nn.Module):

    def __init__(
            self,
            dim,
            ffn_dim,
            num_heads,
            time_embed_dim=512,
            qk_norm="rms_norm_across_heads",
            eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.norm1 = SimpleCogVideoXLayerNormZero(
            time_embed_dim, dim, elementwise_affine=True, eps=1e-5, bias=True
        )
        self.self_attn = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=SimpleAttnProcessor2_0(),
        )
        self.norm2 = SimpleCogVideoXLayerNormZero(
            time_embed_dim, dim, elementwise_affine=True, eps=1e-5, bias=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

    def forward(
            self,
            hidden_states,
            temb,
            rotary_emb,
    ):
        # norm & modulate
        norm_hidden_states, gate_msa = self.norm1(hidden_states, temb)

        # attention
        attn_hidden_states = self.self_attn(hidden_states=norm_hidden_states,
                                            rotary_emb=rotary_emb)

        hidden_states = hidden_states + gate_msa * attn_hidden_states

        # norm & modulate
        norm_hidden_states, gate_ff = self.norm2(hidden_states, temb)

        # feed-forward
        ff_output = self.ffn(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output

        return hidden_states


class WanXControlNet(ModelMixin):
    def __init__(self,
                 conv_out_dim=5120,
                 dim=1024,
                 ffn_dim=4096,
                 num_heads=16,
                 num_layers=20,
                 time_embed_dim=5120,
                 ):
        super().__init__()

        # self.controlnet_cfg = controlnet_cfg
        if conv_out_dim != dim:
            self.proj_in = nn.Linear(conv_out_dim, dim)
        else:
            self.proj_in = nn.Identity()

        self.controlnet_blocks = nn.ModuleList(
            [
                SingleAttentionBlock(
                    dim=dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    time_embed_dim=time_embed_dim,
                    qk_norm="rms_norm_across_heads",
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.ModuleList(
            [
                zero_module(nn.Linear(dim, 5120))
                for _ in range(num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb, rotary_emb):
        hidden_states = self.proj_in(hidden_states)
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        controlnet_states = []
        for i, block in enumerate(self.controlnet_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # hidden_states = self._gradient_checkpointing_func(
                #     block,
                #     hidden_states=hidden_states,
                #     temb=temb,
                #     rotary_emb=rotary_emb
                # )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, temb, rotary_emb,
                    use_reentrant=False,
                )
                # print('Control Gradient checking enabled...')
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    rotary_emb=rotary_emb
                )
            controlnet_states.append(self.proj_out[i](hidden_states))

        return controlnet_states
