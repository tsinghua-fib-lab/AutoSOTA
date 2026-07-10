"""
Model patching utilities from TransFusion.

Copied from TransFusion/src/models.py and TransFusion/src/lora_utils.py.
Patches OpenCLIP visual transformers with split QKV attention, Shortcut layers,
and custom forward methods for permutation-based weight matching.
"""
from __future__ import annotations

import math
import types
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    raise ImportError(
        "open_clip support requires: pip install open_clip_torch"
    )


# ── Shortcut layer ────────────────────────────────────────────────────

class Shortcut(nn.Module):
    """Identity shortcut layer for residual connections."""

    def __init__(self, dim: int):
        super().__init__()
        self.identity = nn.Parameter(torch.eye(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.identity.T


# ── LoRA-compatible linear layer ──────────────────────────────────────

class LoRALinear(nn.Linear):
    """Linear layer supporting LoRA-style and full-rank adaptation via AB argument."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        super().__init__(in_features, out_features, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = True
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def forward(self, x: torch.Tensor, AB: dict = None) -> torch.Tensor:
        def T(w):
            return w.transpose(1, 2) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if AB is not None:
            if isinstance(AB, dict):
                B = AB["B"]
                A = AB.get("A")
            else:
                B = AB
                A = None
            if A is not None:
                return result + (B @ (A @ x.transpose(1, 2).unsqueeze(1))).sum(
                    1
                ).transpose(1, 2)
            return result + (B @ x.transpose(1, 2).unsqueeze(1)).sum(
                1
            ).transpose(1, 2)

        return result


# ── LoRA-compatible attention ─────────────────────────────────────────

class LoRAAttention(nn.Module):
    """Attention layer supporting LoRA-style adaptation via AB argument."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.embed_dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = LoRALinear(dim, dim, 0.0, bias=qkv_bias)
        self.k = LoRALinear(dim, dim, 0.0, bias=qkv_bias)
        self.v = LoRALinear(dim, dim, 0.0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LoRALinear(dim, dim, 0.0, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, xq, xk, xv, AB: dict = None, **kwargs
    ) -> tuple[torch.Tensor, None]:
        B, N, C = xq.shape

        AB_q, AB_k, AB_v = None, None, None
        if AB is not None:
            AB_q = AB.get("q")
            AB_k = AB.get("k")
            AB_v = AB.get("v")

        q = self.q(xq, AB_q)
        k = self.k(xk, AB_k)
        v = self.v(xv, AB_v)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(
            0, 2, 1, 3
        )
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(
            0, 2, 1, 3
        )
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(
            0, 2, 1, 3
        )

        if torch.__version__ >= "2.1.0":
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=1 / math.sqrt(q.shape[-1]),
                dropout_p=self.attn_drop.p,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)

        AB_proj = None
        if AB is not None:
            AB_proj = AB.get("proj")

        x = self.proj(x, AB_proj)
        x = self.proj_drop(x)
        return x, None


# ── LoRA-compatible MLP ───────────────────────────────────────────────

class LoRAMlp(nn.Module):
    """MLP supporting LoRA-style adaptation via AB argument."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias: bool = True,
        drop: float = 0.0,
        use_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert use_conv is False

        self.fc1 = LoRALinear(in_features, hidden_features, bias=bias, lora_dropout=0.0)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = LoRALinear(hidden_features, out_features, bias=bias, lora_dropout=0.0)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, AB: dict = None, **kwargs) -> torch.Tensor:
        AB_fc1, AB_fc2 = None, None
        if AB is not None:
            AB_fc1 = AB.get("fc1")
            AB_fc2 = AB.get("fc2")

        x = self.fc1(x, AB_fc1)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x, AB_fc2)
        x = self.drop2(x)
        return x


# ── Custom forward methods (monkey-patched onto OpenCLIP modules) ─────

def forward_visual(ext, x: torch.Tensor, AB: dict = None) -> torch.Tensor:
    """Forward pass for the visual encoder."""
    x = ext.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    x = torch.cat([_expand_token(ext.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
    x = x + ext.positional_embedding.to(x.dtype)

    x = ext.patch_dropout(x)
    x = ext.ln_pre(x)
    x = ext.transformer(x, AB=AB)

    if ext.attn_pool is not None:
        if ext.attn_pool_contrastive is not None:
            x = ext.ln_post(x)
            tokens = ext.attn_pool(x)
            if ext.attn_pool_type == "parallel":
                pooled = ext.attn_pool_contrastive(x)
            else:
                pooled = ext.attn_pool_contrastive(tokens)
        else:
            x = ext.attn_pool(x)
            x = ext.ln_post(x)
            pooled, tokens = ext._global_pool(x)
    elif ext.final_ln_after_pool:
        pooled, tokens = ext._global_pool(x)
        pooled = ext.ln_post(pooled)
    else:
        x = ext.ln_post(x)
        pooled, tokens = ext._global_pool(x)

    if ext.proj is not None:
        pooled = pooled @ ext.proj

    if ext.output_tokens:
        return pooled, tokens
    return pooled


def _expand_token(token, batch_size: int) -> torch.Tensor:
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def transformer_forward(ext, x: torch.Tensor, attn_mask=None, AB=None) -> torch.Tensor:
    """Forward pass for the transformer encoder."""
    if not ext.batch_first:
        x = x.transpose(0, 1).contiguous()
    for i, r in enumerate(ext.resblocks):
        ab_l = AB.get(i) if AB is not None else None
        x = r(x, attn_mask=attn_mask, AB=ab_l)
    if not ext.batch_first:
        x = x.transpose(0, 1)
    return x


def attention(ext, q_x: torch.Tensor, k_x=None, v_x=None, attn_mask=None, AB=None) -> torch.Tensor:
    """Compute attention using Q, K, V tensors."""
    k_x = k_x if k_x is not None else q_x
    v_x = v_x if v_x is not None else q_x
    attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
    if AB:
        return ext.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask, AB=AB)[0]
    return ext.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]


def block_forward(ext, q_x: torch.Tensor, k_x=None, v_x=None, attn_mask=None, AB=None) -> torch.Tensor:
    """Forward pass for a transformer block with shortcuts."""
    k_x = ext.ln_1_kv(k_x) if hasattr(ext, "ln_1_kv") and k_x is not None else None
    v_x = ext.ln_1_kv(v_x) if hasattr(ext, "ln_1_kv") and v_x is not None else None

    x = ext.attn.shortcut_1(q_x) + ext.ls_1(
        ext.attention(q_x=ext.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask, AB=AB)
    )
    x = ext.mlp.shortcut_2(x) + ext.ls_2(ext.mlp(ext.ln_2(x)))
    return x


# ── setup_visual: patches model in-place ──────────────────────────────

@torch.no_grad()
def setup_visual(model: nn.Module) -> None:
    """
    Patches the visual transformer in-place:
    - Splits fused QKV into separate q/k/v Linear layers
    - Replaces MLP with LoRA-compatible version
    - Adds Shortcut layers for residual connections
    - Monkey-patches forward methods
    """
    device = next(iter(model.parameters())).device

    model.visual.forward = types.MethodType(forward_visual, model.visual)
    model.visual.transformer.forward = types.MethodType(
        transformer_forward, model.visual.transformer
    )

    for block in model.visual.transformer.resblocks:
        block.forward = types.MethodType(block_forward, block)
        block.attention = types.MethodType(attention, block)

        # Replace attention
        dim = block.attn.embed_dim
        n_heads = block.attn.num_heads
        qkv_bias = block.attn.in_proj_bias is not None
        proj_bias = block.attn.out_proj.bias is not None
        attn_drop = block.attn.dropout
        new_attn = LoRAAttention(
            dim, n_heads, attn_drop=attn_drop, qkv_bias=qkv_bias, proj_bias=proj_bias
        ).to(device)
        new_attn.q.weight.data = block.attn.in_proj_weight[:dim]
        new_attn.k.weight.data = block.attn.in_proj_weight[dim : 2 * dim]
        new_attn.v.weight.data = block.attn.in_proj_weight[2 * dim : 3 * dim]
        if qkv_bias:
            new_attn.q.bias.data = block.attn.in_proj_bias[:dim]
            new_attn.k.bias.data = block.attn.in_proj_bias[dim : 2 * dim]
            new_attn.v.bias.data = block.attn.in_proj_bias[2 * dim : 3 * dim]
        new_attn.proj.weight.data = block.attn.out_proj.weight.data
        if proj_bias:
            new_attn.proj.bias.data = block.attn.out_proj.bias.data
        new_attn.shortcut_1 = Shortcut(dim).to(device)
        block.attn = new_attn

        # Replace MLP
        in_features = block.mlp.c_fc.in_features
        out_features = block.mlp.c_proj.out_features
        hidden_features = block.mlp.c_fc.out_features
        new_mlp = LoRAMlp(
            in_features, hidden_features, out_features, bias=block.mlp.c_fc.bias is not None
        ).to(device)
        new_mlp.fc1.weight.data.zero_()
        new_mlp.fc1.weight.data.add_(block.mlp.c_fc.weight)
        if block.mlp.c_fc.bias is not None:
            new_mlp.fc1.bias.data.zero_()
            new_mlp.fc1.bias.data.add_(block.mlp.c_fc.bias)
        new_mlp.fc2.weight.data.zero_()
        new_mlp.fc2.weight.data.add_(block.mlp.c_proj.weight)
        if block.mlp.c_proj.bias is not None:
            new_mlp.fc2.bias.data.zero_()
            new_mlp.fc2.bias.data.add_(block.mlp.c_proj.bias)
        new_mlp.shortcut_2 = Shortcut(dim).to(device)
        block.mlp = new_mlp


# ── OpenCLIPModel wrapper ─────────────────────────────────────────────

class OpenCLIPModel(nn.Module):
    """Wrapper that automatically patches the visual transformer."""

    @torch.no_grad()
    def __init__(self, clip_model: open_clip.CLIP, args=None) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.args = args
        setup_visual(self.clip_model)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.clip_model.visual(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.clip_model.transformer(text)

    def forward(self, image: torch.Tensor = None, text: torch.Tensor = None):
        image_features = (
            self.clip_model.encode_image(image, normalize=True)
            if image is not None
            else None
        )
        text_features = (
            self.clip_model.encode_text(text, normalize=True)
            if text is not None
            else None
        )

        if getattr(self.clip_model, "output_dict", False):
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.clip_model.logit_scale.exp(),
            }
            if getattr(self.clip_model, "logit_bias", None) is not None:
                out_dict["logit_bias"] = self.clip_model.logit_bias
            return out_dict

        if getattr(self.clip_model, "logit_bias", None) is not None:
            return image_features, text_features, self.clip_model.logit_scale.exp(), self.clip_model.logit_bias
        return image_features, text_features, self.clip_model.logit_scale.exp()
