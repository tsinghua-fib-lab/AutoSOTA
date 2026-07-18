import types
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from timm.layers.attention import maybe_add_mask
from timm.models.vision_transformer import VisionTransformer


class GELUDetached(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        y = self.gelu(x)
        multiplier = torch.where(x != 0, y / x, torch.ones_like(x))
        multiplier_detached = multiplier.detach()
        return x * multiplier_detached


def make_detach_attn_forward():
    def new_forward(self, x: Tensor, attn_mask: Tensor = None, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn.detach() @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    return new_forward


def make_centering(n: int, device=None, dtype=None) -> Tensor:
    identity = torch.eye(n, device=device, dtype=dtype)
    ones = torch.ones((n, n), device=device, dtype=dtype)
    c = identity - ones / float(n)
    return c


def make_centering_layer_norm(ln: nn.LayerNorm):
    def new_forward(self, x: Tensor):
        nd = len(self.normalized_shape)
        if nd == 0:
            raise RuntimeError(
                "LayerNorm with empty normalized_shape is not supported!"
            )

        n = 1
        for s in self.normalized_shape:
            n *= s

        if not hasattr(self, "C"):
            C = make_centering(n, device=x.device, dtype=x.dtype)
            self.register_buffer("C", C)

        C = self.C

        in_shape = x.shape
        left = x.numel() // n
        x = x.contiguous().view(left, n)
        x = x @ C

        var = x.pow(2).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)

        x = x * inv_std.detach()
        x = x.view(*in_shape)

        if self.elementwise_affine:
            left_axes = x.dim() - len(self.normalized_shape) - 1
            view_shape = (1,) + (1,) * left_axes + tuple(self.normalized_shape)
            w_view = self.weight.view(view_shape)
            b_view = self.bias.view(view_shape)
            x = x * w_view + b_view

        return x
    return new_forward


def detach_attention(model: VisionTransformer, layer_range: tuple = None):
    if layer_range is None:
        layer_range = (0, len(model.blocks) - 1)
    start, end = layer_range
    for i, blk in enumerate(model.blocks):
        if i < start or i > end:
            continue
        attn_module = blk.attn
        attn_module._orig_forward = attn_module.forward
        forward = make_detach_attn_forward()
        attn_module.forward = types.MethodType(forward, attn_module)


def attach_attention(model: VisionTransformer):
    for blk in model.blocks:
        attn_module = blk.attn
        if hasattr(attn_module, '_orig_forward'):
            attn_module.forward = attn_module._orig_forward


def detach_gelu(model, layer_range: tuple = None):
    if layer_range is None:
        layer_range = (0, len(model.blocks) - 1)
    start, end = layer_range
    for i, blk in enumerate(model.blocks):
        if i < start or i > end:
            continue
        for name, module in blk.named_children():
            if isinstance(module, nn.GELU):
                setattr(blk, name, GELUDetached())
            else:
                _detach_gelu_recursive(blk, name, module)


def _detach_gelu_recursive(parent, name, module):
    for child_name, child_module in module.named_children():
        if isinstance(child_module, nn.GELU):
            setattr(module, child_name, GELUDetached())
        else:
            _detach_gelu_recursive(module, child_name, child_module)


def attach_gelu(model):
    for blk in model.blocks:
        for name, module in blk.named_children():
            if isinstance(module, GELUDetached):
                setattr(blk, name, nn.GELU())
            else:
                _attach_gelu_recursive(blk, name, module)


def _attach_gelu_recursive(parent, name, module):
    for child_name, child_module in module.named_children():
        if isinstance(child_module, GELUDetached):
            setattr(module, child_name, nn.GELU())
        else:
            _attach_gelu_recursive(module, child_name, child_module)


def detach_layer_norm(model, layer_range: tuple = None):
    is_full_detach = layer_range is None
    if layer_range is None:
        layer_range = (0, len(model.blocks) - 1)
    start, end = layer_range
    for i, blk in enumerate(model.blocks):
        if i < start or i > end:
            continue
        _detach_ln_recursive(blk)
    # Also handle model-level LayerNorms (e.g. model.norm before classifier)
    if is_full_detach:
        for name, child in model.named_children():
            if name == 'blocks':
                continue
            if isinstance(child, nn.LayerNorm):
                new_forward = make_centering_layer_norm(child)
                child._orig_forward = child.forward
                child.forward = types.MethodType(new_forward, child)
            else:
                _detach_ln_recursive(child)


def _detach_ln_recursive(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            new_forward = make_centering_layer_norm(child)
            child._orig_forward = child.forward
            child.forward = types.MethodType(new_forward, child)
        else:
            _detach_ln_recursive(child)


def attach_layer_norm(model):
    for blk in model.blocks:
        _attach_ln_recursive(blk)
    # Also handle model-level and non-block LayerNorms
    for name, child in model.named_children():
        if name == 'blocks':
            continue
        _attach_ln_recursive(child)
    # Handle model.norm directly if it has _orig_forward
    if hasattr(model, 'norm') and hasattr(model.norm, '_orig_forward'):
        model.norm.forward = model.norm._orig_forward


def _attach_ln_recursive(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            if hasattr(child, '_orig_forward'):
                child.forward = child._orig_forward
        else:
            _attach_ln_recursive(child)
