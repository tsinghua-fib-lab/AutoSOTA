import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MLP_HIDDEN_DIM = 256
ATTN_DIM = 128
NUM_HEADS = 8
WINDOW_SIZE = 7


def _first_param_dtype(module: nn.Module, default_dtype: torch.dtype = torch.float32) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return default_dtype


def _infer_grid_hw(num_patches: int) -> Tuple[int, int]:
    root = int(math.sqrt(num_patches))
    best_hw = (1, num_patches)
    best_gap = num_patches
    for height in range(max(1, root - 8), root + 9):
        if num_patches % height == 0:
            width = num_patches // height
            gap = abs(height - width)
            if gap < best_gap:
                best_gap = gap
                best_hw = (height, width)
    return best_hw


class WindowSelfAttention(nn.Module):
    """Local window self-attention with a reduced attention dimension."""

    def __init__(self, dim: int, attn_dim: int, num_heads: int, window_size: int):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")

        self.dim = int(dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attn_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_h = int(window_size)
        self.window_w = int(window_size)

        self.qkv = nn.Linear(self.dim, 3 * self.attn_dim)
        self.proj = nn.Linear(self.attn_dim, self.dim)
        self.beta_attn = nn.Parameter(torch.tensor(1.0))

        window_h, window_w = self.window_h, self.window_w
        self.relpos_table = nn.Parameter(
            torch.zeros(self.num_heads, (2 * window_h - 1) * (2 * window_w - 1))
        )

        coords_h = torch.arange(window_h)
        coords_w = torch.arange(window_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords[0] += window_h - 1
        rel_coords[1] += window_w - 1
        rel_coords[0] *= 2 * window_w - 1
        rel_index = rel_coords[0] + rel_coords[1]
        self.register_buffer("relpos_index", rel_index, persistent=False)

    def forward(self, patches_2d: torch.Tensor) -> torch.Tensor:
        batch, dim, height, width = patches_2d.shape
        window_h = min(self.window_h, height)
        window_w = min(self.window_w, width)

        pad_h = (window_h - height % window_h) % window_h
        pad_w = (window_w - width % window_w) % window_w
        if pad_h or pad_w:
            patches_2d = F.pad(patches_2d, (0, pad_w, 0, pad_h), mode="replicate")
            padded_h, padded_w = height + pad_h, width + pad_w
        else:
            padded_h, padded_w = height, width

        num_windows_h = padded_h // window_h
        num_windows_w = padded_w // window_w
        tokens = patches_2d.permute(0, 2, 3, 1).contiguous()
        tokens = tokens.reshape(
            batch, num_windows_h, window_h, num_windows_w, window_w, dim
        ).permute(0, 1, 3, 2, 4, 5).contiguous()
        tokens = tokens.reshape(batch * num_windows_h * num_windows_w, window_h * window_w, dim)

        qkv = self.qkv(tokens)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.reshape(tokens.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(tokens.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(tokens.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (query * self.scale) @ key.transpose(-2, -1)
        if window_h == self.window_h and window_w == self.window_w:
            bias = self.relpos_table[:, self.relpos_index.view(-1)].view(
                self.num_heads, window_h * window_w, window_h * window_w
            )
            attn = attn + bias.unsqueeze(0)

        out = (attn.softmax(dim=-1) @ value).transpose(1, 2).contiguous()
        out = out.reshape(tokens.shape[0], window_h * window_w, self.attn_dim)
        out = self.beta_attn * self.proj(out)

        out = out.reshape(
            batch, num_windows_h, num_windows_w, window_h, window_w, dim
        ).permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.reshape(batch, padded_h, padded_w, dim)
        if pad_h or pad_w:
            out = out[:, :height, :width, :]
        return out.reshape(batch, height * width, dim)


class MAAAdapter(nn.Module):
    """Three-branch Manifold-Adversarial Adapter used in the vision encoder."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.dim = int(dim)
        hidden = MLP_HIDDEN_DIM
        attn_dim = ATTN_DIM
        num_heads = NUM_HEADS
        window_size = WINDOW_SIZE
        attn_dim = max((attn_dim // num_heads) * num_heads, num_heads)

        self.norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dim),
        )
        self.window_attn = WindowSelfAttention(
            dim=self.dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.pool = nn.MaxPool2d(kernel_size=int(kernel_size), stride=1, padding=int(kernel_size) // 2)
        self.alpha_mlp = nn.Parameter(torch.zeros(1))
        self.alpha_pool = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected hidden dimension {self.dim}, got {dim}.")
        if length <= 1:
            return x

        input_dtype = x.dtype
        adapter_dtype = _first_param_dtype(self)

        mlp_input = x.to(dtype=adapter_dtype)
        mlp_residual = self.mlp(self.norm(mlp_input))

        patch_tokens = x[:, 1:, :]
        num_patches = patch_tokens.size(1)
        height, width = _infer_grid_hw(num_patches)
        if height * width != num_patches:
            height, width = 1, num_patches

        patches_2d = patch_tokens.transpose(1, 2).reshape(batch, dim, height, width)
        patches_2d = patches_2d.to(dtype=adapter_dtype)
        attn_residual = self.window_attn(patches_2d)

        fused_mlp = mlp_residual.clone()
        fused_mlp[:, 1:, :] = fused_mlp[:, 1:, :] + attn_residual

        pooled = self.pool(patches_2d)
        pool_residual = pooled.reshape(batch, dim, height * width).transpose(1, 2)

        out = x.to(torch.float32) + self.alpha_mlp.to(torch.float32) * fused_mlp.to(torch.float32)
        out[:, 1:, :] = out[:, 1:, :] + self.alpha_pool.to(torch.float32) * pool_residual.to(torch.float32)
        return out.to(dtype=input_dtype)
