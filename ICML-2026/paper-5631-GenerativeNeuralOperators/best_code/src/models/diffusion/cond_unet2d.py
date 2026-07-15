import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embedding import SinusoidalPositionalEmbedding
from ..unet1d import PreNorm, group_norm
from ..unet2d import AttentionBlock2d, Downsample2d, Upsample2d


class TimeAwareResBlock2d(nn.Module):
    """
    2D ResBlock with FiLM modulation from a time embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        activation: str = "SiLU",
        dropout: float = 0.0,
        padding_mode: str = "circular",
    ):
        super().__init__()
        self.act = getattr(nn, activation)() if hasattr(nn, activation) else nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = group_norm(int(in_channels), max_groups=32)
        self.time_proj1 = nn.Linear(time_emb_dim, in_channels * 2)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, padding_mode=str(padding_mode)
        )

        self.norm2 = group_norm(int(out_channels), max_groups=32)
        self.time_proj2 = nn.Linear(time_emb_dim, out_channels * 2)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, padding_mode=str(padding_mode)
        )

        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.time_proj1.weight)
        nn.init.zeros_(self.time_proj1.bias)
        nn.init.zeros_(self.time_proj2.weight)
        nn.init.zeros_(self.time_proj2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        h = self.norm1(x)
        style1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale1, shift1 = style1.chunk(2, dim=1)
        h = h * (1 + scale1) + shift1
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        style2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale2, shift2 = style2.chunk(2, dim=1)
        h = h * (1 + scale2) + shift2
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class CondUnet2d(nn.Module):
    """
    Conditional 2D U-Net denoiser.

    - x: noisy target [B, x_ch, H, W]
    - c: condition [B, c_ch, H, W] or [B, c_ch] (broadcast) or [B, c_ch, h', w'] (resized)
    - t: timesteps [B]
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: Optional[int] = None,
        c_dim: int = 1,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4, 8),
        time_emb_dim: int = 32,
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),
        resolution: int = 128,
        activation: str = "SiLU",
        dropout: float = 0.0,
        padding_mode: str = "circular",
    ):
        super().__init__()

        self.output_channels = int(
            output_channels if output_channels is not None else input_channels
        )
        self.c_dim = int(c_dim)
        if self.c_dim <= 0:
            raise ValueError(f"CondUnet2d: `c_dim` must be positive, got {self.c_dim}")

        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            getattr(nn, activation)(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(
            input_channels + self.c_dim,
            dim,
            7,
            padding=3,
            padding_mode=str(padding_mode),
        )

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        curr_res = int(resolution)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            blocks = nn.ModuleList()
            current_dim = dim_in

            for _ in range(int(num_res_blocks)):
                blocks.append(
                    TimeAwareResBlock2d(
                        current_dim,
                        dim_out,
                        time_emb_dim,
                        activation,
                        dropout,
                        padding_mode=padding_mode,
                    )
                )
                current_dim = dim_out
                if curr_res in attn_resolutions:
                    blocks.append(PreNorm(dim_out, AttentionBlock2d(dim_out)))

            self.downs.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "downsample": (
                            Downsample2d(dim_out, dim_out, padding_mode=padding_mode)
                            if not is_last
                            else nn.Conv2d(
                                dim_out,
                                dim_out,
                                3,
                                padding=1,
                                padding_mode=str(padding_mode),
                            )
                        ),
                    }
                )
            )

            if not is_last:
                curr_res //= 2

        mid_dim = dims[-1]
        self.mid_block1 = TimeAwareResBlock2d(
            mid_dim,
            mid_dim,
            time_emb_dim,
            activation,
            dropout,
            padding_mode=padding_mode,
        )
        self.mid_attn = (
            PreNorm(mid_dim, AttentionBlock2d(mid_dim))
            if curr_res in attn_resolutions
            else nn.Identity()
        )
        self.mid_block2 = TimeAwareResBlock2d(
            mid_dim,
            mid_dim,
            time_emb_dim,
            activation,
            dropout,
            padding_mode=padding_mode,
        )

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)

            blocks = nn.ModuleList()
            for i in range(int(num_res_blocks) + 1):
                if i == 0:
                    blocks.append(
                        TimeAwareResBlock2d(
                            dim_out * 2,
                            dim_in,
                            time_emb_dim,
                            activation,
                            dropout,
                            padding_mode=padding_mode,
                        )
                    )
                else:
                    blocks.append(
                        TimeAwareResBlock2d(
                            dim_in,
                            dim_in,
                            time_emb_dim,
                            activation,
                            dropout,
                            padding_mode=padding_mode,
                        )
                    )
                if curr_res in attn_resolutions:
                    blocks.append(PreNorm(dim_in, AttentionBlock2d(dim_in)))

            self.ups.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "upsample": (
                            Upsample2d(dim_in, padding_mode=padding_mode)
                            if not is_last
                            else nn.Identity()
                        ),
                    }
                )
            )

            if not is_last:
                curr_res *= 2

        self.final_res = TimeAwareResBlock2d(
            dim, dim, time_emb_dim, activation, dropout, padding_mode=padding_mode
        )
        self.final_conv = nn.Conv2d(dim, self.output_channels, 1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"CondUnet2d: expected x with shape (B,C,H,W), got {tuple(x.shape)}"
            )
        B, _, H, W = x.shape

        t_emb = self.time_embed(t)

        c_emb = c
        if c_emb.ndim == 2:
            c_emb = c_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        elif c_emb.ndim == 4:
            if c_emb.shape[-2:] != (H, W):
                c_emb = F.interpolate(c_emb, size=(H, W), mode="nearest")
        else:
            raise ValueError(
                f"CondUnet2d: expected c with shape (B,C) or (B,C,H,W), got {tuple(c.shape)}"
            )

        if c_emb.shape[0] != B:
            raise ValueError(
                f"CondUnet2d: batch mismatch: x.B={B} vs c.B={c_emb.shape[0]}"
            )
        if c_emb.shape[1] != self.c_dim:
            raise ValueError(
                f"CondUnet2d: expected c with {self.c_dim} channels, got {c_emb.shape[1]}"
            )

        h = self.init_conv(torch.cat([x, c_emb], dim=1))
        skips = []

        for layer in self.downs:
            for block in layer["blocks"]:
                if isinstance(block, TimeAwareResBlock2d):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            skips.append(h)
            h = layer["downsample"](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for layer in self.ups:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)

            for block in layer["blocks"]:
                if isinstance(block, TimeAwareResBlock2d):
                    h = block(h, t_emb)
                else:
                    h = block(h)

            h = layer["upsample"](h)

        h = self.final_res(h, t_emb)
        return self.final_conv(h)
