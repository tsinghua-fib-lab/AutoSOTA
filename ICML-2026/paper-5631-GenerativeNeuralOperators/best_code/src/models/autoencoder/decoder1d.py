import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unet1d import ResBlock1d, AttentionBlock1d, Upsample1d, PreNorm, group_norm


class Decoder1d(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        hidden_channels: int = 32,
        z_channels: int = 4,
        ch_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: tuple | None = None,
        resolution: int = 256,
        dropout: float = 0.0,
        double_z: bool = False,
        cond_channels: int = 0,
        tanh_out: bool = False,
        padding_mode: str = "circular",
        use_attn: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.z_channels = int(z_channels)
        self.resolution = int(resolution)
        ch_mult = tuple(ch_mult)
        if attn_resolutions is None:
            n_down = max(0, len(ch_mult) - 1)
            attn_resolutions = (int(resolution) // (2**n_down),)
        else:
            attn_resolutions = tuple(attn_resolutions)
        self.cond_channels = int(cond_channels)
        self.tanh_out = bool(tanh_out)
        self.use_attn = bool(use_attn)
        _ = float(dropout)
        _ = bool(double_z)
        dims = [hidden_channels * m for m in (1,) + ch_mult]
        dims_reversed = list(reversed(dims))
        in_out = list(zip(dims_reversed[:-1], dims_reversed[1:]))
        curr_res = resolution // (2 ** (len(ch_mult) - 1))
        block_in = dims_reversed[0]
        self.conv_in = nn.Conv1d(
            z_channels + self.cond_channels,
            block_in,
            3,
            padding=1,
            padding_mode=str(padding_mode),
        )
        self.mid_block1 = ResBlock1d(block_in, block_in, padding_mode=padding_mode)
        self.mid_attn = (
            PreNorm(block_in, AttentionBlock1d(block_in))
            if self.use_attn
            else nn.Identity()
        )
        self.mid_block2 = ResBlock1d(block_in, block_in, padding_mode=padding_mode)
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            blocks = nn.ModuleList()
            current_dim = dim_in
            for _ in range(num_res_blocks + 1):
                blocks.append(
                    ResBlock1d(current_dim, dim_out, padding_mode=padding_mode)
                )
                current_dim = dim_out

                if curr_res in attn_resolutions:
                    blocks.append(PreNorm(dim_out, AttentionBlock1d(dim_out)))

            level = nn.ModuleDict(
                {
                    "blocks": blocks,
                    "upsample": (
                        Upsample1d(dim_out, padding_mode=padding_mode)
                        if not is_last
                        else nn.Identity()
                    ),
                }
            )
            self.ups.append(level)

            if not is_last:
                curr_res *= 2
        self.norm_out = group_norm(int(dims[0]), max_groups=32)
        self.final_conv = nn.Conv1d(
            dims[0],
            out_channels,
            3,
            padding=1,
            padding_mode=str(padding_mode),
        )

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.cond_channels > 0:
            if cond is None:
                raise ValueError(
                    "Decoder1d was configured with cond_channels>0, but no cond tensor was provided."
                )
            z = torch.cat([z, cond], dim=1)

        x = self.conv_in(z)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for level in self.ups:
            for block in level["blocks"]:
                x = block(x)
            x = level["upsample"](x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.final_conv(x)
        if self.tanh_out:
            x = torch.tanh(x)
        return x
