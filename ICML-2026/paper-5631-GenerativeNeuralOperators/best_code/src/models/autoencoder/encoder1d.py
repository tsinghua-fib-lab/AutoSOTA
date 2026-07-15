import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unet1d import ResBlock1d, AttentionBlock1d, Downsample1d, PreNorm, group_norm


class Encoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        z_channels: int = 4,
        ch_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: tuple | None = None,
        resolution: int = 256,
        double_z: bool = True,
        dropout: float = 0.0,
        cond_channels: int = 0,
        tanh_out: bool = False,
        padding_mode: str = "circular",
        use_attn: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.z_channels = int(z_channels)
        self.double_z = bool(double_z)
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
        dims = [hidden_channels * m for m in (1,) + ch_mult]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.init_conv = nn.Conv1d(
            in_channels + self.cond_channels,
            hidden_channels,
            3,
            padding=1,
            padding_mode=str(padding_mode),
        )
        self.downs = nn.ModuleList([])
        curr_res = resolution

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            blocks = nn.ModuleList()
            current_dim = dim_in
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock1d(current_dim, dim_out, padding_mode=padding_mode)
                )
                current_dim = dim_out

                if self.use_attn and (curr_res in attn_resolutions):
                    blocks.append(PreNorm(dim_out, AttentionBlock1d(dim_out)))
            level = nn.ModuleDict(
                {
                    "blocks": blocks,
                    "downsample": (
                        Downsample1d(dim_out, dim_out, padding_mode=padding_mode)
                        if not is_last
                        else nn.Identity()
                    ),
                }
            )
            self.downs.append(level)

            if not is_last:
                curr_res //= 2
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock1d(mid_dim, mid_dim, padding_mode=padding_mode)
        self.mid_attn = (
            PreNorm(mid_dim, AttentionBlock1d(mid_dim))
            if self.use_attn
            else nn.Identity()
        )
        self.mid_block2 = ResBlock1d(mid_dim, mid_dim, padding_mode=padding_mode)
        self.norm_out = group_norm(int(mid_dim), max_groups=32)
        self.conv_out = nn.Conv1d(
            mid_dim,
            2 * z_channels if double_z else z_channels,
            3,
            padding=1,
            padding_mode=str(padding_mode),
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of encoder.

        When double_z=True: Returns [B, 2*C_z, L_z] where first half is mean, second half is logvar.
        When double_z=False: Returns [B, C_z, L_z] (deterministic encoding).
        """
        if self.cond_channels > 0:
            if cond is None:
                raise ValueError(
                    "Encoder1d was configured with cond_channels>0, but no cond tensor was provided."
                )
            x = torch.cat([x, cond], dim=1)

        x = self.init_conv(x)

        for level in self.downs:
            for block in level["blocks"]:
                x = block(x)
            x = level["downsample"](x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        if self.tanh_out:
            x = torch.tanh(x)
        return x
