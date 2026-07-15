import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet1d import PreNorm, group_norm


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode: str = "circular"):
        super().__init__()
        self.net = nn.Sequential(
            group_norm(int(in_channels), max_groups=32),
            nn.SiLU(),
            nn.Conv2d(
                in_channels, out_channels, 3, padding=1, padding_mode=str(padding_mode)
            ),
            group_norm(int(out_channels), max_groups=32),
            nn.SiLU(),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=1, padding_mode=str(padding_mode)
            ),
        )

        self.proj = nn.Identity()
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.proj(x) + self.net(x)


class AttentionBlock2d(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h d i", attn, v)
        out = out.reshape(b, -1, h, w)

        return self.to_out(out) + x


def Upsample2d(dim, padding_mode: str = "circular"):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim, 3, padding=1, padding_mode=str(padding_mode)),
    )


def Downsample2d(dim_in, dim_out, padding_mode: str = "circular"):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1, padding_mode=str(padding_mode))


class Unet2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4, 8),
        padding_mode: str = "circular",
        resolution: int = 128,
        attn_resolutions: tuple | None = None,
        use_attn: bool = True,
    ):
        super().__init__()

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        self.init_conv = nn.Conv2d(
            in_channels, dim, 7, padding=3, padding_mode=str(padding_mode)
        )
        dim_mults = tuple(dim_mults)
        if attn_resolutions is None:
            n_down = max(0, len(dim_mults) - 1)
            attn_resolutions = (int(resolution) // (2**n_down),)
        else:
            attn_resolutions = tuple(attn_resolutions)
        coarsest_res = int(resolution) // (2 ** max(0, len(dim_mults) - 1))

        self.downs = nn.ModuleList([])
        in_out = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResBlock2d(dim_in, dim_in, padding_mode=padding_mode),
                        ResBlock2d(dim_in, dim_in, padding_mode=padding_mode),
                        (
                            Downsample2d(dim_in, dim_out, padding_mode=padding_mode)
                            if not is_last
                            else nn.Conv2d(
                                dim_in,
                                dim_out,
                                3,
                                padding=1,
                                padding_mode=str(padding_mode),
                            )
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResBlock2d(mid_dim, mid_dim, padding_mode=padding_mode)
        self.mid_attn = (
            PreNorm(mid_dim, AttentionBlock2d(mid_dim))
            if (bool(use_attn) and (coarsest_res in attn_resolutions))
            else nn.Identity()
        )
        self.mid_block2 = ResBlock2d(mid_dim, mid_dim, padding_mode=padding_mode)

        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResBlock2d(dim_out + dim_in, dim_in, padding_mode=padding_mode),
                        ResBlock2d(dim_in, dim_in, padding_mode=padding_mode),
                        (
                            Upsample2d(dim_in, padding_mode=padding_mode)
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        self.final_res = ResBlock2d(dim, dim, padding_mode=padding_mode)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x):
        x = self.init_conv(x)
        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, upsample in self.ups:
            skip = h.pop()

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

            x = torch.cat((x, skip), dim=1)
            x = block1(x)
            x = block2(x)
            x = upsample(x)

        x = self.final_res(x)
        out = self.final_conv(x)
        return out
