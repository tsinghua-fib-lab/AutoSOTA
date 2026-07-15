import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Return a valid GroupNorm for any positive channel count."""
    c = int(num_channels)
    if c <= 0:
        raise ValueError(f"group_norm: num_channels must be positive, got {c}")
    g = min(int(max_groups), c)
    while g > 1 and (c % g) != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=c)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = group_norm(int(dim), max_groups=32)

    def forward(self, x):
        return self.fn(self.norm(x))


class ResBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        dilation: int = 1,
        padding_mode: str = "circular",
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.net = nn.Sequential(
            group_norm(int(in_channels), max_groups=32),
            nn.SiLU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode=str(padding_mode),
            ),
            group_norm(int(out_channels), max_groups=32),
            nn.SiLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode=str(padding_mode),
            ),
        )

        self.proj = nn.Identity()
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.proj(x) + self.net(x)


class AttentionBlock1d(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, l), qkv)

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h d i", attn, v)
        out = out.reshape(b, -1, l)

        return self.to_out(out) + x


def Upsample1d(dim, padding_mode: str = "circular"):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, dim, 3, padding=1, padding_mode=str(padding_mode)),
    )


def Downsample1d(dim_in, dim_out, padding_mode: str = "circular"):
    return nn.Conv1d(dim_in, dim_out, 4, 2, 1, padding_mode=str(padding_mode))


class Unet1d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4, 8),
        padding_mode: str = "circular",
        resolution: int = 256,
        attn_resolutions: tuple | None = None,
        use_attn: bool = True,
    ):
        super().__init__()

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        self.init_conv = nn.Conv1d(
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
                        ResBlock1d(dim_in, dim_in, padding_mode=padding_mode),
                        ResBlock1d(dim_in, dim_in, padding_mode=padding_mode),
                        (
                            Downsample1d(dim_in, dim_out, padding_mode=padding_mode)
                            if not is_last
                            else nn.Conv1d(
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
        self.mid_block1 = ResBlock1d(mid_dim, mid_dim, padding_mode=padding_mode)
        self.mid_attn = (
            PreNorm(mid_dim, AttentionBlock1d(mid_dim))
            if (bool(use_attn) and (coarsest_res in attn_resolutions))
            else nn.Identity()
        )
        self.mid_block2 = ResBlock1d(mid_dim, mid_dim, padding_mode=padding_mode)

        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResBlock1d(dim_out + dim_in, dim_in, padding_mode=padding_mode),
                        ResBlock1d(dim_in, dim_in, padding_mode=padding_mode),
                        (
                            Upsample1d(dim_in, padding_mode=padding_mode)
                            if not is_last
                            else nn.Identity()
                        ),
                    ]
                )
            )

        self.final_res = ResBlock1d(dim, dim, padding_mode=padding_mode)
        self.final_conv = nn.Conv1d(dim, out_channels, 1)

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
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")

            x = torch.cat((x, skip), dim=1)
            x = block1(x)
            x = block2(x)
            x = upsample(x)

        x = self.final_res(x)
        out = self.final_conv(x)
        return out
