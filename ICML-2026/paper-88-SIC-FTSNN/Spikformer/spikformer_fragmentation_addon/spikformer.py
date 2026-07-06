from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .fragmentation import entropy_weighted_decode
from .spiking_core import make_multistep_lif_node


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    # torch.nn.init.trunc_normal_ is available in modern PyTorch, but keep a small wrapper.
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class TokenBatchNorm1d(nn.Module):
    """BatchNorm over token channels for inputs shaped [T,B,N,C]."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x [T,B,N,C], got {tuple(x.shape)}")
        t, b, n, c = x.shape
        y = self.bn(x.reshape(t * b * n, c))
        return y.reshape(t, b, n, c)


class MultiStepLinearBNLIF(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        spike_backend: str = "native",
        tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_alpha: float = 4.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = TokenBatchNorm1d(out_features)
        self.lif = make_multistep_lif_node(
            spike_backend=spike_backend,
            tau=tau,
            v_threshold=v_threshold,
            v_reset=0.0,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x [T,B,N,C], got {tuple(x.shape)}")
        t, b, n, c = x.shape
        y = self.linear(x.reshape(t * b * n, c)).reshape(t, b, n, -1)
        y = self.bn(y)
        return self.lif(y)


class MultiStepConvBNLIF(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: bool = False,
        spike_backend: str = "native",
        tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
        surrogate_alpha: float = 4.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = make_multistep_lif_node(
            spike_backend=spike_backend,
            tau=tau,
            v_threshold=v_threshold,
            v_reset=0.0,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected x [T,B,C,H,W], got {tuple(x.shape)}")
        t, b, c, h, w = x.shape
        y = self.conv(x.flatten(0, 1))
        y = self.bn(y)
        y = y.reshape(t, b, -1, y.shape[-2], y.shape[-1]).contiguous()
        y = self.lif(y)
        if self.pool is not None:
            pooled = self.pool(y.flatten(0, 1))
            y = pooled.reshape(t, b, pooled.shape[1], pooled.shape[2], pooled.shape[3]).contiguous()
        return y


class SpikingPatchSplitting(nn.Module):
    """The SPS stem used by Spikformer.

    Parameters
    ----------
    pool_schedule:
        A 4-element boolean tuple. The official ImageNet model pools after all
        4 conv blocks; the official CIFAR model pools only after the last 2.
    """

    def __init__(
        self,
        *,
        image_size: Tuple[int, int],
        in_channels: int,
        embed_dim: int,
        pool_schedule: Tuple[bool, bool, bool, bool],
        spike_backend: str = "native",
        tau: float = 2.0,
        detach_reset: bool = True,
        surrogate_alpha: float = 4.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        if embed_dim % 8 != 0:
            raise ValueError(f"embed_dim must be divisible by 8, got {embed_dim}")
        if len(pool_schedule) != 4:
            raise ValueError("pool_schedule must have length 4")

        c1, c2, c3, c4 = embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim
        self.blocks = nn.ModuleList(
            [
                MultiStepConvBNLIF(
                    in_channels,
                    c1,
                    pool=pool_schedule[0],
                    spike_backend=spike_backend,
                    tau=tau,
                    detach_reset=detach_reset,
                    surrogate_alpha=surrogate_alpha,
                    backend=backend,
                ),
                MultiStepConvBNLIF(
                    c1,
                    c2,
                    pool=pool_schedule[1],
                    spike_backend=spike_backend,
                    tau=tau,
                    detach_reset=detach_reset,
                    surrogate_alpha=surrogate_alpha,
                    backend=backend,
                ),
                MultiStepConvBNLIF(
                    c2,
                    c3,
                    pool=pool_schedule[2],
                    spike_backend=spike_backend,
                    tau=tau,
                    detach_reset=detach_reset,
                    surrogate_alpha=surrogate_alpha,
                    backend=backend,
                ),
                MultiStepConvBNLIF(
                    c3,
                    c4,
                    pool=pool_schedule[3],
                    spike_backend=spike_backend,
                    tau=tau,
                    detach_reset=detach_reset,
                    surrogate_alpha=surrogate_alpha,
                    backend=backend,
                ),
            ]
        )
        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = make_multistep_lif_node(
            spike_backend=spike_backend,
            tau=tau,
            v_threshold=1.0,
            v_reset=0.0,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.image_size = tuple(int(v) for v in image_size)
        self.embed_dim = int(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected x [T,B,C,H,W], got {tuple(x.shape)}")
        y = x
        for block in self.blocks:
            y = block(y)

        t, b, c, h, w = y.shape
        feat = y
        rpe = self.rpe_conv(y.flatten(0, 1))
        rpe = self.rpe_bn(rpe)
        rpe = rpe.reshape(t, b, c, h, w).contiguous()
        rpe = self.rpe_lif(rpe)
        y = feat + rpe
        return y.flatten(-2).transpose(-1, -2).contiguous()  # [T,B,N,C]


class SpikingMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        *,
        spike_backend: str = "native",
        tau: float = 2.0,
        detach_reset: bool = True,
        surrogate_alpha: float = 4.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or dim)
        self.fc1 = MultiStepLinearBNLIF(
            dim,
            hidden_dim,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.fc2 = MultiStepLinearBNLIF(
            hidden_dim,
            dim,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class SpikingSelfAttention(nn.Module):
    """SSA from the Spikformer paper and official implementation.

    The implementation uses spike-form Q/K/V, removes softmax, applies the paper's
    fixed scaling factor (default 0.125), and ends with an additional spike neuron
    after the attention product and after the output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        scale: float = 0.125,
        use_ktv_reordering: Optional[bool] = None,
        spike_backend: str = "native",
        tau: float = 2.0,
        detach_reset: bool = True,
        surrogate_alpha: float = 4.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = float(scale)
        self.use_ktv_reordering = use_ktv_reordering

        self.q = MultiStepLinearBNLIF(
            dim,
            dim,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.k = MultiStepLinearBNLIF(
            dim,
            dim,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.v = MultiStepLinearBNLIF(
            dim,
            dim,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.attn_lif = make_multistep_lif_node(
            spike_backend=spike_backend,
            tau=tau,
            v_threshold=0.5,
            v_reset=0.0,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.proj = MultiStepLinearBNLIF(
            dim,
            dim,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        t, b, n, c = x.shape
        return x.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x [T,B,N,C], got {tuple(x.shape)}")
        t, b, n, c = x.shape
        q = self._reshape_heads(self.q(x))
        k = self._reshape_heads(self.k(x))
        v = self._reshape_heads(self.v(x))

        use_ktv = self.use_ktv_reordering
        if use_ktv is None:
            use_ktv = n > self.head_dim

        if use_ktv:
            kv = k.transpose(-2, -1) @ v              # [T,B,H,d,d]
            attn_out = (q @ kv) * self.scale          # [T,B,H,N,d]
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [T,B,H,N,N]
            attn_out = attn @ v                             # [T,B,H,N,d]

        out = attn_out.permute(0, 1, 3, 2, 4).reshape(t, b, n, c).contiguous()
        out = self.attn_lif(out)
        out = self.proj(out)
        return out


class SpikformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        scale: float = 0.125,
        use_ktv_reordering: Optional[bool] = None,
        spike_backend: str = "native",
        tau: float = 2.0,
        detach_reset: bool = True,
        surrogate_alpha: float = 4.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.attn = SpikingSelfAttention(
            dim,
            num_heads,
            scale=scale,
            use_ktv_reordering=use_ktv_reordering,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.mlp = SpikingMLP(
            dim,
            hidden_dim,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Spikformer(nn.Module):
    """Paper-faithful Spikformer implementation with a clean modern API.

    Notes
    -----
    - No LayerNorm is used; the paper and official repo switch to BatchNorm.
    - No softmax is used inside SSA.
    - Static images are converted to a T-step sequence by repeating them.
    - `forward_sequence` accepts an externally created sequence, which makes the
      model directly compatible with learnable fragmentation add-ons.
    """

    def __init__(
        self,
        *,
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        time_steps: int = 4,
        scale: float = 0.125,
        pool_schedule: Tuple[bool, bool, bool, bool] = (True, True, True, True),
        use_ktv_reordering: Optional[bool] = None,
        spike_backend: str = "native",
        tau: float = 2.0,
        detach_reset: bool = True,
        surrogate_alpha: float = 4.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.image_size = tuple(int(v) for v in image_size)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.depth = int(depth)
        self.mlp_ratio = float(mlp_ratio)
        self.time_steps = int(time_steps)
        self.scale = float(scale)
        self.pool_schedule = tuple(bool(v) for v in pool_schedule)

        self.patch_embed = SpikingPatchSplitting(
            image_size=self.image_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            pool_schedule=self.pool_schedule,
            spike_backend=spike_backend,
            tau=tau,
            detach_reset=detach_reset,
            surrogate_alpha=surrogate_alpha,
            backend=backend,
        )
        self.blocks = nn.ModuleList(
            [
                SpikformerBlock(
                    self.embed_dim,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    scale=self.scale,
                    use_ktv_reordering=use_ktv_reordering,
                    spike_backend=spike_backend,
                    tau=tau,
                    detach_reset=detach_reset,
                    surrogate_alpha=surrogate_alpha,
                    backend=backend,
                )
                for _ in range(self.depth)
            ]
        )
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x_seq: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x_seq)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.mean(dim=2)  # [T,B,C]

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        *,
        decode: Optional[str] = "mean",
        gamma: float = 1.0,
        return_logits_seq: bool = False,
    ):
        features_seq = self.forward_features(x_seq)          # [T,B,C]
        logits_seq = self.head(features_seq)                 # [T,B,K]
        if return_logits_seq:
            return logits_seq

        if decode is None:
            return logits_seq

        decode = decode.lower().strip()
        if decode == "mean":
            return logits_seq.mean(dim=0)
        if decode == "entropy":
            return entropy_weighted_decode(logits_seq, gamma=gamma)
        raise ValueError(f"Unknown decode mode: {decode!r}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        decode: Optional[str] = "mean",
        gamma: float = 1.0,
        return_logits_seq: bool = False,
    ):
        if x.dim() != 4:
            raise ValueError(f"Expected x [B,C,H,W], got {tuple(x.shape)}")
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        return self.forward_sequence(x_seq, decode=decode, gamma=gamma, return_logits_seq=return_logits_seq)


PAPER_PRESETS: Dict[str, Dict[str, object]] = {
    # ImageNet variants listed in Table 2 of the paper.
    "spikformer-8-384": dict(embed_dim=384, depth=8, num_heads=8, mlp_ratio=4.0, time_steps=4, pool_schedule=(True, True, True, True)),
    "spikformer-6-512": dict(embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.0, time_steps=4, pool_schedule=(True, True, True, True)),
    "spikformer-8-512": dict(embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, time_steps=4, pool_schedule=(True, True, True, True)),
    "spikformer-10-512": dict(embed_dim=512, depth=10, num_heads=8, mlp_ratio=4.0, time_steps=4, pool_schedule=(True, True, True, True)),
    "spikformer-8-768": dict(embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0, time_steps=4, pool_schedule=(True, True, True, True)),
    # Practical CIFAR-style variant aligned to the official CIFAR code path.
    "spikformer-cifar": dict(embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0, time_steps=4, pool_schedule=(False, False, True, True)),
}


def build_spikformer_preset(
    preset: str,
    *,
    image_size: Tuple[int, int],
    in_channels: int,
    num_classes: int,
    spike_backend: str = "native",
    backend: str = "torch",
    **overrides,
) -> Spikformer:
    preset_key = preset.lower().strip()
    if preset_key not in PAPER_PRESETS:
        raise ValueError(f"Unknown preset={preset!r}. Available: {sorted(PAPER_PRESETS.keys())}")
    cfg = dict(PAPER_PRESETS[preset_key])
    cfg.update(overrides)
    return Spikformer(
        image_size=image_size,
        in_channels=in_channels,
        num_classes=num_classes,
        spike_backend=spike_backend,
        backend=backend,
        **cfg,
    )


__all__ = [
    "Spikformer",
    "build_spikformer_preset",
    "PAPER_PRESETS",
]
