import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unet1d import ResBlock1d
from ..unet2d import ResBlock2d

try:
    from neuralop.models import FNO
except ImportError:
    raise ImportError("neuralop is required. Install with: pip install neuralop")


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1 if half_dim > 1 else 1)
        self.register_buffer(
            "emb", torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.float()[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


class CnnEmbedding1d(nn.Module):
    """
    1D Convolutional Embedder using Pre-Norm Residual Blocks.
    Input: (B, Length) or (B, Channels, Length)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        use_coords: bool = False,
        padding_mode: str = "circular",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_coords = use_coords

        if hidden_dim < 32:
            raise ValueError("hidden_dim must be >= 32 for GroupNorm(32).")
        effective_in_dim = input_dim + 1 if use_coords else input_dim
        self.input_proj = nn.Conv1d(effective_in_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2**i
            self.blocks.append(
                ResBlock1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding_mode=padding_mode,
                )
            )
        self.blocks = nn.Sequential(*self.blocks)
        self.post_norm = nn.GroupNorm(32, hidden_dim)
        self.post_act = nn.SiLU()
        self.final_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.use_coords:
            batch, _, length = x.shape
            grid = torch.linspace(0, 1, length, device=x.device, dtype=x.dtype)
            grid = grid.view(1, 1, length).expand(batch, 1, length)
            x = torch.cat([x, grid], dim=1)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.post_act(self.post_norm(x))
        pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.final_proj(pooled)


class FnoEmbedding1d(nn.Module):
    """
    Fourier Neural Operator (FNO) based Embedder.

    Uses neuralop FNO followed by average pooling to get embeddings.

    Input: (B, Length) or (B, Channels, Length)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        n_modes: int | tuple[int, ...] = (32,),
        n_layers: int = 4,
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if isinstance(n_modes, (list, tuple)) or (
            hasattr(n_modes, "__iter__") and not isinstance(n_modes, (str, bytes))
        ):
            modes_tuple = tuple(int(x) for x in n_modes)
        else:
            modes_tuple = (int(n_modes),)
        self.fno = FNO(
            n_modes=modes_tuple,
            hidden_channels=hidden_dim,
            in_channels=input_dim,
            out_channels=output_dim,  # Output directly to embedding dimension
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.fno(x)
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)


class CnnEmbedding2d(nn.Module):
    """
    2D Convolutional Embedder using 2D ResBlocks.
    Input: (B, H, W) or (B, C, H, W)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        use_coords: bool = False,
        padding_mode: str = "circular",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_coords = bool(use_coords)

        if hidden_dim < 32:
            raise ValueError("hidden_dim must be >= 32 for GroupNorm(32).")

        effective_in_dim = self.input_dim + (2 if self.use_coords else 0)
        self.input_proj = nn.Conv2d(effective_in_dim, hidden_dim, kernel_size=1)

        blocks = []
        for _ in range(int(num_blocks)):
            blocks.append(ResBlock2d(hidden_dim, hidden_dim, padding_mode=padding_mode))
        self.blocks = nn.Sequential(*blocks)

        self.post_norm = nn.GroupNorm(32, hidden_dim)
        self.post_act = nn.SiLU()
        self.final_proj = nn.Linear(hidden_dim, self.output_dim)

    def _make_coords(
        self, batch: int, height: int, width: int, device, dtype
    ) -> torch.Tensor:
        xs = torch.linspace(0, 1, width, device=device, dtype=dtype)
        ys = torch.linspace(0, 1, height, device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")  # (H, W) each
        grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)  # (1, 2, H, W)
        return grid.expand(batch, 2, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"CnnEmbedding2d expected x with shape (B,C,H,W) or (B,H,W), got {tuple(x.shape)}"
            )

        if self.use_coords:
            b, _, h, w = x.shape
            coords = self._make_coords(b, h, w, x.device, x.dtype)
            x = torch.cat([x, coords], dim=1)

        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.post_act(self.post_norm(x))

        pooled = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (B, hidden_dim)
        return self.final_proj(pooled)


class FnoEmbedding2d(nn.Module):
    """
    2D FNO-based embedder.
    Uses neuralop FNO followed by average pooling to get embeddings.

    Input: (B, H, W) or (B, C, H, W)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        n_modes: int | tuple[int, int] = (32, 32),
        n_layers: int = 4,
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if isinstance(n_modes, (list, tuple)) or (
            hasattr(n_modes, "__iter__") and not isinstance(n_modes, (str, bytes))
        ):
            modes_list = [int(x) for x in n_modes]
            if len(modes_list) == 1:
                modes_tuple = (modes_list[0], modes_list[0])
            elif len(modes_list) == 2:
                modes_tuple = (modes_list[0], modes_list[1])
            else:
                raise ValueError(
                    f"FnoEmbedding2d expected n_modes as int or (2,) / (2,) sequence, got {modes_list}"
                )
        else:
            n = int(n_modes)
            modes_tuple = (n, n)
        self.fno = FNO(
            n_modes=modes_tuple,
            hidden_channels=int(hidden_dim),
            in_channels=self.input_dim,
            out_channels=self.output_dim,  # Output directly to embedding dimension
            n_layers=int(n_layers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"FnoEmbedding2d expected x with shape (B,C,H,W) or (B,H,W), got {tuple(x.shape)}"
            )
        x = self.fno(x)
        return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)


class FnoMlpEmbedding2d(nn.Module):
    """
    2D FNO-based embedder with a pooled MLP head.

    Uses neuralop FNO (n_layers=2 by default), global average pooling,
    then a single MLP block to output embeddings.

    Input: (B, H, W) or (B, C, H, W)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        n_modes: int | tuple[int, int] = (32, 32),
        n_layers: int = 2,
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if isinstance(n_modes, (list, tuple)) or (
            hasattr(n_modes, "__iter__") and not isinstance(n_modes, (str, bytes))
        ):
            modes_list = [int(x) for x in n_modes]
            if len(modes_list) == 1:
                modes_tuple = (modes_list[0], modes_list[0])
            elif len(modes_list) == 2:
                modes_tuple = (modes_list[0], modes_list[1])
            else:
                raise ValueError(
                    f"FnoMlpEmbedding2d expected n_modes as int or (2,) / (2,) sequence, got {modes_list}"
                )
        else:
            n = int(n_modes)
            modes_tuple = (n, n)

        self.fno = FNO(
            n_modes=modes_tuple,
            hidden_channels=int(hidden_dim),
            in_channels=self.input_dim,
            out_channels=int(hidden_dim),
            n_layers=int(n_layers),
        )

        self.mlp = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"FnoMlpEmbedding2d expected x with shape (B,C,H,W) or (B,H,W), got {tuple(x.shape)}"
            )

        x = self.fno(x)
        pooled = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return self.mlp(pooled)


class MlpEmbedding2d(nn.Module):
    """
    2D flatten-then-MLP embedder.

    Intended for sparse observation maps where a simple vectorized representation
    is preferable to convolutional/Fourier encoders.

    Input: (B, H, W) or (B, C, H, W)
    Output: (B, output_dim)
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        height: int = 64,
        width: int = 64,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.height = int(height)
        self.width = int(width)

        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"height/width must be positive, got ({self.height}, {self.width})"
            )

        in_features = self.input_dim * self.height * self.width
        dims = [in_features, *[int(h) for h in hidden_dims], self.output_dim]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if float(dropout) > 0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"MlpEmbedding2d expected x with shape (B,C,H,W) or (B,H,W), got {tuple(x.shape)}"
            )

        b, c, h, w = x.shape
        if c != self.input_dim:
            raise ValueError(
                f"MlpEmbedding2d expected input_dim={self.input_dim}, got x.shape={tuple(x.shape)}"
            )
        if h != self.height or w != self.width:
            raise ValueError(
                f"MlpEmbedding2d expected spatial size ({self.height}, {self.width}), got ({h}, {w})"
            )

        x_flat = x.view(b, -1)
        return self.net(x_flat)
