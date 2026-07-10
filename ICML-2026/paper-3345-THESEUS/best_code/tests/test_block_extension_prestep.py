from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.block_extension import (
    BlockExtensionConfig,
    InputAlignedBlock,
    InputAlignedFinalLayer,
    resolve_block_extension_config,
    run_block_extension,
    select_loader,
)


class _TinyAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(x)


class _TinyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.relu(self.fc(x)))


class _TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = _TinyAttn(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = _TinyMLP(dim)

    def forward(self, x: torch.Tensor, attn_mask=None, **kwargs):
        del attn_mask, kwargs
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _TinyVisual(nn.Module):
    def __init__(self, in_dim: int = 6, width: int = 8, depth: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_TinyBlock(width) for _ in range(depth)])
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, 4, 1)
        x = self.input_proj(x)
        for block in self.transformer.resblocks:
            x = block(x)
        x = self.ln_post(x)
        return x.mean(dim=1)


class _TinyModel(nn.Module):
    def __init__(self, in_dim: int = 6, width: int = 8, depth: int = 3):
        super().__init__()
        self.visual = _TinyVisual(in_dim=in_dim, width=width, depth=depth)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _make_loader(n_samples: int = 16, in_dim: int = 6, batch_size: int = 4):
    x = torch.randn(n_samples, in_dim)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_resolve_block_extension_config_defaults() -> None:
    enabled, cfg = resolve_block_extension_config({})
    assert not enabled
    assert isinstance(cfg, BlockExtensionConfig)
    assert cfg.extension_strategy == "interpolate"
    assert cfg.insertion_order == "bottom-top"


def test_select_loader_split_precedence() -> None:
    train = object()
    test = object()
    val = object()

    assert select_loader("train", train, test, val) is train
    assert select_loader("val", train, test, val) is val
    assert select_loader("val", train, test, None) is test
    assert select_loader("test", train, test, val) is test


def test_run_block_extension_increases_depth_and_wraps_modules() -> None:
    source_base = _TinyModel(depth=3)
    source_ft = _TinyModel(depth=3)
    loader = _make_loader()

    cfg = BlockExtensionConfig(
        blocks_to_add=2,
        insertion_order="bottom-top",
        extension_density="spread",
        extension_strategy="duplicate",
        dampening_factor=1.0,
        n_batches_act=1,
        skip_correction=True,
        skip_final_ln=True,
    )

    final_depth = run_block_extension(
        source_base_model=source_base,
        source_ft_model=source_ft,
        calibration_loader=loader,
        target_layers_total=None,
        config=cfg,
        device="cpu",
    )

    assert final_depth == 5
    assert len(source_base.visual.transformer.resblocks) == 5
    assert len(source_ft.visual.transformer.resblocks) == 5
    assert isinstance(source_base.visual.transformer.resblocks[0], InputAlignedBlock)
    assert isinstance(source_base.visual.ln_post, InputAlignedFinalLayer)
