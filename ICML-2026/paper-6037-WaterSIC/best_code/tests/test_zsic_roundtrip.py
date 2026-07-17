"""Pure-CPU smoke tests for ZSIC helpers.

The full `compress_zsic_with_binary_search` pipeline has GPU-unconditional
sync calls and is exercised by real eval runs on the actual models. Here we
test the pure-CPU utility functions that the pipeline composes — those are
the ones a refactor / cleanup is most likely to break silently.
"""
import math

import torch

from quant_layerwise.methods.zsic import (
    ZSICConfig,
    compute_entropy,
    find_dead_dimensions,
)


def test_zsicconfig_default_construction():
    """Config dataclass instantiates with required + default fields."""
    cfg = ZSICConfig(target_rate_bits=3.0)
    assert cfg.target_rate_bits == 3.0
    # Defaults that the pipeline relies on
    assert cfg.binary_search is False
    assert cfg.qronos is False
    assert cfg.residual_compensation is False
    assert cfg.apply_rescaler is True
    assert cfg.dead_dim_threshold == 0.001


def test_compute_entropy_uniform():
    """Entropy of a uniform 2-symbol distribution is exactly 1 bit."""
    z = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int64)
    assert math.isclose(compute_entropy(z), 1.0, abs_tol=1e-6)


def test_compute_entropy_skewed():
    """Entropy of a 4-symbol uniform is exactly 2 bits."""
    z = torch.arange(64, dtype=torch.int64) % 4
    assert math.isclose(compute_entropy(z), 2.0, abs_tol=1e-6)


def test_compute_entropy_constant_zero():
    """Constant (single-symbol) distribution → entropy 0."""
    z = torch.zeros(100, dtype=torch.int64)
    assert math.isclose(compute_entropy(z), 0.0, abs_tol=1e-6)


def test_find_dead_dimensions_flags_zeros():
    """Dimensions with near-zero diagonal variance are flagged dead; live ones are not."""
    n = 10
    Sig = torch.eye(n, dtype=torch.float64)
    Sig[0, 0] = 1e-12   # explicitly dead
    Sig[3, 3] = 1e-12   # explicitly dead
    Sig[7, 7] = -1e-8   # negative — should also count as dead
    mask = find_dead_dimensions(Sig, threshold_ratio=1e-3)
    assert mask.shape == (n,)
    assert mask.dtype == torch.bool
    assert mask[0].item() is True
    assert mask[3].item() is True
    assert mask[7].item() is True
    # Live dims with diag=1 are well above 1e-3 * median(=1) → not flagged
    assert mask[1].item() is False
    assert mask[5].item() is False
    # Count matches
    assert int(mask.sum()) == 3


def test_find_dead_dimensions_respects_median_not_mean():
    """A few high-variance dims should not pull the threshold up enough to flag healthy
    dims as dead (the very motivation for using median, not mean, in find_dead_dimensions)."""
    n = 20
    diag = torch.ones(n, dtype=torch.float64)
    diag[0] = 1e6  # one huge outlier (mean → ~50000, median → 1)
    Sig = torch.diag(diag)
    mask = find_dead_dimensions(Sig, threshold_ratio=1e-3)
    # If we used mean, threshold would be ~50, and most diag=1 dims would be flagged.
    # With median, threshold is ~1e-3, so only the outlier escapes (n_dead=0).
    assert int(mask.sum()) == 0
