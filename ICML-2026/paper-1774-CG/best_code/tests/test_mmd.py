"""Tests for the (biased) squared MMD in ``calibrated_guidance.mmd``.

MMD^2 is 0 iff the two sample sets are identical (for this biased estimator,
exactly 0 when x is y), small for samples from the same distribution, and
strictly positive for well-separated distributions.  These are the properties
the evaluation code relies on when comparing generated vs. reference samples.
"""

import pytest
import torch

mmd = pytest.importorskip("calibrated_guidance.mmd")

maximum_mean_discrepancy = mmd.maximum_mean_discrepancy
gaussian_kernel_matrix = mmd.gaussian_kernel_matrix


def test_mmd_identical_sets_is_zero():
    """MMD^2(x, x) == 0 exactly for the biased estimator (all three means equal)."""
    torch.manual_seed(0)
    x = torch.randn(64, 3)
    val = maximum_mean_discrepancy(x, x)
    torch.testing.assert_close(val, torch.zeros_like(val), rtol=0, atol=1e-5)


def test_mmd_same_distribution_is_small():
    """Two independent draws from the same distribution give a small MMD^2."""
    torch.manual_seed(0)
    x = torch.randn(256, 2)
    y = torch.randn(256, 2)
    val = maximum_mean_discrepancy(x, y)
    assert val.item() >= -1e-5  # biased estimator can dip very slightly negative
    assert val.item() < 0.5


def test_mmd_separated_sets_is_positive_and_larger():
    """Well-separated sample sets yield a clearly positive MMD^2.

    And the separated-vs-baseline MMD must exceed the same-distribution MMD.
    """
    torch.manual_seed(0)
    x = torch.randn(256, 2)
    y_same = torch.randn(256, 2)
    y_far = torch.randn(256, 2) + 10.0

    mmd_same = maximum_mean_discrepancy(x, y_same).item()
    mmd_far = maximum_mean_discrepancy(x, y_far).item()

    assert mmd_far > 0.0
    assert mmd_far > mmd_same


def test_gaussian_kernel_matrix_shape_and_diagonal():
    """Kernel matrix has shape [N, M]; identical points hit the max (sum of 1s).

    Each of the (default) bandwidths contributes exp(0)=1 on a zero-distance
    pair, so the self-kernel equals the number of bandwidths.
    """
    x = torch.randn(5, 3)
    k = gaussian_kernel_matrix(x, x)
    assert k.shape == (5, 5)
    n_bandwidths = len(mmd.MMD_BANDWIDTH_LIST)
    torch.testing.assert_close(
        torch.diagonal(k),
        torch.full((5,), float(n_bandwidths)),
        rtol=0,
        atol=1e-4,
    )
