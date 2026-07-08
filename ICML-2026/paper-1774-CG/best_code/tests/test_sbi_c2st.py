"""C2ST metric sanity (needs scikit-learn)."""

import pytest
import torch

pytest.importorskip("sklearn")
from experiments.sbi.c2st import c2st  # noqa: E402


@pytest.mark.slow  # identical dists give no signal -> MLP runs all 10k iters
def test_c2st_identical_distributions_is_chance():
    torch.manual_seed(0)
    X = torch.randn(300, 3)
    Y = torch.randn(300, 3)
    score = float(c2st(X, Y).item())
    assert 0.4 <= score <= 0.62  # indistinguishable -> ~0.5


def test_c2st_separated_distributions_is_high():
    torch.manual_seed(0)
    X = torch.randn(300, 3)
    Y = torch.randn(300, 3) + 5.0  # far apart
    score = float(c2st(X, Y).item())
    assert score > 0.9


def test_c2st_returns_scalar_tensor_in_unit_interval():
    out = c2st(torch.randn(150, 2), torch.randn(150, 2))
    assert isinstance(out, torch.Tensor) and out.numel() == 1
    assert 0.0 <= float(out.item()) <= 1.0
