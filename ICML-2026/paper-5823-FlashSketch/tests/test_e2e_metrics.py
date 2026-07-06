from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import pytest
import torch

from bench.e2e.metrics import gram_errors, ose_errors, relative_residual

pytestmark = pytest.mark.small


def test_relative_residual_zero_denom():
    """Relative residual should fall back to absolute residual when ||b|| is zero."""
    device = torch.device("cpu")
    A = torch.eye(3, device=device)
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.zeros(3, device=device)

    residual = relative_residual(A, x, b)
    assert abs(residual - torch.linalg.norm(x).item()) < 1e-6


def test_gram_errors_zero():
    """Gram errors should be zero when SA equals A."""
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    A = torch.randn((5, 3), generator=generator, device=device)

    metrics = gram_errors(A, A)
    assert metrics["gram_fro_error"] < 1e-6
    assert metrics["gram_fro_rel"] < 1e-6
    assert metrics["gram_spec_error"] < 1e-6
    assert metrics["gram_spec_rel"] < 1e-6


def test_ose_errors_identity():
    """OSE errors should be zero for an orthonormal subspace."""
    device = torch.device("cpu")
    SQ = torch.eye(4, device=device)
    metrics = ose_errors(SQ)
    assert metrics["ose_fro_err"] < 1e-6
    assert metrics["ose_spec_err"] < 1e-6
    assert metrics["ose_max_sv_dev"] < 1e-6
