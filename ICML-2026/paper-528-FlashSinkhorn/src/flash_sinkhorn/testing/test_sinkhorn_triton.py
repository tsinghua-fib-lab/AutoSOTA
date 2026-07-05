import pytest
import torch

from flash_sinkhorn.testing.reference_sinkhorn import (
    apply_lse_kernel_ref,
    apply_transport_from_potentials_ref,
    update_potential_ref,
    sinkhorn_potentials_ref,
    sqeuclid_cost,
)
from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_alternating,
)
# Keep low-level kernel imports for LSE and transport tests
from flash_sinkhorn.kernels.sinkhorn_triton_ott_sqeuclid import (
    apply_lse_kernel_sqeuclid,
    apply_transport_from_potentials_sqeuclid,
    update_potential,
)


def _rand_inputs(n, m, d, device):
    torch.manual_seed(0)
    x = torch.randn(n, d, device=device, dtype=torch.float16)
    y = torch.randn(m, d, device=device, dtype=torch.float16)
    f = torch.randn(n, device=device, dtype=torch.float32)
    g = torch.randn(m, device=device, dtype=torch.float32)
    return x, y, f, g


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_apply_lse_kernel_matches_ref(axis):
    device = torch.device("cuda")
    n, m, d = 16, 20, 8
    x, y, f, g = _rand_inputs(n, m, d, device)
    eps = 0.5

    out_triton, sgn_triton = apply_lse_kernel_sqeuclid(x, y, f, g, eps, axis)
    out_ref, sgn_ref = apply_lse_kernel_ref(x, y, f, g, eps, axis)

    torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(sgn_triton, sgn_ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_apply_lse_kernel_exp2_matches_exp(axis):
    device = torch.device("cuda")
    n, m, d = 64, 48, 32
    x, y, f, g = _rand_inputs(n, m, d, device)
    eps = 0.5

    out_exp2, sgn_exp2 = apply_lse_kernel_sqeuclid(
        x, y, f, g, eps, axis, use_exp2=True
    )
    out_exp, sgn_exp = apply_lse_kernel_sqeuclid(
        x, y, f, g, eps, axis, use_exp2=False
    )

    torch.testing.assert_close(out_exp2, out_exp, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(sgn_exp2, sgn_exp, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_update_potential_matches_ref():
    device = torch.device("cuda")
    n, m, d = 16, 12, 8
    x, y, f, g = _rand_inputs(n, m, d, device)
    loga = torch.log(torch.full((n,), 1.0 / n, device=device))
    logb = torch.log(torch.full((m,), 1.0 / m, device=device))
    eps = 0.7

    g_triton = update_potential(x, y, f, g, logb, eps, axis=0)
    g_ref = update_potential_ref(x, y, f, g, logb, eps, axis=0)
    torch.testing.assert_close(g_triton, g_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_sinkhorn_potentials_matches_ref():
    device = torch.device("cuda")
    n, m, d = 12, 10, 8
    x, y, _, _ = _rand_inputs(n, m, d, device)
    a = torch.full((n,), 1.0 / n, device=device)
    b = torch.full((m,), 1.0 / m, device=device)
    loga = torch.log(a)
    logb = torch.log(b)
    eps = 1.0

    # Use new FlashSinkhorn alternating solver with OTT convention
    # to match the reference implementation which absorbs log marginals into potentials
    f_triton, g_triton = sinkhorn_flashstyle_alternating(
        x, y, a, b, eps=eps, n_iters=5, ott_convention=True
    )
    f_ref, g_ref = sinkhorn_potentials_ref(x, y, loga, logb, eps, n_iters=5)

    torch.testing.assert_close(f_triton, f_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(g_triton, g_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_apply_transport_matches_explicit(axis):
    device = torch.device("cuda")
    n, m, d = 10, 12, 8
    x, y, _, _ = _rand_inputs(n, m, d, device)
    loga = torch.log(torch.full((n,), 1.0 / n, device=device))
    logb = torch.log(torch.full((m,), 1.0 / m, device=device))
    eps = 0.8
    f, g = sinkhorn_potentials_ref(x, y, loga, logb, eps, n_iters=4)

    if axis == 1:
        vec = torch.linspace(-1.0, 1.0, m, device=device)
    else:
        vec = torch.linspace(-1.0, 1.0, n, device=device)

    out_triton = apply_transport_from_potentials_sqeuclid(
        x, y, f, g, vec, eps, axis
    )

    cost = sqeuclid_cost(x, y)
    P = torch.exp((f[:, None] + g[None, :] - cost) / eps)
    if axis == 1:
        out_ref = P @ vec
    else:
        out_ref = P.transpose(0, 1) @ vec

    torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
def test_apply_transport_matches_ref(axis):
    device = torch.device("cuda")
    n, m, d = 9, 7, 8
    x, y, _, _ = _rand_inputs(n, m, d, device)
    loga = torch.log(torch.full((n,), 1.0 / n, device=device))
    logb = torch.log(torch.full((m,), 1.0 / m, device=device))
    eps = 0.6
    f, g = sinkhorn_potentials_ref(x, y, loga, logb, eps, n_iters=3)

    if axis == 1:
        vec = torch.randn(m, device=device)
    else:
        vec = torch.randn(n, device=device)

    out_triton = apply_transport_from_potentials_sqeuclid(
        x, y, f, g, vec, eps, axis
    )
    out_ref = apply_transport_from_potentials_ref(x, y, f, g, vec, eps, axis)

    torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)
