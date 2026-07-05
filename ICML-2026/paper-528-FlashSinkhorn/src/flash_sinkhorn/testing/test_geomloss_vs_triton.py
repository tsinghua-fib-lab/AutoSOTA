import pytest
import torch

from flash_sinkhorn.kernels.sinkhorn_triton_ott_sqeuclid import apply_lse_kernel_sqeuclid


geomloss = pytest.importorskip("geomloss")
from geomloss.sinkhorn_samples import softmin_tensorized  # noqa: E402


def _sqeuclid_cost(x, y):
    x_f = x.float()
    y_f = y.float()
    x2 = (x_f * x_f).sum(dim=1, keepdim=True)
    y2 = (y_f * y_f).sum(dim=1, keepdim=True).transpose(0, 1)
    return x2 + y2 - 2.0 * x_f @ y_f.transpose(0, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_vec", [False, True])
def test_apply_lse_kernel_matches_geomloss_softmin(axis, with_vec):
    device = torch.device("cuda")
    torch.manual_seed(0)

    n, m, d = 32, 24, 16
    eps = 0.7
    x = torch.randn(n, d, device=device, dtype=torch.float16)
    y = torch.randn(m, d, device=device, dtype=torch.float16)
    f = torch.randn(n, device=device, dtype=torch.float32)
    g = torch.randn(m, device=device, dtype=torch.float32)

    if axis == 1:
        vec_len = m
        other_potential = g
    else:
        vec_len = n
        other_potential = f

    vec = None
    if with_vec:
        vec = torch.rand(vec_len, device=device, dtype=torch.float32) + 0.1

    out_triton, sgn_triton = apply_lse_kernel_sqeuclid(
        x, y, f, g, eps, axis=axis, vec=vec
    )

    if with_vec:
        logw = vec.log()
    else:
        logw = torch.zeros(vec_len, device=device, dtype=torch.float32)

    C = _sqeuclid_cost(x, y)
    h = logw + other_potential / eps

    if axis == 1:
        smin = softmin_tensorized(eps, C.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
    else:
        smin = softmin_tensorized(
            eps, C.transpose(0, 1).unsqueeze(0), h.unsqueeze(0)
        ).squeeze(0)

    torch.testing.assert_close(out_triton, -smin, rtol=1e-3, atol=1e-3)
    if with_vec:
        torch.testing.assert_close(
            sgn_triton, torch.ones_like(sgn_triton), rtol=0, atol=0
        )

