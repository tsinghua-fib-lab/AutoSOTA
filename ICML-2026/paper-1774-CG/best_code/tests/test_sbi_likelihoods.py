"""The SBI task log-likelihoods log p(y|x) (copied from the eval scripts) checked
against independent reference implementations (paper Appendix F)."""

import math

import torch

from experiments.sbi.tasks import (
    gaussian_linear_loglik,
    gaussian_mixture_loglik,
    slcp_loglik,
    two_moons_loglik,
)


def test_gaussian_linear_matches_independent_normal():
    """y ~ N(x, 0.1 I_d): compare to torch.distributions.Normal summed over dims."""
    torch.manual_seed(0)
    theta = torch.randn(3, 5, 10)
    y = torch.randn(10)
    got = gaussian_linear_loglik(theta, y)
    ref = torch.distributions.Normal(theta, math.sqrt(0.1)).log_prob(y).sum(-1)
    torch.testing.assert_close(got.double(), ref.double(), atol=1e-4, rtol=1e-4)
    assert got.shape == (3, 5)


def test_gaussian_mixture_matches_logsumexp_of_components():
    """0.5 N(y;x,I) + 0.5 N(y;x,0.01 I)."""
    torch.manual_seed(0)
    theta = torch.randn(4, 6, 2)
    y = torch.randn(2)
    got = gaussian_mixture_loglik(theta, y)
    c1 = torch.distributions.Normal(theta, 1.0).log_prob(y).sum(-1) + math.log(0.5)
    c2 = torch.distributions.Normal(theta, 0.1).log_prob(y).sum(-1) + math.log(0.5)
    ref = torch.logsumexp(torch.stack([c1, c2], 0), dim=0)
    torch.testing.assert_close(got.double(), ref.double(), atol=1e-4, rtol=1e-4)


def test_slcp_matches_multivariate_normal_over_four_points():
    """SLCP: 4 iid 2D Gaussians with mean (m1,m2) and cov from (s1,s2,rho)."""
    torch.manual_seed(0)
    theta = torch.randn(2, 4, 5)
    y = torch.randn(8)
    got = slcp_loglik(theta, y)

    th = theta.double()
    m = th[..., :2]
    s1 = (th[..., 2] ** 2).clamp_min(1e-3)
    s2 = (th[..., 3] ** 2).clamp_min(1e-3)
    rho = torch.tanh(th[..., 4]).clamp(-0.999, 0.999)
    cov = torch.stack([
        torch.stack([s1 ** 2, rho * s1 * s2], -1),
        torch.stack([rho * s1 * s2, s2 ** 2], -1),
    ], -2)  # [B,K,2,2]
    pts = y.view(4, 2).double()
    mvn = torch.distributions.MultivariateNormal(m, covariance_matrix=cov)
    ref = sum(mvn.log_prob(pts[i]) for i in range(4))
    torch.testing.assert_close(got.double(), ref, atol=1e-3, rtol=1e-3)


def test_two_moons_support_and_density():
    """Two-moons: out-of-support -> -inf-like; in-support -> finite radius density."""
    # An on-manifold point: build x such that y - s(x) is at radius ~0.1, angle 0.
    theta = torch.zeros(1, 1, 2)  # s(theta)=0
    # y = [0.25 + r, 0] with r=0.1 -> zx=0.1, zy=0, rho=0.1, phi=0 (in support)
    y_in = torch.tensor([0.35, 0.0])
    ll_in = two_moons_loglik(theta, y_in)
    assert torch.isfinite(ll_in).all() and ll_in.item() > -50

    # angle outside (-pi/2, pi/2): zx negative -> phi ~ pi -> out of support
    y_out = torch.tensor([0.25 - 0.1, 0.0])  # zx=-0.1 -> phi=pi
    ll_out = two_moons_loglik(theta, y_out)
    assert ll_out.item() <= -1e299  # the -1e300 out-of-support sentinel


def test_likelihoods_return_finite_for_in_support_batches():
    torch.manual_seed(0)
    for fn, dim, ydim in [
        (gaussian_linear_loglik, 10, 10),
        (gaussian_mixture_loglik, 2, 2),
        (slcp_loglik, 5, 8),
    ]:
        theta = 0.3 * torch.randn(2, 16, dim)
        y = 0.3 * torch.randn(ydim)
        out = fn(theta, y)
        assert out.shape == (2, 16) and torch.isfinite(out).all()
