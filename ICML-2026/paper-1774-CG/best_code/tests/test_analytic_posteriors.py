"""Tests for the analytic diffusion posteriors p(x0 | xt).

These distributions are the ``p(x0 | xt)`` term that the guided-mean estimators
sample from.  Forward process: ``xt = (1 - t) x0 + t * noise`` with
``a_t = 1 - t``, ``b_t = t``.  We check sample/log-prob shapes, the closed-form
means/variances, and the boundary behaviour (concentration as t -> 0).
"""

import math

import pytest
import torch

from calibrated_guidance.diffusion_posterior.analytic.gaussian import (
    GaussianDiffusionPosterior,
)
from calibrated_guidance.diffusion_posterior.analytic.gaussian_mixture import (
    GaussianMixtureDiffusionPosterior,
)
from calibrated_guidance.diffusion_posterior.analytic.uniform import (
    TruncatedNormal,
    UniformDiffusionPosterior,
)
from calibrated_guidance.diffusion_posterior.base import SampleOutput


# --------------------------------------------------------------------------- #
# GaussianDiffusionPosterior
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
def test_gaussian_posterior_closed_form_mean_var(d, t):
    """diffusion_posterior(xt, t) matches the analytic Gaussian-product form."""
    loc = torch.linspace(-1.0, 1.0, d)
    scale = torch.linspace(0.5, 1.5, d)
    post = GaussianDiffusionPosterior(loc=loc, scale=scale)

    xt = torch.randn(4, d)
    dist = post.diffusion_posterior(xt, t)

    a_t, b_t = 1 - t, t
    var = 1.0 / (a_t**2 / b_t**2 + 1.0 / scale**2)
    mean = var * (a_t / b_t**2 * xt + loc / scale**2)

    torch.testing.assert_close(dist.mean, mean, rtol=1e-5, atol=1e-5)
    # Independent(Normal) variance == base variance, broadcast over batch.
    torch.testing.assert_close(
        dist.variance, var.expand(4, d), rtol=1e-5, atol=1e-5
    )


def test_gaussian_posterior_concentrates_on_xt_as_t_small():
    """As t -> 0 (a_t -> 1) the diffusion posterior concentrates near xt.

    With xt = a_t x0 + b_t noise, when t is tiny xt ~= x0, so E[x0|xt] -> xt and
    the posterior variance -> 0.
    """
    loc = torch.tensor([0.0, 0.0])
    scale = torch.tensor([1.0, 1.0])
    post = GaussianDiffusionPosterior(loc=loc, scale=scale)
    xt = torch.tensor([[0.3, -0.7]])

    dist = post.diffusion_posterior(xt, 1e-3)
    torch.testing.assert_close(dist.mean, xt, rtol=0, atol=1e-2)
    assert dist.variance.max().item() < 1e-4


def test_gaussian_posterior_tweedie_for_standard_normal():
    """For a standard normal prior the posterior mean equals Tweedie's formula.

    For prior N(0, I) the marginal of xt is N(0, a_t^2 + b_t^2), giving
    E[x0 | xt] = a_t / (a_t^2 + b_t^2) * xt.
    """
    d = 2
    post = GaussianDiffusionPosterior(loc=torch.zeros(d), scale=torch.ones(d))
    xt = torch.randn(5, d)
    for t in (0.2, 0.5, 0.8):
        a_t, b_t = 1 - t, t
        tweedie = a_t / (a_t**2 + b_t**2) * xt
        torch.testing.assert_close(
            post.diffusion_posterior(xt, t).mean, tweedie, rtol=1e-5, atol=1e-5
        )


def test_gaussian_posterior_sample_and_logprob_shapes():
    """sample() returns SampleOutput with [B, K, D]; log_prob handles [B,K,D]."""
    d = 3
    B, K = 4, 16
    post = GaussianDiffusionPosterior(loc=torch.zeros(d), scale=torch.ones(d))
    xt = torch.randn(B, d)

    out = post.sample(xt, 0.5, K)
    assert isinstance(out, SampleOutput)
    assert out.samples.shape == (B, K, d)
    # log_probs are produced and laid out [B, K].
    assert out.log_probs.shape == (B, K)
    # Default uniform log-weights, normalised across the K axis (each = -log K).
    torch.testing.assert_close(
        torch.logsumexp(out.log_weights, dim=-1), torch.zeros(B)
    )

    assert post.mean(xt, 0.5).shape == (B, d)

    lp = post.log_prob(out.samples, xt, 0.5)
    assert lp.shape == (B, K)


def test_gaussian_posterior_samples_have_grad_via_rsample():
    """rsample keeps the autograd path open (needed by the reparam estimator)."""
    d = 2
    post = GaussianDiffusionPosterior(loc=torch.zeros(d), scale=torch.ones(d))
    xt = torch.randn(3, d, requires_grad=True)
    out = post.sample(xt, 0.5, 8)
    assert out.samples.requires_grad


# --------------------------------------------------------------------------- #
# GaussianMixtureDiffusionPosterior
# --------------------------------------------------------------------------- #
def _simple_mixture(d=1):
    weights = torch.tensor([0.5, 0.5])
    means = torch.tensor([[-3.0] * d, [3.0] * d])
    scales = torch.tensor([[0.5] * d, [0.5] * d])
    return GaussianMixtureDiffusionPosterior(weights, means, scales)


@pytest.mark.parametrize("d", [1, 2])
def test_mixture_posterior_sample_shapes(d):
    """MixtureSameFamily posterior yields [B, K, D] samples and [B, K] logp."""
    post = _simple_mixture(d)
    B, K = 4, 10
    xt = torch.randn(B, d)
    out = post.sample(xt, 0.5, K)
    assert isinstance(out, SampleOutput)
    assert out.samples.shape == (B, K, d)
    assert out.log_probs.shape == (B, K)
    assert post.mean(xt, 0.5).shape == (B, d)


def test_mixture_weights_form_valid_distribution():
    """The (responsibility) mixing weights are a proper distribution.

    They are non-negative and sum to one along the component axis, exactly as a
    posterior over the latent component should.
    """
    post = _simple_mixture(d=1)
    t = 0.4
    a_t = 1 - t
    # Responsibilities are computed from the marginal N(xt; a_t*mean_i, ...),
    # so to land "inside" a component we place xt at its marginal centre a_t*mean.
    xt = torch.tensor([[a_t * -3.0], [a_t * 3.0], [0.0]])
    dist = post.diffusion_posterior(xt, t)
    probs = dist.mixture_distribution.probs
    assert (probs >= 0).all()
    torch.testing.assert_close(
        probs.sum(-1), torch.ones(probs.shape[0]), rtol=1e-5, atol=1e-5
    )
    # A point on a component's marginal centre is essentially fully explained
    # by that component (the two components are ~7 sigma apart here).
    assert probs[0, 0] > 0.99  # near the -3 component
    assert probs[1, 1] > 0.99  # near the +3 component


def test_mixture_posterior_mean_near_xt_as_t_small():
    """At small t the mixture posterior mean tracks xt (a_t -> 1)."""
    post = _simple_mixture(d=2)
    xt = torch.tensor([[-2.95, -3.05], [2.9, 3.1]])
    mean = post.mean(xt, 1e-3)
    torch.testing.assert_close(mean, xt, rtol=0, atol=1e-2)


# --------------------------------------------------------------------------- #
# TruncatedNormal
# --------------------------------------------------------------------------- #
def test_truncated_normal_samples_within_support():
    """Draws never leave [low, high]."""
    torch.manual_seed(0)
    low = torch.tensor(-1.0)
    high = torch.tensor(2.0)
    tn = TruncatedNormal(torch.tensor(0.5), torch.tensor(3.0), low, high)
    s = tn.sample((10000,))
    assert (s >= low).all() and (s <= high).all()


def test_truncated_normal_logprob_outside_support_is_neg_inf():
    """log_prob is -inf strictly outside [low, high]."""
    tn = TruncatedNormal(
        torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    pts = torch.tensor([-2.0, 2.0, 5.0])
    lp = tn.log_prob(pts)
    assert torch.isneginf(lp).all()


def test_truncated_normal_normalization_matches_normal_over_mass():
    """log_prob == Normal.log_prob - log(cdf(high) - cdf(low)) inside support.

    This is the defining renormalisation of a truncated Gaussian.
    """
    loc, scale = torch.tensor(0.3), torch.tensor(1.2)
    low, high = torch.tensor(-1.0), torch.tensor(1.5)
    tn = TruncatedNormal(loc, scale, low, high)
    normal = torch.distributions.Normal(loc, scale)
    logZ = torch.log(normal.cdf(high) - normal.cdf(low))

    x = torch.tensor([-0.5, 0.0, 0.7, 1.2])
    expected = normal.log_prob(x) - logZ
    torch.testing.assert_close(tn.log_prob(x), expected, rtol=1e-5, atol=1e-5)

    # And the density integrates to ~1 over the support (trapezoid check).
    grid = torch.linspace(low.item(), high.item(), 4001)
    dens = tn.log_prob(grid).exp()
    integral = torch.trapz(dens, grid)
    torch.testing.assert_close(integral, torch.tensor(1.0), rtol=0, atol=1e-3)


# --------------------------------------------------------------------------- #
# UniformDiffusionPosterior
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
def test_uniform_posterior_samples_within_bounds(t):
    """Samples from the uniform-prior diffusion posterior stay in [low, high]."""
    torch.manual_seed(0)
    low = torch.tensor([-1.0, -2.0])
    high = torch.tensor([1.0, 2.0])
    post = UniformDiffusionPosterior(low=low, high=high)
    xt = torch.tensor([[0.0, 0.0], [0.2, -0.3]]) * (1 - t)
    out = post.sample(xt, t, 200)
    assert out.samples.shape == (2, 200, 2)
    assert (out.samples >= low).all()
    assert (out.samples <= high).all()


def test_uniform_posterior_logprob_outside_support_neg_inf():
    """log_prob assigns -inf to points outside the box (truncated normal)."""
    low = torch.tensor([-1.0])
    high = torch.tensor([1.0])
    post = UniformDiffusionPosterior(low=low, high=high)
    xt = torch.tensor([[0.0]]) * 0.5
    dist = post.diffusion_posterior(xt, 0.5)
    # Independent over the (size-1) event dim; outside-support -> -inf.
    out_pt = torch.tensor([[5.0]])
    assert torch.isneginf(dist.log_prob(out_pt)).all()


@pytest.mark.xfail(
    reason="upstream: UniformDiffusionPosterior divides by a_t=1-t=0 at t=1",
    strict=False,
)
def test_uniform_posterior_t1_edge_case_is_well_defined():
    """KNOWN upstream bug: at exactly t=1 the truncated-normal scale b_t/a_t
    divides by a_t=0 and the posterior mean is NaN.

    Documented here as an xfail so the suite stays green while flagging it.
    (Note: the t==1.0 *float* branch returns a Uniform, but the tensor branch
    and the non-degenerate scale arithmetic blow up; we exercise the tensor
    path which is what samplers feed in.)
    """
    low = torch.tensor([-1.0])
    high = torch.tensor([1.0])
    post = UniformDiffusionPosterior(low=low, high=high)
    xt = torch.tensor([[0.0]])
    mean = post.mean(xt, torch.tensor(1.0))
    assert torch.isfinite(mean).all()


# --------------------------------------------------------------------------- #
# AnalyticDiffusionPosterior.sample axis order
# --------------------------------------------------------------------------- #
def test_sample_output_axis_order_is_batch_k_first():
    """AnalyticDiffusionPosterior.sample swapaxes to [B, K, ...].

    rsample produces [K, B, ...]; the wrapper must return batch-first so the
    rest of the pipeline (softmax over dim=1) is correct.
    """
    d = 4
    B, K = 7, 3
    post = GaussianDiffusionPosterior(loc=torch.zeros(d), scale=torch.ones(d))
    xt = torch.randn(B, d)
    out = post.sample(xt, 0.5, K)
    assert out.samples.shape[0] == B
    assert out.samples.shape[1] == K
    assert out.samples.shape[2] == d
