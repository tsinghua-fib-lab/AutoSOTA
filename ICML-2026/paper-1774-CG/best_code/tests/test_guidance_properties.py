"""Property-based tests for the guided-mean estimators (hypothesis).

These encode invariants that must hold for *any* inputs, complementing the
fixed-example unit tests in ``test_guidance_core.py``:

* the REINFORCE guided mean is a convex combination of the x0 samples, so it can
  never leave their per-coordinate convex hull (bounding box), and
* a uniform (constant) log-likelihood leaves the plain sample mean unchanged.

Examples are small and the deadline is disabled to keep them fast and robust on
CI hardware.
"""

import hypothesis
import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from calibrated_guidance import build_reinforce_mean_fn
from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput


class FixedSamplePosterior(DiffusionPosterior):
    """Returns a fixed [B, K, D] set of x0 samples (no Monte-Carlo noise)."""

    def __init__(self, samples: torch.Tensor):
        self._samples = samples

    def sample(self, xt, t, num_samples) -> SampleOutput:
        return SampleOutput(self._samples)

    def mean(self, xt, t):
        return self._samples.mean(dim=1)


_SETTINGS = settings(max_examples=25, deadline=None)


@_SETTINGS
@given(
    B=st.integers(min_value=1, max_value=3),
    K=st.integers(min_value=2, max_value=6),
    D=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_reinforce_mean_in_convex_hull(B, K, D, seed):
    """The guided mean lies in the per-coordinate bounding box of the samples.

    softmax weights are non-negative and sum to 1, so the weighted average is a
    convex combination -> bounded by the coordinate-wise min/max of the samples.
    """
    torch.manual_seed(seed)
    samples = torch.randn(B, K, D)
    logits = torch.randn(B, K)  # arbitrary log-likelihoods
    post = FixedSamplePosterior(samples)

    mean_fn = build_reinforce_mean_fn(post, lambda x: logits, num_samples_per_step=K)
    guided = mean_fn(torch.zeros(B, D), 0.5)

    lo = samples.min(dim=1).values
    hi = samples.max(dim=1).values
    # Allow a tiny epsilon for float round-off at the hull boundary.
    assert (guided >= lo - 1e-5).all()
    assert (guided <= hi + 1e-5).all()
    assert guided.shape == (B, D)


@_SETTINGS
@given(
    B=st.integers(min_value=1, max_value=3),
    K=st.integers(min_value=2, max_value=8),
    D=st.integers(min_value=1, max_value=4),
    const=st.floats(min_value=-5.0, max_value=5.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_reinforce_uniform_likelihood_is_plain_mean(B, K, D, const, seed):
    """A constant log-likelihood -> uniform softmax -> plain sample mean."""
    torch.manual_seed(seed)
    samples = torch.randn(B, K, D)
    post = FixedSamplePosterior(samples)

    def const_loglik(x):
        return torch.full((x.shape[0], x.shape[1]), const)

    mean_fn = build_reinforce_mean_fn(post, const_loglik, num_samples_per_step=K)
    guided = mean_fn(torch.zeros(B, D), 0.5)
    torch.testing.assert_close(guided, samples.mean(dim=1), rtol=1e-5, atol=1e-5)


@_SETTINGS
@given(
    scale=st.floats(min_value=1.0, max_value=50.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_guidance_scale_monotonically_sharpens(scale, seed):
    """Larger guidance scale moves the mean at least as far toward the argmax.

    With two samples and a likelihood preferring the second, increasing the
    guidance scale never moves the guided mean *away* from that preferred sample.
    """
    torch.manual_seed(seed)
    samples = torch.tensor([[[0.0], [1.0]]])  # x=0 vs x=1
    post = FixedSamplePosterior(samples)

    def loglik(x):
        return x.sum(-1)  # prefers x=1

    base = build_reinforce_mean_fn(post, loglik, 2, guidance_scale=1.0)(
        torch.zeros(1, 1), 0.5
    ).item()
    sharp = build_reinforce_mean_fn(post, loglik, 2, guidance_scale=1.0 + scale)(
        torch.zeros(1, 1), 0.5
    ).item()
    assert sharp >= base - 1e-6
    assert 0.0 <= sharp <= 1.0
