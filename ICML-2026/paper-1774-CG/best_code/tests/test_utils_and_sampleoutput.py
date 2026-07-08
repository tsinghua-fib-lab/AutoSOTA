"""Tests for ``broadcast_time`` and the ``SampleOutput`` container."""

import pytest
import torch

from calibrated_guidance.diffusion_posterior.base import SampleOutput
from calibrated_guidance.utils import broadcast_time


# --------------------------------------------------------------------------- #
# broadcast_time
# --------------------------------------------------------------------------- #
def test_broadcast_time_float_passes_through():
    """A plain float t is returned unchanged (scalar broadcasting handles it)."""
    x = torch.randn(4, 3, 8, 8)
    assert broadcast_time(0.3, x) == 0.3


def test_broadcast_time_scalar_tensor_passes_through():
    """A 0-dim tensor t (shape ()) is returned unchanged."""
    x = torch.randn(2, 5)
    t = torch.tensor(0.7)
    out = broadcast_time(t, x)
    assert out is t
    assert out.shape == ()


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_broadcast_time_batch_vector_broadcasts(ndim):
    """A [B] time vector gains trailing singleton dims to match x.dim()."""
    shape = (6,) + (3,) * (ndim - 1)
    x = torch.randn(*shape)
    t = torch.arange(6, dtype=torch.float32)
    out = broadcast_time(t, x)
    assert out.dim() == x.dim()
    assert out.shape == (6,) + (1,) * (ndim - 1)
    # Values are preserved, just reshaped.
    torch.testing.assert_close(out.reshape(-1), t)
    # And it actually broadcasts against x.
    assert (out * x).shape == x.shape


def test_broadcast_time_batch_x1_broadcasts():
    """A [B, 1] time tensor also broadcasts to match x.dim()."""
    x = torch.randn(6, 3, 8, 8)
    t = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    out = broadcast_time(t, x)
    assert out.shape == (6, 1, 1, 1)
    assert (out * x).shape == x.shape


# --------------------------------------------------------------------------- #
# SampleOutput
# --------------------------------------------------------------------------- #
def test_sample_output_default_log_weights_uniform():
    """Default log_weights are uniform *normalised* log-weights.

    The constructor starts from zeros and subtracts logsumexp over the K axis, so
    every entry becomes -log(K): the weights are equal and normalised (their
    logsumexp is 0, i.e. they exp-sum to 1 = a uniform categorical over K).
    """
    B, K = 3, 7
    samples = torch.randn(B, K, 2)
    out = SampleOutput(samples)
    assert out.log_weights.shape == (B, K)
    import math

    torch.testing.assert_close(out.log_weights, torch.full((B, K), -math.log(K)))
    # Normalised: logsumexp over the K axis is 0 -> exp-weights sum to 1.
    torch.testing.assert_close(torch.logsumexp(out.log_weights, dim=-1), torch.zeros(B))
    torch.testing.assert_close(out.log_weights.exp().sum(-1), torch.ones(B))


def test_sample_output_fields_round_trip():
    """All optional fields are stored verbatim."""
    samples = torch.randn(2, 4, 3)
    log_weights = torch.randn(2, 4)
    log_probs = torch.randn(2, 4)
    metadata = {"foo": 1}
    known_ll = torch.randn(2, 4)
    mask = torch.tensor([[True, False, True, False], [False, False, True, True]])

    out = SampleOutput(
        samples,
        log_weights=log_weights,
        log_probs=log_probs,
        metadata=metadata,
        known_log_likelihoods=known_ll,
        known_likelihoods_mask=mask,
    )
    assert out.samples is samples
    assert out.log_weights is log_weights
    assert out.log_probs is log_probs
    assert out.metadata is metadata
    assert out.known_log_likelihoods is known_ll
    assert out.known_likelihoods_mask is mask


def test_sample_output_provided_log_weights_not_overwritten():
    """Explicitly supplied (unnormalised) log_weights are kept as-is."""
    samples = torch.randn(1, 3, 1)
    lw = torch.tensor([[1.0, 2.0, 3.0]])
    out = SampleOutput(samples, log_weights=lw)
    torch.testing.assert_close(out.log_weights, lw)
