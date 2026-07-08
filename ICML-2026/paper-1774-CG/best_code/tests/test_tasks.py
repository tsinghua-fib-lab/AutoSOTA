"""Tests for ``GaussianMixtureTask.log_likelihood``.

The task is a "which component generated this point?" classification likelihood:
log p(component = target_idx | x) computed as the normalised log-responsibility
of the target Gaussian.  Hence the values are log-probabilities (<= 0), and for a
point sitting deep inside the target component the responsibility -> 1 so the
log-likelihood -> 0.
"""

import math

import pytest
import torch

from calibrated_guidance.diffusion_posterior.analytic.gaussian_mixture import (
    GaussianMixtureDiffusionPosterior,
)
from calibrated_guidance.tasks.toy.gaussian_mixture import GaussianMixtureTask


def _two_component_task(target_idx=0, d=1):
    """Two well-separated equally-weighted unit-ish Gaussians at -/+ 5."""
    weights = torch.tensor([0.5, 0.5])
    means = torch.tensor([[-5.0] * d, [5.0] * d])
    scales = torch.tensor([[1.0] * d, [1.0] * d])
    mixture = GaussianMixtureDiffusionPosterior(weights, means, scales)
    return GaussianMixtureTask(mixture, target_idx=target_idx)


@pytest.mark.parametrize("d", [1, 3])
def test_log_likelihood_shape_and_nonpositive(d):
    """Output is [B, K] and every entry is a log-probability (<= 0)."""
    task = _two_component_task(target_idx=0, d=d)
    x = torch.randn(4, 6, d)
    ll = task.log_likelihood(x)
    assert ll.shape == (4, 6)
    assert (ll <= 1e-6).all()  # tiny tolerance for float round-off at 0


def test_log_likelihood_requires_3d_input():
    """Inputs must be [batch, num_samples, dimensions]."""
    task = _two_component_task()
    with pytest.raises(AssertionError):
        task.log_likelihood(torch.randn(4, 2))  # missing the K axis


def test_responsibility_saturates_inside_target_component():
    """A point deep inside the target component has log-likelihood ~ 0.

    Far inside component 0 the responsibility for component 0 -> 1, so its log
    -> 0, while the responsibility for the *other* component (a non-target task)
    -> 0, giving a strongly negative log-likelihood.
    """
    x = torch.tensor([[[-5.0]]])  # right on top of component 0's mean

    task0 = _two_component_task(target_idx=0)
    ll0 = task0.log_likelihood(x)
    torch.testing.assert_close(ll0, torch.zeros(1, 1), rtol=0, atol=1e-3)

    task1 = _two_component_task(target_idx=1)
    ll1 = task1.log_likelihood(x)
    # Component 1 essentially never explains a point at -5 (10 sigma away).
    assert ll1.item() < -40.0


def test_responsibility_equal_at_midpoint():
    """At the symmetric midpoint both components are equally responsible.

    With equal weights and symmetric components, the responsibility at x=0 is
    1/2, so the log-likelihood is log(1/2) for either target.
    """
    x = torch.tensor([[[0.0]]])
    task = _two_component_task(target_idx=0)
    ll = task.log_likelihood(x)
    torch.testing.assert_close(
        ll, torch.full((1, 1), math.log(0.5)), rtol=0, atol=1e-4
    )


def test_responsibilities_of_both_targets_sum_to_one():
    """The two target tasks' responsibilities (in prob space) sum to 1.

    They are a partition of the latent component assignment.
    """
    x = torch.randn(3, 5, 1)
    p0 = _two_component_task(target_idx=0).log_likelihood(x).exp()
    p1 = _two_component_task(target_idx=1).log_likelihood(x).exp()
    torch.testing.assert_close(p0 + p1, torch.ones(3, 5), rtol=0, atol=1e-5)
