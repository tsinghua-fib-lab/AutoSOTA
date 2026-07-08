"""Core correctness tests for the calibrated-guidance estimators.

These exercise the mathematical contract of ``build_reinforce_mean_fn`` /
``build_reparam_mean_fn`` directly (one denoising step), independent of any
sampler. End-to-end consistency (convergence to the true posterior as K grows)
is covered in ``test_estimator_consistency.py``.
"""

import pytest
import torch

from calibrated_guidance import build_reinforce_mean_fn, build_reparam_mean_fn
from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput
from calibrated_guidance.guidance import masked_likelihood_compute


class FixedSamplePosterior(DiffusionPosterior):
    """A posterior that returns a fixed, prescribed set of x0 samples.

    Lets us assert the exact value of the guided mean for known weights, with no
    Monte-Carlo noise. ``samples`` has shape [B, K, D].
    """

    def __init__(self, samples: torch.Tensor):
        self._samples = samples

    def sample(self, xt, t, num_samples) -> SampleOutput:
        s = self._samples
        if s.requires_grad or (torch.is_tensor(xt) and xt.requires_grad):
            # Reparam path: tie samples to xt so autograd has a path.
            s = s + 0.0 * xt[:, None]
        return SampleOutput(s)

    def mean(self, xt, t):
        return self._samples.mean(dim=1)


def test_reinforce_returns_convex_combination_within_hull():
    # 1D samples; guided mean must lie within [min, max] of the samples.
    samples = torch.tensor([[[-2.0], [0.0], [1.0], [3.0]]])  # [1, 4, 1]
    post = FixedSamplePosterior(samples)

    def loglik(x):  # prefer larger x
        return (5.0 * x).sum(-1)

    mean_fn = build_reinforce_mean_fn(post, loglik, num_samples_per_step=4)
    guided = mean_fn(torch.zeros(1, 1), 0.5)
    assert guided.shape == (1, 1)
    assert samples.min() <= guided.item() <= samples.max()
    # Heavily upweighting large x should push the mean above the plain average.
    assert guided.item() > samples.mean().item()


def test_reinforce_uniform_likelihood_equals_unguided_mean():
    # A constant log-likelihood must leave the (uniform-weight) mean untouched.
    samples = torch.randn(3, 16, 2)
    post = FixedSamplePosterior(samples)

    def const_loglik(x):
        return torch.zeros(x.shape[0], x.shape[1])

    mean_fn = build_reinforce_mean_fn(post, const_loglik, num_samples_per_step=16)
    guided = mean_fn(torch.zeros(3, 2), 0.5)
    torch.testing.assert_close(guided, samples.mean(dim=1))


def test_reinforce_softmax_weighting_matches_manual():
    samples = torch.tensor([[[0.0], [1.0], [2.0]]])  # [1,3,1]
    post = FixedSamplePosterior(samples)
    logits = torch.tensor([[0.3, -0.7, 1.1]])

    def loglik(x):
        return logits

    mean_fn = build_reinforce_mean_fn(post, loglik, num_samples_per_step=3)
    guided = mean_fn(torch.zeros(1, 1), 0.5)

    w = torch.softmax(logits, dim=1)
    expected = (w[..., None] * samples).sum(dim=1)
    torch.testing.assert_close(guided, expected)


def test_reinforce_guidance_scale_sharpens_weights():
    samples = torch.tensor([[[0.0], [1.0]]])
    post = FixedSamplePosterior(samples)

    def loglik(x):
        return x.sum(-1)  # second sample has higher likelihood

    low = build_reinforce_mean_fn(post, loglik, 2, guidance_scale=1.0)(torch.zeros(1, 1), 0.5)
    high = build_reinforce_mean_fn(post, loglik, 2, guidance_scale=50.0)(torch.zeros(1, 1), 0.5)
    # Larger guidance scale concentrates weight on the high-likelihood sample (x=1).
    assert high.item() > low.item()
    assert high.item() > 0.9


def test_reinforce_rejects_grad_input():
    samples = torch.randn(1, 4, 1)
    post = FixedSamplePosterior(samples)
    mean_fn = build_reinforce_mean_fn(post, lambda x: x.sum(-1), 4)
    xt = torch.zeros(1, 1, requires_grad=True)
    with pytest.raises(AssertionError):
        mean_fn(xt, 0.5)


def test_masked_likelihood_uses_known_values():
    x0 = torch.zeros(2, 3, 1)
    known = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    mask = torch.tensor([[True, False, True], [False, True, False]])

    def loglik(x):  # masked path: x is flattened to [N_unmasked, D] -> return [N_unmasked]
        return torch.zeros(x.shape[0])

    out = masked_likelihood_compute(loglik, 1.0, x0, known, mask)
    # Masked-in entries take the known value; the rest are computed (zero here).
    assert out[0, 0] == 10.0 and out[0, 2] == 12.0 and out[0, 1] == 0.0
    assert out[1, 1] == 21.0 and out[1, 0] == 0.0


def test_masked_likelihood_applies_guidance_scale_when_no_mask():
    x0 = torch.ones(2, 4, 1)
    out = masked_likelihood_compute(lambda x: x.sum(-1), 3.0, x0, None, None)
    torch.testing.assert_close(out, 3.0 * torch.ones(2, 4))


def test_reparam_mean_fn_shape_and_grad_path():
    samples = torch.randn(2, 8, 3)
    post = FixedSamplePosterior(samples)

    def loglik(x):
        return -(x**2).sum(-1)

    mean_fn = build_reparam_mean_fn(post, loglik, num_samples_per_step=8)
    out = mean_fn(torch.randn(2, 3), 0.4)
    assert out.shape == (2, 3)
    assert not out.requires_grad  # the guidance is detached before returning


def test_reparam_at_t1_returns_unguided_mean():
    samples = torch.randn(2, 8, 3)
    post = FixedSamplePosterior(samples)
    mean_fn = build_reparam_mean_fn(post, lambda x: x.sum(-1), 8)
    out = mean_fn(torch.randn(2, 3), 1.0)
    torch.testing.assert_close(out, samples.mean(dim=1))
