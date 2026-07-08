"""The SBI tasks' analytic diffusion posteriors p(x0|xt) (from the library) must
match the closed forms in the paper (Appendix F) and the original eval scripts.

Forward process: xt = (1-t) x0 + t eps. The integration relies on these being
exact, so we check the moments and the truncated-normal sampler directly.
"""

import math

import pytest
import torch

from experiments.sbi.tasks import TASKS


def test_task1_gaussian_posterior_matches_paper_closed_form():
    """Paper F.1: p(x0|xt) = N( (1-t)/(10 t^2 + (1-t)^2) xt,  t^2/(10 t^2 + (1-t)^2) I ).
    (10 = 1 / prior_var with prior_var = 0.1.)"""
    post = TASKS["task1"].build_posterior(device="cpu")
    xt = torch.randn(4, 10)
    for t in [0.1, 0.3, 0.5, 0.9]:
        dist = post.diffusion_posterior(xt, t)
        denom = 10 * t**2 + (1 - t) ** 2
        expected_mean = (1 - t) / denom * xt
        expected_var = t**2 / denom
        torch.testing.assert_close(dist.mean, expected_mean, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(dist.variance, torch.full_like(xt, expected_var),
                                   atol=1e-5, rtol=1e-4)


def test_task1_posterior_at_t1_equals_prior():
    """At t=1 the Gaussian posterior collapses to the prior N(0, 0.1 I)."""
    post = TASKS["task1"].build_posterior(device="cpu")
    dist = post.diffusion_posterior(torch.randn(3, 10), 1.0)
    torch.testing.assert_close(dist.mean, torch.zeros(3, 10), atol=1e-5, rtol=0)
    torch.testing.assert_close(dist.variance, torch.full((3, 10), 0.1), atol=1e-5, rtol=0)


@pytest.mark.parametrize("task_key,low,high,dim", [
    ("task2", -1, 1, 10), ("task3", -3, 3, 5),
    ("task4", -10, 10, 2), ("task5", -1, 1, 2),
])
def test_uniform_posterior_is_truncated_normal(task_key, low, high, dim):
    """Paper F.2-F.5: p(x0|xt) ~ TruncN(mean=xt/(1-t), std=t/(1-t)) on [low,high]^d."""
    post = TASKS[task_key].build_posterior(device="cpu")
    xt = 0.3 * torch.randn(4, dim)
    t = 0.4
    dist = post.diffusion_posterior(xt, t)
    inner = dist.base_dist  # Independent -> TruncatedNormal
    torch.testing.assert_close(inner.loc, xt / (1 - t), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(inner.scale, torch.full_like(xt, t / (1 - t)), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("task_key,low,high,dim", [
    ("task2", -1, 1, 10), ("task4", -10, 10, 2), ("task5", -1, 1, 2),
])
def test_uniform_posterior_samples_in_bounds_and_finite(task_key, low, high, dim):
    post = TASKS[task_key].build_posterior(device="cpu")
    xt = 0.5 * torch.randn(8, dim)
    for t in [0.05, 0.5, 0.95, 0.999]:  # including near t=1 (the TruncatedNormal fix)
        out = post.sample(xt, t, 200)
        s = out.samples
        assert torch.isfinite(s).all(), f"{task_key} t={t}: non-finite samples"
        assert s.shape == (8, 200, dim)
        assert s.min() >= low - 1e-3 and s.max() <= high + 1e-3


def test_gaussian_posterior_sample_moments_match_distribution():
    """Sampled moments of the Gaussian posterior match the closed-form mean/var."""
    torch.manual_seed(0)
    post = TASKS["task1"].build_posterior(device="cpu")
    xt = torch.randn(1, 10)
    t = 0.5
    s = post.sample(xt, t, 200_000).samples[0]   # [K, 10]
    dist = post.diffusion_posterior(xt, t)
    torch.testing.assert_close(s.mean(0), dist.mean[0], atol=0.01, rtol=0)
    torch.testing.assert_close(s.var(0), dist.variance[0], atol=0.01, rtol=0)


def test_uniform_posterior_at_small_t_concentrates_near_xt():
    """As t -> 0 the posterior collapses onto xt (within the box)."""
    post = TASKS["task4"].build_posterior(device="cpu")  # [-10,10]^2
    xt = torch.tensor([[1.0, -2.0]])
    s = post.sample(xt, 0.01, 1000).samples[0]
    torch.testing.assert_close(s.mean(0), xt[0], atol=0.05, rtol=0)
