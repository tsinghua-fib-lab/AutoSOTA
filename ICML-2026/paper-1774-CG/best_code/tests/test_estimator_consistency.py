"""End-to-end consistency tests for the calibrated-guidance estimators.

This file encodes the central *scientific* claim of "Calibrated Bayesian
Guidance": as the number of per-step Monte-Carlo samples ``K`` grows, running
the diffusion sampler with the guided-mean estimator converges to the TRUE
Bayesian posterior ``p(x | y) ∝ p(x) p(y | x)``.

We make this checkable in closed form by using a Gaussian prior (via
``GaussianDiffusionPosterior``) together with a Gaussian observation likelihood
``p(y | x) = N(y; x, sigma_obs^2)``.  The product of two Gaussians is Gaussian:

    post_var  = 1 / (1 / scale^2 + 1 / sigma_obs^2)
    post_mean = post_var * (loc / scale^2 + y / sigma_obs^2)

The tempered posterior ``p(x) p(y | x)^gamma`` is again Gaussian, with the
likelihood *precision* scaled by ``gamma``:

    post_var(gamma)  = 1 / (1 / scale^2 + gamma / sigma_obs^2)
    post_mean(gamma) = post_var(gamma) * (loc / scale^2 + gamma * y / sigma_obs^2)

All runs are 1-D / low-D and CPU-only so the whole file stays fast.
"""

import math

import pytest
import torch

from calibrated_guidance import build_reinforce_mean_fn, build_reparam_mean_fn
from calibrated_guidance.diffusion_posterior.analytic.gaussian import (
    GaussianDiffusionPosterior,
)
from calibrated_guidance.inference import flow_matching


# --------------------------------------------------------------------------- #
# closed-form helpers
# --------------------------------------------------------------------------- #
def gaussian_posterior(loc, scale, y, sigma_obs, gamma=1.0):
    """Closed-form mean/std of ``p(x) p(y|x)^gamma`` for Gaussian prior+lik."""
    prior_prec = 1.0 / scale**2
    lik_prec = gamma / sigma_obs**2
    post_var = 1.0 / (prior_prec + lik_prec)
    post_mean = post_var * (loc / scale**2 + gamma * y / sigma_obs**2)
    return post_mean, math.sqrt(post_var)


def make_gaussian_loglik(y, sigma_obs):
    """Log N(y; x, sigma_obs^2) summed over the event (last) dimension.

    Accepts x of shape [B, K, D] (sampler-time) and returns [B, K].
    """
    y = torch.as_tensor(y, dtype=torch.float32)

    def loglik(x):
        # -0.5 * ((x - y)/sigma)^2 - log(sigma) - 0.5 log(2 pi), summed over D.
        z = (x - y) / sigma_obs
        return (-0.5 * z**2 - math.log(sigma_obs) - 0.5 * math.log(2 * math.pi)).sum(-1)

    return loglik


# Default time schedule: t = 1 (pure noise) -> t ~ 0 (clean).  We stop just shy
# of zero to avoid the t=0 singularity in flow matching's v = (mean - x)/t.
def default_time_steps(num_steps=100):
    return torch.linspace(1.0, 1e-3, num_steps)


# --------------------------------------------------------------------------- #
# REINFORCE estimator: convergence to the true posterior
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [1, 2])
def test_reinforce_recovers_gaussian_posterior_mean(d):
    """With enough per-step samples the REINFORCE sampler matches p(x|y).

    Verifies the paper's headline consistency claim for a closed-form Gaussian
    target: empirical sample mean/std of the generated samples track the
    analytic posterior mean/std.
    """
    torch.manual_seed(0)
    loc, scale = 1.0, 2.0
    sigma_obs = 0.7
    y_val = -1.5

    prior = GaussianDiffusionPosterior(
        loc=torch.full((d,), loc), scale=torch.full((d,), scale)
    )
    loglik = make_gaussian_loglik(torch.full((d,), y_val), sigma_obs)
    mean_fn = build_reinforce_mean_fn(prior, loglik, num_samples_per_step=512)

    samples = flow_matching(
        mean_fn,
        default_time_steps(100),
        shape=(d,),
        n_samples=2000,
        use_tqdm=False,
    )
    assert samples.shape == (2000, d)

    post_mean, post_std = gaussian_posterior(loc, scale, y_val, sigma_obs)
    emp_mean = samples.mean(dim=0)
    emp_std = samples.std(dim=0)

    torch.testing.assert_close(
        emp_mean, torch.full((d,), post_mean), rtol=0, atol=0.15
    )
    torch.testing.assert_close(
        emp_std, torch.full((d,), post_std), rtol=0, atol=0.15
    )


def test_reinforce_consistency_trend_more_samples_lower_error():
    """Error must SHRINK as K grows -- the defining consistency property.

    Tiny K (biased, high-variance softmax estimate) should give a noticeably
    worse posterior-mean estimate than large K.
    """
    loc, scale = 0.0, 1.5
    sigma_obs = 0.5
    y_val = 1.0
    post_mean, _ = gaussian_posterior(loc, scale, y_val, sigma_obs)

    prior = GaussianDiffusionPosterior(
        loc=torch.tensor([loc]), scale=torch.tensor([scale])
    )
    loglik = make_gaussian_loglik(torch.tensor([y_val]), sigma_obs)

    def run(K, seed):
        torch.manual_seed(seed)
        mean_fn = build_reinforce_mean_fn(prior, loglik, num_samples_per_step=K)
        s = flow_matching(
            mean_fn, default_time_steps(100), shape=(1,), n_samples=2000,
            use_tqdm=False,
        )
        return abs(s.mean().item() - post_mean)

    # Average a couple of seeds to keep the comparison robust.
    err_small = sum(run(2, seed) for seed in (0, 1, 2)) / 3
    err_large = sum(run(512, seed) for seed in (0, 1, 2)) / 3

    assert err_large < err_small, (
        f"expected large-K error ({err_large:.3f}) < small-K error "
        f"({err_small:.3f})"
    )
    # Large K should land very close to the analytic mean.
    assert err_large < 0.1


# --------------------------------------------------------------------------- #
# Reparam estimator: convergence to the true posterior
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [1, 2])
def test_reparam_recovers_gaussian_posterior_mean(d):
    """The gradient-based (reparam) estimator also targets p(x|y).

    GaussianDiffusionPosterior uses rsample, so gradients flow through the
    samples and the reparameterised guidance term is well defined.
    """
    torch.manual_seed(0)
    loc, scale = 1.0, 2.0
    sigma_obs = 0.7
    y_val = -1.5

    prior = GaussianDiffusionPosterior(
        loc=torch.full((d,), loc), scale=torch.full((d,), scale)
    )
    loglik = make_gaussian_loglik(torch.full((d,), y_val), sigma_obs)
    mean_fn = build_reparam_mean_fn(prior, loglik, num_samples_per_step=256)

    samples = flow_matching(
        mean_fn,
        default_time_steps(100),
        shape=(d,),
        n_samples=2000,
        use_tqdm=False,
    )
    assert samples.shape == (2000, d)

    post_mean, post_std = gaussian_posterior(loc, scale, y_val, sigma_obs)
    emp_mean = samples.mean(dim=0)
    emp_std = samples.std(dim=0)

    torch.testing.assert_close(
        emp_mean, torch.full((d,), post_mean), rtol=0, atol=0.15
    )
    torch.testing.assert_close(
        emp_std, torch.full((d,), post_std), rtol=0, atol=0.15
    )


def test_reparam_consistency_trend_more_samples_lower_error():
    """Reparam error also shrinks as the per-step sample count grows."""
    loc, scale = 0.0, 1.5
    sigma_obs = 0.5
    y_val = 1.0
    post_mean, _ = gaussian_posterior(loc, scale, y_val, sigma_obs)

    prior = GaussianDiffusionPosterior(
        loc=torch.tensor([loc]), scale=torch.tensor([scale])
    )
    loglik = make_gaussian_loglik(torch.tensor([y_val]), sigma_obs)

    def run(K, seed):
        torch.manual_seed(seed)
        mean_fn = build_reparam_mean_fn(prior, loglik, num_samples_per_step=K)
        s = flow_matching(
            mean_fn, default_time_steps(100), shape=(1,), n_samples=2000,
            use_tqdm=False,
        )
        return abs(s.mean().item() - post_mean)

    err_small = sum(run(2, seed) for seed in (0, 1, 2)) / 3
    err_large = sum(run(256, seed) for seed in (0, 1, 2)) / 3

    assert err_large < err_small
    assert err_large < 0.1


# --------------------------------------------------------------------------- #
# Tempered posteriors: guidance_scale = gamma
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gamma", [1.0, 2.0, 4.0])
def test_reinforce_tempered_posterior_mean(gamma):
    """guidance_scale=gamma must target p(x) p(y|x)^gamma.

    For Gaussians this sharpens the likelihood (precision *= gamma), so the
    posterior mean moves toward y as gamma grows.
    """
    torch.manual_seed(0)
    loc, scale = 0.0, 2.0
    sigma_obs = 1.0
    y_val = 3.0

    prior = GaussianDiffusionPosterior(
        loc=torch.tensor([loc]), scale=torch.tensor([scale])
    )
    loglik = make_gaussian_loglik(torch.tensor([y_val]), sigma_obs)
    mean_fn = build_reinforce_mean_fn(
        prior, loglik, num_samples_per_step=512, guidance_scale=gamma
    )

    samples = flow_matching(
        mean_fn, default_time_steps(100), shape=(1,), n_samples=2000,
        use_tqdm=False,
    )
    post_mean, _ = gaussian_posterior(loc, scale, y_val, sigma_obs, gamma=gamma)
    torch.testing.assert_close(
        samples.mean(), torch.tensor(post_mean), rtol=0, atol=0.15
    )


def test_tempering_pulls_mean_toward_observation():
    """Larger gamma pulls the posterior mean closer to the observation y.

    Direct test of the tempered-target ordering claimed in the paper.
    """
    loc, scale = 0.0, 2.0
    sigma_obs = 1.0
    y_val = 3.0

    prior = GaussianDiffusionPosterior(
        loc=torch.tensor([loc]), scale=torch.tensor([scale])
    )
    loglik = make_gaussian_loglik(torch.tensor([y_val]), sigma_obs)

    def run(gamma):
        torch.manual_seed(0)
        mean_fn = build_reinforce_mean_fn(
            prior, loglik, num_samples_per_step=512, guidance_scale=gamma
        )
        return flow_matching(
            mean_fn, default_time_steps(100), shape=(1,), n_samples=2000,
            use_tqdm=False,
        ).mean().item()

    m1 = run(1.0)
    m2 = run(2.0)
    m4 = run(4.0)
    # Monotonically approaching y=3 as gamma increases.
    assert m1 < m2 < m4 < y_val
    # And each empirical mean should be close to its closed-form target.
    for gamma, m in ((1.0, m1), (2.0, m2), (4.0, m4)):
        pm, _ = gaussian_posterior(loc, scale, y_val, sigma_obs, gamma=gamma)
        assert abs(m - pm) < 0.2
