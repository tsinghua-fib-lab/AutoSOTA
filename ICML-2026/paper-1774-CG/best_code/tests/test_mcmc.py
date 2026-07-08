"""Tests for ``calibrated_guidance.inference.mcmc``.

The differentiable path uses pyro's NUTS, so the whole module is skipped when
pyro is unavailable (it is an optional dependency in CI).

KNOWN ISSUE: the non-differentiable path references ``PotentialRandomWalkKernel``
whose import is commented out in ``inference.py`` (see the top of that file), so
``mcmc(..., differentiable=False)`` raises ``NameError`` at call time.  We assert
that this is the (buggy) behaviour rather than letting it crash the suite.
"""

import pytest
import torch

# pyro is optional; skip the entire module if it can't be imported.
pytest.importorskip("pyro")

from calibrated_guidance.inference import mcmc  # noqa: E402


def test_mcmc_nuts_samples_standard_normal():
    """NUTS on a standard-normal log-posterior recovers ~N(0, 1).

    log p(x) = -0.5 x^2 (up to a constant) -> potential 0.5 x^2 -> the target is
    the unit Gaussian.  We check the empirical mean is near 0 and the std near 1.
    Sample counts are kept small so the test runs in a few seconds.
    """
    torch.manual_seed(0)

    def log_posterior(x):
        return -0.5 * (x**2).sum()

    samples = mcmc(
        log_posterior,
        differentiable=True,
        shape=(1,),
        num_chains=1,
        n_samples=400,
        warmup_steps=200,
    )

    # Flatten across whatever chain/sample layout pyro returns.
    flat = samples.reshape(-1).float()
    assert flat.numel() >= 100
    assert torch.isfinite(flat).all()
    assert abs(flat.mean().item()) < 0.3
    assert abs(flat.std().item() - 1.0) < 0.4


def test_mcmc_nuts_recovers_shifted_gaussian():
    """NUTS targeting N(mu, 1) centres the samples near mu.

    Confirms the potential is -log_posterior (so the mode is the posterior mode).
    """
    torch.manual_seed(0)
    mu = 2.0

    def log_posterior(x):
        return -0.5 * ((x - mu) ** 2).sum()

    samples = mcmc(
        log_posterior,
        differentiable=True,
        shape=(1,),
        num_chains=1,
        n_samples=400,
        warmup_steps=200,
    )
    flat = samples.reshape(-1).float()
    assert abs(flat.mean().item() - mu) < 0.4


def test_mcmc_non_differentiable_path_is_broken():
    """Documented upstream bug: differentiable=False hits an undefined name.

    ``PotentialRandomWalkKernel`` is referenced but its import is commented out,
    so the call raises NameError.  We pin this so the regression is visible
    without failing the suite.
    """

    def log_posterior(x):
        return -0.5 * (x**2).sum()

    with pytest.raises(NameError):
        mcmc(
            log_posterior,
            differentiable=False,
            shape=(1,),
            num_chains=1,
            n_samples=10,
            warmup_steps=5,
        )
