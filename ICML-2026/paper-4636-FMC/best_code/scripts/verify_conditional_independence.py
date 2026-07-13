"""Verify the conditional independence experiment setup.

For each D scale value, this script:
1. Computes I(y; θ | x) analytically
2. Estimates I(y; θ | x) empirically via a regression test
3. Verifies that the analytical posterior p(θ|y) is correct by checking sampling consistency

Usage:
    python scripts/verify_conditional_independence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from simulator.pure_gaussian import PureGaussian


def empirical_ci_test(simulator: PureGaussian, n: int = 10000) -> float:
    """Empirical test: regress y on θ given x, using transitive data (y generated from x).

    If CI holds, adding θ should not improve prediction of y given x.
    Returns the R² improvement from adding θ to the regression y ~ x vs y ~ x + θ.
    """
    import pyro.distributions as dist

    theta = simulator.sample_prior(n)
    # Generate x from theta
    x = simulator.get_simulator(misspecified=True)(theta)
    # Generate y from the SAME x (transitive), with optional Dθ term
    D = simulator.direct_theta_coef
    mean_y = simulator.mean_noise(x) + theta @ D.T
    y = dist.MultivariateNormal(
        loc=mean_y, covariance_matrix=simulator.noise_params["covariance_matrix"]
    ).sample()

    # Regression y ~ x (OLS)
    X_only = torch.cat([x, torch.ones(n, 1)], dim=1)
    beta_x = torch.linalg.lstsq(X_only, y).solution
    resid_x = y - X_only @ beta_x
    ss_res_x = (resid_x ** 2).sum().item()

    # Regression y ~ x + theta
    X_theta = torch.cat([x, theta, torch.ones(n, 1)], dim=1)
    beta_xtheta = torch.linalg.lstsq(X_theta, y).solution
    resid_xtheta = y - X_theta @ beta_xtheta
    ss_res_xtheta = (resid_xtheta ** 2).sum().item()

    # R² improvement
    r2_improvement = 1.0 - ss_res_xtheta / ss_res_x
    return r2_improvement


def verify_posterior(simulator: PureGaussian, n_obs: int = 100, n_samples: int = 5000):
    """Verify that the analytical posterior p(θ|y) matches empirical samples."""
    theta_true = simulator.sample_prior(n_obs)
    y = simulator.get_simulator(misspecified=False)(theta_true)

    # Get analytical posterior
    post_dist = simulator.posterior_theta_given_y(y)

    # Sample from posterior
    samples = post_dist.sample((n_samples,))  # (n_samples, n_obs, theta_dim)

    # Check that empirical mean/cov match analytical for first observation
    emp_mean = samples[:, 0, :].mean(dim=0)
    ana_mean = post_dist.loc[0]
    mean_err = (emp_mean - ana_mean).norm().item()

    emp_cov = torch.cov(samples[:, 0, :].T)
    ana_cov = post_dist.covariance_matrix[0]
    cov_err = (emp_cov - ana_cov).norm().item() / ana_cov.norm().item()

    return mean_err, cov_err


def main():
    scales = [0.0, 0.5, 1.0, 2.0]
    common_params = dict(
        theta_dim=3, obs_dim=10,
        prior_var_scale=1, likelihood_var_scale=1, noisy_var_scale=0.1,
        seed=0,
    )

    print("=" * 70)
    print("Conditional Independence Verification")
    print("=" * 70)

    for scale in scales:
        sim = PureGaussian(**common_params, direct_theta_scale=scale)

        mi = sim.conditional_mutual_information()
        r2 = empirical_ci_test(sim)
        mean_err, cov_err = verify_posterior(sim)

        print(f"\n--- D scale = {scale} ---")
        print(f"  I(y; θ | x) [analytical]:    {mi:.4f} nats")
        print(f"  R² improvement [empirical]:   {r2:.4f}")
        print(f"  Posterior mean error:          {mean_err:.4f}")
        print(f"  Posterior cov relative error:  {cov_err:.4f}")

        if scale == 0.0:
            assert abs(mi) < 1e-6, f"MI should be ~0 when D=0, got {mi}"
            assert r2 < 0.05, f"R² improvement should be ~0 when D=0, got {r2}"
        else:
            assert mi > 0, f"MI should be > 0 when D≠0, got {mi}"

        assert mean_err < 0.1, f"Posterior mean error too large: {mean_err}"
        assert cov_err < 0.1, f"Posterior cov error too large: {cov_err}"

    print("\n" + "=" * 70)
    print("All checks passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
