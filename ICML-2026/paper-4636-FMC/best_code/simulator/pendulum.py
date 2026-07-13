from scipy.signal import correlate
import numpy as np
import torch
from pyro import distributions as dist
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as ndist
from numpyro.infer import MCMC, NUTS
from simulator.base import Simulator
from utils.transform import LogitBoxTransform


class Pendulum(Simulator):
    def __init__(
        self,
        obs_dim: int = 200,
        noise: float = 0.1,
        low_w0: float = 0.0,
        high_w0: float = 3.0,
        low_A: float = 0.5,
        high_A: float = 10.0,
        tmax: float = 10.0,
        no_misspecification: bool = False,
    ):
        obs_dim = int(obs_dim)
        super().__init__(obs_dim=obs_dim, theta_dim=2, name="pendulum")
        self.obs_dim = obs_dim
        self.noise = noise
        self.low_w0 = low_w0
        self.high_w0 = high_w0
        self.low_A = low_A
        self.high_A = high_A
        self.tmax = tmax
        self.no_misspecification = no_misspecification

        self.callable_simulator = True
        self.callable_dgp = True
        self.supported_generation = ["independent", "transitive"]

        # prior distribution
        self.prior_params = {
            "low": torch.Tensor([self.low_w0, self.low_A]),
            "high": torch.Tensor([self.high_w0, self.high_A]),
        }
        self.prior_dist = dist.Independent(dist.Uniform(**self.prior_params), 1)
        # self.prior_dist = dist.Normal(loc=torch.Tensor([1.5, 5.0]), scale=torch.Tensor([0.5, 2.5]))
        self.prior_dist.set_default_validate_args(False)

        # Transform for theta
        self.transform = LogitBoxTransform(a=self.prior_params["low"], b=self.prior_params["high"])

        self.like_dist = None
        self.post_dist = None
        self.denoise_dist = None

    def get_simulator(self, misspecified: bool):
        # In no-misspec mode, both simulators are identical (alpha=0, no damping)
        effective_misspecified = misspecified or self.no_misspecification

        def simulator(theta: torch.Tensor) -> torch.Tensor:
            phi = torch.zeros(theta.shape[0], 1)
            wo, A = theta[:, 0].reshape(-1, 1), theta[:, 1].reshape(-1, 1)
            t = torch.linspace(0, self.tmax, self.obs_dim).reshape(1, -1)
            alpha = (1 - effective_misspecified) * torch.rand(theta.shape[0], 1)
            x = torch.exp(-alpha @ t) * A * torch.cos(wo @ t + phi) + self.noise * torch.randn(
                theta.shape[0], self.obs_dim
            )
            return x

        return simulator

    # ----------------------------------------------
    # MCMC using NUTS in NumPyro
    # ----------------------------------------------
    def sample_reference_posterior(
        self,
        num_samples: int,
        observations: torch.Tensor,
        misspecified: bool = False,
        num_warmup: int = 500,
        rng_seed: int = 0,
    ) -> torch.Tensor:
        """Run NUTS posterior inference in NumPyro.

        Args:
            num_samples: number of posterior samples
            observations: torch.Tensor (N, D), observed trajectories
            misspecified: if True, alpha=0, else alpha~Uniform(0,1)
            num_warmup: NUTS warmup iterations
            rng_seed: random seed

        Returns:
            torch.Tensor of shape (num_samples, N, 2) containing posterior samples for (w0, A)
        """
        y_jnp = jnp.asarray(observations.detach().cpu().numpy())
        N, D = y_jnp.shape
        t = jnp.linspace(0.0, float(self.tmax), D)

        def numpyro_model(y_obs):
            w0 = numpyro.sample("w0", ndist.Uniform(self.low_w0, self.high_w0).expand([N]))  # type: ignore
            A = numpyro.sample("A", ndist.Uniform(self.low_A, self.high_A).expand([N]))  # type: ignore

            if misspecified or self.no_misspecification:
                alpha = jnp.zeros((N, 1))
            else:
                alpha = numpyro.sample("alpha", ndist.Uniform(0.0, 1.0).expand([N]))[:, None]  # type: ignore

            t_row = t[None, :]
            mean = jnp.exp(-alpha * t_row) * (A[:, None]) * jnp.cos(w0[:, None] * t_row)  # type: ignore

            numpyro.sample("y", ndist.Normal(mean, self.noise).to_event(1), obs=y_obs)  # type: ignore

        kernel = NUTS(numpyro_model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        rng_key = jax.random.PRNGKey(rng_seed)
        mcmc.run(rng_key, y_obs=y_jnp)

        samples = mcmc.get_samples()
        w0_samps = samples["w0"]  # (num_samples, N)
        A_samps = samples["A"]  # (num_samples, N)
        theta_samps = jnp.stack([w0_samps, A_samps], axis=-1)  # (num_samples, N, 2)

        return torch.from_numpy(np.asarray(theta_samps))

    def get_noisy_process(self):
        def noisy_process(x: torch.Tensor) -> torch.Tensor:
            t = torch.linspace(0, self.tmax, self.obs_dim).reshape(1, -1)
            alpha = torch.rand(x.shape[0], 1)
            y = torch.exp(-alpha @ t) * x + 0.1 * self.noise * torch.randn(x.shape[0], self.obs_dim)
            return y

        return noisy_process

    def sample_denoiser(self, num_samples: int, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def estimate_A_wo_batch(
    x_batch: np.ndarray, t: np.ndarray, A_low: float, A_high: float, w_low: float, w_high: float
) -> np.ndarray:
    """
    Estimate A and w_o for a batch of signals using autocorrelation.

    Parameters:
    - x_batch: np.ndarray of shape (N, T), batch of N signals each with T time points.
    - t: np.ndarray of shape (T,), time points corresponding to the signals.

    Returns:
    - estimates: np.ndarray of shape (N, 2), containing (A, w_o) for each signal.
    """
    N, T = x_batch.shape
    dt = t[1] - t[0]  # Time step (assumed constant)
    estimates = np.zeros(
        (N, 2)
    )  # Placeholder for (A, w_o) estimates

    for i in range(N):
        x = x_batch[i]

        # Compute autocorrelation
        corr = correlate(x, x, mode="full")[T - 1 :]  # Keep only positive lags
        lags = np.arange(T) * dt  # Convert lag indices to time

        # Find peaks in autocorrelation
        peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1

        if len(peaks) > 0:
            if corr[peaks[0]] < 0:
                T_half = lags[peaks[0]]  # First peak corresponds to T_o / 2
                T_o_est = 2 * T_half  # Correcting for half-period peak
                w_o_est = 2 * np.pi / T_o_est
            else:
                w_o_est = 2 * np.pi / lags[peaks[0]]
        else:
            w_o_est = np.nan  # Assign NaN if no peak is found

        # Estimate A from the max autocorrelation value
        A_est = np.sqrt(2 * np.max(corr) / T)
        th = 1e-4
        estimates[i] = [
            np.clip(A_est, A_low + th, A_high - th),
            np.clip(w_o_est, w_low + th, w_high - th),
        ]

    return estimates
