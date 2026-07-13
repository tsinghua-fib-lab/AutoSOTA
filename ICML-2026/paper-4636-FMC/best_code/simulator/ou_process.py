from typing import Optional
from sbi.samplers.rejection.rejection import rejection_sample

import numpy as np
import torch
from pyro import distributions as dist

from simulator.base import Simulator


class OUProcess(Simulator):
    def __init__(
        self,
        obs_dim: int = 10,
        theta_dim: int = 4,
        length_total_trace: int = 101,
        subsample_rate: int = 10,
        first_n_samples: int = 100,
    ):
        # Necessary when loading from configs
        obs_dim = int(obs_dim)
        theta_dim = int(theta_dim)
        length_total_trace = int(length_total_trace)
        subsample_rate = int(subsample_rate)
        first_n_samples = int(first_n_samples)
        super().__init__(obs_dim=obs_dim, theta_dim=theta_dim, name="ou_process")
        self.callable_simulator = True
        self.callable_dgp = True
        self.supported_generation = ["independent"]

        self.obs_dim = obs_dim
        self.theta_dim = theta_dim
        self.length_total_trace = length_total_trace
        self.subsample_rate = subsample_rate
        self.first_n_samples = first_n_samples
        self.dt = 0.1

        # prior distribution
        self.low_mu, self.high_mu = (0.1, 3.0)
        self.low_sigma, self.high_sigma = (0.1, 0.6)
        self.low_gamma, self.high_gamma = (0.1, 1.0)
        self.low_mu_offset, self.high_mu_offset = (0.0, 4.0)
        self.prior_params = {
            "low": torch.Tensor([self.low_mu, self.low_sigma, self.low_gamma, self.low_mu_offset]),
            "high": torch.Tensor(
                [self.high_mu, self.high_sigma, self.high_gamma, self.high_mu_offset]
            ),
        }
        self.prior_dist = dist.Independent(dist.Uniform(**self.prior_params), 1)
        # self.prior_dist = dist.Normal(loc=torch.Tensor([1.5, 5.0]), scale=torch.Tensor([0.5, 2.5]))
        self.prior_dist.set_default_validate_args(False)
        self.like_dist = None
        self.post_dist = None
        self.denoise_dist = None

    def xt0_dist(self, xto, mu):
        """
        Define the distribution of the initial condition.
        The mean_xt is {true) mu + a certain value, since it should be adjusted
        to the true theta.
        """
        sigma_xto = 1
        rel_init_value = mu + xto
        xto_dist = dist.Normal(rel_init_value, sigma_xto)

        return xto_dist

    def _dW(self, delta_t: float) -> float:
        """Sample a random number at each call."""
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

    def _drift(self, x: float, mu: float, gamma: float) -> float:
        """Drift term of the Ornstein-Uhlenbeck process."""
        return gamma * (mu - x)

    def _diffusion(self, sigma: float) -> float:
        """Diffusion term of the Ornstein-Uhlenbeck process."""
        return sigma

    def _OU_timestep(self, theta, x_prev) -> torch.Tensor:
        """Run the Ornstein-Uhlenbeck process.

        Each simulation is the likelihood under specific theta configuration
        Runs the OU process for a given theta
        """
        mu = theta[0]
        sigma = theta[1]

        gamma = theta[2]  # The steeper (smaller gamma), the more similar the HF model is

        x_t = (
            x_prev
            + self._drift(x_prev, mu, gamma) * self.dt
            + self._diffusion(sigma) * self._dW(self.dt)
        )
        x_t = x_t.clone().detach()  # Detach the tensor to avoid backpropagation

        return x_t

    def integrator(self, theta):
        # print("samples theta", theta)

        # length of the trace is equal to the dimension of x
        trace = torch.zeros(self.length_total_trace)

        mu_offset = theta[3]

        mu = theta[0]
        xto_dist = self.xt0_dist(mu_offset, mu)
        xto = xto_dist.sample()
        trace[0] = xto

        for delta_t in range(1, self.length_total_trace):
            trace[delta_t] = self._OU_timestep(theta, trace[delta_t - 1])

        return trace

    def get_simulator(self, misspecified: bool):
        def simulator(theta: torch.Tensor) -> torch.Tensor:
            if misspecified:
                # Gaussian process
                mu = theta[:, 0]
                sigma = theta[:, 1]
                xt_dist = dist.Normal(loc=mu, scale=sigma)
                x = torch.cat([xt_dist.sample().unsqueeze(1) for _ in range(self.obs_dim)], dim=1)
            else:
                # OU process
                simulations = [self.integrator(theta[i]).flatten() for i in range(theta.shape[0])]
                x_raw = torch.stack(simulations)
                idx = torch.arange(0, self.length_total_trace, self.subsample_rate)
                assert len(idx) >= self.obs_dim, "Subsampled trace is shorter than obs_dim."
                x = x_raw[:, idx[: self.obs_dim]]
            return x

        return simulator

    def true_log_likelihood(self, thetas, true_x) -> torch.Tensor:
        """Analytical solution for likelihood: p(x|theta) = p(x|mu, sigma, gamma)

        Source: A Multiresolution Method for Parameter Estimation of Diffusion Processes
                DOI:10.1080/01621459.2012.720899
                Supeng Kou, Benjamin P. Olding

        Expectes multiple thetas (e.g. 1000) and a fixed x,
        goes into rejection sampling function that requires a tensor
        """
        mu = thetas[:, 0]
        sigma = thetas[:, 1]

        gamma = thetas[:, 2]
        mu_offset = thetas[:, 3]

        loglik = torch.zeros(len(mu))

        idx = np.arange(0, self.first_n_samples, self.subsample_rate)
        idx[0] = 0  # We dont want dt to be 0, otherwise we cannot compute true likelihood
        # self.subsample_rate
        # TODO: Fix data generation: Not 1 1 2 ... distance should be at least 1

        xto_dist = self.xt0_dist(mu_offset, mu)
        xto = true_x[0]

        xto_log_prob = xto_dist.log_prob(xto)  # Because log(1) = 0. # xto_dist.log_prob(x0)
        logsum = xto_log_prob

        for j in range(1, self.obs_dim):  # length_total_trace
            # print(idx)
            # print(idx[j]-idx[j-1])
            g = (
                1 - torch.exp(-2 * gamma * self.dt * (idx[j] - idx[j - 1]))
            ) / gamma  # self.dt * sample_rate

            x_prev = true_x[j - 1]
            x_curr = true_x[j]

            term = 1 / (torch.sqrt(torch.pi * g) * sigma)
            exp1 = -(1 / (g * sigma**2))
            exp2 = (mu - x_curr) - torch.sqrt(1 - gamma * g) * (mu - x_prev)
            exp = torch.exp(exp1 * exp2**2)

            logsum = logsum + torch.log(term * exp)
        loglik = logsum

        return loglik

    def sample_reference_posterior(
        self,
        num_samples: int,
        observations: torch.Tensor,
        misspecified: bool,
        theta_true: Optional[torch.Tensor] = None,
        num_warmup: int = 1000,
    ) -> torch.Tensor:
        true_posterior_samples = []
        for i, x_o in enumerate(observations):
            # Generate samples from the true posterior
            true_posterior_s, _ = rejection_sample(
                potential_fn=lambda theta: self.prior_dist.log_prob(theta)
                + self.true_log_likelihood(theta, x_o),
                proposal=self.prior_dist,
                m=2.0,
                num_samples=num_samples,
            )
            true_posterior_samples.append(true_posterior_s)

        true_posterior_samples = torch.stack(true_posterior_samples)

        return true_posterior_samples

    def get_noisy_process(self):
        def noisy_process(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError("Noisy process is not implemented for OUProcess.")

        return noisy_process

    def sample_denoiser(self, num_samples: int, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
