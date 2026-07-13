import numpyro
from numpyro.infer import MCMC, NUTS
import pyro.distributions as dist
import numpyro.distributions as ndist
import jax.numpy as jnp
import jax
import torch

from simulator.base import Simulator


class AdaptiveGaussian(Simulator):
    def __init__(
        self,
        theta_dim: int = 2,
        obs_dim: int = 2,
        prior_var_scale: float = 5.0,
        likelihood_var_scale: float = 2.0,
        noisy_var_scale: float = 5.0,
        s: float = 0.0,
        t: float = 1.0,
        complex: str = "simulator",
        seed: int = 0,
    ):
        # Make sure inputs are integers
        theta_dim = int(theta_dim)
        obs_dim = int(obs_dim)
        super().__init__(obs_dim=obs_dim, theta_dim=theta_dim, name="adaptive_gaussian")
        self.callable_simulator = True
        self.callable_dgp = True
        self.supported_generation = ["independent", "transitive"]

        # Prior parameters
        torch.manual_seed(seed)
        prior_loc = torch.rand((theta_dim,)) * 10 - 5
        cov_theta_sqrt = torch.normal(0.0, prior_var_scale, (theta_dim, theta_dim))
        prior_cov = cov_theta_sqrt @ cov_theta_sqrt.T + torch.eye(theta_dim)

        self.prior_params = {"loc": prior_loc, "covariance_matrix": prior_cov}

        self.prior_dist = dist.MultivariateNormal(**self.prior_params)
        self.prior_dist.set_default_validate_args(False)

        assert complex in ["simulator", "misspecification"]

        # Likelihood parameters
        cov_likelihood_sqrt = torch.normal(0.0, likelihood_var_scale, (obs_dim, obs_dim))
        A = torch.normal(0.0, 1.0, (obs_dim, theta_dim))
        b = torch.normal(0.0, 1.0, (obs_dim,))
        likelihood_cov = cov_likelihood_sqrt @ cov_likelihood_sqrt.T

        def mean_likelihood(theta):
            return theta @ A.T + b

        self.mean_likelihood = mean_likelihood
        self.simulator_params = {
            "coef": A,
            "bias": b,
            "covariance_matrix": likelihood_cov,
        }

        # Noisy process parameters
        cov_noisy_sqrt = torch.normal(0.0, noisy_var_scale, (obs_dim, obs_dim))
        C = torch.normal(1.0, 1.0, (obs_dim, obs_dim))
        d = torch.rand((obs_dim,)) * 5 + 5
        noise_cov = cov_noisy_sqrt @ cov_noisy_sqrt.T

        def mean_noise(x):
            return x @ C.T + d

        self.mean_noise = mean_noise
        self.noise_params = {
            "coef": C,
            "bias": d,
            "covariance_matrix": noise_cov,
        }
        self.like_dist = None
        self.post_dist = None
        self.denoise_dist = None

        self.s = s
        self.t = t
        self.complex_simulator = True if complex == "simulator" else False
        self.complex_misspecification = True if complex == "misspecification" else False

    def transform_theta(self, theta: torch.Tensor) -> torch.Tensor:
        if self.complex_simulator:
            return torch.sinh((torch.arcsinh(theta) + self.s) / self.t)
        elif self.complex_misspecification:
            return theta
        else:
            raise ValueError("Invalid complex type specified.")

    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.complex_simulator:
            return x
        elif self.complex_misspecification:
            return torch.sinh((torch.arcsinh(x) + self.s) / self.t)
        else:
            raise ValueError("Invalid complex type specified.")

    def get_simulator(self, misspecified: bool):
        def simulator(theta: torch.Tensor) -> torch.Tensor:
            theta_transform = self.transform_theta(theta)
            parameters = {
                "loc": self.mean_likelihood(theta_transform),
                "covariance_matrix": self.simulator_params["covariance_matrix"],
            }
            x = dist.MultivariateNormal(**parameters).sample()
            x_transform = self.transform_x(x)
            parameters_noise = {
                "loc": self.mean_noise(x_transform),
                "covariance_matrix": self.noise_params["covariance_matrix"],
            }
            y = dist.MultivariateNormal(**parameters_noise).sample()
            if misspecified:
                return x
            return y

        return simulator

    def sample_reference_posterior(
        self,
        num_samples: int,
        observations: torch.Tensor,
        misspecified: bool,
        num_warmup: int = 1000,
        seed=0,
    ) -> torch.Tensor:
        def T(theta, s, t):
            return jnp.sinh((jnp.arcsinh(theta) + s) / t)

        def model(y_obs, mu_theta, cov_theta, A, b, cov_x, C, d, cov_y, s, t):
            N, d = y_obs.shape

            # Prior over theta (N, d)
            prior_dist_numpyro = ndist.MultivariateNormal(mu_theta, cov_theta)
            theta = numpyro.sample("theta", prior_dist_numpyro.expand([N]))

            # Linear transformation of theta
            x_mean = theta @ A.T + b  # shape: (N, d)
            x = numpyro.sample("x", ndist.MultivariateNormal(x_mean, cov_x))

            # Transformed theta
            Tx = T(x, s, t)  # shape: (N, d)

            # Model mean
            mean = Tx @ C.T + d  # shape: (N, d)

            # Likelihood
            numpyro.sample("y", ndist.MultivariateNormal(mean, cov_y), obs=y_obs)

        # Tensor to array
        y_obs = observations.numpy()
        y_obs = jnp.array(y_obs)
        mu_theta = self.prior_params["loc"].numpy()
        cov_theta = self.prior_params["covariance_matrix"].numpy()
        A = self.simulator_params["coef"].numpy()
        b = self.simulator_params["bias"].numpy()
        cov_x = self.simulator_params["covariance_matrix"].numpy()
        C = self.noise_params["coef"].numpy()
        d = self.noise_params["bias"].numpy()
        cov_y = self.noise_params["covariance_matrix"].numpy()
        s = self.s
        t = self.t

        rng = jax.random.PRNGKey(seed)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(
            rng,
            y_obs=y_obs,
            mu_theta=mu_theta,
            cov_theta=cov_theta,
            A=A,
            b=b,
            cov_x=cov_x,
            C=C,
            d=d,
            cov_y=cov_y,
            s=s,
            t=t,
        )
        samples = mcmc.get_samples()["theta"]  # shape: (nsamples, N, d)

        # Convert to torch tensor
        samples = torch.tensor(samples, dtype=torch.float32)
        return samples

    def get_noisy_process(self):
        def noisy_process(x: torch.Tensor) -> torch.Tensor:
            x_transform = self.transform_x(x)
            parameters = {
                "loc": self.mean_noise(x_transform),
                "covariance_matrix": self.noise_params["covariance_matrix"],
            }
            y = dist.MultivariateNormal(**parameters).sample()
            return y

        return noisy_process

    def sample_denoiser(self, num_samples: int, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
