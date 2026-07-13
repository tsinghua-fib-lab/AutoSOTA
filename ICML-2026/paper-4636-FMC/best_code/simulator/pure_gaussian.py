import pyro.distributions as dist
import torch
from torch import Tensor

from simulator.base import Simulator


class PureGaussian(Simulator):
    def __init__(
        self,
        theta_dim: int = 2,
        obs_dim: int = 2,
        prior_var_scale: float = 5.0,
        likelihood_var_scale: float = 2.0,
        noisy_var_scale: float = 5.0,
        seed: int = 0,
        no_misspecification: bool = False,
        direct_theta_scale: float = 0.0,
    ):
        obs_dim = int(obs_dim)
        theta_dim = int(theta_dim)
        super().__init__(obs_dim=obs_dim, theta_dim=theta_dim, name="pure_gaussian")
        self.callable_simulator = True
        self.callable_dgp = True
        # Transitive generation not supported when D≠0 (noisy_process doesn't receive θ)
        if direct_theta_scale > 0.0:
            self.supported_generation = ["independent"]
        else:
            self.supported_generation = ["independent", "transitive"]

        # Prior parameters
        torch.manual_seed(seed)
        prior_loc = torch.rand((theta_dim,)) * 10 - 5
        cov_theta_sqrt = torch.normal(0.0, prior_var_scale, (theta_dim, theta_dim))
        prior_cov = cov_theta_sqrt @ cov_theta_sqrt.T + torch.eye(theta_dim)

        self.prior_params = {"loc": prior_loc, "covariance_matrix": prior_cov}

        self.prior_dist = dist.MultivariateNormal(**self.prior_params)
        self.prior_dist.set_default_validate_args(False)

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

        self.likelihood_dist = lambda theta: dist.MultivariateNormal(
            loc=mean_likelihood(theta), covariance_matrix=likelihood_cov
        )

        # Noisy process parameters: C=I, d=0, noise≈0 when no_misspecification=True
        if no_misspecification:
            C = torch.eye(obs_dim)
            d = torch.zeros(obs_dim)
            noise_cov = torch.eye(obs_dim) * 1e-6
        else:
            cov_noisy_sqrt = torch.normal(0.0, noisy_var_scale, (obs_dim, obs_dim))
            C = torch.normal(1.0, 1.0, (obs_dim, obs_dim))
            d = torch.rand((obs_dim,)) * 5 + 5
            noise_cov = cov_noisy_sqrt @ cov_noisy_sqrt.T

        # Direct theta -> y influence (breaks conditional independence when nonzero)
        if direct_theta_scale > 0.0:
            D = torch.normal(0.0, direct_theta_scale, (obs_dim, theta_dim))
        else:
            D = torch.zeros(obs_dim, theta_dim)
        self.direct_theta_coef = D

        def mean_noise(x):
            return x @ C.T + d

        self.mean_noise = mean_noise
        self.noise_params = {
            "coef": C,
            "bias": d,
            "covariance_matrix": noise_cov,
        }

    def get_simulator(self, misspecified: bool):
        D = self.direct_theta_coef

        def simulator(
            theta: Tensor,
        ) -> Tensor:
            parameters = {
                "loc": self.mean_likelihood(theta),
                "covariance_matrix": self.simulator_params["covariance_matrix"],
            }
            if not misspecified:
                x = dist.MultivariateNormal(**parameters).sample()
                mean_y = self.mean_noise(x) + theta @ D.T
                noise_parameters = {
                    "loc": mean_y,
                    "covariance_matrix": self.noise_params["covariance_matrix"],
                }
                noise = dist.MultivariateNormal(**noise_parameters).sample()
                return noise
            else:
                return dist.MultivariateNormal(**parameters).sample()

        return simulator

    def denoise_dist(self, y: Tensor) -> dist.MultivariateNormal:
        """
        Given a batch of noisy observations y of shape (batch_size, obs_dim),
        return the batch of conditional distributions p(x | y).
        """
        # Prior over theta
        mu_theta = self.prior_params["loc"]
        Sigma_theta = self.prior_params["covariance_matrix"]

        # Likelihood parameters
        A = self.simulator_params["coef"]
        b = self.simulator_params["bias"]
        Sigma_lik = self.simulator_params["covariance_matrix"]

        # Noise parameters
        C = self.noise_params["coef"]
        d = self.noise_params["bias"]
        Sigma_noise = self.noise_params["covariance_matrix"]

        # Direct theta influence
        D = self.direct_theta_coef

        # With D: y = Cx + Dθ + d + ε_y = (CA+D)θ + Cb + d + Cε_x + ε_y
        F = C @ A + D  # combined linear map θ -> y

        # Mean and covariance of x
        mu_x = A @ mu_theta + b
        Sigma_x = A @ Sigma_theta @ A.T + Sigma_lik

        # Cov(x, y) = A Σ_θ F^T + Σ_lik C^T
        Sigma_xy = A @ Sigma_theta @ F.T + Sigma_lik @ C.T

        # Covariance of y = F Σ_θ F^T + C Σ_lik C^T + Σ_noise
        Sigma_y = F @ Sigma_theta @ F.T + C @ Sigma_lik @ C.T + Sigma_noise
        Sigma_y_inv = torch.linalg.inv(Sigma_y)

        # Mean of y
        mu_y = F @ mu_theta + C @ b + d

        # Centered y: shape (batch_size, obs_dim)
        y_centered = y - mu_y

        # Compute conditional mean: μ_x|y = μ_x + Σ_xy Σ_yy^{-1} (y - μ_y)
        correction = y_centered @ Sigma_y_inv.T @ Sigma_xy.T
        mu_x_given_y = mu_x + correction  # (batch_size, x_dim)

        # Conditional covariance
        Sigma_x_given_y = Sigma_x - Sigma_xy @ Sigma_y_inv @ Sigma_xy.T

        return dist.MultivariateNormal(
            loc=mu_x_given_y, covariance_matrix=Sigma_x_given_y.expand(y.shape[0], -1, -1)
        )

    def posterior_theta_given_x(self, x: Tensor) -> dist.MultivariateNormal:
        """
        Given a batch of clean observations x of shape (batch_size, obs_dim),
        return the batch of posterior distributions p(theta | x).
        """
        # Prior
        mu_theta = self.prior_params["loc"]  # (theta_dim,)
        Sigma_theta = self.prior_params["covariance_matrix"]  # (theta_dim, theta_dim)

        # Likelihood
        A = self.simulator_params["coef"]  # (obs_dim, theta_dim)
        b = self.simulator_params["bias"]  # (obs_dim,)
        Sigma_lik = self.simulator_params["covariance_matrix"]  # (obs_dim, obs_dim)

        # Covariance between theta and x
        Sigma_thetay = Sigma_theta @ A.T

        # Covariance of x
        Sigma_x = A @ Sigma_theta @ A.T + Sigma_lik
        Sigma_x_inv = torch.linalg.inv(Sigma_x)

        # Mean of x
        mu_x = A @ mu_theta + b

        # Centered y: shape (batch_size, obs_dim)
        x_centered = x - mu_x

        # Compute conditional mean: μ_x|y = μ_x + Σ_xy Σ_yy^{-1} (y - μ_y)
        correction = x_centered @ Sigma_x_inv.T @ Sigma_thetay.T
        mu_theta_given_x = mu_theta + correction  # (batch_size, x_dim)

        # Conditional covariance
        Sigma_theta_given_x = Sigma_theta - Sigma_thetay @ Sigma_x_inv @ Sigma_thetay.T

        return dist.MultivariateNormal(
            loc=mu_theta_given_x, covariance_matrix=Sigma_theta_given_x.expand(x.shape[0], -1, -1)
        )

    def posterior_theta_given_y(self, y: Tensor) -> dist.MultivariateNormal:
        """
        Given a batch of noisy observations y of shape (batch_size, obs_dim),
        return the batch of posterior distributions p(theta | y).
        """
        # Prior
        mu_theta = self.prior_params["loc"]
        Sigma_theta = self.prior_params["covariance_matrix"]

        # Likelihood
        A = self.simulator_params["coef"]
        b = self.simulator_params["bias"]
        Sigma_lik = self.simulator_params["covariance_matrix"]

        # Noise
        C = self.noise_params["coef"]
        d = self.noise_params["bias"]
        Sigma_noise = self.noise_params["covariance_matrix"]

        # Direct theta influence
        D = self.direct_theta_coef

        # Linear mapping from theta to y: E[y|theta] = (CA + D)theta + Cb + d
        F = C @ A + D
        c = C @ b + d
        Sigma_y = C @ Sigma_lik @ C.T + Sigma_noise

        # Inverses
        Sigma_y_inv = torch.linalg.inv(Sigma_y)
        Sigma_theta_inv = torch.linalg.inv(Sigma_theta)

        # Posterior covariance
        Sigma_post = torch.linalg.inv(Sigma_theta_inv + F.T @ Sigma_y_inv @ F)

        # Center y
        y_centered = y - c
        mean_post = (
            Sigma_post @ (Sigma_theta_inv @ mu_theta + y_centered @ (Sigma_y_inv @ F)).T
        ).T  # shape (batch_size, theta_dim)

        return dist.MultivariateNormal(
            loc=mean_post, covariance_matrix=Sigma_post.expand(y.shape[0], -1, -1)
        )

    def post_dist(self, x_y: Tensor, misspecified: bool) -> dist.MultivariateNormal:
        if misspecified:
            return self.posterior_theta_given_x(x_y)
        else:
            return self.posterior_theta_given_y(x_y)

    def sample_reference_posterior(
        self,
        num_samples: int,
        observations: Tensor,
        misspecified: bool,
    ) -> Tensor:
        return self.post_dist(observations, misspecified).sample((num_samples,))

    def get_noisy_process(self):
        def noisy_process(x: Tensor) -> Tensor:
            parameters = {
                "loc": self.mean_noise(x),
                "covariance_matrix": self.noise_params["covariance_matrix"],
            }
            return dist.MultivariateNormal(**parameters).sample()

        return noisy_process

    def conditional_mutual_information(self) -> float:
        """Compute I(y; θ | x) analytically.

        When D=0 (no direct theta->y link), this is 0 (conditional independence holds).
        When D≠0, this measures the information y carries about θ beyond what x provides.

        I(y; θ | x) = 0.5 * log|Σ_{y|x}| - 0.5 * log|Σ_{y|x,θ}|

        where Σ_{y|x,θ} = Σ_noise (since y | x, θ ~ N(Cx + Dθ + d, Σ_noise))
        and Σ_{y|x} = D Σ_{θ|x} D^T + Σ_noise
        """
        D = self.direct_theta_coef
        Sigma_noise = self.noise_params["covariance_matrix"]

        # Σ_{θ|x} (posterior covariance of θ given x, same for all x in Gaussian case)
        Sigma_theta = self.prior_params["covariance_matrix"]
        A = self.simulator_params["coef"]
        Sigma_lik = self.simulator_params["covariance_matrix"]
        Sigma_x = A @ Sigma_theta @ A.T + Sigma_lik
        Sigma_theta_x = Sigma_theta - Sigma_theta @ A.T @ torch.linalg.inv(Sigma_x) @ A @ Sigma_theta

        # Σ_{y|x} = D Σ_{θ|x} D^T + Σ_noise
        Sigma_y_given_x = D @ Sigma_theta_x @ D.T + Sigma_noise

        # I(y; θ | x) = 0.5 * (log|Σ_{y|x}| - log|Σ_noise|)
        mi = 0.5 * (
            torch.linalg.slogdet(Sigma_y_given_x)[1]
            - torch.linalg.slogdet(Sigma_noise)[1]
        )
        return mi.item()

    def sample_denoiser(self, num_samples: int, y: Tensor) -> Tensor:
        raise NotImplementedError
