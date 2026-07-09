# Distributions based on Gaussians

# Libraries
import torch
import math
from .base import Distribution
from .utils import cov_of_mog, log_prob_and_grad_gauss, log_prob_and_grad_gauss_full, log_prob_and_grad_mog, \
    log_prob_and_grad_mog_full, log_prob_and_grad_and_hess_gauss, log_prob_and_grad_and_hess_gauss_full, \
    log_prob_and_grad_and_hess_mog, log_prob_and_grad_and_hess_mog_full, cov_of_mog


class Gauss(Distribution):

    def __init__(self, mean, variance):
        """Builds a diagonal covariance Gaussian distribution

        Args:
            * mean (torch.Tensor of shape (dim,)): Mean
            * variance (torch.Tensor of shape (dim,)): Diagonal of the covariance
        """
        # Call the parent constructor
        super().__init__(build_score=False, build_log_prob_and_grad=False,
                         build_laplacian=False, build_log_prob_and_grad_and_laplacian=False
                         )
        # Register everything to the buffer
        self.register_buffer('mean_', mean)
        self.register_buffer('variance_', variance)
        # Build the distribution
        self.build_dist()

    def build_dist(self):
        """Builds the inner torch.distributions.Normal object"""
        self.dist = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(
                loc=self.mean_, scale=torch.sqrt(self.variance_), validate_args=False
            ),
            reinterpreted_batch_ndims=1, validate_args=False
        )

    def mean(self):
        """Returns the mean of the distribution"""
        return self.mean_

    def variance(self):
        """Returns the variance of the distribution"""
        return self.variance_

    def covariance(self):
        """Returns the variance of the distribution"""
        return torch.diag(self.variance_)

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the distribution at x"""
        return log_prob_and_grad_gauss(x, self.mean_, self.variance_, return_log_prob=True)

    def score(self, x):
        """Evaluates the score of the distribution at x"""
        return -(x - self.mean_) / self.variance_

    def laplacian(self, x):
        """Evaluates the laplacian of the log-likelihood at x"""
        return torch.sum(log_prob_and_grad_and_hess_gauss(x, self.mean_, self.variance_, return_only_diag=True,
                                                          return_log_prob=False, return_grad=False), dim=-1)

    def log_prob_and_grad_and_laplacian(self, x):
        """Evaluates the log-likelihood as well as its gradient and laplacian at x"""
        log_prob, grad, diag_hessian = log_prob_and_grad_and_hess_gauss(x, self.mean_, self.variance_, return_only_diag=True,
                                                                        return_log_prob=True, return_grad=True)
        return log_prob, grad, torch.sum(diag_hessian, dim=-1)

    def marginal_distr(self, t, sde):
        """Get the marginal distribution at time t"""
        return Gauss(*sde.marginal_gauss_params(t, self.mean_, self.variance_))

    def marginal_log_prob_and_grad(self, t, x, sde):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        return sde.log_prob_and_grad_gauss(t, x, self.mean_, self.variance_)

    def marginal_log_prob_and_grad_and_hess(self, t, x, sde, return_only_diag=False):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        return sde.log_prob_and_grad_and_hess_gauss(t, x, self.mean_, self.variance_, return_only_diag)

    def sample_exact_denoising_kernel(self, s, t, x_t, sde, return_log_prob=False):
        """Sample from the exact denoising kernel

        Args:
            * s (torch.Tensor of shape (batch_size,)): Time s
            * t (torch.Tensor of shape (batch_size,)): Time t (s < t)
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * return_log_prob (bool): Whether to return the likelihood of the obtained sample
                (default is False)

        Returns:
            * x_s (torch.Tensor of the same shape as x_t): Denoised sample
            if return_log_prob:
                * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        return sde.exact_denoising_kernel_gauss_sample(s, t, x_t,
                                                       self.mean_, self.variance_, return_log_prob=return_log_prob)

    def log_prob_exact_denoising_kernel(self, s, t, x_s, x_t, sde):
        """Evaluate the log-likelihood of the exact denoising kernel

        Args:
            * s (torch.Tensor of shape (batch_size,)): Time s
            * t (torch.Tensor of shape (batch_size,)): Time t (s < t)
            * x_s (torch.Tensor of shape (batch_size, dim)): Evaluation point
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * sde (LinearSDE): SDE

        Returns:
            * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        return sde.exact_denoising_kernel_gauss_log_prob(x_s,
                                                         *sde.exact_denoising_kernel_gauss_params(s, t, x_t, self.mean_, self.variance_))

    def sample_exact_posterior(self, t, x_t, sde, return_log_prob=False):
        """Sample from the posterior

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * return_log_prob (bool): Whether to return the likelihood of the obtained sample
                (default is False)

        Returns:
            * x_s (torch.Tensor of the same shape as x_t): Denoised sample
            if return_log_prob:
                * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        return sde.exact_posterior_gauss_sample(t, x_t, self.mean_, self.variance_,
                                                return_log_prob=return_log_prob)

    def log_prob_exact_posterior(self, x, t, x_t, sde):
        """Evaluate the log-likelihood of the posterior

        Args:
            * x (torch.Tensor of shape (batch_size, dim)): Evaluation point
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * sde (LinearSDE): SDE

        Returns:
            * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        return sde.exact_denoising_kernel_gauss_log_prob(x, *sde.exact_posterior_gauss_params(
            t, x_t, self.mean_, self.variance_))

    def denoiser(self, t, x_t, sde):
        """Evaluate the denoiser

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point

        Returns:
            * x (torch.Tensor of the same shape as x_t): Denoised sample
        """
        return sde.denoiser_gauss(t, x_t, self.mean_, self.variance_)

    def posterior_covariance(self, t, x_t, sde, return_diag=False):
        """Evaluate the posterior's covariance

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * return_diag (bool): Whether to return only the diagonal of the covariance
                (default is False)

        Returns:
            * cov (torch.Tensor of shape (batch_size, dim, dim) or (batch_size, dim)): Covariance
        """
        return sde.posterior_covariance_gauss(t, x_t, self.mean_, self.variance_, return_diag=return_diag)


class GaussFull(Distribution):

    def __init__(self, mean, covariance, precompute=False):
        """Builds a full covariance Gaussian distribution

        Args:
            * mean (torch.Tensor of shape (dim,)): Mean
            * variance (torch.Tensor of shape (dim,)): Covariance
            * precompute (bool): Precompute the precision and the log determinant
                (default is False)
        """
        # Call the parent constructor
        super().__init__(build_score=False, build_log_prob_and_grad=False,
                         build_laplacian=False, build_log_prob_and_grad_and_laplacian=False
                         )
        # Register everything to the buffer
        self.register_buffer('mean_', mean)
        self.register_buffer('covariance_', covariance)
        if precompute:
            self.register_buffer('precision', torch.linalg.inv(covariance))
            self.register_buffer('covariance_log_det', torch.logdet(covariance))
        else:
            self.precision = None
            self.covariance_log_det = None
        # Build the distribution
        self.build_dist()

    def build_dist(self):
        """Builds the inner torch.distributions.MultivariateNormal object"""
        self.dist = torch.distributions.MultivariateNormal(
            loc=self.mean_, covariance_matrix=self.covariance_, validate_args=False
        )

    def mean(self):
        """Returns the mean of the distribution"""
        return self.mean_

    def variance(self):
        """Returns the variance of the distribution"""
        return torch.diag(self.covariance_)

    def covariance(self):
        """Returns the variance of the distribution"""
        return self.covariance_

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the distribution at x"""
        return log_prob_and_grad_gauss_full(x, self.mean_, self.covariance_,
                                            precision=self.precision, covariance_log_det=self.covariance_log_det,
                                            return_log_prob=True)

    def score(self, x):
        """Evaluates the score of the distribution at x"""
        diff = x - self.mean_.unsqueeze(0)
        if self.precision is None:
            return -torch.linalg.solve(self.covariance_.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)
        else:
            return -torch.matmul(self.precision.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)

    def laplacian(self, x):
        """Evaluates the laplacian of the log-likelihood at x"""
        return torch.sum(log_prob_and_grad_and_hess_gauss_full(x, self.mean_, self.covariance_, precision=self.precision,
                                                               return_only_diag=True, return_log_prob=False, return_grad=False), dim=-1)

    def log_prob_and_grad_and_laplacian(self, x):
        """Evaluates the log-likelihood as well as its gradient and laplacian at x"""
        log_prob, grad, diag_hessian = log_prob_and_grad_and_hess_gauss_full(x, self.mean_, self.covariance_, precision=self.precision,
                                                                             return_only_diag=True, return_log_prob=True, return_grad=True)
        return log_prob, grad, torch.sum(diag_hessian, dim=-1)

    def marginal_distr(self, t, sde):
        """Get the marginal distribution at time t"""
        return GaussFull(*sde.marginal_gauss_params(t, self.mean_, self.covariance_))

    def marginal_log_prob_and_grad(self, t, x, sde):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        return sde.log_prob_and_grad_gauss(t, x, self.mean_, self.covariance_)

    def marginal_log_prob_and_grad_and_hess(self, t, x, sde, return_only_diag=False):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        return sde.log_prob_and_grad_and_hess_gauss(t, x, self.mean_, self.covariance_, return_only_diag)

    def sample_exact_denoising_kernel(self, s, t, x_t, sde, return_log_prob=False):
        raise NotImplementedError('sample_exact_denoising_kernel is not implemented for full covariance.')

    def log_prob_exact_denoising_kernel(self, s, t, x_s, x_t, sde):
        raise NotImplementedError('log_prob_exact_denoising_kernel is not implemented for full covariance.')

    def sample_exact_posterior(self, t, x_t, sde, return_log_prob=False):
        raise NotImplementedError('sample_exact_posterior is not implemented for full covariance.')

    def log_prob_exact_posterior(self, x, t, x_t, sde):
        raise NotImplementedError('log_prob_exact_posterior is not implemented for full covariance.')

    def denoiser(self, t, x_t, sde):
        raise NotImplementedError('denoiser is not implemented for full covariance.')

    def posterior_covariance(self, t, x_t, sde, return_diag=False):
        raise NotImplementedError('posterior_covariance is not implemented for full covariance.')


class MeanFreeGauss(Gauss):

    def __init__(self, scalar_variance, n_particles, dim_phys):
        """Builds a mean-free Gaussian distribution

        Args:
            * scalar_variance (float): Scalar variance
            * n_particles (int): Number of particles
            * dim_phys (int): Spatial dimension of the space
        """
        # Call the parent constructor
        if isinstance(scalar_variance, torch.Tensor):
            variance = scalar_variance.cpu()
        variance = variance * torch.ones((n_particles * dim_phys,))
        super().__init__(
            mean=torch.zeros((n_particles * dim_phys,)),
            variance=variance
        )
        self.register_buffer('scalar_variance',
            torch.tensor(scalar_variance) if isinstance(scalar_variance, float) else scalar_variance)
        self.n_particles = n_particles
        self.dim_phys = dim_phys
        self.dim = n_particles * dim_phys

    def sample(self, sample_shape):
        """Returns samples from the distribution of shape sample_shape"""
        samples = super().sample(sample_shape)
        samples = samples.view((*sample_shape, self.n_particles, self.dim_phys))
        samples -= samples.mean(dim=-2, keepdims=True)
        return samples

    def log_prob(self, x):
        """Evaluates the log-likelihood of the distribution at x"""
        log_prob = super().log_prob(x.view((-1, self.dim)))
        log_prob -= -0.5 * self.dim * torch.log(2 * torch.pi * self.scalar_variance)
        log_prob += -0.5 * (self.n_particles-1) * self.dim_phys * torch.log(2 * torch.pi * self.scalar_variance)
        return log_prob

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the distribution at x"""
        log_prob, grad = super().log_prob_and_grad(x.view((-1, self.dim)))
        grad = grad.view((-1, self.n_particles, self.dim_phys))
        log_prob -= -0.5 * self.dim * torch.log(2 * torch.pi * self.scalar_variance)
        log_prob += -0.5 * (self.n_particles-1) * self.dim_phys * torch.log(2 * torch.pi * self.scalar_variance)
        return log_prob, grad

    def log_prob_and_grad_and_laplacian(self, x):
        """Evaluates the log-likelihood as well as its gradient and laplacian at x"""
        log_prob, grad, diag_hessian = super().log_prob_and_grad_and_laplacian(x.view((-1, self.dim)))
        grad = grad.view((-1, self.n_particles, self.dim_phys))
        log_prob -= -0.5 * self.dim * torch.log(2 * torch.pi * self.scalar_variance)
        log_prob += -0.5 * (self.n_particles-1) * self.dim_phys * torch.log(2 * torch.pi * self.scalar_variance)
        return log_prob, grad, torch.sum(diag_hessian, dim=-1)

    def marginal_distr(self, t, sde):
        raise NotImplementedError('marginal_distr is not implemented for mean-free.')

    def marginal_log_prob_and_grad(self, t, x, sde):
        raise NotImplementedError('marginal_log_prob_and_grad is not implemented for mean-free.')

    def marginal_log_prob_and_grad_and_hess(self, t, x, sde, return_only_diag=False):
        raise NotImplementedError('marginal_log_prob_and_grad_and_hess is not implemented for mean-free.')

    def sample_exact_denoising_kernel(self, s, t, x_t, sde, return_log_prob=False):
        raise NotImplementedError('sample_exact_denoising_kernel is not implemented for mean-free.')

    def log_prob_exact_denoising_kernel(self, s, t, x_s, x_t, sde):
        raise NotImplementedError('log_prob_exact_denoising_kernel is not implemented for mean-free.')

    def sample_exact_posterior(self, t, x_t, sde, return_log_prob=False):
        raise NotImplementedError('sample_exact_posterior is not implemented for mean-free.')

    def log_prob_exact_posterior(self, x, t, x_t, sde):
        raise NotImplementedError('log_prob_exact_posterior is not implemented for mean-free.')

    def denoiser(self, t, x_t, sde):
        raise NotImplementedError('denoiser is not implemented for mean-free.')

    def posterior_covariance(self, t, x_t, sde, return_diag=False):
        raise NotImplementedError('posterior_covariance is not implemented for mean-free.')


class MOG(Distribution):

    def __init__(self, weights, means, variances):
        """Builds a mixture of Gaussians with diagonal covariances

        Args:
            * weights (torch.Tensor of shape (n_modes,)): Weights
            * means (torch.Tensor of shape (n_modes,dim)): Means
            * variances (torch.Tensor of shape (n_modes, dim)): Diagonal of the covariances
        """

        # Call the parent constructor
        super().__init__(build_score=False, build_log_prob_and_grad=False,
                         build_laplacian=False, build_log_prob_and_grad_and_laplacian=False
                         )
        # Register everything to the buffer
        self.n_modes = weights.shape[0]
        self.register_buffer('weights', weights / weights.sum())
        self.register_buffer('means', means)
        self.register_buffer('variances', variances)
        # Build the distribution
        self.build_dist()

    def build_dist(self):
        """Builds the inner torch.distributions.MixtureSameFamily object"""
        self.dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(self.weights),
            component_distribution=torch.distributions.Independent(
                base_distribution=torch.distributions.Normal(
                    loc=self.means, scale=torch.sqrt(self.variances),
                    validate_args=False
                ),
                reinterpreted_batch_ndims=1,
                validate_args=False
            ), validate_args=False
        )

    def get_bad_distribution(self):
        """Build a bad version of the distribution"""
        return MOG(torch.ones_like(self.weights), self.means, self.variances)

    def mean(self):
        """Returns the mean of the distribution"""
        return self.dist.mean

    def variance(self):
        """Returns the variance of the distribution"""
        return self.dist.variance

    def covariance(self, return_diag=False):
        """Compute the covariance matrix"""
        return cov_of_mog(self.weights, self.means, self.variances, return_diag=return_diag)

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the distribution at x"""
        return log_prob_and_grad_mog(x, self.weights, self.means, self.variances, return_log_prob=True)

    def score(self, x):
        """Evaluates the score of the distribution at x"""
        return self.log_prob_and_grad(x)[1]

    def laplacian(self, x):
        """Evaluates the laplacian of the log-likelihood at x"""
        return torch.sum(log_prob_and_grad_and_hess_mog(x, self.weights, self.means, self.variances,
                                                        return_only_diag=True, return_log_prob=False, return_grad=False), dim=-1)

    def log_prob_and_grad_and_laplacian(self, x):
        """Evaluates the log-likelihood as well as its gradient and laplacian at x"""
        log_prob, grad, diag_hessian = log_prob_and_grad_and_hess_mog(x, self.weights, self.means, self.variances,
                                                                      return_only_diag=True, return_log_prob=True, return_grad=True)
        return log_prob, grad, torch.sum(diag_hessian, dim=-1)

    def marginal_distr(self, t, sde):
        """Get the marginal distribution at time t"""
        return MOG(*sde.marginal_mog_params(t, self.weights, self.means, self.variances))

    def marginal_log_prob_and_grad(self, t, x, sde):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        return sde.log_prob_and_grad_mog(t, x, self.weights, self.means, self.variances)

    def marginal_log_prob_and_grad_and_hess(self, t, x, sde, return_only_diag=False):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        return sde.log_prob_and_grad_and_hess_mog(t, x, self.weights, self.means, self.variances, return_only_diag)

    def sample_exact_denoising_kernel(self, s, t, x_t, sde, return_log_prob=False):
        """Sample from the exact denoising kernel

        Args:
            * s (torch.Tensor of shape (batch_size,)): Time s
            * t (torch.Tensor of shape (batch_size,)): Time t (s < t)
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * sde (LinearSDE): SDE
            * return_log_prob (bool): Whether to return the likelihood of the obtained sample
                (default is False)

        Returns:
            * x_s (torch.Tensor of the same shape as x_t): Denoised sample
            if return_log_prob:
                * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        return sde.exact_denoising_kernel_mog_sample(s, t, x_t,
                                                     self.weights, self.means, self.variances, return_log_prob=return_log_prob)

    def log_prob_exact_denoising_kernel(self, s, t, x_s, x_t, sde):
        """Evaluate the log-likelihood of the exact denoising kernel

        Args:
            * s (torch.Tensor of shape (batch_size,)): Time s
            * t (torch.Tensor of shape (batch_size,)): Time t (s < t)
            * x_s (torch.Tensor of shape (batch_size, dim)): Evaluation point
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * sde (LinearSDE): SDE

        Returns:
            * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        return sde.exact_denoising_kernel_mog_log_prob(x_s,
                                                       *sde.exact_denoising_kernel_mog_params(s, t, x_t, self.weights, self.means, self.variances))

    def sample_exact_posterior(self, t, x_t, sde, return_log_prob=False):
        """Sample from the posterior

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * sde (LinearSDE): SDE
            * return_log_prob (bool): Whether to return the likelihood of the obtained sample
                (default is False)

        Returns:
            * x (torch.Tensor of the same shape as x_t): Denoised sample
            if return_log_prob:
                * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        return sde.exact_posterior_mog_sample(t, x_t, self.weights, self.means, self.variances,
                                              return_log_prob=return_log_prob)

    def log_prob_exact_posterior(self, x, t, x_t, sde):
        """Evaluate the log-likelihood of the posterior

        Args:
            * x (torch.Tensor of shape (batch_size, dim)): Evaluation point
            * t (torch.Tensor of shape (batch_size,)): Time t (s < t)
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * sde (LinearSDE): SDE

        Returns:
            * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        return sde.exact_posterior_mog_log_prob(x, *sde.exact_posterior_mog_params(
            t, x_t, self.weights, self.means, self.variances))

    def denoiser(self, t, x_t, sde):
        """Evaluate the denoiser

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point

        Returns:
            * x (torch.Tensor of the same shape as x_t): Denoised sample
        """
        return sde.denoiser_mog(t, x_t, self.weights, self.means, self.variances)

    def posterior_covariance(self, t, x_t, sde, return_diag=False):
        """Evaluate the posterior's covariance

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * return_diag (bool): Whether to return only the diagonal of the covariance
                (default is False)

        Returns:
            * cov (torch.Tensor of shape (batch_size, dim, dim) or (batch_size, dim)): Covariance
        """
        return sde.posterior_covariance_mog(t, x_t, self.weights, self.means, self.variances,
                                            return_diag=return_diag)

    def compute_mode_weights(self, samples, weights=None):
        """Compute the empirical mode weights"""
        log_prob_per_mode = self.dist.component_distribution.log_prob(
            samples.unsqueeze(1).to(self.weights.device)
        )
        idx = torch.argmax(log_prob_per_mode, dim=-1)
        if weights is None:
            counts = torch.FloatTensor([
                (idx == k).sum() for k in range(self.n_modes)
            ]).to(self.weights.device)
            return counts.flatten() / counts.sum()
        else:
            return torch.FloatTensor([
                weights[idx == k].sum() for k in range(self.n_modes)
            ]).to(self.weights.device)

    def entropy(self, samples, weights=None, hist=None):
        """Compute the entropy of the mode weights"""
        if hist is None:
            hist = self.compute_mode_weights(samples, weights=weights)
        return -torch.sum(hist * (torch.log(hist) / math.log(self.n_modes))).item()

    def kl_weights(self, samples, weights=None, hist=None):
        """Compute the KL between the empirical weights and the true ones"""
        if hist is None:
            hist = self.compute_mode_weights(samples, weights=weights)
        true_hist = self.weights.flatten() / self.weights.sum()
        return torch.sum(true_hist * torch.log(true_hist / hist)).item()

    def tv_weights(self, samples, weights=None, hist=None):
        """Compute the TV between the empirical weights and the true ones"""
        if hist is None:
            hist = self.compute_mode_weights(samples, weights=weights)
        true_hist = self.weights.flatten() / self.weights.sum()
        return torch.sum(torch.abs(hist - true_hist)).item()

    def compute_forgotten_modes(self, samples, tol=0.05, weights=None, hist=None):
        """Compute the number of forgotten modes"""
        if hist is None:
            hist = self.compute_mode_weights(samples, weights=weights)
        true_hist = self.weights.flatten() / self.weights.sum()
        return torch.sum(hist < tol * true_hist.min()).item() / self.n_modes

    def compute_metrics(self, samples, weights=None, ref_samples=None, compute_standard_metrics=False,
                        skip_costly_metrics=True, return_hist=False):
        """Compute various metrics based on samples

        Args:
            * samples (torch.Tensor of shape (batch_size, *data_shape)): Samples to compare against
            * weights (torch.Tensor of shape (batch_size,)): Weights of the samples (default is None)
            * ref_samples (torch.Tensor of same shape as samples): Reference samples
                If None, they will be sampled manually. (default is None)
            * compute_standard_metrics (bool): Whether to compute standard metrics (KS, MMD, W2, ..)
                (default is False)
            * skip_costly_metrics (bool): Whether to skip costly metrics (involving OT computations)
                (default is True)
            * return_hist (bool): Whether to return the empirical mode weights (default is False)

        Returns:
            * metrics (dict): All the computed metrics
            if return_hist:
                * hist (torch.Tensor of shape (n_modes,)): Empirical mode weights
        """
        # Get the standard statistics
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                      compute_standard_metrics=compute_standard_metrics,
                                      skip_costly_metrics=skip_costly_metrics
                                      )
        # Compute empirical mode weights
        hist = self.compute_mode_weights(samples, weights=weights)
        # Compute the metrics
        ret['emc'] = self.entropy(samples, hist=hist)
        ret['kl_weights'] = self.kl_weights(samples, hist=hist)
        ret['tv_weights'] = self.tv_weights(samples, hist=hist)
        ret['perc_forgotten_modes'] = self.compute_forgotten_modes(samples, hist=hist)
        if return_hist:
            return ret, hist
        else:
            return ret


class MOGFull(MOG):

    def __init__(self, weights, means, covariances, precompute=False):
        """Builds a mixture of Gaussians with full covariances

        Args:
            * weights (torch.Tensor of shape (n_modes,)): Weights
            * means (torch.Tensor of shape (n_modes,dim)): Means
            * covariances (torch.Tensor of shape (n_modes, dim)): Covariances
            * precompute (bool): Precompute the precisions and the log determinants
                (default is False)
        """

        # Call the parent constructor
        super(MOG, self).__init__(build_score=False, build_log_prob_and_grad=False,
                                  build_laplacian=False, build_log_prob_and_grad_and_laplacian=False
                                  )
        # Register everything to the buffer
        self.n_modes = weights.shape[0]
        self.register_buffer('weights', weights / weights.sum())
        self.register_buffer('means', means)
        self.register_buffer('covariances', covariances)
        if precompute:
            self.register_buffer('precisions', torch.linalg.inv(covariances))
            self.register_buffer('covariances_log_det', torch.logdet(covariances))
        else:
            self.precisions = None
            self.covariances_log_det = None
        # Build the distribution
        self.build_dist()

    def build_dist(self):
        """Builds the inner torch.distributions.MixtureSameFamily object"""
        self.dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(self.weights),
            component_distribution=torch.distributions.MultivariateNormal(
                loc=self.means, covariance_matrix=self.covariances, validate_args=False
            ), validate_args=False
        )

    def get_bad_distribution(self):
        """Build a bad version of the distribution"""
        return MOGFull(torch.ones_like(self.weights), self.means, self.covariances)

    def mean(self):
        """Returns the mean of the distribution"""
        return self.dist.mean

    def variance(self):
        """Returns the variance of the distribution"""
        return self.dist.variance

    def covariance(self, return_diag=False):
        """Compute the covariance matrix"""
        return cov_of_mog(self.weights, self.means, self.covariances, return_diag=return_diag)

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the distribution at x"""
        return log_prob_and_grad_mog_full(x, self.weights, self.means, self.covariances,
                                          precisions=self.precisions, covariances_log_det=self.covariances_log_det,
                                          return_log_prob=True)

    def laplacian(self, x):
        """Evaluates the laplacian of the log-likelihood at x"""
        return torch.sum(log_prob_and_grad_and_hess_mog_full(x, self.weights, self.means, self.covariances, precisions=self.precisions,
                                                             return_only_diag=True, return_log_prob=False, return_grad=False), dim=-1)

    def log_prob_and_grad_and_laplacian(self, x):
        """Evaluates the log-likelihood as well as its gradient and laplacian at x"""
        log_prob, grad, diag_hessian = log_prob_and_grad_and_hess_mog_full(x, self.weights, self.means, self.covariances, precisions=self.precisions,
                                                                           return_only_diag=True, return_log_prob=True, return_grad=True)
        return log_prob, grad, torch.sum(diag_hessian, dim=-1)

    def marginal_distr(self, t, sde):
        """Get the marginal distribution at time t"""
        return MOGFull(*sde.marginal_mog_params(t, self.weights, self.means, self.covariances))

    def marginal_log_prob_and_grad(self, t, x, sde):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        return sde.log_prob_and_grad_mog(t, x, self.weights, self.means, self.covariances)

    def marginal_log_prob_and_grad_and_hess(self, t, x, sde, return_only_diag=False):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        return sde.log_prob_and_grad_and_hess_mog(t, x, self.weights, self.means, self.covariances, return_only_diag)

    def sample_exact_denoising_kernel(self, s, t, x_t, sde, return_log_prob=False):
        raise NotImplementedError('sample_exact_denoising_kernel is not implemented for full covariance.')

    def log_prob_exact_denoising_kernel(self, s, t, x_s, x_t, sde):
        raise NotImplementedError('log_prob_exact_denoising_kernel is not implemented for full covariance.')

    def sample_exact_posterior(self, t, x_t, sde, return_log_prob=False):
        raise NotImplementedError('sample_exact_posterior is not implemented for full covariance.')

    def log_prob_exact_posterior(self, x, t, x_t, sde):
        raise NotImplementedError('log_prob_exact_posterior is not implemented for full covariance.')

    def denoiser(self, t, x_t, sde):
        raise NotImplementedError('denoiser is not implemented for full covariance.')

    def posterior_covariance(self, t, x_t, sde, return_diag=False):
        raise NotImplementedError('posterior_covariance is not implemented for full covariance.')


class TwoModes(MOG):

    def __init__(self, dim, a=1.0, centered=False, ill_conditioned='not'):
        """Builds a mixture of two unequally weighted Gaussians with diagonal covariances

            p = (2/3) N(-a 1_d, C) + (1/3) N(+a 1_d, C)

        Args:
            * dim (int): Dimension of the mixture
            * a (float): Half distance between the modes (default is 1.0)
            * centered (bool): Whether to center the distribution (default is False)
            * ill_conditioned (str): Type of conditioning (either 'not', 'medium' or 'hard')
                (default is False)
        """
        # Build the parameters
        assert ill_conditioned in ['not', 'medium', 'hard']
        # 1. Mixture weights : 2/3, 1/3
        weights = torch.FloatTensor([2., 1.])
        # 2. Means : -a, a
        means = torch.stack([-a * torch.ones((dim,)), a * torch.ones((dim,))])
        if centered:
            means += (a / 3.) * torch.ones((dim,))
        # 3. Standard deviation
        if ill_conditioned == 'medium':
            variances = 0.05 * torch.logspace(-1, 0., dim).unsqueeze(0).expand(2, -1)
        elif ill_conditioned == 'hard':
            variances = 0.05 * torch.logspace(-2., 0., dim).unsqueeze(0).expand(2, -1)
        else:
            variances = 0.05 * torch.ones_like(means)
        # Call the param constructor
        super().__init__(weights=weights, means=means, variances=variances)

    def compute_mode_weight(self, samples, weights=None, hist=None):
        """Compute the weight of the strongest mode"""
        if hist is None:
            hist = self.compute_mode_weights(samples, weights=weights)
        return 100. * hist[0].item()

    def compute_metrics(self, samples, weights=None, ref_samples=None, compute_standard_metrics=False,
                        skip_costly_metrics=True):
        """Compute various metrics based on samples

        Args:
            * samples (torch.Tensor of shape (batch_size, *data_shape)): Samples to compare against
            * weights (torch.Tensor of shape (batch_size,)): Weights of the samples (default is None)
            * ref_samples (torch.Tensor of same shape as samples): Reference samples
                If None, they will be sampled manually. (default is None)
            * compute_standard_metrics (bool): Whether to compute standard metrics (KS, MMD, W2, ..)
                (default is False)
            * skip_costly_metrics (bool): Whether to skip costly metrics (involving OT computations)
                (default is True)

        Returns:
            * metrics (dict): All the computed metrics
        """
        # Get the standard statistics
        ret, hist = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                            compute_standard_metrics=compute_standard_metrics,
                                            skip_costly_metrics=skip_costly_metrics,
                                            return_hist=True
                                            )
        # Add the mode weight
        ret['mode_weight'] = self.compute_mode_weight(samples, weights=weights, hist=hist)
        ret['mode_weight_abs_err'] = abs(ret['mode_weight'] - 100. * self.weights[0].item())
        return ret


class TwoModesFull(MOGFull):

    def __init__(self, dim=2, a=1.0, centered=False, ill_conditioned='medium', rand_factor=5., seed_q=42,
                 precompute=False):
        """Builds a mixture of two unequally weighted Gaussians with full covariances

            p = (2/3) N(-a 1_d, C) + (1/3) N(+a 1_d, C)

        Args:
            * dim (int): Dimension of the mixture
            * a (float): Half distance between the modes (default is 1.0)
            * centered (bool): Whether to center the distribution (default is False)
            * ill_conditioned (str): Type of conditioning (either 'not', 'medium' or 'hard')
                (default is False)
            * rand_factor (float): Uniform spread of the random matrix used to build the covariance via QR
                (default is 5.)
            * seed_q (int): Seed for the sampling of the random matrix (default is 42)
            * precompute (bool): Precompute the precisions and the log determinants
                (default is False)
        """
        # Build the parameters
        assert ill_conditioned in ['medium', 'hard']
        # 1. Mixture weights : 2/3, 1/3
        weights = torch.FloatTensor([2., 1.])
        # 2. Means : -a, a
        means = torch.stack([-a * torch.ones((dim,)), a * torch.ones((dim,))])
        if centered:
            means += (a / 3.) * torch.ones((dim,))
        # 3. Build a orthonormal matrix
        generator = torch.Generator()
        generator.manual_seed(seed_q)
        q = torch.linalg.qr(rand_factor * torch.rand((dim, dim), generator=generator), mode='complete').Q
        # 4. Covariance
        if ill_conditioned == 'hard':
            covariances = torch.diag(0.05 * torch.logspace(-2., 0., dim))
        else:
            covariances = torch.diag(0.05 * torch.logspace(-1, 0., dim))
        covariances = torch.matmul(q, torch.matmul(covariances, q.T))
        covariances = torch.stack([covariances, covariances.clone()], dim=0)
        # Call the param constructor
        super().__init__(weights=weights, means=means, covariances=covariances, precompute=precompute)

    def compute_mode_weight(self, samples, hist=None):
        """Compute the weight of the strongest mode"""
        if hist is None:
            hist = self.compute_mode_weights(samples)
        return 100. * hist[0].item()

    def compute_metrics(self, samples, weights=None, ref_samples=None, compute_standard_metrics=False,
                        skip_costly_metrics=True):
        """Compute various metrics based on samples

        Args:
            * samples (torch.Tensor of shape (batch_size, *data_shape)): Samples to compare against
            * weights (torch.Tensor of shape (batch_size,)): Weights of the samples (default is None)
            * ref_samples (torch.Tensor of same shape as samples): Reference samples
                If None, they will be sampled manually. (default is None)
            * compute_standard_metrics (bool): Whether to compute standard metrics (KS, MMD, W2, ..)
                (default is False)
            * skip_costly_metrics (bool): Whether to skip costly metrics (involving OT computations)
                (default is True)

        Returns:
            * metrics (dict): All the computed metrics
        """
        # Get the standard statistics
        ret, hist = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
                                            compute_standard_metrics=compute_standard_metrics,
                                            skip_costly_metrics=skip_costly_metrics,
                                            return_hist=True
                                            )
        # Add the mode weight
        ret['mode_weight'] = self.compute_mode_weight(samples, hist=hist)
        ret['mode_weight_abs_err'] = abs(ret['mode_weight'] - 100. * self.weights[0].item())
        return ret


class FourtyModesMOG(MOG):

    def __init__(self, dim=2, n_modes=40, loc_scaling=40, log_var_scaling=1.0, seed=0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        weights = torch.ones((n_modes,)) / n_modes
        means = (torch.rand((n_modes, dim), generator=generator) - 0.5)*2 * loc_scaling
        log_variances = torch.ones((n_modes, dim)) * log_var_scaling
        super().__init__(weights, means, torch.nn.functional.softplus(log_variances))


class FourtyLinearModesMOG(MOG):

    def __init__(self, dim=2, n_modes=40, loc_scaling=40, log_var_scaling=1.0, seed=0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        weights = torch.linspace(1., 40., 40)
        weights = weights / weights.sum()
        means = (torch.rand((n_modes, dim), generator=generator) - 0.5)*2 * loc_scaling
        log_variances = torch.ones((n_modes, dim)) * log_var_scaling
        super().__init__(weights, means, torch.nn.functional.softplus(log_variances))


def full_standardize_gauss_params(L, b, means, covariances, upper=False):
    """Apply the transform T(X) = L^{-1}(X - b)

    Args:
        * L (torch.Tensor of shape (dim, dim)): Triangular matrix
        * b (torch.Tensor of shape (dim,)): Shift vector
        * means (torch.Tensor of shape (n_modes, dim)): Means
        * covariances (torch.Tensor of shape (n_modes, dim, dim)): Covariances

    Returns:
        * means_new (torch.Tensor of shape (n_modes, dim)): New means
        * covariances_new (torch.Tensor of shape (n_modes, dim, dim)): New covariances
    """
    # Transform the means
    means_new = torch.linalg.solve_triangular(L.unsqueeze(0), means.unsqueeze(-1) - b.view((1, -1, 1)),
                                              upper=upper).squeeze(-1)
    # Transform the covariances
    L_inv_covariances = torch.linalg.solve_triangular(L, covariances, upper=upper)
    covariances_new = torch.linalg.solve_triangular(L, L_inv_covariances.transpose(-1, -2), upper=upper)
    covariances_new = covariances_new.transpose(-1, -2)
    # Return everything
    return means_new, covariances_new


def diag_standardize_gauss_params(S, b, means, covariances, upper=False):
    """Apply the transform T(X) = S^{-1}(X - b)

    Args:
        * S (torch.Tensor of shape (dim,)): Scaling vector
        * b (torch.Tensor of shape (dim,)): Shift vector
        * means (torch.Tensor of shape (n_modes, dim)): Means
        * covariances (torch.Tensor of shape (n_modes, dim, dim) or (n_modes, dim)): (Co)variances

    Returns:
        * means_new (torch.Tensor of shape (n_modes, dim)): New means
        * covariances_new (torch.Tensor of same shape as covariances): New covariances
    """
    # Transform the means
    means_new = (means - b.unsqueeze(0)) / S.unsqueeze(0)
    # Transform the covariances
    if len(covariances.shape) == 3:
        covariances_new = covariances / S[:, None] / S[None, :]
    else:
        covariances_new = covariances / torch.square(S)
    # Return everything
    return means_new, covariances_new


def standardize_gauss(dist, mode='diag'):
    """Return a standardized version of the input distribution

    Args:
        * dist (Distribution): Distribution in the Gaussian familly
        * mode (str): Type of standardisation (default is 'diag')

    Returns:
        * standard_dist (Distribution): Standardized distribution
    """
    # Work on the full mode
    if mode == 'full':
        if isinstance(dist, GaussFull):
            mean_new, covariance_new = full_standardize_gauss_params(
                L=torch.linalg.cholesky(dist.covariance),
                b=dist.dist.mean,
                means=dist.mean.unsqueeze(0),
                covariances=dist.covariance.unsqueeze(0)
            )
            return GaussFull(mean_new.squeeze(0), covariance_new.squeeze(0))
        elif isinstance(dist, Gauss):
            mean_new, covariance_new = full_standardize_gauss_params(
                L=torch.linalg.cholesky(torch.diag(dist.variance)),
                b=dist.dist.mean,
                means=dist.mean.unsqueeze(0),
                covariances=torch.diag(dist.variance).unsqueeze(0)
            )
            return GaussFull(mean_new.squeeze(0), covariance_new.squeeze(0))
        elif isinstance(dist, MOGFull):
            means_new, covariances_new = full_standardize_gauss_params(
                L=torch.linalg.cholesky(dist.covariance()),
                b=dist.dist.mean,
                means=dist.means,
                covariances=dist.covariances
            )
            return MOGFull(dist.weights, means_new, covariances_new)
        else:
            covariances = torch.stack([torch.diag(M) for M in dist.variances], dim=0)
            means_new, covariances_new = full_standardize_gauss_params(
                L=torch.linalg.cholesky(dist.covariance()),
                b=dist.dist.mean,
                means=dist.means,
                covariances=covariances
            )
            return MOGFull(dist.weights, means_new, covariances_new)
    else:
        # Get the variance
        variance = dist.dist.variance
        # Build S and b
        if mode == 'diag':
            S = torch.sqrt(variance)
        else:
            S = torch.sqrt(torch.mean(variance) * torch.ones_like(variance))
        b = dist.dist.mean
        # Apply S and b
        if isinstance(dist, GaussFull):
            mean_new, covariance_new = diag_standardize_gauss_params(S=S, b=b,
                                                                     means=dist.mean.unsqueeze(0),
                                                                     covariances=dist.covariance.unsqueeze(0)
                                                                     )
            return GaussFull(mean_new.squeeze(0), covariance_new.squeeze(0))
        elif isinstance(dist, Gauss):
            mean_new, variance_new = diag_standardize_gauss_params(S=S, b=b,
                                                                   means=dist.mean.unsqueeze(0),
                                                                   covariances=dist.variance.unsqueeze(0)
                                                                   )
            return Gauss(mean_new.squeeze(0), variance_new.squeeze(0))
        elif isinstance(dist, MOGFull):
            means_new, covariances_new = diag_standardize_gauss_params(S=S, b=b,
                                                                       means=dist.means, covariances=dist.covariances
                                                                       )
            return MOGFull(dist.weights, means_new, covariances_new)
        else:
            means_new, variances_new = diag_standardize_gauss_params(S=S, b=b,
                                                                     means=dist.means, covariances=dist.variances
                                                                     )
            return MOG(dist.weights, means_new, variances_new)
