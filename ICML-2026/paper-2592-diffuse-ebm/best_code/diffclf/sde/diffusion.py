# Implementation of the linear SDEs used in Diffusion Models

# Libraries
import math
import torch
import numpy as np
from ..utils.se3_utils import remove_mean
from ..distr.gauss import Gauss
from ..distr.utils import log_prob_and_grad_gauss, log_prob_and_grad_gauss_full, \
    log_prob_and_grad_mog, \
    log_prob_and_grad_mog_full, \
    log_prob_and_grad_and_hess_gauss, \
    log_prob_and_grad_and_hess_gauss_full, \
    log_prob_and_grad_and_hess_mog, \
    log_prob_and_grad_and_hess_mog_full, \
    mean_of_mog_batch, \
    cov_of_mog_batch, \
    ReshapeWrapper


class LinearSDE(torch.nn.Module):
    """Implementation of a linear SDE
        d X_t = f(t) X_t dt + g(t) dW_t
    """

    def __init__(self, T):
        super().__init__()
        self.register_buffer("T", torch.tensor(T, dtype=torch.float), persistent=False)
        # Vectorize the log_prob_and_grad
        self.log_prob_and_grad_gauss = torch.vmap(
            lambda t, x, mean, variance: tuple(y.squeeze(0).squeeze(0) for y in
                                               self.log_prob_and_grad_gauss_scalar(t, x.unsqueeze(0), mean, variance)),
            in_dims=(0, 0, None, None)
        )
        self.log_prob_and_grad_mog = torch.vmap(
            lambda t, x, weights, means, variances: tuple(y.squeeze(0).squeeze(0) for y in
                                                          self.log_prob_and_grad_mog_scalar(t, x.unsqueeze(0), weights, means, variances)),
            in_dims=(0, 0, None, None, None)
        )
        self.log_prob_and_grad_and_hess_gauss = torch.vmap(
            lambda t, x, mean, variance, return_diag_hess: tuple(y.squeeze(0).squeeze(0) for y in
                                                                 self.log_prob_and_grad_and_hess_gauss_scalar(t, x.unsqueeze(0), mean, variance, return_diag_hess)),
            in_dims=(0, 0, None, None, None)
        )
        self.log_prob_and_grad_and_hess_mog = torch.vmap(
            lambda t, x, weights, means, variances, return_diag_hess: tuple(y.squeeze(0).squeeze(0) for y in
                                                                            self.log_prob_and_grad_and_hess_mog_scalar(t, x.unsqueeze(0), weights, means, variances, return_diag_hess)),
            in_dims=(0, 0, None, None, None, None)
        )
        self.exact_denoising_kernel_gauss_params = torch.vmap(
            self.exact_denoising_kernel_gauss_params_scalar, in_dims=(0, 0, 0, None, None)
        )
        self.exact_denoising_kernel_mog_params = torch.vmap(
            self.exact_denoising_kernel_mog_params_scalar, in_dims=(0, 0, 0, None, None, None)
        )
        self.exact_posterior_gauss_params = torch.vmap(
            self.exact_posterior_gauss_params_scalar, in_dims=(0, 0, None, None)
        )
        self.exact_posterior_mog_params = torch.vmap(
            self.exact_posterior_mog_params_scalar, in_dims=(0, 0, None, None, None)
        )

    def get_base_dist(self, data_shape):
        """Returns the base distribution (i.e., the marginal distribution at time T)"""
        raise NotImplementedError

    def sample_base_dist(self, shape, data_shape):
        """Sample the base distribution (i.e., the marginal distribution at time T)"""
        raise NotImplementedError

    def f(self, t):
        """Function f"""
        raise NotImplementedError

    def g(self, t):
        """Function g"""
        raise NotImplementedError

    def s(self, t):
        """Value of exp(int_0^t f(u) du)"""
        raise NotImplementedError

    def s_dot(self, t):
        """Derivative of s"""
        return self.s(t) * self.f(t)

    def sigma_sq(self, t):
        """Value of int_0^t g^2(u) / s^2(u) du"""
        raise NotImplementedError

    def sigma_sq_dot(self, t):
        """Derivative of sigma_sq"""
        return torch.square(self.g(t) / self.s(t))

    def sigma(self, t):
        """Square root of sigma_sq"""
        return torch.sqrt(self.sigma_sq(t))

    def sigma_dot(self, t):
        """Derivative of sigma"""
        return 0.5 * self.sigma_sq_dot(t) / self.sigma(t)

    def sigma_inv(self, sigma):
        """Inverse of the sigma function"""
        raise NotImplementedError
    
    def gamma_sq(self, t):
        """Product between s^2 and sigma_sq"""
        return torch.square(self.s(t)) * self.sigma_sq(t)
    
    def gamma_sq_dot(self, t):
        """Derivative of gamma_sq"""
        return 2. * self.f(t) * self.gamma_sq(t) + torch.square(self.g(t))

    def gamma(self, t):
        """Product between s and sigma"""
        return self.s(t) * self.sigma(t)

    def gamma_dot(self, t):
        """Derivate of gamma"""
        gamma_ = self.gamma(t)
        return self.f(t) * gamma_ + 0.5 * (torch.square(self.g(t)) / gamma_)

    def s_dot_over_gamma(self, t):
        """Ratio of s_dot and gamma"""
        return self.f(t) / self.sigma(t)

    def gamma_dot_over_gamma(self, t):
        """Ratio of gamma_dot and gamma"""
        return self.f(t) + 0.5 * torch.square(self.g(t)) / self.gamma_sq(t)

    def sigma_dot_over_sigma(self, t):
        """Ratio of sigma_dot and sigma"""
        return 0.5 * self.sigma_sq_dot(t) / self.sigma_sq(t)

    def s_dot_over_s(self, t):
        """Ratio of s_dot and s"""
        return self.f(t)

    def log_snr(self, t):
        """Get the log-SNR at time t"""
        alphas_bar_t, sigmas_sq_bar_t = self.transition_params_from_data(t)
        return 2. * torch.log(alphas_bar_t) - torch.log(sigmas_sq_bar_t)

    def log_snr_inv(self, l):
        """Get the log-SNR at time t"""
        return self.sigma_inv(torch.exp(-l / 2.))

    def get_snr_time_discretization(self, start, end, n_steps, n_attemps=1024):
        """Get SNR adapted time discretization

        Args:
            * start (float or torch.Tensor): Start time
            * end (float or torch.Tensor): End time
            * n_steps (int): Number of intermediate times
            * n_attemps (int): Number of bisection attemps (default is 1024)

        Returns:
            * ts (torch.Tensor of shape (n_steps,)): Time discretization
        """
        if isinstance(start, float):
            start = torch.tensor(start)
        if isinstance(end, float):
            end = torch.tensor(end)
        log_snr_start = self.log_snr(start)
        if torch.isnan(log_snr_start):
            raise ValueError('NaN SNR at t_0')
        log_snr_end = self.log_snr(end)
        if torch.isnan(log_snr_end):
            raise ValueError('NaN SNR at t_K')
        log_snr_range = torch.linspace(log_snr_start, log_snr_end, steps=n_steps, device=self.T.device)
        return torch.concat([
            torch.FloatTensor([start]).to(self.T.device),
            self.log_snr_inv(log_snr_range[1:-1]),
            torch.FloatTensor([end]).to(self.T.device)
        ], dim=0).sort().values

    def transition_params(self, s, t):
        """Mean and variance parameters for noising transition kernel s -> t (s < t)

        We have that X_t = alpha_s X_s + gamma_s Z with Z ~ N(0,I)
        where
            * alpha_s = exp(int_s^t f(u) du ) = exp( log s(t) - log s(s) )
            * (gamma_s)^2 = s^2(t) * (sigma^2(t) - sigma^2(s))

        This function returns alpha_s and (gamma_s)^2.
        """
        mean_factor = torch.exp(torch.log(self.s(t)) - torch.log(self.s(s)))
        var_factor = self.s(t) ** 2 * (self.sigma_sq(t) - self.sigma_sq(s))
        return mean_factor, var_factor

    def transition_params_from_data(self, t):
        """Mean and variance parameters for noising transition kernel 0 -> t

        We have that X_t = alpha_0 X_0 + gamma_0 Z with Z ~ N(0,I)

        This function returns alpha_0 and (gamma_0)^2.
        """
        mean_factor = self.s(t)
        var_factor = torch.square(mean_factor) * self.sigma_sq(t)
        return mean_factor, var_factor

    def noise_sample_params(self, t, x_0):
        """Mean and variance of p_{t|0}( . | x_0)

        Args:
            * t (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Times
            * x_0 (torch.Tensor of shape (batch_size, *data_shape)): Conditioning points

        Returns:
            * mean (torch.Tensor of shape (batch_size, *data_shape)): Mean of the conditional distribution
            * var (torch.Tensor of shape (batch_size, *data_shape)): Variance of the conditional distribution
        """

        mean_factor, var_factor = self.transition_params_from_data(t)
        return mean_factor * x_0, var_factor

    def marginal_gauss_params(self, t, mean, variance, is_mixture=False):
        """Mean and variance of the marginal with Gaussian data distribution

        Args:
            * t (float): Time
            * mean (torch.Tensor of shape (dim,)): Mean of the Gaussian distribution
            * variance (torch.Tensor of shape (dim,) or (dim, dim)): Variance of the Gaussian distribution
            * is_mixture (bool): Indicates whether mean and variance have an extra dimension (default is False)

        Returns:
            * mean_t (torch.Tensor of the same shape as mean): Mean of the marginal at time t
            * variance_t (torch.Tensor of the same shape as variance): Variance of the marginal at time t
        """
        mean_t, variance_t = self.noise_sample_params(t, mean)
        if is_mixture:
            if len(variance.shape) == 3:
                variance_t = variance_t * torch.eye(variance.shape[-1], device=variance.device)
                variance_t = variance_t.unsqueeze(0)
        else:
            if len(variance.shape) == 2:
                variance_t = variance_t * torch.eye(variance.shape[-1], device=variance.device)
        variance_t = variance_t + torch.square(self.s(t)) * variance
        return mean_t, variance_t

    def marginal_mog_params(self, t, weights, means, variances):
        """Mean and variance of the marginal with Gaussian mixture data distribution

        Args:
            * t (float): Time
            * weights (torch.Tensor of shape (n_modes,)): Weights of the modes
            * means (torch.Tensor of shape (n_modes, dim): Means of the modes
            * variances (torch.Tensor of shape (n_modes, dim)): Variances of the modes

        Returns:
            * weights_t (torch.Tensor of the same shape as weights): Weights of the modes at time t
            * means_t (torch.Tensor of the same shape as mean): Mean of the marginal at time t
            * variances_t (torch.Tensor of the same shape as variance): Variance of the marginal at time t
        """
        means_t, variances_t = self.marginal_gauss_params(t, means, variances, is_mixture=True)
        return weights, means_t, variances_t

    def log_prob_and_grad_gauss_scalar(self, t, x, mean, variance):
        """Log-likelihood and score of the marginal distribution with Gaussian data

        Args:
            * t (float): Time
            * x (torch.Tensor of shape (batch_size, dim)): Evaluation points
            * mean (torch.Tensor of shape (dim,)): Mean of the Gaussian distribution
            * variance (torch.Tensor of shape (dim,) or (dim, dim)): Variance of the Gaussian distribution

        Returns:
            * log_prob (torch.Tensor of shape (batch_size,)): Marginal log-prob at time t
            * score (torch.Tensor of the same shape as x): Marginal score at time t
        """
        # Compute the parameters
        is_full_cov = len(variance.shape) == 2
        mean_t, variance_t = self.marginal_gauss_params(t, mean, variance)
        # Compute the log_prob and score
        if is_full_cov:
            return log_prob_and_grad_gauss_full(x, mean_t, variance_t, return_log_prob=True)
        else:
            return log_prob_and_grad_gauss(x, mean_t, variance_t, return_log_prob=True)

    def log_prob_and_grad_mog_scalar(self, t, x, weights, means, variances):
        """Log-likelihood and score of the marginal distribution with Gaussian mixture data

        Args:
            * t (float): Time
            * x (torch.Tensor of shape (batch_size, dim)): Evaluation points
            * weights (torch.Tensor of shape (n_modes,)): Weights of the modes
            * means (torch.Tensor of shape (n_modes, dim): Means of the modes
            * variances (torch.Tensor of shape (n_modes, dim) or (n_modes, dim, dim)): Variances of the modes

        Returns:
            * log_prob (torch.Tensor of shape (batch_size,)): Marginal log-prob at time t
            * score (torch.Tensor of the same shape as x): Marginal score at time t
        """
        # Compute the parameters
        is_full_cov = len(variances.shape) == 3
        weights_t, means_t, variances_t = self.marginal_mog_params(t, weights, means, variances)
        weights_t = weights_t / weights_t.sum()
        # Compute the log_prob and score
        if is_full_cov:
            return log_prob_and_grad_mog_full(x, weights_t, means_t, variances_t, return_log_prob=True)
        else:
            return log_prob_and_grad_mog(x, weights_t, means_t, variances_t, return_log_prob=True)

    def log_prob_and_grad_and_hess_gauss_scalar(self, t, x, mean, variance, return_diag_hess=False):
        """Log-likelihood, score and hessian of the marginal distribution with Gaussian data

        Args:
            * t (float): Time
            * x (torch.Tensor of shape (batch_size, dim)): Evaluation points
            * mean (torch.Tensor of shape (dim,)): Mean of the Gaussian distribution
            * variance (torch.Tensor of shape (dim,) or (dim, dim)): Variance of the Gaussian distribution
            * return_diag_hess (bool): Whether to only return the diagonal of the hessian
                (default is False)

        Returns:
            * log_prob (torch.Tensor of shape (batch_size,)): Marginal log-prob at time t
            * score (torch.Tensor of the same shape as x): Marginal score at time t
            * hessian (torch.Tensor of the shape (batch_size, dim) or (batch_size, dim, dim)): Marginal hessian at time t
        """
        # Compute the parameters
        is_full_cov = len(variance.shape) == 2
        mean_t, variance_t = self.marginal_gauss_params(t, mean, variance)
        # Compute the log_prob and score
        if is_full_cov:
            return log_prob_and_grad_and_hess_gauss_full(x, mean_t, variance_t,
                                                         return_only_diag=return_diag_hess, return_log_prob=True, return_grad=True)
        else:
            return log_prob_and_grad_and_hess_gauss(x, mean_t, variance_t,
                                                    return_only_diag=return_diag_hess, return_log_prob=True, return_grad=True)

    def log_prob_and_grad_and_hess_mog_scalar(self, t, x, weights, means, variances, return_diag_hess=False):
        """Log-likelihood, score and hessian of the marginal distribution with Gaussian mixture data

        Args:
            * t (float): Time
            * x (torch.Tensor of shape (batch_size, dim)): Evaluation points
            * weights (torch.Tensor of shape (n_modes,)): Weights of the modes
            * means (torch.Tensor of shape (n_modes, dim): Means of the modes
            * variances (torch.Tensor of shape (n_modes, dim) or (n_modes, dim, dim)): Variances of the modes
            * return_diag_hess (bool): Whether to only return the diagonal of the hessian (default is False)

        Returns:
            * log_prob (torch.Tensor of shape (batch_size,)): Marginal log-prob at time t
            * score (torch.Tensor of the same shape as x): Marginal score at time t
            * hessian (torch.Tensor of the shape (batch_size, dim) or (batch_size, dim, dim)): Marginal hessian at time t
        """
        # Compute the parameters
        is_full_cov = len(variances.shape) == 3
        weights_t, means_t, variances_t = self.marginal_mog_params(t, weights, means, variances)
        weights_t = weights_t / weights_t.sum()
        # Compute the log_prob and score
        if is_full_cov:
            return log_prob_and_grad_and_hess_mog_full(x, weights_t, means_t, variances_t,
                                                       return_only_diag=return_diag_hess, return_log_prob=True, return_grad=True)
        else:
            return log_prob_and_grad_and_hess_mog(x, weights_t, means_t, variances_t,
                                                  return_only_diag=return_diag_hess, return_log_prob=True, return_grad=True)

    def em_integration_step(self, x, t_k, t_k_p_1, s, return_z=False, return_log_prob=False,
            return_mean_var=False, is_particles=False):
        """Denoising EM transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        
        delta = t_k_p_1 - t_k
        mean = (1. - self.f(t_k_p_1) * delta) * x + torch.square(self.g(t_k_p_1)) * delta * s
        std = self.g(t_k_p_1) * torch.sqrt(delta) 
        if return_mean_var:
            return mean, torch.square(std)
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            ret = mean + std * z
            if return_log_prob:
                data_shape = x.shape[1:]
                dim = np.prod(data_shape)
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                log_prob -= dim * torch.log(std).flatten()
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

    def ei_integration_step(self, x, t_k, t_k_p_1, s, return_z=False, return_log_prob=False,
            return_mean_var=False, is_particles=False):
        """Denoising EI transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        raise NotImplementedError

    def ddpm_integration_step(self, x, t_k, t_k_p_1, s, h=None, use_forward_var=False,
                              return_z=False, return_log_prob=False, return_mean_var=False,
                              is_particles=False):
        """Denoising DDPM transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * h (torch.Tensor of shape (batch_size, *data_shape) or (batch_size, *data_shape, *data_shape)): Hessian at t_k_p_1
            * use_forward_var (bool): Whether to use the variance of the forward kernel (default is False)
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        alpha_s_t, sigma_sq_s_t = self.transition_params(t_k, t_k_p_1)
        mean = (x + sigma_sq_s_t * s) / alpha_s_t
        full_h = (not use_forward_var) and (h is not None) and (h.shape == (x.shape[0], *x.shape[1:], *x.shape[1:]))
        if full_h or return_log_prob:
            data_shape = x.shape[1:]
            dim = math.prod(data_shape)
            if full_h:
                I = torch.eye(dim, device=x.device).unsqueeze(0)
        if use_forward_var:
            var = sigma_sq_s_t
        else:
            if full_h:
                sigma_sq_s_t = sigma_sq_s_t.view((-1, 1, 1))
                alpha_s_t = alpha_s_t.view((-1, 1, 1))
                var = sigma_sq_s_t * (I + sigma_sq_s_t * h.view((-1, dim, dim))) / torch.square(alpha_s_t)
            else:
                var = sigma_sq_s_t * (1 + sigma_sq_s_t * h) / torch.square(alpha_s_t)
        if return_mean_var:
            if full_h:
                var = var.view((-1, *data_shape, *data_shape))
            return mean, var
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            if full_h:
                ret = mean + torch.matmul(torch.linalg.cholesky(var), z.view((-1, dim, 1))).view((-1, *data_shape))
            else:
                ret = mean + torch.sqrt(var) * z
            if return_log_prob:
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                if full_h:
                    log_prob -= 0.5 * torch.logdet(var)
                elif use_forward_var:
                    log_prob -= 0.5 * dim * torch.log(var).flatten()
                else:
                    log_prob -= 0.5 * torch.log(var).sum(dim=sum_indexes)
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

    def ddim_integration_step(self, x, t_k, t_k_p_1, post_sampler_fn, return_z=False,
            return_log_prob=False, return_mean_var=False, is_particles=False):
        """Denoising DDIM transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * post_sampler_fn (function): Function sampling the posterior
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored. (WARNING: ONLY WITH DETERMISTIC POSTERIOR)
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
                (WARNING: ONLY WITH DETERMISTIC POSTERIOR)            
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """

        # Sample the posterior
        x0 = post_sampler_fn(t_k_p_1, x)
        if is_particles:
            x0 = remove_mean(x0)
        # Sample the bridge
        alpha_s_t, sigma_sq_s_t = self.transition_params(t_k, t_k_p_1)
        alpha_0_s, sigma_sq_0_s = self.transition_params_from_data(t_k)
        var = (sigma_sq_s_t * sigma_sq_0_s) / (sigma_sq_s_t + torch.square(alpha_s_t) * sigma_sq_0_s)
        mean = var * ((alpha_s_t / sigma_sq_s_t) * x + (alpha_0_s / sigma_sq_0_s) * x0)
        if return_mean_var:
            return mean, var
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            ret = mean + torch.sqrt(var) * z
            if return_log_prob:
                data_shape = x.shape[1:]
                dim = np.prod(data_shape)
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                log_prob -= 0.5 * dim * torch.log(var).flatten()
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

    def second_order_heun_integration_step(self, x, t_k, t_k_p_1, denoiser_fn, apply_correction=False):
        """Deterministic sampling using Heun’s 2nd order method from t_k_p_1 to t_k
            conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * denoiser_fn (function): Function evaluating the denoiser
            * apply_correction (bool): Whether to apply the correction (default is True)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """

        # Evaluate the denoiser
        x_hat_k_p_1 = denoiser_fn(t_k_p_1, x / self.s(t_k_p_1))
        d_k_p_1 = (self.sigma_dot_over_sigma(t_k_p_1) + self.s_dot_over_s(t_k_p_1)) * x
        d_k_p_1 -= self.sigma_dot_over_sigma(t_k_p_1) * self.s(t_k_p_1) * x_hat_k_p_1
        x_next = x + (t_k - t_k_p_1) * d_k_p_1
        if apply_correction:
            x_hat_k = denoiser_fn(t_k, x_next / self.s(t_k))
            d_k_p_1_new = (self.sigma_dot_over_sigma(t_k) + self.s_dot_over_s(t_k)) * x_next
            d_k_p_1_new -= self.sigma_dot_over_sigma(t_k) * self.s(t_k) * x_hat_k
            x_next = x + 0.5 * (t_k - t_k_p_1) * (d_k_p_1 + d_k_p_1_new)
        return x_next

    def exact_denoising_kernel_gauss_params_scalar(self, s, t, x_t, mean, variance,
                                                   return_alphas_sigmas=False):
        """Compute the mean and variance of the exact denoising kernel associated to a Gaussian

        Args:
            * s (float): Time s
            * t (float): Time t (s < t)
            * x_t (torch.Tensor of shape (dim,)): Conditioning point
            * mean (torch.Tensor of shape (dim,) or (n_modes, dim)): Mean of the Gaussian
            * variance (torch.Tensor of shape (dim,) or (n_modes, dim)): Variance of the Gaussian
            * return_alphas_sigmas (bool): Whether to return auxiliary quantities (default is False)

        Returns:
            * mean_t (torch.Tensor of the same shape as mean): Mean of the Gaussian kernel
            * var_t (torch.Tensor of the same shape as variance): Variance of the Gaussian kernel
            if return_alphas_sigmas:
                * params (tuple of size 5): Various parameters
        """
        # Compute the alphas and sigmas
        alphas_s_t, gammas_sq_s_t = self.transition_params(s, t)
        alphas_s, gammas_sq_s = self.transition_params_from_data(s)
        # Unsqueeze x_t if it is a mixture
        if (mean.ndim == 2) and (variance.ndim == 2):
            x_t = x_t.unsqueeze(0)
        # Precompute certain quantities
        variances_s = torch.square(alphas_s) * variance + gammas_sq_s
        sum_var = gammas_sq_s_t + torch.square(alphas_s_t) * variances_s
        # Compute the new mean
        mean_t = alphas_s * gammas_sq_s_t * mean
        mean_t += alphas_s_t * variances_s * x_t
        mean_t /= sum_var
        # Compute the new variance
        var_t = gammas_sq_s_t * variances_s / sum_var
        # Return everything
        if return_alphas_sigmas:
            return mean_t, var_t, (alphas_s_t, alphas_s, sum_var)
        else:
            return mean_t, var_t

    def exact_denoising_kernel_mog_params_scalar(self, s, t, x_t, weights, means, variances):
        """Compute the weights, the means and variances of the exact denoising kernel
           associated to Gaussian mixture.

        Args:
            * s (float): Time s
            * t (float): Time t (s < t)
            * x_t (torch.Tensor of shape (dim,)): Conditioning point
            * weights (torch.Tensor of shape (n_modes,)): Weights of the mixture
            * means (torch.Tensor of shape (n_modes, dim)): Means of the mixture
            * variances (torch.Tensor of shape (n_modes, dim)): Variances of the mixture

        Returns:
            * weights_t (torch.Tensor of the same shape as mean): Weights of the mixture kernel
            * means_t (torch.Tensor of the same shape as mean): Mean of the mixture kernel
            * vars_t (torch.Tensor of the same shape as variance): Variance of the mixture kernel
        """
        # Compute the means and variances
        means_t, vars_t, (alphas_s_t, alphas_s, sum_var) = self.exact_denoising_kernel_gauss_params_scalar(
            s, t, x_t, means, variances, return_alphas_sigmas=True
        )
        # Compute the weights
        log_weights_t = -0.5 * torch.sum(
            torch.square(x_t.unsqueeze(0) - alphas_s * alphas_s_t * means) / sum_var,
            dim=-1
        )
        log_weights_t += torch.log(weights)
        weights_t = torch.nn.functional.softmax(log_weights_t, dim=0)
        # Return everything
        return weights_t, means_t, vars_t

    def exact_denoising_kernel_gauss_log_prob(self, x_s, mean_t, variance_t, z_s=None):
        """Computes the log-likelihood of the exact denoising kernel in the Gaussian case

        Args:
            * x_s (torch.Tensor of shape (batch_size, dim)): Evaluation point
            * mean_t (torch.Tensor of shape (batch_size, dim)): Mean of the kernel
            * variance_t (torch.Tensor of shape (batch_size, dim)): Variance of the kernel
            * z_s (torch.Tensor of same shape as x_s): Value of (x_s - mean_t) / sqrt(variance_t)
                (default is None)

        Returns:
            * log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood
        """
        # Compute the log_prob
        if z_s is None:
            log_prob = -0.5 * torch.sum(torch.square(x_s - mean_t) / variance_t, dim=-1)
        else:
            log_prob = -0.5 * torch.sum(torch.square(z_s), dim=-1)
        log_prob -= 0.5 * mean_t.shape[-1] * math.log(2. * math.pi)
        log_prob -= 0.5 * torch.log(variance_t).sum(dim=-1)
        return log_prob

    def exact_denoising_kernel_mog_log_prob(self, x_s, weights_t, means_t, variances_t):
        """Computes the log-likelihood of the exact denoising kernel in the Gaussian mixture case

        Args:
            * x_s (torch.Tensor of shape (batch_size, dim)): Evaluation point
            * weights_t (torch.Tensor of shape (batch_size, n_modes)): Weights of the kernel
            * means_t (torch.Tensor of shape (batch_size, n_modes, dim)): Means of the kernel
            * variances_t (torch.Tensor of shape (batch_size, n_modes, dim)): Variances of the kernel

        Returns:
            * log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood
        """
        # Compute the log_prob of each gaussian
        log_probs = -0.5 * torch.sum(torch.square(x_s.unsqueeze(1) - means_t) / variances_t, dim=-1)
        log_probs -= 0.5 * means_t.shape[-1] * math.log(2. * math.pi)
        log_probs -= 0.5 * torch.log(variances_t).sum(dim=-1)
        log_probs += torch.log(weights_t)
        # Return the log_prob
        return torch.logsumexp(log_probs, dim=-1)

    def exact_denoising_kernel_gauss_sample(self, s, t, x_t, mean, variance, return_log_prob=False):
        """Sample from the exact denoising kernel associated to a Gaussian

        Args:
            * s (torch.Tensor of shape (batch_size,)): Time s
            * t (torch.Tensor of shape (batch_size,)): Time t (s < t)
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * mean (torch.Tensor of shape (dim,)): Mean of the Gaussian
            * variance (torch.Tensor of shape (dim,)): Variance of the Gaussian
            * return_log_prob (bool): Whether to return the likelihood of the obtained sample
                (default is False)

        Returns:
            * x_s (torch.Tensor of the same shape as x_t): Denoised sample
            if return_log_prob:
                * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        # Get the parameters
        mean_t, variance_t = self.exact_denoising_kernel_gauss_params(s, t, x_t, mean, variance)
        # Get the sample
        z_s = torch.randn_like(x_t)
        x_s = mean_t + torch.sqrt(variance_t) * z_s
        # Return everything
        if return_log_prob:
            return x_s, self.exact_denoising_kernel_gauss_log_prob(x_s, mean_t, variance_t, z_s=z_s)
        else:
            return x_s

    def exact_denoising_kernel_mog_sample(self, s, t, x_t, weights, means, variances, return_log_prob=False):
        """Sample from the exact denoising kernel associated to a Gaussian mixture

        Args:
            * s (torch.Tensor of shape (batch_size,)): Time s
            * t (torch.Tensor of shape (batch_size,)): Time t (s < t)
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * weights (torch.Tensor of shape (n_modes,)): Weights of the mixture
            * mean (torch.Tensor of shape (n_modes, dim)): Means of the mixture
            * variance (torch.Tensor of shape (n_modes, dim)): Variances of the mixture
            * return_log_prob (bool): Whether to return the likelihood of the obtained sample
                (default is False)

        Returns:
            * x_s (torch.Tensor of the same shape as x_t): Denoised sample
            if return_log_prob:
                * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        # Get the parameters
        weights_t, means_t, variances_t = self.exact_denoising_kernel_mog_params(s, t, x_t, weights, means, variances)
        # Sample the weights
        mode_idx = torch.multinomial(weights_t, 1).flatten()
        batch_idx = torch.arange(x_t.shape[0])
        z_s = torch.randn_like(x_t)
        x_s = means_t[batch_idx, mode_idx] + torch.sqrt(variances_t[batch_idx, mode_idx]) * z_s
        # Return everything
        if return_log_prob:
            return x_s, self.exact_denoising_kernel_mog_log_prob(x_s, weights_t, means_t, variances_t)
        else:
            return x_s

    def exact_posterior_gauss_params_scalar(self, t, x_t, mean, variance, return_aux=False):
        """Compute the mean and variance of the exact posterior associated to a Gaussian

        Args:
            * t (float): Time t
            * x_t (torch.Tensor of shape (dim,)): Conditioning point
            * mean (torch.Tensor of shape (dim,) or (n_modes, dim)): Mean of the Gaussian
            * variance (torch.Tensor of shape (dim,) or (n_modes, dim)): Variance of the Gaussian
            * return_aux (bool): Whether to return auxiliary quantities (default is False)

        Returns:
            * mean_t (torch.Tensor of the same shape as mean): Mean of the posterior
            * var_t (torch.Tensor of the same shape as variance): Variance of the posterior
            if return_aux:
                * aux (tuple of 2 torch.Tensor): Auxiliary quantities
        """
        # Compute the s and sigma
        alpha_t, gamma_sq_t = self.transition_params_from_data(t)
        sigma_sq_t = self.sigma_sq(t)
        # Unsqueeze x_t if it is a mixture
        if (mean.ndim == 2) and (variance.ndim == 2):
            x_t = x_t.unsqueeze(0)
        # Compute the mean (I multiplied by s on top and bottom for stability)
        denom = torch.square(alpha_t) * variance + gamma_sq_t
        mean_t = gamma_sq_t * mean + alpha_t * variance * x_t
        mean_t /= denom
        # Compute the variance
        var_t = sigma_sq_t * variance / (variance + sigma_sq_t)
        # var_t = gamma_sq_t * variance / denom
        # Return everything
        if return_aux:
            return mean_t, var_t, (alpha_t, denom)
        else:
            return mean_t, var_t

    def exact_posterior_mog_params_scalar(self, t, x_t, weights, means, variances):
        """Compute the weights, the means and variances of the exact posterior associated
            to Gaussian mixture.

        Args:
            * t (float): Time t
            * x_t (torch.Tensor of shape (dim,)): Conditioning point
            * weights (torch.Tensor of shape (n_modes,)): Weights of the mixture
            * means (torch.Tensor of shape (n_modes, dim)): Means of the mixture
            * variances (torch.Tensor of shape (n_modes, dim)): Variances of the mixture

        Returns:
            * weights_t (torch.Tensor of the same shape as (n_modes,)): Weights of the posterior
            * means_t (torch.Tensor of the same shape as mean): Mean of the posterior
            * vars_t (torch.Tensor of the same shape as variance): Variance of the posterior
        """
        # Compute the s and sigma
        means_t, vars_t, (alpha_t, denom) = self.exact_posterior_gauss_params_scalar(
            t, x_t, means, variances, return_aux=True)
        # Compute the weights
        log_weights_t = -0.5 * torch.sum(
            torch.square(x_t.unsqueeze(0) - alpha_t * means) / denom,
            dim=-1
        )
        log_weights_t += torch.log(weights)
        weights_t = torch.nn.functional.softmax(log_weights_t, dim=0)
        # Return everything
        return weights_t, means_t, vars_t

    def exact_posterior_gauss_log_prob(self, x, mean_t, variance_t, z=None):
        """Computes the log-likelihood of the exact posterior in the Gaussian case"""
        return self.exact_denoising_kernel_gauss_log_prob(x, mean_t, variance_t, z=None)

    def exact_posterior_mog_log_prob(self, x, weights_t, means_t, variances_t):
        """Computes the log-likelihood of the exact posterior in the Gaussian mixture case"""
        return self.exact_denoising_kernel_mog_log_prob(x, weights_t, means_t, variances_t)

    def exact_posterior_gauss_sample(self, t, x_t, mean, variance, return_log_prob=False):
        """Sample from the exact posterior associated to a Gaussian

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * mean (torch.Tensor of shape (dim,)): Mean of the Gaussian
            * variance (torch.Tensor of shape (dim,)): Variance of the Gaussian
            * return_log_prob (bool): Whether to return the likelihood of the obtained sample
                (default is False)

        Returns:
            * x (torch.Tensor of the same shape as x_t): Denoised sample
            if return_log_prob:
                * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        # Get the parameters
        mean_t, variance_t = self.exact_posterior_gauss_params(t, x_t, mean, variance)
        # Get the sample
        z = torch.randn_like(x_t)
        x = mean_t + torch.sqrt(variance_t) * z
        # Return everything
        if return_log_prob:
            return x, self.exact_denoising_kernel_gauss_log_prob(x, mean_t, variance_t, z=z)
        else:
            return x

    def exact_posterior_mog_sample(self, t, x_t, weights, means, variances, return_log_prob=False):
        """Sample from the exact posterior associated to a Gaussian mixture

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * weights (torch.Tensor of shape (n_modes,)): Weights of the mixture
            * means (torch.Tensor of shape (n_modes, dim)): Means of the mixture
            * variance (torch.Tensor of shape (n_modes, dim)): Variances of the mixture
            * return_log_prob (bool): Whether to return the likelihood of the obtained sample
                (default is False)

        Returns:
            * x (torch.Tensor of the same shape as x_t): Denoised sample
            if return_log_prob:
                * log_prob (torch.Tensor of the shape (batch_size,)): Log-likelihood
        """
        # Get the parameters
        weights_t, means_t, variances_t = self.exact_posterior_mog_params(t, x_t, weights, means, variances)
        # Sample the weights
        mode_idx = torch.multinomial(weights_t, 1).flatten()
        batch_idx = torch.arange(x_t.shape[0])
        z = torch.randn_like(x_t)
        x = means_t[batch_idx, mode_idx] + torch.sqrt(variances_t[batch_idx, mode_idx]) * z
        # Return everything
        if return_log_prob:
            return x, self.exact_denoising_kernel_mog_log_prob(x, weights_t, means_t, variances_t)
        else:
            return x

    def denoiser_gauss(self, t, x_t, mean, variance):
        """Compute the exact denoiser associated to a Gaussian

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * mean (torch.Tensor of shape (dim,)): Mean of the Gaussian
            * variance (torch.Tensor of shape (dim,)): Variance of the Gaussian

        Returns:
            * x (torch.Tensor of the same shape as x_t): Denoised sample
        """
        return self.exact_posterior_gauss_params(t, x_t, mean, variance)[0]

    def denoiser_mog(self, t, x_t, weights, means, variances):
        """Compute the exact denoiser associated to a Gaussian mixture

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * weights (torch.Tensor of shape (n_modes,)): Weights of the mixture
            * means (torch.Tensor of shape (n_modes, dim)): Means of the mixture

        Returns:
            * x (torch.Tensor of the same shape as x_t): Denoised sample
        """
        weights_t, means_t, _ = self.exact_posterior_mog_params(t, x_t, weights, means, variances)
        return mean_of_mog_batch(weights_t, means_t)

    def posterior_covariance_gauss(self, t, x_t, mean, variance, return_diag=False):
        """Compute the posterior's covariance associated to a Gaussian

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * mean (torch.Tensor of shape (dim,)): Mean of the Gaussian
            * variance (torch.Tensor of shape (dim,)): Variance of the Gaussian
            * return_diag (bool): Whether to return only the diagonal of the covariance
                (default is False)

        Returns:
            * cov (torch.Tensor of shape (batch_size, dim, dim) or (batch_size, dim)): Covariance
        """
        post_var = self.exact_posterior_gauss_params(t, x_t, mean, variance)[1]
        if return_diag:
            return post_var
        else:
            return torch.diag_embed(post_var)

    def posterior_covariance_mog(self, t, x_t, weights, means, variances, return_diag=False):
        """Compute the posterior's covariance associated to a Gaussian mixture

        Args:
            * t (torch.Tensor of shape (batch_size,)): Time t
            * x_t (torch.Tensor of shape (batch_size, dim)): Conditioning point
            * weights (torch.Tensor of shape (n_modes,)): Weights of the mixture
            * means (torch.Tensor of shape (n_modes, dim)): Means of the mixture
            * return_diag (bool): Whether to return only the diagonal of the covariance
                (default is False)

        Returns:
            * cov (torch.Tensor of shape (batch_size, dim, dim) or (batch_size, dim)): Covariance
        """
        weights_t, means_t, vars_t = self.exact_posterior_mog_params(t, x_t, weights, means, variances)
        return cov_of_mog_batch(weights_t, means_t, vars_t, return_diag=return_diag)

    def score_from_denoiser(self, ts, ys, ds):
        """Compute the score from the denoiser with Tweedie

        Args:
            * ts (torch.Tensor of shape (batch_size, *data_shape_ones)): Times
            * ys (torch.Tensor of shape (batch_size, *data_shape)): Noisy samples
            * ds (torch.Tensor of shape (batch_size, *data_shape)): Denoiser evaluation

        Returns:
            * s (torch.Tensor of the same shape as ys): Scores
        """
        alpha_t, gamma_sq_t = self.transition_params_from_data(ts)
        return (alpha_t * ds - ys) / gamma_sq_t

    def hessian_from_posterior_covariance(self, ts, ys, covs):
        """Compute the hessian from the posterior's covariance with second order Tweedie

        Args:
            * ts (torch.Tensor of shape (batch_size, *data_shape_ones)): Times
            * ys (torch.Tensor of shape (batch_size, *data_shape)): Noisy samples
            * covs (torch.Tensor of shape (batch_size, *data_shape) or (batch_size, *data_shape, *data_shape): Covariances

        Returns:
            * h (torch.Tensor of the same shape as covs): Hessians
        """
        _, gamma_sq_t = self.transition_params_from_data(ts)
        sigma_sq_t = self.sigma_sq(ts)
        if covs.shape == ys.shape:
            return (covs - sigma_sq_t) / (sigma_sq_t * gamma_sq_t)
        else:
            data_shape = ys.shape[1:]
            dim = math.prod(data_shape)
            I = torch.eye(dim, device=ys.device).view((1, *data_shape, *data_shape))
            return (covs - sigma_sq_t.unsqueeze(-1) * I) / (sigma_sq_t * gamma_sq_t).unsqueeze(-1)


class VP(LinearSDE):
    """Implement Variance Preserving

        dX_t = -0.5 * beta(t) X_t dt + sigma sqrt(beta(t)) dW_t

    """

    def __init__(self, sigma=1.0, T=1.0):
        super().__init__(T=T)
        self.register_buffer("sigma_", torch.tensor(sigma, dtype=torch.float), persistent=False)

    def get_base_dist(self, data_shape):
        """Returns the base distribution (i.e., the marginal distribution at time T)"""
        if len(data_shape) > 1:
            dim = math.prod(data_shape)
            return ReshapeWrapper(Gauss(
                mean=torch.zeros((dim,), device=self.sigma_.device),
                variance=torch.square(self.sigma_) * torch.ones((dim,), device=self.sigma_.device)
            ), data_shape=data_shape)
        else:
            return Gauss(
                mean=torch.zeros(data_shape, device=self.sigma_.device),
                variance=torch.square(self.sigma_) * torch.ones(data_shape, device=self.sigma_.device)
            )

    def sample_base_dist(self, shape, data_shape):
        """Sample the base distribution (i.e., the marginal distribution at time T)"""
        return self.sigma_ * torch.randn((*shape, *data_shape), device=self.sigma_.device)

    def beta(self, t):
        """Function beta"""
        raise NotImplementedError

    def alpha(self, t):
        """Value of int_0^t beta(u) du"""
        raise NotImplementedError

    def alpha_inv(self, a):
        """Inverse of the alpha function"""
        raise NotImplementedError

    def f(self, t):
        """Function f"""
        return -0.5 * self.beta(t)

    def g(self, t):
        """Function f"""
        return self.sigma_ * torch.sqrt(self.beta(t))

    def s(self, t):
        """Value of exp(int_0^t f(u) du)"""
        return torch.exp(-0.5 * self.alpha(t))

    def s_dot(self, t):
        """Derivative of s"""
        return -0.5 * self.beta(t) * self.s(t)

    def sigma_sq(self, t):
        """Value of int_0^t g^2(u) / s^2(u) du"""
        return torch.square(self.sigma_) * torch.expm1(self.alpha(t))

    def sigma_sq_dot(self, t):
        """Derivative of sigma_sq"""
        return torch.square(self.sigma_) * self.beta(t) * torch.exp(self.alpha(t))

    def sigma_inv(self, sigma):
        """Inverse of the sigma function"""
        return self.alpha_inv(torch.log1p(torch.square(sigma / self.sigma_)))

    def gamma_sq(self, t):
        """Product between s^2 and sigma_sq"""
        return -torch.square(self.sigma_) * torch.expm1(-self.alpha(t))
    
    def gamma_sq_dot(self, t):
        """Derivative of gamma_sq"""
        return torch.square(self.sigma_) * self.beta(t) * torch.exp(-self.alpha(t))

    def gamma(self, t):
        """Product between s and sigma"""
        return torch.sqrt(self.gamma_sq(t))

    def gamma_dot(self, t):
        """Derivate of gamma"""
        return 0.5 * self.gamma_sq_dot(t) / self.gamma(t)

    def s_dot_over_gamma(self, t):
        """Ratio of s_dot and gamma"""
        return -0.5 * self.beta(t) * self.s(t) / (self.sigma_ * torch.sqrt(-torch.expm1(-self.alpha(t))))

    def gamma_dot_over_gamma(self, t):
        """Ratio of gamma_dot and gamma"""
        return -0.5 * self.beta(t) / torch.expm1(self.alpha(t))

    def sigma_dot_over_sigma(self, t):
        """Ratio of sigma_dot and sigma"""
        return -0.5 * self.beta(t) / torch.expm1(-self.alpha(t))

    def transition_params(self, s, t):
        """Mean and variance parameters for noising transition kernel s -> t (s < t)

        We have that X_t = alpha_s X_s + gamma_s Z with Z ~ N(0,I)

        This function returns alpha_s and (sigma_s)^2.
        """
        lambda_s_t = -torch.expm1(self.alpha(s) - self.alpha(t))
        mean_factor = torch.sqrt(1. - lambda_s_t)
        var_factor = self.sigma_ ** 2 * lambda_s_t
        return mean_factor, var_factor

    def transition_params_from_data(self, t):
        """Mean and variance parameters for noising transition kernel 0 -> t

        We have that X_t = alpha_0 X_0 + gamma_0 Z with Z ~ N(0,I)

        This function returns alpha_0 and (gamma_0)^2.
        """
        lambda_t = -torch.expm1(-self.alpha(t))
        mean_factor = torch.sqrt(1. - lambda_t)
        var_factor = self.sigma_ ** 2 * lambda_t
        return mean_factor, var_factor

    def ei_integration_step(self, x, t_k, t_k_p_1, s, return_z=False, return_log_prob=False,
            return_mean_var=False, is_particles=False):
        """Denoising EI transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        lambda_k = torch.expm1(self.alpha(t_k_p_1) - self.alpha(t_k))
        mean = torch.sqrt(1. + lambda_k) * x + 2. * torch.square(self.sigma_) * (torch.sqrt(1. + lambda_k) - 1.) * s
        std = self.sigma_ * torch.sqrt(lambda_k)
        if return_mean_var:
            return mean, torch.square(std)
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            ret = mean + std * z
            if return_log_prob:
                data_shape = x.shape[1:]
                dim = np.prod(data_shape)
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                log_prob -= dim * torch.log(std).flatten()
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

    def ddpm_integration_step(self, x, t_k, t_k_p_1, s, h=None, use_forward_var=False,
                              return_z=False, return_log_prob=False, return_mean_var=False,
                              is_particles=False):
        """Denoising DDPM transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * h (torch.Tensor of shape (batch_size, *data_shape) or (batch_size, *data_shape, *data_shape)): Hessian at t_k_p_1
            * use_forward_var (bool): Whether to use the variance of the forward kernel (default is False)
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        sigma_sq = torch.square(self.sigma_)
        lambda_s_t_f = -torch.expm1(self.alpha(t_k) - self.alpha(t_k_p_1))
        lambda_s_t_b = torch.expm1(self.alpha(t_k_p_1) - self.alpha(t_k))
        mean = torch.sqrt(1 + lambda_s_t_b) * x
        mean += 2. * sigma_sq * torch.sinh(0.5 * (self.alpha(t_k_p_1) - self.alpha(t_k))) * s
        full_h = (not use_forward_var) and (h is not None) and (h.shape == (x.shape[0], *x.shape[1:], *x.shape[1:]))
        if full_h or return_log_prob:
            data_shape = x.shape[1:]
            dim = math.prod(data_shape)
            if full_h:
                I = torch.eye(dim, device=x.device).unsqueeze(0)
        if use_forward_var:
            var = sigma_sq * lambda_s_t_f
        else:
            if full_h:
                lambda_s_t_f, lambda_s_t_b = lambda_s_t_f.view((-1, 1, 1)), lambda_s_t_b.view((-1, 1, 1))
                var = sigma_sq * (lambda_s_t_b * I + sigma_sq * (lambda_s_t_b - lambda_s_t_f) * h.view((-1, dim, dim)))
            else:
                var = sigma_sq * lambda_s_t_b + torch.square(sigma_sq) * (lambda_s_t_b - lambda_s_t_f) * h
        if return_mean_var:
            if full_h:
                var = var.view((-1, *data_shape, *data_shape))
            return mean, var
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            if full_h:
                ret = mean + torch.matmul(torch.linalg.cholesky(var), z.view((-1, dim, 1))).view((-1, *data_shape))
            else:
                ret = mean + torch.sqrt(var) * z
            if return_log_prob:
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                if full_h:
                    log_prob -= 0.5 * torch.logdet(var)
                elif use_forward_var:
                    log_prob -= 0.5 * dim * torch.log(var).flatten()
                else:
                    log_prob -= 0.5 * torch.log(var).sum(dim=sum_indexes)
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret


class LinearVP(VP):
    """Implement VP with a linear schedule of form

        beta(t) = beta_min + (t / T) * (beta_max - beta_min)

    """

    def __init__(self, beta_min=0.1, beta_max=20.0, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.register_buffer("beta_min", torch.tensor(beta_min, dtype=torch.float), persistent=False)
        self.register_buffer("beta_max", torch.tensor(beta_max, dtype=torch.float), persistent=False)

    def beta(self, t):
        """Function beta"""
        return torch.lerp(self.beta_min, self.beta_max, t / self.T)

    def alpha(self, t):
        """Value of int_0^t beta(u) du"""
        return self.beta_min * t + (0.5 * torch.square(t) / self.T) * (self.beta_max - self.beta_min)

    def alpha_inv(self, a):
        """Inverse of the alpha function"""
        beta_diff = self.beta_max - self.beta_min
        delta = torch.square(self.beta_min) + 4. * beta_diff * a / (2. * self.T)
        return (-self.beta_min + torch.sqrt(delta)) * self.T / beta_diff


class CosineVP(VP):
    """Implement VP with a cosine schedule of form

        beta(t) = pi * tan(0.5 * pi * ((t / T) + c) / (1. + c)) / (T * (1. + c))

    """

    def __init__(self, c=0.008, sigma=1.0, T=1.0):
        super().__init__(sigma=sigma, T=T)
        self.register_buffer("c", torch.tensor(c, dtype=torch.float), persistent=False)

    def beta(self, t):
        """Function beta"""
        return torch.pi * torch.tan(0.5 * torch.pi * ((t / self.T) + self.c) / (1. + self.c)) / (self.T * (1. + self.c))

    def alpha(self, t):
        """Value of int_0^t beta(u) du"""
        return -2. * torch.log(torch.cos(0.5 * torch.pi * ((t / self.T) + self.c) / (1. + self.c)))

    def alpha_inv(self, a):
        """Inverse of the alpha function"""
        return self.T * ((2. * (1. + self.c) * torch.arccos(torch.exp(-0.5 * a)) / torch.pi) - self.c)


class VE(LinearSDE):

    """Implementation of a VE SDE
        d X_t = g(t) dW_t
    """

    def __init__(self, sigma_min=1e-1, sigma_max=1e1):
        """Constructor for the Variance Exploding SDE

        Args:
            * sigma_min (float): Minimum value of sigma (default is 1e-1)
            * sigma_max (float): Maximum value of sigma (default is 1e+1)
            * T (float): Terminal time (default is 1.0)
        """
        super().__init__(T=1.0)
        self.register_buffer("sigma_min", torch.tensor(sigma_min, dtype=torch.float),
                             persistent=False)
        self.register_buffer("sigma_max", torch.tensor(sigma_max, dtype=torch.float),
                             persistent=False)
        self.register_buffer("sigma_ratio", self.sigma_max / self.sigma_min,
            persistent=False)
        self.T = self.sigma_inv(self.sigma_max)

    def get_base_dist(self, data_shape):
        """Returns the base distribution (i.e., the marginal distribution at time T)"""
        if len(data_shape) > 1:
            dim = math.prod(data_shape)
            return ReshapeWrapper(Gauss(
                mean=torch.zeros((dim,), device=self.sigma_max.device),
                variance=torch.square(self.sigma_max) * torch.ones((dim,), device=self.sigma_max.device)
            ), data_shape=data_shape)
        else:
            return Gauss(
                mean=torch.zeros(data_shape, device=self.sigma_max.device),
                variance=torch.square(self.sigma_max) * torch.ones(data_shape, device=self.sigma_max.device)
            )

    def sample_base_dist(self, shape, data_shape):
        """Sample the base distribution (i.e., the marginal distribution at time T)"""
        return self.sigma_max * torch.randn((*shape, *data_shape), device=self.sigma_max.device)


    def f(self, t):
        """Function f"""
        if isinstance(t, float):
            return torch.tensor(0.0).to(self.sigma_min.device)
        else:
            return torch.zeros_like(t)

    def g(self, t):
        """Function g"""
        ret = self.sigma_min * torch.pow(self.sigma_ratio, t)
        ret *= torch.sqrt(2. * torch.log(self.sigma_ratio))
        return ret

    def s(self, t):
        """Value of exp(int_0^t f(u) du)"""
        if isinstance(t, float):
            return torch.tensor(1.0).to(self.sigma_min.device)
        else:
            return torch.ones_like(t)

    def s_dot(self, t):
        """Derivative of s"""
        if isinstance(t, float):
            return torch.tensor(0.0).to(self.sigma_min.device)
        else:
            return torch.zeros_like(t)

    def sigma_sq(self, t):
        """Value of int_0^t g^2(u) / s^2(u) du"""
        return torch.square(self.sigma_min) * (torch.pow(self.sigma_ratio, 2. * t) - 1.)

    def sigma_sq_dot(self, t):
        """Derivative of sigma_sq"""
        return torch.square(self.g(t))

    def sigma_dot(self, t):
        """Derivative of sigma"""
        sigma_ratio_2t = torch.pow(self.sigma_ratio, 2. * t)
        return self.sigma_min * sigma_ratio_2t * torch.log(self.sigma_ratio) \
            / torch.sqrt(sigma_ratio_2t - 1.)

    def sigma_inv(self, sigma):
        """Inverse of the sigma function"""
        return torch.log1p(torch.square(sigma / self.sigma_min)) / (2. * torch.log(self.sigma_ratio))

    def gamma_sq(self, t):
        """Product between s^2 and sigma_sq"""
        return self.sigma_sq(t)
    
    def gamma_sq_dot(self, t):
        """Derivative of gamma_sq"""
        return self.sigma_sq_dot(t)

    def gamma(self, t):
        """Product between s and sigma"""
        return self.sigma(t)

    def gamma_dot(self, t):
        """Derivate of gamma"""
        return self.sigma_dot(t)

    def s_dot_over_gamma(self, t):
        """Ratio of s_dot and gamma"""
        return torch.zeros_like(t)

    def gamma_dot_over_gamma(self, t):
        """Ratio of gamma_dot and gamma"""
        sigma_ratio_2t = torch.pow(self.sigma_ratio, 2. * t)
        return sigma_ratio_2t * torch.log(self.sigma_ratio) / (sigma_ratio_2t - 1.)

    def get_snr_time_discretization(self, start, end, n_steps, n_attemps=1024):
        """Get SNR adapted time discretization

        Args:
            * start (float or torch.Tensor): Start time
            * end (float or torch.Tensor): End time
            * n_steps (int): Number of intermediate times

        Returns:
            * ts (torch.Tensor of shape (n_steps,)): Time discretization
        """
        if isinstance(start, float):
            start = torch.tensor(start, device=self.sigma_min.device)
        if isinstance(end, float):
            end = torch.tensor(end, device=self.sigma_min.device)
        # return self.sigma_inv(torch.linspace(self.sigma_min, self.sigma_max, n_steps))
        ts = torch.log(torch.pow(self.sigma_ratio, 2 * torch.arange(n_steps, device=self.sigma_min.device) / (n_steps - 1)) + 1.)
        ts /= (2. * torch.log(self.sigma_ratio))
        return ts

    def transition_params(self, s, t):
        """Mean and variance parameters for noising transition kernel s -> t (s < t)

        We have that X_t = alpha_s X_s + gamma_s Z with Z ~ N(0,I)

        This function returns alpha_s and (sigma_s)^2.
        """
        mean_factor = torch.ones_like(t)
        var_factor = torch.square(self.sigma_min) * torch.pow(self.sigma_ratio, 2. * s)
        var_factor *= torch.pow(self.sigma_ratio, 2. * (t - s)) - 1.0
        return mean_factor, var_factor

    def transition_params_from_data(self, t):
        """Mean and variance parameters for noising transition kernel 0 -> t

        We have that X_t = alpha_0 X_0 + gamma_0 Z with Z ~ N(0,I)

        This function returns alpha_0 and (gamma_0)^2.
        """
        mean_factor = torch.ones_like(t)
        var_factor = self.sigma_sq(t)
        return mean_factor, var_factor

    def ei_integration_step(self, x, t_k, t_k_p_1, s, return_z=False, return_log_prob=False,
            return_mean_var=False, is_particles=False):
        """Denoising EI transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        lambda_k = torch.square(self.sigma_min) * torch.pow(self.sigma_ratio, 2. * t_k)
        lambda_k *= torch.pow(self.sigma_ratio, 2. * (t_k_p_1 - t_k)) - 1.0
        mean = x + lambda_k * s
        std = torch.sqrt(lambda_k)
        if return_mean_var:
            return mean, torch.square(std)
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            ret = mean + std * z
            if return_log_prob:
                data_shape = x.shape[1:]
                dim = np.prod(data_shape)
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                log_prob -= dim * torch.log(std).flatten()
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

    def ddpm_integration_step(self, x, t_k, t_k_p_1, s, h=None, use_forward_var=False,
                              return_z=False, return_log_prob=False, return_mean_var=False,
                              is_particles=False):
        """Denoising DDPM transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * h (torch.Tensor of shape (batch_size, *data_shape) or (batch_size, *data_shape, *data_shape)): Hessian at t_k_p_1
            * use_forward_var (bool): Whether to use the variance of the forward kernel (default is False)
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        sigma_sq_s_t = self.transition_params(t_k, t_k_p_1)[1]
        mean = x + sigma_sq_s_t * s
        full_h = (not use_forward_var) and (h is not None) and (h.shape == (x.shape[0], *x.shape[1:], *x.shape[1:]))
        if full_h or return_log_prob:
            data_shape = x.shape[1:]
            dim = math.prod(data_shape)
            if full_h:
                I = torch.eye(dim, device=x.device).unsqueeze(0)
        if use_forward_var:
            var = sigma_sq_s_t
        else:
            if full_h:
                sigma_sq_s_t = sigma_sq_s_t.view((-1, 1, 1))
                var = sigma_sq_s_t * (I + sigma_sq_s_t * h.view((-1, dim, dim)))
            else:
                var = sigma_sq_s_t * (1 + sigma_sq_s_t * h)
        if return_mean_var:
            if full_h:
                var = var.view((-1, *data_shape, *data_shape))
            return mean, var
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            if full_h:
                ret = mean + torch.matmul(torch.linalg.cholesky(var), z.view((-1, dim, 1))).view((-1, *data_shape))
            else:
                ret = mean + torch.sqrt(var) * z
            if return_log_prob:
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                if full_h:
                    log_prob -= 0.5 * torch.logdet(var)
                elif use_forward_var:
                    log_prob -= 0.5 * dim * torch.log(var).flatten()
                else:
                    log_prob -= 0.5 * torch.log(var).sum(dim=sum_indexes)
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

    def ddim_integration_step(self, x, t_k, t_k_p_1, post_sampler_fn, return_z=False,
            return_log_prob=False, return_mean_var=False, is_particles=False):
        """Denoising DDIM transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * post_sampler_fn (function): Function sampling the posterior
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored. (WARNING: ONLY WITH DETERMISTIC POSTERIOR)
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
                (WARNING: ONLY WITH DETERMISTIC POSTERIOR)            
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """

        # Sample the posterior
        x0 = post_sampler_fn(t_k_p_1, x)
        if is_particles:
            x0 = remove_mean(x0)
        # Sample the bridge
        _, sigma_sq_s_t = self.transition_params(t_k, t_k_p_1)
        _, sigma_sq_0_s = self.transition_params(torch.zeros_like(t_k), t_k)
        var = (sigma_sq_s_t * sigma_sq_0_s) / (sigma_sq_s_t + sigma_sq_0_s)
        mean = (sigma_sq_0_s * x + sigma_sq_s_t * x0) / (sigma_sq_s_t + sigma_sq_0_s)
        if return_mean_var:
            return mean, var
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            ret = mean + torch.sqrt(var) * z
            if return_log_prob:
                data_shape = x.shape[1:]
                dim = np.prod(data_shape)
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                log_prob -= 0.5 * dim * torch.log(var).flatten()
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

class EDM(LinearSDE):

    """Implementation of a EDM schedule
        d X_t = sqrt(t) dW_t
    """

    def __init__(self, sigma_min=2e-3, sigma_max=8e1):
        """Constructor for the EDM SDE

        Args:
            * sigma_min (float): Minimum value of sigma (default is 2e-3)
            * sigma_max (float): Maximum value of sigma (default is 8e+1)
        """
        super().__init__(T=1.0)
        self.register_buffer("sigma_min", torch.tensor(sigma_min, dtype=torch.float),
                             persistent=False)
        self.register_buffer("sigma_max", torch.tensor(sigma_max, dtype=torch.float),
                             persistent=False)
        self.T = self.sigma_inv(self.sigma_max)

    def get_base_dist(self, data_shape):
        """Returns the base distribution (i.e., the marginal distribution at time T)"""
        if len(data_shape) > 1:
            dim = math.prod(data_shape)
            return ReshapeWrapper(Gauss(
                mean=torch.zeros((dim,), device=self.sigma_max.device),
                variance=torch.square(self.sigma_max) * torch.ones((dim,), device=self.sigma_max.device)
            ), data_shape=data_shape)
        else:
            return Gauss(
                mean=torch.zeros(data_shape, device=self.sigma_max.device),
                variance=torch.square(self.sigma_max) * torch.ones(data_shape, device=self.sigma_max.device)
            )

    def sample_base_dist(self, shape, data_shape):
        """Sample the base distribution (i.e., the marginal distribution at time T)"""
        return self.sigma_max * torch.randn((*shape, *data_shape), device=self.sigma_max.device)


    def f(self, t):
        """Function f"""
        if isinstance(t, float):
            return torch.tensor(0.0).to(self.sigma_min.device)
        else:
            return torch.zeros_like(t)

    def g(self, t):
        """Function g"""
        return torch.sqrt(2. * t)

    def s(self, t):
        """Value of exp(int_0^t f(u) du)"""
        if isinstance(t, float):
            return torch.tensor(1.0).to(self.sigma_min.device)
        else:
            return torch.ones_like(t)

    def s_dot(self, t):
        """Derivative of s"""
        if isinstance(t, float):
            return torch.tensor(0.0).to(self.sigma_min.device)
        else:
            return torch.zeros_like(t)

    def sigma_sq(self, t):
        """Value of int_0^t g^2(u) / s^2(u) du"""
        return torch.square(t)

    def sigma_sq_dot(self, t):
        """Derivative of sigma_sq"""
        return 2. * t

    def sigma(self, t):
        """Square root of sigma_sq"""
        return t

    def sigma_dot(self, t):
        """Derivative of sigma"""
        if isinstance(t, float):
            return torch.tensor(1.0).to(self.sigma_min.device)
        else:
            return torch.ones_like(t)

    def sigma_inv(self, sigma):
        """Inverse of the sigma function"""
        return sigma

    def gamma_sq(self, t):
        """Product between s^2 and sigma_sq"""
        return self.sigma_sq(t)
    
    def gamma_sq_dot(self, t):
        """Derivative of gamma_sq"""
        return self.sigma_sq_dot(t)

    def gamma(self, t):
        """Product between s and sigma"""
        return self.sigma(t)

    def gamma_dot(self, t):
        """Derivate of gamma"""
        return self.sigma_dot(t)

    def s_dot_over_gamma(self, t):
        """Ratio of s_dot and gamma"""
        return torch.zeros_like(t)

    def gamma_dot_over_gamma(self, t):
        """Ratio of gamma_dot and gamma"""
        return 1. / t

    def sigma_dot_over_sigma(self, t):
        """Ratio of sigma_dot and sigma"""
        return 1. / t

    def get_snr_time_discretization(self, start, end, n_steps, rho=7., n_attemps=1024):
        """Get SNR adapted time discretization

        Follows the recommendations of Karras et al. 2022

        Args:
            * start (float or torch.Tensor): Start time (not used)
            * end (float or torch.Tensor): End time (not used)
            * n_steps (int): Number of intermediate times

        Returns:
            * ts (torch.Tensor of shape (n_steps,)): Time discretization
        """
        sigma_min_pow_one_over_rho = torch.pow(self.sigma_min, 1. / rho)
        sigma_max_pow_one_over_rho = torch.pow(self.sigma_max, 1. / rho)
        arr = n_steps-1-torch.arange(n_steps, device=self.sigma_min.device)
        return torch.pow(sigma_max_pow_one_over_rho \
            + (arr / (n_steps - 1)) * (sigma_min_pow_one_over_rho - sigma_max_pow_one_over_rho), rho)

    def transition_params(self, s, t):
        """Mean and variance parameters for noising transition kernel s -> t (s < t)

        We have that X_t = alpha_s X_s + gamma_s Z with Z ~ N(0,I)

        This function returns alpha_s and (sigma_s)^2.
        """
        mean_factor = torch.ones_like(t)
        var_factor = torch.square(t) - torch.square(s)
        return mean_factor, var_factor

    def transition_params_from_data(self, t):
        """Mean and variance parameters for noising transition kernel 0 -> t

        We have that X_t = alpha_0 X_0 + gamma_0 Z with Z ~ N(0,I)

        This function returns alpha_0 and (gamma_0)^2.
        """
        mean_factor = torch.ones_like(t)
        var_factor = self.sigma_sq(t)
        return mean_factor, var_factor

    def ei_integration_step(self, x, t_k, t_k_p_1, s, return_z=False, return_log_prob=False,
            return_mean_var=False, is_particles=False):
        """Denoising EI transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        lambda_k = (t_k_p_1 - t_k) * (t_k_p_1 + t_k)
        mean = x + lambda_k * s
        std = torch.sqrt(lambda_k)
        if return_mean_var:
            return mean, torch.square(std)
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            ret = mean + std * z
            if return_log_prob:
                data_shape = x.shape[1:]
                dim = np.prod(data_shape)
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                log_prob -= dim * torch.log(std).flatten()
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

    def ddpm_integration_step(self, x, t_k, t_k_p_1, s, h=None, use_forward_var=False,
                              return_z=False, return_log_prob=False, return_mean_var=False,
                              is_particles=False):
        """Denoising DDPM transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * s (torch.Tensor of shape (batch_size, *data_shape)): Marginal score at t_k_p_1
            * h (torch.Tensor of shape (batch_size, *data_shape) or (batch_size, *data_shape, *data_shape)): Hessian at t_k_p_1
            * use_forward_var (bool): Whether to use the variance of the forward kernel (default is False)
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored.
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """
        sigma_sq_s_t = self.transition_params(t_k, t_k_p_1)[1]
        mean = x + sigma_sq_s_t * s
        full_h = (not use_forward_var) and (h is not None) and (h.shape == (x.shape[0], *x.shape[1:], *x.shape[1:]))
        if full_h or return_log_prob:
            data_shape = x.shape[1:]
            dim = math.prod(data_shape)
            if full_h:
                I = torch.eye(dim, device=x.device).unsqueeze(0)
        if use_forward_var:
            var = sigma_sq_s_t
        else:
            if full_h:
                sigma_sq_s_t = sigma_sq_s_t.view((-1, 1, 1))
                var = sigma_sq_s_t * (I + sigma_sq_s_t * h.view((-1, dim, dim)))
            else:
                var = sigma_sq_s_t * (1 + sigma_sq_s_t * h)
        if return_mean_var:
            if full_h:
                var = var.view((-1, *data_shape, *data_shape))
            return mean, var
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            if full_h:
                ret = mean + torch.matmul(torch.linalg.cholesky(var), z.view((-1, dim, 1))).view((-1, *data_shape))
            else:
                ret = mean + torch.sqrt(var) * z
            if return_log_prob:
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                if full_h:
                    log_prob -= 0.5 * torch.logdet(var)
                elif use_forward_var:
                    log_prob -= 0.5 * dim * torch.log(var).flatten()
                else:
                    log_prob -= 0.5 * torch.log(var).sum(dim=sum_indexes)
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret

    def ddim_integration_step(self, x, t_k, t_k_p_1, post_sampler_fn, return_z=False,
            return_log_prob=False, return_mean_var=False, is_particles=False):
        """Denoising DDIM transition kernel from t_k_p_1 to t_k conditioned on x

        Here t_k < t_k_p_1 and p_{t_k_p_1} is more noisy than p_{t_k}.

        Args:
            * x (torch.Tensor of shape (batch_size, *data_shape)): Conditioning point
            * t_k (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Noisy time t_k
            * t_k_p_1 (torch.Tensor of shape (batch_size, *data_shape_ones) or float): Less noisy time t_k_p_1
            * post_sampler_fn (function): Function sampling the posterior
            * return_z (bool): Whether to return the Gaussian noise used
            * return_log_prob (bool): Whether to return the log-likelihood of the transition.
                Note that return_z will be ignored. (WARNING: ONLY WITH DETERMISTIC POSTERIOR)
            * return_mean_var (bool): Only return the mean and variance of the Gaussian kernel
                (WARNING: ONLY WITH DETERMISTIC POSTERIOR)            
            * is_particles (bool): Whether if it is a particle system (default is False)

        Returns:
            * x_next (torch.Tensor of the same shape as x): Denoised sample
        """

        # Sample the posterior
        x0 = post_sampler_fn(t_k_p_1, x)
        if is_particles:
            x0 = remove_mean(x0)
        # Sample the bridge
        t_k_over_t_k_p_1_sq = torch.square(t_k / t_k_p_1)
        var = (1. - t_k_over_t_k_p_1_sq) * torch.square(t_k)
        mean = x0 + t_k_over_t_k_p_1_sq * (x - x0)
        if return_mean_var:
            return mean, var
        else:
            z = torch.randn_like(x)
            if is_particles:
                z = remove_mean(z)
            ret = mean + torch.sqrt(var) * z
            if return_log_prob:
                data_shape = x.shape[1:]
                dim = np.prod(data_shape)
                sum_indexes = tuple(range(1, len(data_shape)+1))
                log_prob = -0.5 * torch.sum(torch.square(z), dim=sum_indexes)
                log_prob -= 0.5 * dim * math.log(2. * math.pi)
                log_prob -= 0.5 * dim * torch.log(var).flatten()
                return ret, log_prob
            elif return_z:
                return ret, z
            else:
                return ret
