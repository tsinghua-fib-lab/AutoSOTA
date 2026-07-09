# Closed-form helpers for Gaussian and mixture-of-Gaussian densities

# Libraries
import math
import torch
from diffclf.distr.base import Distribution


def mean_of_mog(weights, means):
    return torch.sum(weights.unsqueeze(-1) * means, dim=0)


mean_of_mog_batch = torch.vmap(mean_of_mog)


def cov_of_mog(weights, means, variances, return_diag=True, return_mean=False):
    mean = mean_of_mog(weights, means)
    means_diff = means - mean.unsqueeze(0)
    if return_diag:
        if means.shape == variances.shape:
            cov = variances
        else:
            arr = torch.arange(means.shape[-1], device=means.device)
            cov = variances[:, arr, arr]
        cov += torch.square(means_diff)
        cov = torch.sum(weights.unsqueeze(-1) * cov, dim=0)
    else:
        means_diff = means_diff.unsqueeze(-1)
        if means.shape == variances.shape:
            cov = torch.diag_embed(variances)
        else:
            cov = variances
        cov += torch.matmul(means_diff, means_diff.transpose(1, 2))
        cov = torch.sum(weights.view((-1, 1, 1)) * cov, dim=0)
    if return_mean:
        return mean, cov
    else:
        return cov


cov_of_mog_batch_ = torch.vmap(cov_of_mog, in_dims=(0, 0, 0, None, None))


def cov_of_mog_batch(weights, means, variances, return_diag=True, return_mean=False):
    return cov_of_mog_batch_(weights, means, variances, return_diag, return_mean)


def log_prob_gaussian(x, mean, variance):
    log_prob = -0.5 * torch.sum(torch.square(x - mean.unsqueeze(0)) / variance.unsqueeze(0), dim=-1)
    log_prob -= 0.5 * mean.shape[-1] * math.log(2. * math.pi)
    log_prob -= 0.5 * torch.log(variance).sum(dim=-1).unsqueeze(0)
    return log_prob


def log_prob_gaussian_full(x, means, covariances, precisions=None, covariances_log_det=None,
                           return_precision_times_diff=False):
    diff = x - means.unsqueeze(0)
    if precisions is None:
        precision_times_diff = torch.linalg.solve(covariances.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)
    else:
        precision_times_diff = torch.matmul(precisions.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)
    log_prob = -0.5 * torch.sum(diff * precision_times_diff, dim=-1)
    log_prob -= 0.5 * means.shape[-1] * math.log(2. * math.pi)
    if covariances_log_det is None:
        log_prob -= 0.5 * torch.logdet(covariances).unsqueeze(0)
    else:
        log_prob -= 0.5 * covariances_log_det.unsqueeze(0)
    if return_precision_times_diff:
        return log_prob, precision_times_diff
    else:
        return log_prob


def log_prob_and_grad_gauss(x, mean, variance, return_log_prob=False):
    log_prob = log_prob_gaussian(x, mean, variance)
    grad = -(x - mean) / variance
    # Return the score
    if return_log_prob:
        return log_prob, grad
    else:
        return grad


def log_prob_and_grad_gauss_full(x, mean, covariance, precision=None, covariance_log_det=None,
                                 return_log_prob=False):
    log_prob, precision_times_diff = log_prob_gaussian_full(x, mean, covariance, precisions=precision,
                                                            covariances_log_det=covariance_log_det, return_precision_times_diff=True)
    grad = -precision_times_diff
    # Return the score
    if return_log_prob:
        return log_prob, grad
    else:
        return grad


def log_prob_and_grad_mog(x, weights, means, variances, return_log_prob=False):
    # Normalize the weights
    weights = weights / weights.sum()
    # Compute the individual gaussian probs
    log_probs = log_prob_gaussian(x.unsqueeze(1), means, variances)
    log_probs += torch.log(weights.unsqueeze(0))
    probs = torch.nn.functional.softmax(log_probs, dim=-1).unsqueeze(-1)
    # Compute the score
    grad = -torch.sum(probs * (x.unsqueeze(1) - means.unsqueeze(0)) / variances.unsqueeze(0), dim=1)
    # Return the score
    if return_log_prob:
        return torch.logsumexp(log_probs, dim=-1), grad
    else:
        return grad


def log_prob_and_grad_mog_full(x, weights, means, covariances, precisions=None, covariances_log_det=None,
                               return_log_prob=False):
    # Normalize the weights
    weights = weights / weights.sum()
    # Compute the individual gaussian probs
    log_probs, precision_times_diff = log_prob_gaussian_full(x.unsqueeze(1), means, covariances,
                                                             precisions=precisions,
                                                             covariances_log_det=covariances_log_det,
                                                             return_precision_times_diff=True)
    log_probs += torch.log(weights.unsqueeze(0))
    probs = torch.nn.functional.softmax(log_probs, dim=-1).unsqueeze(-1)
    # Compute the score
    grad = -torch.sum(probs * precision_times_diff, dim=1)
    # Return the score
    if return_log_prob:
        return torch.logsumexp(log_probs, dim=-1), grad
    else:
        return grad


def log_prob_and_grad_and_hess_gauss(x, mean, variance, return_only_diag=True, return_log_prob=False, return_grad=False):
    log_prob = log_prob_gaussian(x, mean, variance)
    grad = -(x - mean) / variance
    if return_only_diag:
        hessian = -1. / variance.unsqueeze(0).expand((x.shape[0], -1))
    else:
        hessian = -torch.eye(variance.shape[0], device=variance.device).unsqueeze(0) \
            * (1. / variance).unsqueeze(0).unsqueeze(-1)
        hessian = hessian.expand((x.shape[0], -1, -1))
    # Return the everything
    if return_log_prob:
        if return_grad:
            return log_prob, grad, hessian
        else:
            return log_prob, hessian
    else:
        if return_grad:
            return grad, hessian
        else:
            return hessian


def log_prob_and_grad_and_hess_gauss_full(x, mean, covariance, precision=None, covariance_log_det=None,
                                          return_only_diag=True, return_log_prob=False, return_grad=False):
    log_prob, precision_times_diff = log_prob_gaussian_full(x, mean, covariance,
                                                            precisions=precision, covariances_log_det=covariance_log_det, return_precision_times_diff=True)
    grad = -precision_times_diff
    if return_only_diag:
        if precision is not None:
            hessian = -torch.diag(precision).unsqueeze(0).expand((x.shape[0], -1))
        else:
            hessian = -torch.diag(torch.linalg.inv(covariance)).unsqueeze(0).expand((x.shape[0], -1))
    else:
        if precision is not None:
            hessian = -precision.unsqueeze(0).expand((x.shape[0], -1, -1))
        else:
            hessian = -torch.linalg.inv(covariance).unsqueeze(0).expand((x.shape[0], -1, -1))
    # Return the everything
    if return_log_prob:
        if return_grad:
            return log_prob, grad, hessian
        else:
            return log_prob, hessian
    else:
        if return_grad:
            return grad, hessian
        else:
            return hessian


def log_prob_and_grad_and_hess_mog(x, weights, means, variances, return_only_diag=True,
                                   return_log_prob=False, return_grad=False):
    # Normalize the weights
    weights = weights / weights.sum()
    # Compute the individual gaussian probs
    log_probs = log_prob_gaussian(x.unsqueeze(1), means, variances)
    log_probs += torch.log(weights.unsqueeze(0))
    probs = torch.nn.functional.softmax(log_probs, dim=-1).unsqueeze(-1)
    scores = -(x.unsqueeze(1) - means.unsqueeze(0)) / variances.unsqueeze(0)
    # Compute the score
    grad = torch.sum(probs * scores, dim=1)
    # Compute the hessian
    if return_only_diag:
        hess_aux = (scores - grad.unsqueeze(1)) * scores - (1. / variances.unsqueeze(0))
        hessian = torch.sum(probs * hess_aux, dim=1)
    else:
        arr = torch.arange(means.shape[-1], device=means.device)
        hess_aux = torch.einsum('bkd,bke->bkde', scores - grad.unsqueeze(1), scores)
        hess_aux[:, :, arr, arr] -= 1. / variances.unsqueeze(0)
        hessian = torch.sum(probs.unsqueeze(-1) * hess_aux, dim=1)
    # Return everything
    if return_log_prob:
        log_prob = torch.logsumexp(log_probs, dim=-1)
        if return_grad:
            return log_prob, grad, hessian
        else:
            return log_prob, hessian
    else:
        if return_grad:
            return grad, hessian
        else:
            return hessian


def log_prob_and_grad_and_hess_mog_full(x, weights, means, covariances, precisions=None, covariances_log_det=None,
                                        return_only_diag=True, return_log_prob=False, return_grad=False):
    # Normalize the weights
    weights = weights / weights.sum()
    # Compute the individual gaussian probs
    log_probs, precision_times_diff = log_prob_gaussian_full(x.unsqueeze(1), means, covariances,
                                                             precisions=precisions,
                                                             covariances_log_det=covariances_log_det,
                                                             return_precision_times_diff=True)
    log_probs += torch.log(weights.unsqueeze(0))
    probs = torch.nn.functional.softmax(log_probs, dim=-1).unsqueeze(-1)
    # Compute the score
    grad = -torch.sum(probs * precision_times_diff, dim=1)
    # Compute the hessian
    if return_only_diag:
        arr = torch.arange(means.shape[-1], device=means.device)
        hess_aux = -(-precision_times_diff - grad.unsqueeze(1)) * precision_times_diff
        if precisions is not None:
            hess_aux -= precisions[:, arr, arr].unsqueeze(0)
        else:
            hess_aux -= torch.linalg.inv(covariances)[:, arr, arr].unsqueeze(0)
        hessian = torch.sum(probs * hess_aux, dim=1)
    else:
        hess_aux = torch.einsum('bkd,bke->bkde', -precision_times_diff - grad.unsqueeze(1), -precision_times_diff)
        if precisions is not None:
            hess_aux -= precisions.unsqueeze(0)
        else:
            hess_aux -= torch.linalg.inv(covariances).unsqueeze(0)
        hessian = torch.sum(probs.unsqueeze(-1) * hess_aux, dim=1)
    # Return everything
    if return_log_prob:
        log_prob = torch.logsumexp(log_probs, dim=-1)
        if return_grad:
            return log_prob, grad, hessian
        else:
            return log_prob, hessian
    else:
        if return_grad:
            return grad, hessian
        else:
            return hessian

def rejection_sampling(n_samples, proposal, target_log_prob_fn, k, sampling_factor=10):
    """Draw samples from an unnormalized target distribution using rejection sampling.

    It is assumed to be 1D.

    Args:

        * n_samples (int):  Number of samples to draw from the target distribution
        * proposal (Distribution): Proposal distribution to sample
        * target_log_prob_fn (Callable): Target log-likelihood
        * k (float): Scaling constant
        * sampling_factor (int): Sampling factor when sampling the proposal (default is 10)

    Returns:
        * samples:  Samples drawn from the target distribution
    """
    z_0 = proposal.sample((sampling_factor * n_samples,)).flatten()
    proposal_z_0 = proposal.log_prob(z_0).flatten()
    u_0 = torch.rand_like(proposal_z_0) * torch.exp(proposal_z_0) * k
    accept = torch.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(required_samples, proposal, target_log_prob_fn, k)
        samples = torch.cat([samples, new_samples], dim=0)
        return samples


def importance_resampling(n_samples, n_particles, proposal, target_log_prob_fn):
    z = proposal.sample((n_particles, ))
    log_q_z = proposal.log_prob(z)
    log_p_z = target_log_prob_fn(z)
    log_w = log_p_z - log_q_z
    w = torch.softmax(log_w, dim=0)
    idx = torch.multinomial(w, n_samples, replacement=True)
    ess = torch.exp(2. * torch.logsumexp(log_w, dim=0) - torch.logsumexp(2. * log_w, dim=0)) / n_particles
    return z[idx], ess

class ReshapeWrapper(Distribution):

    def __init__(self, base_dist, data_shape):
        super().__init__(build_score=False, build_log_prob_and_grad=False, build_laplacian=False,
                 build_log_prob_and_grad_and_laplacian=False, data_path=None)
        self.base_dist = base_dist
        self.data_shape = data_shape
        self.data_shape_ones = (1,) * len(data_shape)
        self.dim = math.prod(data_shape)

    def build_dist(self):
        """Builds the inner dist object"""
        self.base_dist.build_dist()

    def get_bad_distribution(self):
        """Build a bad version of the distribution"""
        return ReshapeWrapper(self.base_dist.get_bad_distribution(), data_shape=self.data_shape)

    def mean(self):
        """Returns the mean of the distribution"""
        return self.base_dist.mean().view(self.data_shape)

    def variance(self):
        """Returns the variance of the distribution"""
        return self.base_dist.variance().view(self.data_shape)

    def log_prob(self, x):
        """Evaluates the log-likelihood of the distribution at x"""
        return self.base_dist.log_prob(x.view((x.shape[0], self.dim)))

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the distribution at x"""
        log_prob, grad = self.base_dist.log_prob_and_grad(x.view((x.shape[0], self.dim)))
        grad = grad.view(x.shape)
        return log_prob, grad

    def score(self, x):
        """Evaluates the score of the distribution at x"""
        return self.base_dist.score(x.view((x.shape[0], self.dim))).view(x.shape)

    def laplacian(self, x):
        """Evaluates the laplacian of the log-likelihood at x"""
        return self.base_dist.laplacian(x.view((x.shape[0], self.dim)))

    def log_prob_and_grad_and_laplacian(self, x):
        """Evaluates the log-likelihood as well as its gradient and laplacian at x"""
        log_prob, grad, lap = self.base_dist.log_prob_and_grad_and_laplacian(x.view((x.shape[0],self.dim)))
        grad = grad.view(x.shape)
        return log_prob, grad, lap

    def sample(self, sample_shape):
        """Returns samples from the distribution of shape sample_shape"""
        return self.base_dist.sample(sample_shape).view((-1, *self.data_shape))

    def marginal_distr(self, t, sde):
        """Get the marginal distribution at time t"""
        return ReshapeWrapper(self.base_dist.marginal_distr(t.view((-1, 1)), sde), self.data_shape)

    def marginal_log_prob_and_grad(self, t, x, sde):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        log_prob, grad = self.base_dist.marginal_log_prob_and_grad(t.view((-1, 1)), x.view((-1, self.dim)), sde)
        grad = grad.view(x.shape)
        return log_prob, grad

    def marginal_log_prob_and_grad_and_hess(self, t, x, sde, return_only_diag=False):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        log_prob, grad, hess = self.base_dist.marginal_log_prob_and_grad_and_hess(t.view((-1, 1)), x.view((-1, self.dim)),
            sde, return_only_diag=return_only_diag)
        grad = grad.view(x.shape)
        if return_only_diag:
            hess = hess.view(x.shape)
        else:
            hess = hess.view((-1, *self.data_shape, *self.data_shape))
        return log_prob, grad, hess

    def sample_exact_denoising_kernel(self, s, t, x_t, sde, return_log_prob=False):
        """Sample from the exact denoising kernel"""
        ret = self.base_dist.sample_exact_denoising_kernel(s.view((-1, 1)), t.view((-1, 1)),
            x_t.view((-1, self.dim)), sde, return_log_prob=return_log_prob)
        if return_log_prob:
            return ret[0], ret[1].view(x_t.shape)
        else:
            return ret.view(x_t.shape)

    def log_prob_exact_denoising_kernel(self, s, t, x_s, x_t, sde):
        """Evaluate the log-likelihood of the exact denoising kernel"""
        return self.base_dist.log_prob_exact_denoising_kernel(s.view((-1, 1)), t.view((-1, 1)),
            x_s.view((-1, self.dim)), x_t.view((-1, self.dim)), sde)

    def sample_exact_posterior(self, t, x_t, sde, return_log_prob=False):
        """Sample from the posterior"""
        ret = self.base_dist.sample_exact_posterior(t.view((-1, 1)), x_t.view((-1, self.dim)),
            sde, return_log_prob=return_log_prob)
        if return_log_prob:
            return ret[0], ret[1].view(x_t.shape)
        else:
            return ret.view(x_t.shape)

    def log_prob_exact_posterior(self, x, t, x_t, sde):
        """Evaluate the log-likelihood of the posterior"""
        return self.base_dist.log_prob_exact_posterior(x.view((-1, self.dim)), t.view((-1, 1)),
            x_t.view((-1, self.dim)), x_t.view((-1, self.dim)), sde)

    def denoiser(self, t, x_t, sde):
        """Evaluate the denoiser"""
        return self.base_dist.denoiser(t.view((-1, 1)), x_t.view((-1, self.dim)), sde).view(x_t.shape)

    def posterior_covariance(self, t, x_t, sde, return_diag=False):
        """Evaluate the posterior's covariance"""
        var = self.base_dist.posterior_covariance(t.view((-1, 1)), x_t.view((-1, self.dim)),
            sde, return_diag=return_diag)
        if return_diag:
            var = var.view(x_t.shape)
        else:
            var = var.view((-1, *self.data_shape, *self.data_shape))
        return var

    def plot_samples(self, ax, samples, label="model"):
        """Display the samples"""
        raise NotImplementedError('Plotting is not supported.')

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
        return super().compute_metrics(samples.view((-1, self.dim)), weights=weights,
            ref_samples=ref_samples.view((-1, self.dim)) if ref_samples else None,
            compute_standard_metrics=compute_standard_metrics, skip_costly_metrics=skip_costly_metrics)
