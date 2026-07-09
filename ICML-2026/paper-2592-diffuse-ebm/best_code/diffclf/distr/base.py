# Base distribution

# Libraries
import torch
import warnings
import numpy as np
from ..metrics.ks import compute_sliced_ks
from ..metrics.mmd import compute_mmd
from ..metrics.wasserstein import compute_wasserstein_distance
from ot import sliced_wasserstein_distance as compute_sliced_wasserstein_distance

trace_vec = torch.vmap(torch.trace)


class Distribution(torch.nn.Module):

    dist = None

    def __init__(self, build_score=True, build_log_prob_and_grad=True, build_laplacian=True,
                 build_log_prob_and_grad_and_laplacian=True, data_path=None,
                 ):
        super().__init__()
        if build_score:
            self.score_fn = torch.vmap(torch.func.grad(self.log_prob_non_batched))
        if build_log_prob_and_grad or build_log_prob_and_grad_and_laplacian:
            grad_and_log_prob_non_batched_fn = torch.func.grad_and_value(self.log_prob_non_batched)
        if build_log_prob_and_grad:
            self.grad_and_log_prob_fn = torch.vmap(grad_and_log_prob_non_batched_fn)
        if build_laplacian:
            hessian_fn = torch.vmap(torch.func.hessian(self.log_prob))
            self.laplacian_fn = lambda x: trace_vec(hessian_fn(x))
        if build_log_prob_and_grad_and_laplacian:
            def grad_and_log_prob_non_batched_fn_aux(x):
                grad, value = grad_and_log_prob_non_batched_fn(x)
                return grad, (value, grad)
            log_prob_and_grad_and_hessian_non_batched_fn = torch.func.jacrev(
                grad_and_log_prob_non_batched_fn_aux, has_aux=True)

            def log_prob_and_grad_and_laplacian_non_batched_fn(x):
                hess, (value, grad) = log_prob_and_grad_and_hessian_non_batched_fn(x)
                return value, grad, torch.trace(hess)
            self.log_prob_and_grad_and_laplacian_fn = torch.vmap(
                log_prob_and_grad_and_laplacian_non_batched_fn)
        self.has_data = data_path is not None
        if self.has_data:
            if '.npy' in data_path:
                data = torch.from_numpy(np.load(data_path))
            else:
                data = torch.load(data_path, weights_only=False)
            self.load_data(data)

    def load_data(self, data):
        """Load the dataset"""
        if not self.has_data:
            raise NotImplementedError('load_data not supported is has_data is not set.')
        else:
            self.register_buffer('data', data)
            self.register_buffer('data_mean', torch.mean(self.data, dim=0))
            self.register_buffer('data_var', torch.var(self.data, dim=0))

    def build_dist(self):
        """Builds the inner dist object"""
        raise NotImplementedError

    def get_bad_distribution(self):
        """Build a bad version of the distribution"""
        raise NotImplementedError('The bad distribution is not implemented.')

    def mean(self):
        """Returns the mean of the distribution"""
        if self.has_data:
            return self.data_mean
        else:
            raise NotImplementedError('Mean is not implemented.')

    def variance(self):
        """Returns the variance of the distribution"""
        if self.has_data:
            return self.data_var
        else:
            raise NotImplementedError('Variance is not implemented.')

    def covariance(self):
        """Returns the covariance of the distribution"""
        raise NotImplementedError('Covariance is not implemented.')

    def log_prob(self, x):
        """Evaluates the log-likelihood of the distribution at x"""
        return self.dist.log_prob(x)

    def log_prob_non_batched(self, x):
        """Evaluates the log-likelihood of the distribution at x (non-batched)"""
        return self.log_prob(x.unsqueeze(0)).squeeze(0)

    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the distribution at x"""
        if hasattr(self, 'grad_and_log_prob_fn'):
            grad, log_prob = self.grad_and_log_prob_fn(x)
            return log_prob, grad
        else:
            raise NotImplementedError('log_prob_and_grad is not implemented.')

    def score(self, x):
        """Evaluates the score of the distribution at x"""
        if hasattr(self, 'score_fn'):
            return self.score_fn(x)
        else:
            raise NotImplementedError('score is not implemented.')

    def laplacian(self, x):
        """Evaluates the laplacian of the log-likelihood at x"""
        if hasattr(self, 'laplacian_fn'):
            return self.laplacian_fn(x)
        else:
            raise NotImplementedError('laplacian is not implemented.')

    def log_prob_and_grad_and_laplacian(self, x):
        """Evaluates the log-likelihood as well as its gradient and laplacian at x"""
        if hasattr(self, 'log_prob_and_grad_and_laplacian_fn'):
            return self.log_prob_and_grad_and_laplacian_fn(x)
        else:
            raise NotImplementedError('log_prob_and_grad_and_laplacian is not implemented.')

    def sample(self, sample_shape):
        """Returns samples from the distribution of shape sample_shape"""
        if self.has_data:
            return self.data[torch.randint(0, self.data.shape[0], sample_shape,
                                           device=self.data.device)]
        else:
            return self.dist.sample(sample_shape)

    def marginal_distr(self, t, sde):
        """Get the marginal distribution at time t"""
        raise NotImplementedError('marginal_distr is not implemented.')

    def marginal_sample(self, t, sde):
        """Returns samples from the marginal distribution of sde at times t"""
        x_0 = self.sample((t.shape[0],))
        mean, var = sde.noise_sample_params(t, x_0)
        return mean + torch.sqrt(var) * torch.randn_like(mean)

    def marginal_log_prob_and_grad(self, t, x, sde):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        raise NotImplementedError('marginal_log_prob_and_grad is not implemented.')

    def marginal_log_prob_and_grad_and_hess(self, t, x, sde, return_only_diag=False):
        """Returns the log-likelihood and score of the marginal of sde at t and x"""
        raise NotImplementedError('marginal_log_prob_and_grad_and_hess is not implemented.')

    def sample_exact_denoising_kernel(self, s, t, x_t, sde, return_log_prob=False):
        """Sample from the exact denoising kernel"""
        raise NotImplementedError('sample_exact_denoising_kernel is not implemented.')

    def log_prob_exact_denoising_kernel(self, s, t, x_s, x_t, sde):
        """Evaluate the log-likelihood of the exact denoising kernel"""
        raise NotImplementedError('log_prob_exact_denoising_kernel is not implemented.')

    def sample_exact_posterior(self, t, x_t, sde, return_log_prob=False):
        """Sample from the posterior"""
        raise NotImplementedError('sample_exact_posterior is not implemented.')

    def log_prob_exact_posterior(self, x, t, x_t, sde):
        """Evaluate the log-likelihood of the posterior"""
        raise NotImplementedError('log_prob_exact_posterior is not implemented.')

    def denoiser(self, t, x_t, sde):
        """Evaluate the denoiser"""
        raise NotImplementedError('denoiser is not implemented.')

    def posterior_covariance(self, t, x_t, sde, return_diag=False):
        """Evaluate the posterior's covariance"""
        raise NotImplementedError('posterior_covariance is not implemented.')

    def plot_samples(self, ax, samples, label="model"):
        """Display the samples"""
        true_samples = self.sample((samples.shape[0], )).cpu()
        ax.scatter(samples.detach().cpu()[:, 0], samples.detach().cpu()[:, 1], alpha=0.5, label=label)
        ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, label="true")
        ax.legend()

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

        # Get reference samples
        if ref_samples is None:
            ref_samples = self.sample((samples.shape[0],)).to(samples.device)
        # Reshape the samples if needed
        ref_samples = ref_samples.view((ref_samples.shape[0], -1))
        samples = samples.view((samples.shape[0], -1))
        # Compute standard metrics
        ret = {}
        if compute_standard_metrics:
            # Display a warning
            if weights is not None:
                warnings.warn('WARNING: MDD will ignore the weights.')
            ret['ks'] = compute_sliced_ks(ref_samples, samples, weights=weights)
            ret['mmd'] = compute_mmd(ref_samples, samples)
            ret['w2_sliced'] = compute_sliced_wasserstein_distance(ref_samples, samples,
                b=weights).item()
            if not skip_costly_metrics:
                flatten_dims = True
                if hasattr(self, "n_particles"):
                    ref_samples = ref_samples.view(-1, self.n_particles, self.n_dimensions)
                    samples = samples.view(-1, self.n_particles, self.n_dimensions)
                    flatten_dims = False
                ret['w2'] = compute_wasserstein_distance(ref_samples, samples, weights1=weights,
                    flattn_dims=flatten_dims)
                ret['w2_reg'] = compute_wasserstein_distance(ref_samples, samples, weights1=weights,
                    method='sinkhorn', flattn_dims=flatten_dims)
        return ret

    def _apply(self, fn):
        """Builds the distribution again when using _apply"""
        new_self = super(Distribution, self)._apply(fn)
        new_self.build_dist()
        return new_self
