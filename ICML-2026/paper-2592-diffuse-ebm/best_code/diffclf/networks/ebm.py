# Energy-based models architectures

# Libraries
import torch
import math
from ..utils.se3_utils import remove_mean

class EBM(torch.nn.Module):
    """Base class for EBM"""

    def     __init__(self, build_score=True, build_log_prob_dot=True, build_grad_and_log_prob=True,
        build_log_prob_and_grad_and_dot=True, sde=None):
        super().__init__()
        if build_score:
            score_non_batched_fn = torch.func.grad(self.log_prob_non_batched, argnums=1)
        if build_score:
            self.score_fn = torch.func.vmap(score_non_batched_fn)
        if build_log_prob_dot:
            log_prob_dot_non_batched_fn = torch.func.grad(self.log_prob_non_batched, argnums=0)
            self.log_prob_dot_fn = torch.func.vmap(log_prob_dot_non_batched_fn)
        if build_grad_and_log_prob:
            grad_and_value_non_batched_fn = torch.func.grad_and_value(self.log_prob_non_batched, argnums=1)
        if build_grad_and_log_prob:
            self.grad_and_log_prob_fn = torch.func.vmap(grad_and_value_non_batched_fn)
        if build_log_prob_and_grad_and_dot:
            log_prob_and_grad_and_dot_fn_non_batched_fn = torch.func.grad_and_value(self.log_prob_non_batched,
                argnums=(0, 1))
            self.log_prob_and_grad_and_dot_fn = torch.func.vmap(log_prob_and_grad_and_dot_fn_non_batched_fn)
        self.sde = sde

    def energy(self, t, x):
        """Compute the energy"""
        raise NotImplementedError('energy is not implemented.')

    def log_prob(self, t, x):
        """Compute the log-likelihood (i.e., negative energy)"""
        return -self.energy(t, x)

    def log_prob_non_batched(self, t, x):
        """Compute the log-likelihood with non-batched entries (i.e., negative energy)"""
        return self.log_prob(t.unsqueeze(0), x.unsqueeze(0)).squeeze(0)

    def log_prob_and_grad(self, t, x, return_denoiser=False):
        """Evaluates the log-likelihood and the score of the distribution at (t,x)"""
        if hasattr(self, 'grad_and_log_prob_fn'):
            grad, log_prob = self.grad_and_log_prob_fn(t, x)
            if return_denoiser:
                if self.sde is not None:
                    alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
                    denoiser = (x + gamma_sq_t * grad) / alpha_t
                    return log_prob, denoiser
                else:
                    raise NotImplementedError('Cannot return the denoiser without an SDE.')
            else:
                return log_prob, grad
        else:
            raise NotImplementedError('log_prob_and_grad is not implemented.')
        
    def log_prob_and_grad_and_dot(self, t, x, return_denoiser=False):
        """Evaluates the log-likelihood, x-score, and t-score of the distribution at (t,x)"""
        if hasattr(self, 'log_prob_and_grad_and_dot_fn'):
            (grad_t, grad_x), log_prob = self.log_prob_and_grad_and_dot_fn(t, x)
            if return_denoiser:
                if self.sde is not None:
                    alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
                    denoiser = (x + gamma_sq_t * grad_x) / alpha_t
                    return log_prob, denoiser, grad_t
                else:
                    raise NotImplementedError('Cannot return the denoiser without an SDE.')
            else:
                return log_prob, grad_x, grad_t
        else:
            raise NotImplementedError('log_prob_and_grad is not implemented.')

    def score(self, t, x):
        """Evaluates the score of the distribution at (t,x)"""
        if hasattr(self, 'score_fn'):
            return self.score_fn(t, x)
        else:
            raise NotImplementedError('score is not implemented.')
        
    def log_prob_dot(self, t, x):
        """Evaluates the t-score of the distribution at (t,x)"""
        if hasattr(self, 'log_prob_dot_fn'):
            return self.log_prob_dot_fn(t, x)
        else:
            raise NotImplementedError('log_prob_dot is not implemented.')

    def denoiser(self, t, x):
        """Evaluates the score of the distribution at (t,x)"""
        return self.log_prob_and_grad(t, x, return_denoiser=True)[1]

    def forward(self, t, x):
        """Compute the score"""
        return self.score(t, x)


class BasicEBM(EBM):
    """Basic implementation of an EBM"""

    def __init__(self, base_ebm, sde=None):
        """Constructor for BasicEBM

        Args:
            * base_ebm (torch.nn.Module): Neural network of output shape (batch_size, 1)
        """
        super().__init__(sde=sde)
        self.base_ebm = base_ebm

    def energy(self, t, x):
        """Compute the energy"""
        return self.base_model(t, x).flatten()


class AdvancedEBM(BasicEBM):
    """Advanced implementation of an EBM"""

    def __init__(self, base_ebm, data_shape, energy_type='sq_norm',
            data_mean=None, data_var=None, log_snr_dist=None, sde=None):
        """Constructor for AdvancedEBM

        Args:
            * base_ebm (torch.nn.Module): Neural network of output shape (batch_size, *data_shape)
            * data_shape (tuple of int): Shape of the data
            * energy_type (str): Type of energy function (default is 'sq_norm')
                - dot
                        E(t, x) = NN(t,X)^T X
                - sq_norm
                        E(t, X) = norm(NN(t,X))^2
                - residual_sq_norm
                        E(t, X) = norm(NN(t,X) - X)^2
            * sde (OU): SDE object (for rescaling) (default is None)
            * data_mean (torch.Tensor of shape data_shape): Mean of the data
                    (for rescaling) (default is None)
            * data_scalar_var (torch.Tensor of shape data_shape): Diagonance variance of the data
                    (for rescaling) (default is None)
        """
        super().__init__(base_ebm=base_ebm, sde=sde)
        self.energy_type = energy_type
        self.log_snr_dist = log_snr_dist
        self.sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
        self.do_rescaling = all([(x is not None) for x in [sde, data_mean, data_var]])
        if (not self.do_rescaling) and any([(x is not None) for x in [sde, data_mean, data_var]]):
            raise ValueError('sde, data_mean and data_var have to be all defined to do rescaling.')
        if self.do_rescaling:
            self.register_buffer("data_mean", data_mean)
            self.register_buffer("data_var", data_var)

    def scaling_input(self, t, x):
        """Rescale the input"""
        if self.do_rescaling:
            alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
            c_i = torch.sqrt(torch.square(alpha_t) * self.data_var.unsqueeze(0) + gamma_sq_t)
            c_m = self.sde.s(t) * self.data_mean.unsqueeze(0)
            if self.log_snr_dist:
                t_scaled = 0.3 * self.sde.log_snr(t) / self.log_snr_dist[1]
                t_scaled -= 0.3 * (1. + (self.log_snr_dist[0] / self.log_snr_dist[1]))
            else:
                t_scaled = t
            return t_scaled, (x - c_m) / c_i
        else:
            return x

    def energy(self, t, x):
        """Compute the energy"""
        t_scaled, x_scaled = self.scaling_input(t, x)
        if self.energy_type == 'dot':
            return torch.sum(self.base_ebm(t_scaled, x_scaled) * x_scaled, dim=self.sum_indexes)
        elif self.energy_type == 'sq_norm':
            return 0.5 * torch.sum(torch.square(self.base_ebm(t_scaled, x_scaled)), dim=self.sum_indexes)
        elif self.energy_type == 'residual_sq_norm':
            return 0.5 * torch.sum(torch.square(self.base_ebm(t_scaled, x_scaled) - x_scaled), dim=self.sum_indexes)
        else:
            return self.base_ebm(t, x_scaled).sum(dim=self.sum_indexes)


class AddCteEBM(BasicEBM):
    """Add a small parametrized constant to the EBM"""

    def __init__(self, base_ebm, sde=None):
        super(EBM, self).__init__(base_ebm=base_ebm, sde=sde)
        self.c = torch.nn.Parameter(torch.rand(1,))

    def energy(self, t, x):
        """Compute the energy"""
        return self.base_ebm.energy(t, x) + self.c

    def log_prob_and_grad(self, t, x, return_denoiser=False):
        """Evaluates the log-likelihood and the score of the distribution at (t,x)"""
        log_prob, grad = self.base_ebm.log_prob_and_grad(t, x, return_denoiser=return_denoiser)
        return log_prob + self.c, grad

    def log_prob_and_grad_and_dot(self, t, x, return_denoiser=False):
        """Evaluates the log-likelihood, x-score, and t-score of the distribution at (t,x)"""
        log_prob, grad_x, grad_t = self.base_ebm.log_prob_and_grad_and_dot(t, x, return_denoiser=return_denoiser)
        return log_prob + self.c, grad_x, grad_t

    def score(self, t, x):
        """Evaluates the score of the distribution at (t,x)"""
        return self.base_ebm.score(t, x)

    def log_prob_dot(self, t, x):
        """Evaluates the t-score of the distribution at (t,x)"""
        return self.base_ebm.log_prob_dot(t, x)

class ResidualSquaredNormEBM(BasicEBM):
    """EBM as the squared norm of a network minus the input"""
    def energy(self, t, x):
        data_shape = x.shape[1:]
        sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
        return torch.sum(torch.square(self.base_ebm(t, x) - x), dim=sum_indexes)

class SquaredNormEBM(BasicEBM):
    """EBM as the squared norm of a network"""
    def energy(self, t, x):
        data_shape = x.shape[1:]
        sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
        return torch.sum(torch.square(self.base_ebm(t, x)), dim=sum_indexes)

class DotEBM(BasicEBM):
    """EBM as the dot product of a network"""
    def energy(self, t, x):
        data_shape = x.shape[1:]
        sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
        return torch.sum(self.base_ebm(t, x) * x, dim=sum_indexes)

class SIEnergyDenoiserNet(BasicEBM):

    def __init__(self, base_net, add_net, is_particles=False, gamma_type='brownian', gamma_factor=1.0):
        super().__init__(base_ebm=base_net)
        self.add_net = add_net
        self.is_particles = is_particles
        if gamma_type == 'brownian':
            self.gamma = lambda t: gamma_factor * torch.sqrt(t * (1. - t))
        else:
            raise ValueError('Gamma function {} not implemented.'.format(gamma_type))
    
    def energy(self, t, x):
        """Compute the energy"""
        data_shape = x.shape[1:]
        sum_indexes = tuple([-(i + 1) for i in range(len(data_shape))])
        if self.is_particles:
            x = remove_mean(x)
        ret = torch.sum(x * self.base_ebm(t, x), dim=sum_indexes)
        if self.add_net is not None:
            ret += self.add_net(t, x).flatten()
        return ret

    def log_prob_and_grad(self, t, x, return_denoiser=False):
        """Evaluates the log-likelihood and the score of the distribution at (t,x)"""
        grad, log_prob = self.grad_and_log_prob_fn(t, x)
        if self.is_particles:
            grad = remove_mean(grad)
        if return_denoiser:
            return log_prob, -self.gamma(t) * grad
        else:
            return log_prob, grad
        
    def log_prob_and_grad_and_dot(self, t, x, return_denoiser=False):
        """Evaluates the log-likelihood, x-score, and t-score of the distribution at (t,x)"""
        (grad_t, grad_x), log_prob = self.log_prob_and_grad_and_dot_fn(t, x)
        if self.is_particles:
            grad_x = remove_mean(grad_x)
        if return_denoiser:
            return log_prob, -self.gamma(t) * grad_x, grad_t
        else:
            return log_prob, grad_x, grad_t

    def score(self, t, x):
        """Evaluates the score of the distribution at (t,x)"""
        score = self.score_fn(t, x)
        if self.is_particles:
            score = remove_mean(score)
        return score

    def denoiser(self, t, x):
        """Evaluates the score of the distribution at (t,x)"""
        return -self.gamma(t) * self.score(t, x)

class EDMEnergyPreconditioning(BasicEBM):

    def __init__(self, base_ebm, sde, data_mean, data_var_scalar, is_particles=False, log_snr_dist=None):
        super().__init__(base_ebm=base_ebm, sde=sde)
        self.log_snr_dist = log_snr_dist
        self.data_shape = tuple(data_mean.shape)
        self.dim = math.prod(self.data_shape)
        self.sum_indexes = tuple([-(i + 1) for i in range(len(self.data_shape))])
        self.is_particles = is_particles
        self.register_buffer("data_mean", data_mean.unsqueeze(0))
        self.register_buffer("data_var_scalar", data_var_scalar)
        self.register_buffer("data_mean_norm", torch.sum(torch.square(self.data_mean)))

    def get_parameters(self, t):
        # Compute s and sigma
        alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
        alpha_t_sq = torch.square(alpha_t)
        s_, sigma_sq_ = self.sde.s(t), self.sde.sigma_sq(t)
        sigma_ = torch.sqrt(sigma_sq_)
        # Compute alpha_in and beta_in
        alpha_in_sq = alpha_t_sq * self.data_var_scalar + gamma_sq_t
        alpha_in = torch.sqrt(alpha_in_sq)
        beta_in = alpha_t * self.data_mean
        # Comput alpha_out and beta_out
        alpha_out = sigma_ * torch.sqrt(self.data_var_scalar / (self.data_var_scalar + sigma_sq_))
        beta_out = (1.0 - (self.data_var_scalar / (self.data_var_scalar + sigma_sq_))) * self.data_mean
        # Compute alpha_skip
        alpha_skip = (self.data_var_scalar / (self.data_var_scalar + sigma_sq_)) / s_
        # Compute the log-normalizing constant
        log_z = 0.5 * alpha_t_sq * self.data_mean_norm / (self.data_var_scalar + sigma_sq_)
        log_z += 0.5 * self.dim * torch.log(2. * torch.pi * alpha_in_sq)
        log_z = log_z.flatten()
        # Return everything
        return alpha_in, beta_in, alpha_out, beta_out, alpha_skip, log_z

    def precond_time(self, t):
        """Comprecondition time"""
        if self.log_snr_dist:
            t_precond = 0.3 * self.sde.log_snr(t) / self.log_snr_dist[1]
            t_precond -= 0.3 * (1. + (self.log_snr_dist[0] / self.log_snr_dist[1]))
        else:
            t_precond = t
        return t_precond

    def energy(self, t, x):
        """Compute the energy"""
        raise -self.log_prob(t, x)

    def log_prob(self, t, x):
        """Compute the log-likelihood (i.e., negative energy)"""
        if self.is_particles:
            x = remove_mean(x)
        alpha_in, beta_in, _, _, _, log_z = self.get_parameters(t)
        s_t, sigma_sq_t = self.sde.s(t), self.sde.sigma_sq(t)
        log_prob = -0.5 * torch.sum(torch.square(x), dim=self.sum_indexes) / torch.square(alpha_in).flatten()
        log_prob += torch.sqrt(self.data_var_scalar / sigma_sq_t.flatten()) * self.base_ebm.log_prob(self.precond_time(t), (x - beta_in) / alpha_in)
        log_prob += (s_t * sigma_sq_t).flatten() * torch.sum(self.data_mean * x, dim=self.sum_indexes) / (sigma_sq_t + self.data_var_scalar).flatten()
        log_prob -= log_z
        return log_prob

    def log_prob_and_grad(self, t, x, return_denoiser=False):
        """Evaluates the log-likelihood and the score of the distribution at (t,x)"""
        if self.is_particles:
            x = remove_mean(x)
        alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
        alpha_in, beta_in, alpha_out, beta_out, alpha_skip, log_z = self.get_parameters(t)
        s_t, sigma_sq_t = self.sde.s(t), self.sde.sigma_sq(t)
        base_log_prob, base_grad = self.base_ebm.log_prob_and_grad(self.precond_time(t), (x - beta_in) / alpha_in)
        log_prob = -0.5 * torch.sum(torch.square(x), dim=self.sum_indexes) / torch.square(alpha_in).flatten()
        log_prob += torch.sqrt(self.data_var_scalar / sigma_sq_t.flatten()) * base_log_prob
        log_prob += (s_t * sigma_sq_t).flatten() * torch.sum(self.data_mean * x, dim=self.sum_indexes) / (sigma_sq_t + self.data_var_scalar).flatten()
        log_prob -= log_z
        denoiser = alpha_skip * x + alpha_out * base_grad + beta_out
        if self.is_particles:
            denoiser = remove_mean(denoiser)
        if return_denoiser:
            return log_prob, denoiser
        else:
            return log_prob, (alpha_t * denoiser - x) / gamma_sq_t

    def denoiser(self, t, x):
        """Evaluates the score of the distribution at (t,x)"""
        alpha_in, beta_in, alpha_out, beta_out, alpha_skip, _ = self.get_parameters(t)
        if self.is_particles:
            x = remove_mean(x)
        denoiser = alpha_skip * x + alpha_out * self.base_ebm.score(
            self.precond_time(t), (x - beta_in) / alpha_in) + beta_out
        if self.is_particles:
            denoiser = remove_mean(denoiser)
        return denoiser

    def score(self, t, x):
        """Evaluates the denoiser of the distribution at (t,x)"""
        alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
        if self.is_particles:
            x = remove_mean(x)
        d = self.denoiser(t, x)
        return (alpha_t * d - x) / gamma_sq_t
