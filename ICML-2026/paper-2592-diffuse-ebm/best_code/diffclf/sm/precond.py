# Implement wrappers for the different tricks of EDM

# Libraries
import torch
from ..utils.se3_utils import remove_mean


class TweedieWrapper(torch.nn.Module):

    def __init__(self, denoiser_net, sde):
        super().__init__()
        self.denoiser_net = denoiser_net
        self.sde = sde

    def forward(self, t, x):
        alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
        return (alpha_t * self.denoiser_net(t, x) - x) / gamma_sq_t


class InputPreconditioning(torch.nn.Module):

    def __init__(self, base_net, sde, data_mean, data_var_scalar, log_snr_dist=None):
        super().__init__()
        self.base_net = base_net
        self.sde = sde
        self.log_snr_dist = log_snr_dist
        self.register_buffer("data_mean", data_mean.unsqueeze(0))
        self.register_buffer("data_var_scalar", data_var_scalar)

    def get_parameters(self, t):
        # Compute s and sigma
        alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
        # Compute alpha_in and beta_in
        alpha_in = torch.sqrt(torch.square(alpha_t) * self.data_var_scalar + gamma_sq_t)
        beta_in = alpha_t * self.data_mean
        # Return everything
        return alpha_in, beta_in

    def forward(self, t, x):
        alpha_in, beta_in = self.get_parameters(t)
        if self.log_snr_dist:
            t_precond = 0.3 * self.sde.log_snr(t) / self.log_snr_dist[1]
            t_precond -= 0.3 * (1. + (self.log_snr_dist[0] / self.log_snr_dist[1]))
        else:
            t_precond = t
        return self.base_net(t_precond, (x - beta_in) / alpha_in)


class EDMDenoiserPreconditioning(torch.nn.Module):

    def __init__(self, base_net, sde, data_mean, data_var_scalar, log_snr_dist=None, is_particles=False):
        super().__init__()
        self.base_net = base_net
        self.sde = sde
        self.log_snr_dist = log_snr_dist
        self.is_particles = is_particles
        self.data_shape = tuple(data_mean.shape)
        self.register_buffer("data_mean", data_mean.unsqueeze(0))
        self.register_buffer("data_var_scalar", data_var_scalar)

    def get_parameters(self, t):
        # Compute s and sigma
        alpha_t, gamma_sq_t = self.sde.transition_params_from_data(t)
        s_, sigma_sq_ = self.sde.s(t), self.sde.sigma_sq(t)
        sigma_ = torch.sqrt(sigma_sq_)
        # Compute alpha_in and beta_in
        alpha_in = torch.sqrt(torch.square(alpha_t) * self.data_var_scalar + gamma_sq_t)
        beta_in = alpha_t * self.data_mean
        # Comput alpha_out and beta_out
        alpha_out = sigma_ * torch.sqrt(self.data_var_scalar / (self.data_var_scalar + sigma_sq_))
        beta_out = (1.0 - (self.data_var_scalar / (self.data_var_scalar + sigma_sq_))) * self.data_mean
        # Compute alpha_skip
        alpha_skip = (self.data_var_scalar / (self.data_var_scalar + sigma_sq_)) / s_
        # Return everything
        return alpha_in, beta_in, alpha_out, beta_out, alpha_skip

    def forward(self, t, x):
        if self.is_particles:
            x = remove_mean(x)
        alpha_in, beta_in, alpha_out, beta_out, alpha_skip = self.get_parameters(t)
        if self.log_snr_dist:
            t_precond = 0.3 * self.sde.log_snr(t) / self.log_snr_dist[1]
            t_precond -= 0.3 * (1. + (self.log_snr_dist[0] / self.log_snr_dist[1]))
        else:
            t_precond = t
        ret = alpha_skip * x + alpha_out * self.base_net(t_precond, (x - beta_in) / alpha_in) + beta_out
        if self.is_particles:
            x = remove_mean(x)
        return ret
