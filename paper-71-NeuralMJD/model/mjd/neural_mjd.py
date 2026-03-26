import math
import torch
import torch.nn as nn

from ..utils import mask_nodes


class NeuralMJD(nn.Module):
    def __init__(
        self,
        model, 
        w_cond_mean_loss = 1.0,
        n_runs = 300,
        steps_per_unit_time = 10,
        jump_diffusion = True,
        s_0_from_avg = False,
        cond_mean_raw_scale=False,
    ) -> None:
        super().__init__()

        self.model = model
        self.w_cond_mean_loss = w_cond_mean_loss
        self.n_runs = n_runs
        self.steps_per_unit_time = steps_per_unit_time
        self.jump_diffusion = jump_diffusion
        self.s_0_from_avg = s_0_from_avg
        self.cond_mean_raw_scale = cond_mean_raw_scale

    @torch.no_grad()
    def mjd_sample(self, mjd_params, batched_data, target, steps_per_unit_time, solver_restart):
        """MJD sampling"""
        mus, sigmas, lambdas_logits, nus, gammas = mjd_params.chunk(5, dim=-1)
        mus, sigmas, lambdas_logits, nus, gammas = mus.squeeze(-1), sigmas.squeeze(-1), lambdas_logits.squeeze(-1), nus.squeeze(-1), gammas.squeeze(-1)

        # Clamp parameters
        bound_mus = 10.0
        bound_variance = 1.0
        bound_lambdas = 1.0
        bound_nus = 0.5

        mus = mus.clamp(-bound_mus, bound_mus)
        sigmas = sigmas.clamp(0, bound_variance)
        lambdas_logits = lambdas_logits.clamp(max=math.log(bound_lambdas))
        lambdas = torch.exp(lambdas_logits).clamp(1e-6, bound_lambdas)
        nus = nus.clamp(-bound_nus, bound_nus)
        gammas = gammas.clamp(0, bound_variance)
        ks = torch.exp(nus + gammas.square() / 2.0) - 1.0

        # compute the conditional mean for each time step
        if self.s_0_from_avg:
            s_0 = batched_data['node_past_dyn_data'][:, :, 0, :].mean(dim=-1).clamp(min=1e-6)
        else:
            s_0 = batched_data['node_past_dyn_data'][:, :, 0, -1].clamp(min=1e-6)

        log_s_pred_mean = torch.log(s_0).unsqueeze(-1) + torch.cumsum(mus, dim=-1)  # [B, N, Future]
        s_pred_mean = torch.exp(log_s_pred_mean)  # [B, N, Future]

        # Sample loop
        n_graph, n_node = s_0.size()
        n_runs = self.n_runs
        num_steps = mus.size(-1)
        log_s_cur = torch.log(s_0).unsqueeze(-1).expand(-1, -1, n_runs)
        log_s_ls = [log_s_cur]
        log_s_out = []
        for i in range(num_steps):
            for j in range(steps_per_unit_time):
                delta_log_s_ = torch.empty([n_graph, n_node, n_runs], dtype=torch.float32, device=mus.device)
                alpha_ = (mus[..., i] - lambdas[..., i] * ks[..., i] - sigmas[..., i].square() / 2.0) / steps_per_unit_time  # [B, N]
                # Antithetic variates: generate half samples, mirror the rest
                n_half = n_runs // 2
                eps_beta = torch.randn([n_graph, n_node, n_half], dtype=torch.float32, device=mus.device)
                eps_beta_full = torch.cat([eps_beta, -eps_beta], dim=-1)  # [B, N, n_runs]
                beta_ = sigmas[..., i].unsqueeze(-1).abs() * eps_beta_full / math.sqrt(steps_per_unit_time)  # [B, N, n_runs]

                k_sample_ = torch.poisson(lambdas[..., i].unsqueeze(-1).expand(-1, -1, n_runs) / steps_per_unit_time)  # [B, N, n_runs]
                # k_sample_ = k_sample_.clip(max=steps_per_unit_time)
                eps_zeta = torch.randn([n_graph, n_node, n_half], dtype=torch.float32, device=mus.device)
                eps_zeta_full = torch.cat([eps_zeta, -eps_zeta], dim=-1)  # [B, N, n_runs]
                zeta_ = k_sample_ * nus[..., i].unsqueeze(-1) + k_sample_.sqrt() * gammas[..., i].unsqueeze(-1).abs() * eps_zeta_full  # [B, N, n_runs]

                delta_log_s_ = alpha_.unsqueeze(-1) + beta_ + zeta_

                log_s_cur = log_s_cur + delta_log_s_

                if j == steps_per_unit_time - 1:
                    log_s_out.append(log_s_cur)

                if i > 0 and j == 0 and solver_restart:
                    log_s_cur = log_s_pred_mean[..., i-1].unsqueeze(-1) + delta_log_s_

                log_s_ls.append(log_s_cur)

        log_s_out = torch.stack(log_s_out, dim=-1)  # [B, N, R, Future]
        s_out = torch.exp(log_s_out)

        if batched_data['data_norm'] == 'minmax':
            s_out_demean = s_out * batched_data['node_norm_coef'].unsqueeze(-1).expand(-1, -1, n_runs, num_steps) + batched_data['node_norm_min'].unsqueeze(-1).expand(-1, -1, n_runs, num_steps)
            s_pred_mean_demean = s_pred_mean * batched_data['node_norm_coef'] + batched_data['node_norm_min']
        else:
            s_out_demean = s_out * batched_data['node_norm_coef'].unsqueeze(-1).expand(-1, -1, n_runs, num_steps)
            s_pred_mean_demean = s_pred_mean * batched_data['node_norm_coef']

        # Winner-take-all selection
        s_out_error = torch.abs(s_out_demean - target.unsqueeze(-2))  # [B, N, R, Future]
        s_out_winner_demean = s_out_demean.gather(dim=2, index=s_out_error.argmin(dim=2, keepdim=True)).squeeze(-2)  # [B, N, Future]

        # Probabilistic-based selection
        log_s_pred_mean_offset = torch.zeros_like(log_s_pred_mean)
        log_s_pred_mean_offset[:, :, 0] = torch.log(s_0)
        log_s_pred_mean_offset[:, :, 1:] = log_s_pred_mean[:, :, :-1]

        a_t_base = log_s_pred_mean_offset + mus - lambdas * ks - sigmas.square() / 2.0                                                  # [B, N, Future]
        a_t_n = lambda n: (a_t_base + n * nus).unsqueeze(-2).expand(-1, -1, n_runs, -1)                                                 # [B, N, R, Future]
        b_t_n = lambda n: (sigmas.square() + n * gammas.square()).clip(min=1e-6).sqrt().unsqueeze(-2).expand(-1, -1, n_runs, -1)        # [B, N, R, Future]

        log_prob_gaussian_n = lambda n: self.log_density_gaussian(log_s_out, a_t_n(n), b_t_n(n))

        # Replace direct probability calculation with stabilized version
        max_n = 5
        log_probs_n = []
        for n in range(max_n+1):
            log_p_n = (- lambdas + torch.log(lambdas) * n).unsqueeze(-2).expand(-1, -1, n_runs, -1) + log_prob_gaussian_n(n) - torch.lgamma(torch.tensor(n+1.0))
            log_probs_n.append(log_p_n)
            
        # Use log-sum-exp stabilization
        log_probs_n = torch.stack(log_probs_n, dim=-1)      # [B, N, R, Future, max_n+1]
        log_probs = torch.logsumexp(log_probs_n, dim=-1)    # [B, N, R, Future]

        s_out_prob_demean = s_out_demean.gather(dim=2, index=log_probs.argmax(dim=2, keepdim=True)).squeeze(-2)  # [B, N, Future]

        return s_out_demean, s_out_winner_demean, s_out_prob_demean


    @torch.no_grad()
    def bs_sample(self, bs_params, batched_data, target, steps_per_unit_time, solver_restart):
        """BS sampling"""
        mus, sigmas = bs_params.chunk(2, dim=-1)
        mus, sigmas = mus.squeeze(-1), sigmas.squeeze(-1)

        # Clamp parameters
        bound_mus = 10.0
        bound_variance = 1.0

        mus = mus.clamp(-bound_mus, bound_mus)
        sigmas = sigmas.clamp(1e-3, bound_variance)

        # compute the conditional mean for each time step
        if self.s_0_from_avg:
            s_0 = batched_data['node_past_dyn_data'][:, :, 0, :].mean(dim=-1).clamp(min=1e-6)
        else:
            s_0 = batched_data['node_past_dyn_data'][:, :, 0, -1].clamp(min=1e-6)

        log_s_pred_mean = torch.log(s_0).unsqueeze(-1) + torch.cumsum(mus, dim=-1)  # [B, N, Future]
        s_pred_mean = torch.exp(log_s_pred_mean)  # [B, N, Future]

        # Sample loop
        n_graph, n_node = s_0.size()
        n_runs = self.n_runs
        num_steps = mus.size(-1)
        log_s_cur = torch.log(s_0).unsqueeze(-1).expand(-1, -1, n_runs)
        log_s_ls = [log_s_cur]
        log_s_out = []
        for i in range(num_steps):
            for j in range(steps_per_unit_time):
                delta_log_s_ = torch.empty([n_graph, n_node, n_runs], dtype=torch.float32, device=mus.device)
                alpha_ = (mus[..., i] - sigmas[..., i].square() / 2.0) / steps_per_unit_time  # [B, N]
                beta_ = sigmas[..., i].unsqueeze(-1).abs() * torch.randn_like(delta_log_s_) / math.sqrt(steps_per_unit_time)  # [B, N, n_runs]

                delta_log_s_ = alpha_.unsqueeze(-1) + beta_

                log_s_cur = log_s_cur + delta_log_s_

                if j == steps_per_unit_time - 1:
                    log_s_out.append(log_s_cur)

                if i > 0 and j == 0 and solver_restart:
                    log_s_cur = log_s_pred_mean[..., i-1].unsqueeze(-1) + delta_log_s_

                log_s_ls.append(log_s_cur)

        log_s_out = torch.stack(log_s_out, dim=-1)  # [B, N, R, Future]
        s_out = torch.exp(log_s_out)

        if batched_data['data_norm'] == 'minmax':
            s_out_demean = s_out * batched_data['node_norm_coef'].unsqueeze(-1).expand(-1, -1, n_runs, num_steps) + batched_data['node_norm_min'].unsqueeze(-1).expand(-1, -1, n_runs, num_steps)
            s_pred_mean_demean = s_pred_mean * batched_data['node_norm_coef'] + batched_data['node_norm_min']
        else:
            s_out_demean = s_out * batched_data['node_norm_coef'].unsqueeze(-1).expand(-1, -1, n_runs, num_steps)
            s_pred_mean_demean = s_pred_mean * batched_data['node_norm_coef']

        # Winner-take-all selection
        s_out_error = torch.abs(s_out_demean - target.unsqueeze(-2))  # [B, N, R, Future]
        s_out_winner_demean = s_out_demean.gather(dim=2, index=s_out_error.argmin(dim=2, keepdim=True)).squeeze(-2)  # [B, N, Future]

        # Probabilistic-based selection
        log_s_pred_mean_offset = torch.zeros_like(log_s_pred_mean)
        log_s_pred_mean_offset[:, :, 0] = torch.log(s_0)
        log_s_pred_mean_offset[:, :, 1:] = log_s_pred_mean[:, :, :-1]

        a_t_base = log_s_pred_mean_offset + mus - sigmas.square() / 2.0         # [B, N, Future]
        a_t_base = a_t_base.unsqueeze(-2).expand(-1, -1, n_runs, -1)            # [B, N, R, Future]
        b_t_base = sigmas.clip(min=1e-6)                                        # [B, N, Future]
        b_t_base = b_t_base.unsqueeze(-2).expand(-1, -1, n_runs, -1)            # [B, N, R, Future] 

        log_prob = self.log_density_gaussian(log_s_out, a_t_base, b_t_base)     # [B, N, R, Future]

        s_out_prob_demean = s_out_demean.gather(dim=2, index=log_prob.argmax(dim=2, keepdim=True)).squeeze(-2)  # [B, N, Future]

        return s_out_demean, s_out_winner_demean, s_out_prob_demean


    def log_density_gaussian(self, x, mu, sigma):
        # Add epsilon to sigma and clamp to prevent log(0)
        sigma_safe = sigma.clamp(min=1e-6) + 1e-8
        return -torch.log(sigma_safe) - (x - mu).square() / (2.0 * sigma_safe.square()) - 0.5 * torch.log(2.0 * torch.tensor(torch.pi, device=sigma_safe.device))
        
    def mjd_loss(self, mjd_params, batched_data, target):
        """
        @param mjd_params: [B, N, Future, 5], predicted params of the MJD process
        @param batched_data: dict, batched data
        @param target: [B, N, Future], target data
        """

        """Compute intermediate variables"""
        mus, sigmas, lambdas_logits, nus, gammas = mjd_params.chunk(5, dim=-1)
        mus, sigmas, lambdas_logits, nus, gammas = mus.squeeze(-1), sigmas.squeeze(-1), lambdas_logits.squeeze(-1), nus.squeeze(-1), gammas.squeeze(-1)

        # Clamp parameters
        bound_mus = 10.0
        bound_variance = 1.0
        bound_lambdas = 1.0
        bound_nus = 0.5

        mus = mus.clamp(-bound_mus, bound_mus)
        sigmas = sigmas.clamp(0, bound_variance)
        lambdas_logits = lambdas_logits.clamp(max=math.log(bound_lambdas))
        lambdas = torch.exp(lambdas_logits).clamp(1e-6, bound_lambdas)
        nus = nus.clamp(-bound_nus, bound_nus)
        gammas = gammas.clamp(0, bound_variance)
        ks = torch.exp(nus + gammas.square() / 2.0) - 1.0

        # compute the conditional mean for each time step
        if self.s_0_from_avg:
            s_0 = batched_data['node_past_dyn_data'][:, :, 0, :].mean(dim=-1).clamp(min=1e-6)
        else:
            s_0 = batched_data['node_past_dyn_data'][:, :, 0, -1].clamp(min=1e-6)
        log_s_pred_mean = torch.log(s_0).unsqueeze(-1) + torch.cumsum(mus, dim=-1)  # [B, N, Future]
        s_pred_mean = torch.exp(log_s_pred_mean)  # [B, N, Future]

        # compute the denormalized conditional mean
        if batched_data['data_norm'] == 'minmax':
            # s_0_denorm = s_0 * batched_data['node_norm_coef'].squeeze(-1) + batched_data['node_norm_min'].squeeze(-1)
            target_norm = (target - batched_data['node_norm_min']) / batched_data['node_norm_coef']
            s_pred_mean_demean = s_pred_mean * batched_data['node_norm_coef'] + batched_data['node_norm_min']
        else:
            target_norm = target / batched_data['node_norm_coef']
            s_pred_mean_demean = s_pred_mean * batched_data['node_norm_coef']

        s_pred_mean_out = s_pred_mean_demean.detach()

        """Loss calculation"""
        # Conditional mean loss
        huber_delta = batched_data['huber_delta']
        target_data = target if self.cond_mean_raw_scale else target_norm
        cond_mean_pred = s_pred_mean_demean if self.cond_mean_raw_scale else s_pred_mean
        if huber_delta is None or huber_delta == 'inf':
            cond_mean_loss = torch.nn.functional.mse_loss(cond_mean_pred, target_data, reduction='none')  # [B, Future, 1]
        elif isinstance(huber_delta, (float, int)):
            cond_mean_loss = torch.nn.functional.huber_loss(cond_mean_pred, target_data, reduction='none', delta=huber_delta)  # [B, Future, 1]
        else:
            raise ValueError('huber_delta must be either None, "inf", or a float/int number. Its current value is {}'.format(huber_delta))
        cond_mean_loss = mask_nodes(cond_mean_loss, batched_data['node_mask'])
        cond_mean_loss = cond_mean_loss * self.w_cond_mean_loss

        # log likelihood loss
        log_s_pred_mean_offset = torch.zeros_like(log_s_pred_mean)
        log_s_pred_mean_offset[:, :, 0] = torch.log(s_0)
        log_s_pred_mean_offset[:, :, 1:] = log_s_pred_mean[:, :, :-1]

        a_t_base = log_s_pred_mean_offset + mus - lambdas * ks - sigmas.square() / 2.0      # [B, N, Future]
        a_t_n = lambda n: a_t_base + n * nus                                                # [B, N, Future]
        b_t_n = lambda n: (sigmas.square() + n * gammas.square()).clip(min=1e-6).sqrt()     # [B, N, Future]

        log_target = torch.log(torch.clip(target_norm, min=1e-6))
        log_prob_gaussian_n = lambda n: self.log_density_gaussian(log_target, a_t_n(n), b_t_n(n))

        # Replace direct probability calculation with stabilized version
        max_n = 5
        log_probs_n = []
        for n in range(max_n+1):
            log_p_n = - lambdas + torch.log(lambdas) * n + log_prob_gaussian_n(n) - torch.lgamma(torch.tensor(n+1.0))
            log_probs_n.append(log_p_n)
            
        # Use log-sum-exp stabilization
        log_probs_n = torch.stack(log_probs_n, dim=-1)      # [B, N, Future, max_n+1]
        log_probs = torch.logsumexp(log_probs_n, dim=-1)    # [B, N, Future]

        likelihood_loss = -log_probs
        likelihood_loss = mask_nodes(likelihood_loss, batched_data['node_mask'])
        
        return cond_mean_loss, likelihood_loss, s_pred_mean_out


    def bs_loss(self, bs_params, batched_data, target):
        """
        @param bs_params: [B, N, Future, 2], predicted params of the MJD process
        @param batched_data: dict, batched data
        @param target: [B, N, Future], target data
        """

        """Compute intermediate variables"""
        mus, sigmas = bs_params.chunk(2, dim=-1)
        mus, sigmas = mus.squeeze(-1), sigmas.squeeze(-1)

        # Clamp parameters
        bound_mus = 10.0
        bound_variance = 1.0

        mus = mus.clamp(-bound_mus, bound_mus)
        sigmas = sigmas.clamp(1e-3, bound_variance)

        # compute the conditional mean for each time step
        if self.s_0_from_avg:
            s_0 = batched_data['node_past_dyn_data'][:, :, 0, :].mean(dim=-1).clamp(min=1e-6)
        else:
            s_0 = batched_data['node_past_dyn_data'][:, :, 0, -1].clamp(min=1e-6)
        log_s_pred_mean = torch.log(s_0).unsqueeze(-1) + torch.cumsum(mus, dim=-1)  # [B, N, Future]
        s_pred_mean = torch.exp(log_s_pred_mean)  # [B, N, Future]

        # compute the denormalized conditional mean
        if batched_data['data_norm'] == 'minmax':
            # s_0_denorm = s_0 * batched_data['node_norm_coef'].squeeze(-1) + batched_data['node_norm_min'].squeeze(-1)
            target_norm = (target - batched_data['node_norm_min']) / batched_data['node_norm_coef']
            s_pred_mean_demean = s_pred_mean * batched_data['node_norm_coef'] + batched_data['node_norm_min']
        else:
            target_norm = target / batched_data['node_norm_coef']
            s_pred_mean_demean = s_pred_mean * batched_data['node_norm_coef']

        s_pred_mean_out = s_pred_mean_demean.detach()

        """Loss calculation"""
        # Conditional mean loss
        huber_delta = batched_data['huber_delta']
        target_data = target if self.cond_mean_raw_scale else target_norm
        cond_mean_pred = s_pred_mean_demean if self.cond_mean_raw_scale else s_pred_mean
        if huber_delta is None or huber_delta == 'inf':
            cond_mean_loss = torch.nn.functional.mse_loss(cond_mean_pred, target_data, reduction='none')  # [B, Future, 1]
        elif isinstance(huber_delta, (float, int)):
            cond_mean_loss = torch.nn.functional.huber_loss(cond_mean_pred, target_data, reduction='none', delta=huber_delta)  # [B, Future, 1]
        else:
            raise ValueError('huber_delta must be either None, "inf", or a float/int number. Its current value is {}'.format(huber_delta))
        cond_mean_loss = mask_nodes(cond_mean_loss, batched_data['node_mask'])
        cond_mean_loss = cond_mean_loss * self.w_cond_mean_loss

        # log likelihood loss
        log_s_pred_mean_offset = torch.zeros_like(log_s_pred_mean)
        log_s_pred_mean_offset[:, :, 0] = torch.log(s_0)
        log_s_pred_mean_offset[:, :, 1:] = log_s_pred_mean[:, :, :-1]

        a_t_base = log_s_pred_mean_offset + mus - sigmas.square() / 2.0         # [B, N, Future]
        b_t_base = sigmas.clip(min=1e-6)                                        # [B, N, Future]

        log_target = torch.log(torch.clip(target_norm, min=1e-6))
        log_prob = self.log_density_gaussian(log_target, a_t_base, b_t_base)    # [B, N, Future]

        likelihood_loss = -log_prob
        likelihood_loss = mask_nodes(likelihood_loss, batched_data['node_mask'])
        
        return cond_mean_loss, likelihood_loss, s_pred_mean_out


    def forward(self, batched_data, target=None, flag_sample=False):
        """
        @param batched_data: dict, batched data
        @param target: [B, N, Future], target data
        @param flag_sample: bool, whether to sample
        @return: loss
        """
        model_params = self.model(batched_data)

        if self.jump_diffusion:
            cond_mean_loss, likelihood_loss, cond_mean = self.mjd_loss(model_params, batched_data, target)
        else:
            cond_mean_loss, likelihood_loss, cond_mean = self.bs_loss(model_params, batched_data, target)

        if flag_sample:
            if self.jump_diffusion:
                samples = self.mjd_sample(model_params, batched_data, target, self.steps_per_unit_time, solver_restart=True)
            else:
                samples = self.bs_sample(model_params, batched_data, target, self.steps_per_unit_time, solver_restart=True)
            return cond_mean_loss, likelihood_loss, samples
        else:
            return cond_mean_loss, likelihood_loss, cond_mean
