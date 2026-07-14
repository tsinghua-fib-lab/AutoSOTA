import torch
from torch.autograd import grad
from diffusers.utils.torch_utils import randn_tensor

from .ddim import DDIMSampler
from search.utils import rescale_grad


def expand_like(v, x):
    while v.ndim < x.ndim:
        v = v.unsqueeze(-1)
    return v


class DoobHGuidanceSampler(DDIMSampler):
    """
    Doob h-transform guidance (match SD implementation style):
      1) simulate x_{t-1}^{(m)} = mu_theta(x_t) + sigma_sim * z_m
      2) compute reward r(x0^{(m)}) -> weights w_m = softmax((r-mean)/tau)
      3) v_eff = sum_m w_m * (z_m / sigma_sim)
      4) doob_grad = d mu_theta(x_t) / d x_t  vjp with v_eff
      5) eps <- eps - gamma * sigma_s * doob_grad
    """

    def __init__(self, args, model=None):
        super().__init__(args=args, model=model)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.args.seed)

    @property
    def M(self):
        return int(getattr(self.args, "doob_M", 8))

    @property
    def gamma(self):
        return float(getattr(self.args, "doob_gamma", 1.0))

    def tau_t(self, i: int):
        if hasattr(self.args, "doob_taus"):
            taus = self.args.doob_taus
            if taus is not None:
                if torch.is_tensor(taus):
                    return float(taus[i].item())
                return float(taus[i])
        return float(getattr(self.args, "doob_tau", 0.6))

    def in_window(self, i: int):
        t_star_idx = getattr(self.args, "doob_t_star_idx", None)
        if t_star_idx is None:
            t_star_idx = self.args.inference_steps - 1
        t_star_idx = int(t_star_idx)
        return (i > 0) and (i <= t_star_idx)

    @property
    def antithetic(self):
        return bool(getattr(self.args, "doob_antithetic_sampling", False))

    @property
    def reward_threshold(self):
        return float(getattr(self.args, "doob_reward_threshold", 0.001))

    @property
    def use_reward_gate(self):
        return bool(getattr(self.args, "doob_use_reward_gate", False))

    @property
    def greedy_step(self):
        if hasattr(self, "_doob_greedy_step"):
            return self._doob_greedy_step
        return getattr(self.args, "doob_greedy_step", None)

    @property
    def store_best(self):
        if hasattr(self, "_doob_store_best"):
            return self._doob_store_best
        return bool(getattr(self.args, "doob_store_best", False))

    @property
    def save_rewards(self):
        if hasattr(self, "_doob_save_rewards"):
            return self._doob_save_rewards
        return bool(getattr(self.args, "doob_save_rewards", False))

    def eps_from_x_x0(self, x, x0_hat, alpha, sigma):
        return (x - alpha * x0_hat) / (sigma + 1e-12)

    def reward(self, actions: torch.Tensor, states: torch.Tensor | None = None, M: int | None = None) -> torch.Tensor:
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions_t.dim() > 2:
            actions_t = actions_t.reshape(-1, actions_t.shape[-1])

        states_t = states
        if states_t is None:
            states_t = getattr(self.model, "condition", None)

        repeat_factor = None
        if states_t is not None:
            states_t = torch.as_tensor(states_t, device=self.device, dtype=torch.float32)
            if states_t.dim() > 2:
                states_t = states_t.reshape(-1, states_t.shape[-1])
            if M is not None and actions_t.shape[0] == states_t.shape[0] * M:
                repeat_factor = M

        return self.get_reward(actions_t, states=states_t, repeat_factor=repeat_factor)

    @torch.enable_grad()
    def estimate_doob_grad(self, x, vec_s, vec_t, i, return_sim_info=True, compute_grad=True, **kwargs):
        """
        Returns:
          doob_grad = d mu_theta(x) / d x  vjp with v_eff
          shape: same as x
        """
        M = self.M
        tau = self.tau_t(i)
        eta_val = float(getattr(self.args, "eta", 0.1))
        alpha_s = self.noise_schedule.marginal_alpha(vec_s)
        sigma_s = self.noise_schedule.marginal_std(vec_s)
        alpha_t = self.noise_schedule.marginal_alpha(vec_t)
        sigma_t = self.noise_schedule.marginal_std(vec_t)
        alpha_bar_s = alpha_s**2
        alpha_bar_t = alpha_t**2
        sigma2_sim = (
            eta_val**2
            * (1.0 - alpha_bar_t)
            / torch.clamp(1.0 - alpha_bar_s, min=1e-12)
            * (1.0 - alpha_bar_s / torch.clamp(alpha_bar_t, min=1e-12))
        )
        sigma2_sim = torch.clamp(sigma2_sim, min=0.0)
        sigma_sim = torch.sqrt(sigma2_sim + 1e-12)
        x_leaf = x.clone().detach().requires_grad_(compute_grad)

        _, x0_dict = self.solver_update(x_leaf, vec_s, vec_t)
        x0_hat = x0_dict["model_s"]

        eps_pred = self.eps_from_x_x0(
            x_leaf, x0_hat, expand_like(alpha_s, x_leaf), expand_like(sigma_s, x_leaf)
        )
        eps_s = eps_pred
        alpha_s_sqrt = alpha_s
        alpha_t_sqrt = alpha_t
        one_minus_alpha_bar_s_sqrt = sigma_s
        safe_term = torch.sqrt(torch.clamp(sigma_t**2 - sigma2_sim, min=1e-12)).to(x.dtype)

        mu_theta = (
            (expand_like(alpha_t_sqrt / (alpha_s_sqrt + 1e-12), x_leaf) * x_leaf)
            + (
                - expand_like(alpha_t_sqrt * one_minus_alpha_bar_s_sqrt / (alpha_s_sqrt + 1e-12), x_leaf)
                + expand_like(safe_term, x_leaf)
            ) * eps_pred
        )
        sigma_sim_x = expand_like(sigma_sim, x_leaf)

        z = randn_tensor((M, *x.shape), generator=self.rng, device=self.device, dtype=x.dtype)
        if self.antithetic and M >= 2:
            half = M // 2
            z[:half] = -z[half:half * 2]

        mu_det = mu_theta.detach().unsqueeze(0)
        sigma_det = sigma_sim_x.detach().unsqueeze(0)
        x_sim = mu_det + sigma_det * z

        eps_det = eps_s.detach().unsqueeze(0)
        alpha_t_x = expand_like(alpha_t, x_leaf).detach().unsqueeze(0)
        sigma_t_x = expand_like(sigma_t, x_leaf).detach().unsqueeze(0)
        x0_sim = (x_sim - sigma_t_x * eps_det) / (alpha_t_x + 1e-12)

        K = x.shape[0]
        states_KS = self.model.condition

        flat_x0 = x0_sim.reshape(-1, x0_sim.shape[-1])
        r = self.reward(flat_x0, states=states_KS, M=M).reshape(M, K)

        r_mean = r.mean(dim=0, keepdim=True)
        r_std = r.std(dim=0, keepdim=True, unbiased=False)
        centered = r - r_mean
        denom = torch.clamp(r_std, min=1e-12) * max(tau, 1e-12)
        w = torch.softmax(centered / denom, dim=0)
        sigma_eps = (sigma_sim + 1e-8).view(1, -1, 1)
        v_eff = (w.view(M, -1, 1) * (z / sigma_eps)).sum(dim=0)

        if self.use_reward_gate:
            gate = (r_mean.view(-1) >= self.reward_threshold).to(x.dtype)
            v_eff = v_eff * gate.view(-1, 1)

        doob_grad = None
        if compute_grad:
            doob_grad = grad(
                outputs=mu_theta,
                inputs=x_leaf,
                grad_outputs=v_eff,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        if doob_grad is not None and hasattr(self.args, "doob_clip_scale"):
            doob_grad = rescale_grad(doob_grad, clip_scale=self.args.doob_clip_scale, **kwargs)

        sim_info = None
        if return_sim_info:
            best_rewards, best_idx = r.max(dim=0)
            k_idx = torch.arange(K, device=x.device)
            best_x_sim = x_sim[best_idx, k_idx].detach()
            best_x0_sim = x0_sim[best_idx, k_idx].detach()
            sim_info = {
                "rewards": r.detach(),
                "best_rewards": best_rewards.detach(),
                "best_x_sim": best_x_sim,
                "best_x0_sim": best_x0_sim,
            }

        if doob_grad is None:
            return None, sim_info
        return doob_grad.detach(), sim_info

    def guide_step(self, x, i, **kwargs):
        vec_s = self.timesteps[i].expand(x.shape[0])
        vec_t = self.timesteps[i + 1].expand(x.shape[0])

        alpha_s = self.noise_schedule.marginal_alpha(vec_s)
        sigma_s = self.noise_schedule.marginal_std(vec_s)
        alpha_t = self.noise_schedule.marginal_alpha(vec_t)
        sigma_t = self.noise_schedule.marginal_std(vec_t)

        with torch.no_grad():
            _, x0_dict = self.solver_update(x, vec_s, vec_t)
            x0_hat = x0_dict["model_s"]
            eps_base = self.eps_from_x_x0(
                x, x0_hat, expand_like(alpha_s, x), expand_like(sigma_s, x)
            )

        if self.in_window(i):
            greedy_mode = self.greedy_step is not None and i >= int(self.greedy_step)
            need_sim_info = greedy_mode or self.store_best or self.save_rewards
            doob_grad, sim_info = self.estimate_doob_grad(
                x,
                vec_s,
                vec_t,
                i,
                return_sim_info=True,
                compute_grad=not greedy_mode,
                **kwargs,
            )

            if sim_info is not None:
                if self.save_rewards and self._reward_history is not None:
                    self._reward_history.append(sim_info["rewards"].detach().transpose(0, 1).cpu())

                if self.store_best:
                    if self._best_rewards is None:
                        self._best_rewards = torch.full(
                            (x.shape[0],), -float("inf"), device=x.device, dtype=x.dtype
                        )
                        self._best_actions = torch.zeros_like(x)
                    better = sim_info["best_rewards"] > self._best_rewards
                    if torch.any(better):
                        self._best_rewards[better] = sim_info["best_rewards"][better]
                        self._best_actions[better] = sim_info["best_x0_sim"][better]

                if greedy_mode:
                    return sim_info["best_x_sim"]

            if doob_grad is None:
                eps_doob = eps_base
            else:
                eps_doob = eps_base - self.gamma * expand_like(sigma_s, x) * doob_grad
        else:
            eps_doob = eps_base

        x0_doob = (x - expand_like(sigma_s, x) * eps_doob) / (expand_like(alpha_s, x) + 1e-12)
        eta_val = float(getattr(self.args, "eta", 0.0))
        sigma2 = (
            eta_val**2
            * (sigma_t**2)
            / torch.clamp(sigma_s**2, min=1e-12)
            * (1.0 - (alpha_s**2) / torch.clamp(alpha_t**2, min=1e-12))
        )
        sigma2 = torch.clamp(sigma2, min=0.0)
        sigma = torch.sqrt(sigma2 + 1e-12)
        eps_coeff = torch.sqrt(torch.clamp(sigma_t**2 - sigma2, min=1e-12))
        mu_doob = expand_like(alpha_t, x) * x0_doob + expand_like(eps_coeff, x) * eps_doob
        if eta_val > 0:
            noise = randn_tensor(x.shape, generator=self.rng, device=self.device, dtype=x.dtype)
        else:
            noise = torch.zeros_like(x)
        x_next = mu_doob + expand_like(sigma, x) * noise
        return x_next

    def __call__(self, states, **kwargs):
        self._doob_store_best = bool(kwargs.pop("doob_store_best", getattr(self.args, "doob_store_best", False)))
        self._doob_save_rewards = bool(kwargs.pop("doob_save_rewards", getattr(self.args, "doob_save_rewards", False)))
        self._doob_greedy_step = kwargs.pop("doob_greedy_step", getattr(self.args, "doob_greedy_step", None))

        if self._doob_greedy_step is not None:
            self._doob_greedy_step = int(self._doob_greedy_step)
            if self._doob_greedy_step < 0 or self._doob_greedy_step > self.args.inference_steps:
                raise ValueError(
                    f"doob_greedy_step must be between 0 and {self.args.inference_steps} (inclusive), "
                    f"got {self._doob_greedy_step}."
                )

        self._reward_history = [] if self._doob_save_rewards else None
        self._best_rewards = None
        self._best_actions = None
        self.last_reward_history = None
        self.last_best_rewards = None

        self.model.eval()
        multiple_input = True
        states = torch.FloatTensor(states).to(self.device)
        if states.dim == 1:
            states = states.unsqueeze(0)
            multiple_input = False
        states = states.repeat_interleave(self.args.per_sample_batch_size, dim=0)
        num_states = states.shape[0]
        self.model.condition = states
        x = randn_tensor((states.shape[0], self.model.output_dim), generator=self.rng, device=self.device, dtype=torch.float32)
        for i in range(self.args.inference_steps):
            x = self.guide_step(x, i)

        if self.store_best and self._best_rewards is not None:
            final_rewards = self.get_reward(x, states=self.model.condition)
            better = final_rewards > self._best_rewards
            if torch.any(better):
                self._best_rewards[better] = final_rewards[better]
                self._best_actions[better] = x[better]
            x = self._best_actions
            self.last_best_rewards = self._best_rewards.detach().cpu()

        if self._reward_history is not None and len(self._reward_history) > 0:
            self.last_reward_history = torch.stack(self._reward_history, dim=0)

        return self.tensor_to_obj(x, num_states, multiple_input)

if __name__ == "__main__":
    pass
