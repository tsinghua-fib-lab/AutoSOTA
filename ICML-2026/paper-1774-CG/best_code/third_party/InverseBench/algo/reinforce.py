import copy
import torch
from calibrated_guidance.guidance import build_reinforce_mean_fn, build_reparam_mean_fn
from calibrated_guidance.inference import edm_sde, flow_matching
from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput
from utils.diffusion import DiffusionSampler
from algo.base import Algo
from utils.scheduler import Scheduler
import os
import matplotlib.pyplot as plt
import wandb


class InverseBenchModel(DiffusionPosterior):
    def __init__(self, scheduler, model, inner_loop_steps, output_path):
        self.scheduler = scheduler
        self.model = model
        self.output_path = output_path
        self.desired_number_of_steps_inner_loop = inner_loop_steps

    def sample(self, xt, t, num_samples_per_step):
        B, C, H, W = xt.shape
        assert B == 1, "Only batch size 1 is supported in InverseBenchModel"

        os.makedirs(self.output_path, exist_ok=True)
        if os.path.exists(os.path.join(self.output_path, f'time_{t:.4f}.pt')):
            result = torch.load(os.path.join(self.output_path, f'time_{t:.4f}.pt'))
        else:
            xt = xt.repeat(num_samples_per_step, 1, 1, 1)
            
            this_scheduler = self.scheduler
            use_normal_scheduler = True
            if self.desired_number_of_steps_inner_loop is not None:
                start_step = int((1 - t) * self.scheduler.num_steps)
                if start_step + self.desired_number_of_steps_inner_loop <= self.scheduler.num_steps:
                    this_scheduler = Scheduler.get_evenly_spaced_subset_scheduler(self.scheduler, self.desired_number_of_steps_inner_loop, start_step)
                    use_normal_scheduler = False

            diffusion_sampler = DiffusionSampler(this_scheduler)
            result = diffusion_sampler._euler(self.model, xt, SDE=True, t_begin=(1 - t), skip_filtering=(not use_normal_scheduler))
            result = result.reshape(1, num_samples_per_step, C, H, W)
            torch.save(result, os.path.join(self.output_path, f'time_{t:.4f}.pt'))

        output = SampleOutput(result)
        return output


def plot_torch_image(tensor, filename):
    plt.imshow(tensor.reshape(64, 64).cpu().numpy(), cmap='gray')
    plt.savefig(filename)
    plt.close()


class CalibratedGuidanceReinforce(Algo):
    def __init__(self,
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 sde=True,
                 reinf_k=128,
                 desired_number_of_steps_inner_loop=50,
                 mf_net=None,
                 mf_sibling_renoise_t=None,
                 mf_sigma_min=0.002,
                 reparam=False,
                 use_clamp=True,
                 save_intermediates=False,
                 stop_at_sigma=None,
                 last_hop=False,
                 particle_pool_size=None,
                 pool_no_noise=False,
                 pool_analytic=False):
        super().__init__(net, forward_op)
        self.diffusion_scheduler_config = dict(diffusion_scheduler_config)
        self.scheduler = Scheduler(**self.diffusion_scheduler_config)
        self.last_hop = last_hop
        self.stop_at_sigma = stop_at_sigma
        self.particle_pool_size = int(particle_pool_size) if particle_pool_size else None
        self.pool_no_noise = bool(pool_no_noise)
        self.pool_analytic = bool(pool_analytic)
        self.sde = sde
        self.REINF_K = reinf_k
        self.desired_number_of_steps_inner_loop = desired_number_of_steps_inner_loop
        self.mf_net = mf_net
        self.mf_sibling_renoise_t = mf_sibling_renoise_t
        self.mf_sigma_min = mf_sigma_min
        self.reparam = reparam
        self.use_clamp = use_clamp
        self.save_intermediates = save_intermediates
        self.min_log_likelihoods = []
        self.mean_log_likelihoods = []
        self.max_log_likelihoods = []

    def create_summary_plots(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        plt.plot(self.min_log_likelihoods, label='min')
        plt.plot(self.mean_log_likelihoods, label='mean')
        plt.plot(self.max_log_likelihoods, label='max')
        plt.xlabel('Diffusion Step')
        plt.ylabel('Log Likelihood')
        plt.yscale('symlog')
        plt.legend()
        plt.savefig(os.path.join(output_path, 'summary_log_likelihoods.png'))
        plt.close()

        print(f"At the end of generation at {output_path}, log likelihoods - min: {self.min_log_likelihoods[-1]}, mean: {self.mean_log_likelihoods[-1]}, max: {self.max_log_likelihoods[-1]}")

    def create_plot_samples(self, output_path):
        def plot_samples(xt, x0_samples, log_likelihoods, t):
            B, K, C, H, W = x0_samples.samples.shape
            assert B == 1, "Only batch size 1 is supported in plotting samples"
            self.min_log_likelihoods.append(log_likelihoods.min().item())
            self.mean_log_likelihoods.append(log_likelihoods.mean().item())
            self.max_log_likelihoods.append(log_likelihoods.max().item())

            log_weights = x0_samples.log_weights if x0_samples.log_weights is not None else torch.zeros_like(log_likelihoods)
            weights = torch.softmax(log_likelihoods + log_weights, dim=1)
            x0_hat = (weights[..., None, None, None] * x0_samples.samples).sum(dim=1).detach().cpu()
            self.x0_traj.append(x0_hat)
            self.t_traj.append(float(t))

            if self.save_intermediates:
                self.xt_traj.append(xt.detach().cpu())
                self.x0_samples_traj.append(x0_samples.samples.detach().cpu())
                self.log_lik_traj.append(log_likelihoods.detach().cpu())

            # Log to wandb
            wandb.log({
                'min_log_likelihood': log_likelihoods.min().item(),
                'mean_log_likelihood': log_likelihoods.mean().item(),
                'max_log_likelihood': log_likelihoods.max().item(),
                'diffusion_step': len(self.min_log_likelihoods),
                't': t
            })

        return plot_samples

    def get_log_likelihood_fn(self):
        chunk = int(os.environ.get("LOSS_CHUNK_SIZE", "0") or "0")

        def _ll(x0_samples):
            x = x0_samples.reshape(-1, 1, 64, 64)
            if chunk <= 0 or x.shape[0] <= chunk:
                return -self.forward_op.loss(x, self.observation).reshape(1, -1)
            parts = []
            for i in range(0, x.shape[0], chunk):
                parts.append(-self.forward_op.loss(x[i:i + chunk], self.observation).reshape(-1))
            return torch.cat(parts, dim=0).reshape(1, -1)

        return _ll

    def _inference_particle_pool(self, observation, num_samples=1, gen_idx=None,
                                  guidance_scale=1.0, sample_idx=0):
        """
        Population-SMC variant: maintain P=particle_pool_size independent SDE
        trajectories. Each outer step, pick K=REINF_K random particles, run
        the MeanFlow on those K, REINFORCE-weight them into a single guided
        x0 mean, and apply the deterministic EDM-SDE update with that shared
        mean to *all* P particles. Trajectories start very different and
        converge as guidance pulls them toward the same x0.
        """
        if self.mf_net is None:
            raise NotImplementedError("particle_pool requires mf_net")

        output_path = f'step_samples/gen_idx_{gen_idx}_{sample_idx}'
        os.makedirs(output_path, exist_ok=True)
        self.observation = observation
        self.x0_traj = []
        self.t_traj = []

        self.wandb_run = wandb.init(
            project='reinforce_log_likelihoods',
            name=f'gen_{gen_idx}_sample_{sample_idx}_pool',
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )

        from algo.meanflow_posterior import MeanFlowDiffusionPosterior
        posterior = MeanFlowDiffusionPosterior(
            self.scheduler, self.mf_net, output_path,
            sigma_min=self.mf_sigma_min,
            sibling_renoise_t=self.mf_sibling_renoise_t,
            enable_grad=False,
        )

        device = "cuda"
        P = self.particle_pool_size
        K = self.REINF_K

        x_t = DiffusionSampler(self.scheduler).get_start(
            torch.ones((P, 1, 64, 64), device=device)
        )

        sigma_steps = torch.as_tensor(self.scheduler.sigma_steps, device=device).float()
        factor_steps = torch.as_tensor(self.scheduler.factor_steps, device=device).float()
        scaling_factor = torch.as_tensor(self.scheduler.scaling_factor, device=device).float()
        scaling_steps = torch.as_tensor(self.scheduler.scaling_steps, device=device).float()

        stop_steps = self.scheduler.num_steps
        if self.stop_at_sigma is not None:
            below = (sigma_steps <= float(self.stop_at_sigma)).nonzero(as_tuple=False)
            if below.numel() > 0:
                stop_steps = max(int(below[0].item()), 1)
            print(f"[pool] running {stop_steps}/{self.scheduler.num_steps} SDE steps "
                  f"on P={P} particles (subset K={K})")
        else:
            print(f"[pool] running {stop_steps} SDE steps on P={P} particles (subset K={K})")

        log_lik_fn = self.get_log_likelihood_fn()

        def reinforce_step(x_t_full, sigma, scaling):
            """Pick K particles, MF→x0, REINFORCE-weighted mean."""
            idx = torch.randperm(P, device=device)[:K]
            x_t_sel = x_t_full[idx]
            tau = sigma / (1.0 + sigma)
            z_tau = x_t_sel / (scaling * (1.0 + sigma))
            with torch.no_grad():
                x0_cand = posterior._mf_one_step(z_tau, tau)            # [K,1,H,W]
            log_lik = log_lik_fn(x0_cand).flatten()                      # [K]
            w = torch.softmax(guidance_scale * log_lik, dim=0)
            mean = (w[:, None, None, None] * x0_cand).sum(dim=0, keepdim=True)
            if self.use_clamp:
                mean = mean.clamp(-1.0, 1.0)
            return mean, log_lik

        for i in range(stop_steps):
            sigma = float(sigma_steps[i])
            scaling = float(scaling_steps[i])
            s_factor = float(scaling_factor[i])
            factor = float(factor_steps[i])

            mean, log_lik = reinforce_step(x_t, sigma, scaling)

            self.x0_traj.append(mean.detach().cpu())
            self.t_traj.append(1.0 - i / self.scheduler.num_steps)
            wandb.log({
                'mean_log_likelihood': log_lik.mean().item(),
                'min_log_likelihood': log_lik.min().item(),
                'max_log_likelihood': log_lik.max().item(),
                'pool_x_t_std': x_t.std(dim=0).mean().item(),
                'diffusion_step': i,
                't': 1.0 - i / self.scheduler.num_steps,
            })

            score = (mean - x_t / scaling) / (sigma ** 2) / scaling
            eps = torch.randn_like(x_t)
            noise_term = (factor ** 0.5) * eps if i < stop_steps - 1 else 0
            if self.pool_no_noise:
                noise_term = 0
            x_t = x_t * s_factor + factor * score + noise_term

        if self.last_hop:
            sigma = float(sigma_steps[stop_steps]) if stop_steps < len(sigma_steps) else float(sigma_steps[-1])
            scaling = float(scaling_steps[stop_steps]) if stop_steps < len(scaling_steps) else float(scaling_steps[-1])
            result, _ = reinforce_step(x_t, sigma, scaling)
        else:
            result = x_t.mean(dim=0, keepdim=True)

        if len(self.x0_traj) > 0:
            torch.save(
                {'x0_traj': torch.cat(self.x0_traj, dim=0),
                 't_traj': torch.tensor(self.t_traj)},
                os.path.join(output_path, 'x0_traj.pt'),
            )

        self.wandb_run.finish()
        return result

    def _inference_particle_pool_analytic(self, observation, num_samples=1, gen_idx=None,
                                          guidance_scale=1.0, sample_idx=0):
        """
        Analytic-marginal pool: equivalent in distribution to the explicit
        particle-pool method (up to redrawing the K-subset each step), but
        without storing P particles. The per-step SDE update is linear in x_t
        with shared coefficients, so the marginal of any single particle stays
        Gaussian: x_t[p,i] = mu[i] + xi[p,i] with xi ~ N(0, Sigma[i] * I).
        Sample K fresh particles from N(mu, Sigma * I) each step instead.
        """
        if self.mf_net is None:
            raise NotImplementedError("particle_pool_analytic requires mf_net")

        output_path = f'step_samples/gen_idx_{gen_idx}_{sample_idx}'
        os.makedirs(output_path, exist_ok=True)
        self.observation = observation
        self.x0_traj = []
        self.t_traj = []

        self.wandb_run = wandb.init(
            project='reinforce_log_likelihoods',
            name=f'gen_{gen_idx}_sample_{sample_idx}_pool_analytic',
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )

        from algo.meanflow_posterior import MeanFlowDiffusionPosterior
        posterior = MeanFlowDiffusionPosterior(
            self.scheduler, self.mf_net, output_path,
            sigma_min=self.mf_sigma_min,
            sibling_renoise_t=self.mf_sibling_renoise_t,
            enable_grad=self.reparam,
        )

        device = "cuda"
        K = self.REINF_K

        sigma_steps = torch.as_tensor(self.scheduler.sigma_steps, device=device).float()
        factor_steps = torch.as_tensor(self.scheduler.factor_steps, device=device).float()
        scaling_factor = torch.as_tensor(self.scheduler.scaling_factor, device=device).float()
        scaling_steps = torch.as_tensor(self.scheduler.scaling_steps, device=device).float()

        stop_steps = self.scheduler.num_steps
        if self.stop_at_sigma is not None:
            below = (sigma_steps <= float(self.stop_at_sigma)).nonzero(as_tuple=False)
            if below.numel() > 0:
                stop_steps = max(int(below[0].item()), 1)
            print(f"[pool-analytic] running {stop_steps}/{self.scheduler.num_steps} SDE steps (K={K})")
        else:
            print(f"[pool-analytic] running {stop_steps} SDE steps (K={K})")

        log_lik_fn = self.get_log_likelihood_fn()

        # Initial marginal: x_T = sigma_max * randn  =>  mu_0 = 0, Sigma_0 = sigma_max^2
        mu = torch.zeros((1, 1, 64, 64), device=device)
        Sigma = float(self.scheduler.sigma_max) ** 2

        def reinforce_step(mu_cur, Sigma_cur, sigma, scaling):
            """Sample K particles from N(mu, Sigma*I), MF→x0, REINFORCE-weighted mean."""
            std = max(Sigma_cur, 0.0) ** 0.5
            x_t_sel = mu_cur + std * torch.randn((K, 1, 64, 64), device=device)
            tau = sigma / (1.0 + sigma)
            z_tau = x_t_sel / (scaling * (1.0 + sigma))
            with torch.no_grad():
                x0_cand = posterior._mf_one_step(z_tau, tau)
            log_lik = log_lik_fn(x0_cand).flatten()
            w = torch.softmax(guidance_scale * log_lik, dim=0)
            mean = (w[:, None, None, None] * x0_cand).sum(dim=0, keepdim=True)
            if self.use_clamp:
                mean = mean.clamp(-1.0, 1.0)
            return mean, log_lik

        reparam_chunk = int(os.environ.get("REPARAM_CHUNK", "32"))

        def reparam_step(mu_cur, Sigma_cur, sigma, scaling):
            """K particles from N(mu, Sigma*I); reparam-grad of logsumexp w.r.t. mu.
            Computed in chunks to bound activation memory: with detached softmax
            weights w_k, grad(logsumexp_k g_k) = sum_k w_k * grad(g_k), so we can
            backprop chunk-by-chunk and accumulate the gradient.
            """
            std = max(Sigma_cur, 0.0) ** 0.5
            tau = sigma / (1.0 + sigma)
            eps_k = torch.randn((K, 1, 64, 64), device=device)

            # Pass 1 (no grad): compute softmax weights and unguided mean over full K.
            with torch.no_grad():
                x_t_full = mu_cur + std * eps_k
                z_tau_full = x_t_full / (scaling * (1.0 + sigma))
                x0_full = posterior._mf_one_step(z_tau_full, tau)
                log_lik_full = log_lik_fn(x0_full).flatten()
                w_full = torch.softmax(guidance_scale * log_lik_full, dim=0)  # [K]
                unguided = x0_full.mean(dim=0, keepdim=True)

            # Pass 2 (chunked, grad): accumulate grad of sum_k w_k * (guidance * log_lik_k) w.r.t. mu.
            mu_grad = mu_cur.detach().clone().requires_grad_(True)
            grad = torch.zeros_like(mu_grad)
            M = max(1, reparam_chunk)
            for c in range(0, K, M):
                eps_chunk = eps_k[c:c + M]
                w_chunk = w_full[c:c + M].detach()
                x_t_chunk = mu_grad + std * eps_chunk
                z_tau_chunk = x_t_chunk / (scaling * (1.0 + sigma))
                with torch.enable_grad():
                    x0_chunk = posterior._mf_one_step(z_tau_chunk, tau)
                log_lik_chunk = log_lik_fn(x0_chunk).flatten()
                chunk_loss = (w_chunk * guidance_scale * log_lik_chunk).sum()
                chunk_grad = torch.autograd.grad(chunk_loss, mu_grad, retain_graph=False)[0]
                grad = grad + chunk_grad.detach()

            a_t = 1.0 - tau
            b_t = tau
            mean = unguided + (b_t ** 2 / a_t) * grad
            if self.use_clamp:
                mean = mean.clamp(-1.0, 1.0)
            return mean.detach(), log_lik_full.detach()

        step_fn = reparam_step if self.reparam else reinforce_step

        for i in range(stop_steps):
            sigma = float(sigma_steps[i])
            scaling = float(scaling_steps[i])
            s_factor = float(scaling_factor[i])
            factor = float(factor_steps[i])

            mean, log_lik = step_fn(mu, Sigma, sigma, scaling)

            self.x0_traj.append(mean.detach().cpu())
            self.t_traj.append(1.0 - i / self.scheduler.num_steps)
            wandb.log({
                'mean_log_likelihood': log_lik.mean().item(),
                'min_log_likelihood': log_lik.min().item(),
                'max_log_likelihood': log_lik.max().item(),
                'pool_x_t_std': max(Sigma, 0.0) ** 0.5,
                'diffusion_step': i,
                't': 1.0 - i / self.scheduler.num_steps,
            })

            # Linear-recurrence coefficients (per-particle update, shared across particles):
            #   x_t[p, i+1] = a * x_t[p, i] + b + c * eps[p, i]
            # so the marginal moments evolve as:
            #   mu    <- a * mu + b
            #   Sigma <- a^2 * Sigma + c^2
            a = s_factor - factor / (sigma ** 2 * scaling ** 2)
            b = factor * mean / (sigma ** 2 * scaling)
            c2 = factor if i < stop_steps - 1 else 0.0
            if self.pool_no_noise:
                c2 = 0.0

            mu = a * mu + b
            Sigma = a * a * Sigma + c2

        if self.last_hop:
            sigma = float(sigma_steps[stop_steps]) if stop_steps < len(sigma_steps) else float(sigma_steps[-1])
            scaling = float(scaling_steps[stop_steps]) if stop_steps < len(scaling_steps) else float(scaling_steps[-1])
            result, _ = step_fn(mu, Sigma, sigma, scaling)
        else:
            result = mu

        if len(self.x0_traj) > 0:
            torch.save(
                {'x0_traj': torch.cat(self.x0_traj, dim=0),
                 't_traj': torch.tensor(self.t_traj)},
                os.path.join(output_path, 'x0_traj.pt'),
            )

        self.wandb_run.finish()
        return result

    def inference(self, observation, num_samples=1, gen_idx=None, guidance_scale=1.0, sample_idx=0):
        if self.pool_analytic:
            return self._inference_particle_pool_analytic(
                observation, num_samples=num_samples, gen_idx=gen_idx,
                guidance_scale=guidance_scale, sample_idx=sample_idx,
            )
        if self.particle_pool_size is not None:
            return self._inference_particle_pool(
                observation, num_samples=num_samples, gen_idx=gen_idx,
                guidance_scale=guidance_scale, sample_idx=sample_idx,
            )

        output_path = f'step_samples/gen_idx_{gen_idx}_{sample_idx}'
        self.x0_traj = []
        self.t_traj = []
        self.xt_traj = []
        self.x0_samples_traj = []
        self.log_lik_traj = []

        # Initialize independent wandb session for this inference run
        self.wandb_run = wandb.init(
            project='reinforce_log_likelihoods',
            name=f'gen_{gen_idx}_sample_{sample_idx}',
            reinit=True,
            settings=wandb.Settings(start_method="fork")
        )

        if self.mf_net is None:
            if self.reparam:
                raise NotImplementedError("Reparam estimator currently requires mf_net (gradient-enabled MF posterior).")
            diffusion_posterior = InverseBenchModel(self.scheduler, self.net, self.desired_number_of_steps_inner_loop, output_path)
        else:
            from algo.meanflow_posterior import MeanFlowDiffusionPosterior
            diffusion_posterior = MeanFlowDiffusionPosterior(
                self.scheduler, self.mf_net, output_path,
                sigma_min=self.mf_sigma_min,
                sibling_renoise_t=self.mf_sibling_renoise_t,
                enable_grad=self.reparam,
            )
        self.observation = observation
        build_mean_fn = build_reparam_mean_fn if self.reparam else build_reinforce_mean_fn
        reinforce_mean_fn = build_mean_fn(diffusion_posterior, self.get_log_likelihood_fn(), self.REINF_K, plot_samples=self.create_plot_samples(output_path), guidance_scale=guidance_scale)

        clamp_stats = {'pixels_clamped': 0, 'pixels_total': 0,
                       'steps_with_clamp': 0, 'n_steps': 0,
                       'max_abs_pre_clamp': 0.0}

        def clamp_wrap(fn):
            def wrapped(xt, t):
                m = fn(xt, t)
                oob = (m < -1.0) | (m > 1.0)
                n_oob = int(oob.sum().item())
                clamp_stats['pixels_clamped'] += n_oob
                clamp_stats['pixels_total'] += m.numel()
                clamp_stats['steps_with_clamp'] += int(n_oob > 0)
                clamp_stats['n_steps'] += 1
                clamp_stats['max_abs_pre_clamp'] = max(clamp_stats['max_abs_pre_clamp'], float(m.abs().max().item()))
                return m.clamp(-1.0, 1.0)
            return wrapped

        if self.use_clamp:
            reinforce_mean_fn = clamp_wrap(reinforce_mean_fn)

        sigma_steps = torch.as_tensor(self.scheduler.sigma_steps).float().to("cuda")
        factor_steps = torch.as_tensor(self.scheduler.factor_steps).float().to("cuda")
        scaling_factor = torch.as_tensor(self.scheduler.scaling_factor).float().to("cuda")
        scaling_steps = torch.as_tensor(self.scheduler.scaling_steps).float().to("cuda")

        # Truncate the SDE: stop at the first step where sigma <= stop_at_sigma.
        # `stop_steps` = number of SDE iterations actually run.
        stop_steps = self.scheduler.num_steps
        if self.stop_at_sigma is not None:
            below = (sigma_steps <= float(self.stop_at_sigma)).nonzero(as_tuple=False)
            if below.numel() > 0:
                stop_steps = max(int(below[0].item()), 1)
            sigma_steps = sigma_steps[: stop_steps + 1]
            factor_steps = factor_steps[: stop_steps]
            scaling_factor = scaling_factor[: stop_steps]
            scaling_steps = scaling_steps[: stop_steps + 1]
            print(f"[truncate] running {stop_steps}/{self.scheduler.num_steps} SDE steps "
                  f"(stop sigma={float(sigma_steps[-1]):.4f}, threshold={self.stop_at_sigma})")

        result = edm_sde(
            reinforce_mean_fn,
            sigma_steps,
            factor_steps,
            scaling_factor,
            scaling_steps,
            shape=(1, 64, 64),
            n_samples=num_samples,
            x_t_start=DiffusionSampler(self.scheduler).get_start(torch.ones((1, 1, 64, 64)).to("cuda"))
        )

        if self.last_hop:
            # Use the t corresponding to the last sigma_steps element so the MF
            # posterior reads the matching sigma from the unmodified scheduler.
            last_t = 1.0 - stop_steps / self.scheduler.num_steps
            # Both reinforce_mean_fn and reparam_mean_fn manage their own grad
            # context; an outer torch.no_grad() would prevent reparam from
            # attaching gradients to its xt clone.
            result = reinforce_mean_fn(result, last_t)

        if len(self.x0_traj) > 0:
            os.makedirs(output_path, exist_ok=True)
            torch.save(
                {'x0_traj': torch.cat(self.x0_traj, dim=0),
                 't_traj': torch.tensor(self.t_traj)},
                os.path.join(output_path, 'x0_traj.pt'),
            )

        if self.save_intermediates and len(self.xt_traj) > 0:
            os.makedirs(output_path, exist_ok=True)
            torch.save(
                {'xt_traj': torch.cat(self.xt_traj, dim=0),
                 'x0_samples_traj': torch.cat(self.x0_samples_traj, dim=0),
                 'log_lik_traj': torch.cat(self.log_lik_traj, dim=0),
                 't_traj': torch.tensor(self.t_traj),
                 'observation': observation.detach().cpu()},
                os.path.join(output_path, 'intermediates.pt'),
            )

        if clamp_stats['n_steps'] > 0:
            steps_frac = clamp_stats['steps_with_clamp'] / clamp_stats['n_steps']
            pixel_frac = clamp_stats['pixels_clamped'] / max(clamp_stats['pixels_total'], 1)
            print(f"[clamp] gen={gen_idx} sample={sample_idx}  "
                  f"steps clamped: {clamp_stats['steps_with_clamp']}/{clamp_stats['n_steps']} ({steps_frac:.1%}); "
                  f"pixels clamped: {clamp_stats['pixels_clamped']}/{clamp_stats['pixels_total']} ({pixel_frac:.2%}); "
                  f"max|guided_mean| pre-clamp: {clamp_stats['max_abs_pre_clamp']:.3f}")
            self.last_clamp_stats = dict(clamp_stats)

        self.min_log_likelihoods = []
        self.mean_log_likelihoods = []
        self.max_log_likelihoods = []

        # Finish the independent wandb session
        self.wandb_run.finish()

        return result
