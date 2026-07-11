import abc
import torch
import numpy as np
from . import distribution
from tqdm import tqdm
import gc
import scipy.special as sp
from scipy.optimize import fsolve


class LogBridge(abc.ABC):
    """Logarithm Bridge abstract class. Functions are designed for a mini-batch of inputs."""
    def __init__(self, manifold, scheduler, prior_dist, **kwargs):
        super().__init__()
        self.manifold = manifold
        self.scheduler = scheduler
        self.prior_dist = prior_dist
        
        self.add_mask_token = hasattr(prior_dist, "mask_idx")
        self.preprocess_steps = kwargs.get("preprocess_steps") + 1
        self.eps = kwargs.get("eps", 1e-6)
        self.kwargs = kwargs
        
        self.rho_scale = kwargs.get("rho_scale", 1.0)
        
    @property
    def T(self):
        return self.scheduler.T
    
    # Scale coefficient for the drift
    def scale_by_coeff(self, x, t):
        scale = self.scheduler.drift_coeff(t)
        return torch.einsum("...,...ij->...ij", scale, x)

    # Drift of Logarithm Bridge
    def drift(self, end, cur, t):
        tangent_vec = self.manifold.log(point=end, base_point=cur)
        return self.scale_by_coeff(tangent_vec, t)

    # Diffusion coefficientof Logarithm Bridge
    def diffusion(self, x, t):
        return self.scheduler.beta(t).sqrt()

    # Laplacian term in radial process
    def coord_laplacian(self, x, t):
        laplacian_coeff = -0.5 * self.manifold.dim * self.scheduler.beta(t)
        if len(x.shape) == len(t.shape):
            return laplacian_coeff * x
        elif len(x.shape) == len(t.shape)+1:
            return torch.einsum("...,...i->...i", laplacian_coeff, x)
        else:
            return torch.einsum("...,...ij->...ij", laplacian_coeff, x)

    # Sampling from the prior distribution
    def prior_sample(self, batch_dims, device, **kwargs):
        return self.prior_dist.sample(batch_dims, device)

    # Sampling the interpolant from approximated transition distribution
    @abc.abstractmethod
    def interpolant(self, start, end, t):
        pass
    
    # Simulate the interpolant
    def interpolant_simul(self, start, end, t, simul_steps):
        with torch.no_grad():
            timesteps = torch.linspace(0., 1., simul_steps+1, device=t.device).unsqueeze(1)
            timesteps = t * timesteps
            dt = timesteps[1] - timesteps[0]
            
            # Simulate with geodesic random walk
            x = start
            for i in range(0, timesteps.shape[0]-1):
                t = timesteps[i]
                z = self.manifold.random_normal_tangent(base_point=x)
            
                drift = self.drift(end, x, t)
                diffusion = self.diffusion(x, t)
                tangent_vec = torch.einsum(
                    "...,...ij,...->...ij", diffusion, z, dt.abs().sqrt()
                )
                tangent_vec = tangent_vec + torch.einsum("...ij,...->...ij", drift, dt)
                x = self.manifold.exp(tangent_vec=tangent_vec, base_point=x)
        return x
    
    # Pre-compute the parameters of Riemannian Normal
    @abc.abstractmethod
    def preprocess(self, batch_dims):
        pass
    
    # Compute the covariance of Riemannian Normal
    def solve_rho(self, cos_norm, init=0):
        def f(rho):
            lhs = np.exp(-rho**2 / 2) * sp.hyp1f1((self.manifold.dim)/2, 0.5, -rho**2 / 2)
            return lhs - cos_norm
        return fsolve(f, init)
    
    # Preprocessing the parameters of Riemannian Normal with simulation
    def preprocess_simul(self, batch_dims):
        device=self.device
        print(f"Preprocess LogBridge...")
        with torch.no_grad():
            alphas = torch.zeros(self.preprocess_steps, device=device)
            alphas[-1] = 1.
            diffs = torch.zeros(self.preprocess_steps, device=device)

            start = self.prior_sample(batch_dims, device)
            end_idx = 0
            end = torch.zeros(tuple(batch_dims), device=device)
            end[...,end_idx] = 1.

            x = start
            timesteps = torch.linspace(0., 1., self.preprocess_steps, device=device).unsqueeze(1)
            dt = timesteps[1] - timesteps[0]
            
            # Simulate with geodesic random walk
            pbar = tqdm(range(0, timesteps.shape[0]-2), leave=False)
            for i in pbar:
                tangent_vec = torch.einsum(
                    "...,...ij,...->...ij", 
                    self.scheduler.beta(timesteps[i]).sqrt(), 
                    self.manifold.random_normal_tangent(base_point=x), 
                    dt.sqrt()
                )
                tangent_vec = tangent_vec + torch.einsum("...ij,...->...ij", self.drift(end, x, timesteps[i]), dt)
                x = self.manifold.exp(tangent_vec=tangent_vec, base_point=x)
                
                # compute mean of normal dist
                mu = x.mean(list(range(len(batch_dims)-1)), keepdim=True)
                mu = mu / mu.norm()
                mask_idx = self.prior_dist.mask_idx if self.prior_dist.mask_idx>=0 else batch_dims[-1] + self.prior_dist.mask_idx
                idx_set_zero = list(set(list(range(batch_dims[-1]))) - set([end_idx, mask_idx]))
                mu[..., idx_set_zero] = 0.
                mu = self.manifold.projection(mu)
                alphas[i+1] = mu[0, 0, end_idx]

                # compute std of normal dist
                lift = self.manifold.log(x, mu)
                var = (lift - lift.mean(list(range(len(batch_dims)-1)), keepdim=True)).square().sum(-1) / (mu.shape[-1]-1)
                diffs[i+1] = var.sqrt().mean()

            alphas = torch.where(alphas<0, torch.zeros_like(alphas), alphas)
            diffs[-1] = 0.

            del start, end, x, tangent_vec, mu, lift, var
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Preprocess Done.")
        return alphas, diffs


# Mixture Path from Masked diffusion and Uniform diffusion
class LogBridge_Mixture(LogBridge):
    def __init__(self, manifold, scheduler, prior_dist, **kwargs):
        super().__init__(manifold, scheduler, prior_dist, **kwargs)
        assert isinstance(self.prior_dist, distribution.Mixture)
        self.mix_type = kwargs.get("mix_type")

        self.inner_prod = 1 / np.sqrt(self.manifold.dim+1)
        self.proj_norm = np.sqrt(1 - self.inner_prod**2)
        
        if kwargs.get("preprocessed", None) is None:
            self.device = kwargs.get("device")
            self.alphas, self.rhos = self.preprocess(kwargs.get("dims", 2**14))
        else:
            self.alphas, self.rhos = kwargs.get("preprocessed")
            self.preprocess_steps = len(self.alphas[0])
            self.device = self.alphas.device

    def mixture_prob(self, x=None, t=0.):
        if self.mix_type == "linear":
            return 1 - t
        elif self.mix_type == "sqrt":
            return 1 - t**0.5
        elif "step" in self.mix_type:
            thr = self.kwargs.get("step_thr")
            return torch.where(t < thr, torch.ones_like(t), torch.zeros_like(t))
        else:
            raise ValueError(f"Invalid mixture type: {self.mix_type}")
        
    def prior_sample(self, batch_dims, device, t=None, **kwargs):
        if t is None:
            t = torch.zeros(batch_dims[0], device=device)
        probs = self.mixture_prob(t=t)
        return self.prior_dist.sample(batch_dims, device, probs=probs)

    def preprocess(self, dims):
        device = self.device
        print(f"Preprocess Mixture Path...")
        with torch.no_grad():
            proj_f = torch.zeros((2, self.preprocess_steps), device=device)
            proj_f[1,...] = self.inner_prod
            proj_f[...,-1] = 1.

            proj_0 = torch.zeros((2, self.preprocess_steps), device=device)
            proj_0[1,...] = self.inner_prod
            proj_0[...,0] = 1.

            x = torch.stack([
                torch.zeros(dims, device=device),
                torch.ones(dims, device=device),
                torch.ones(dims, device=device) * self.inner_prod,
                torch.ones(dims, device=device),
            ], dim=-1)
            timesteps = torch.linspace(0., 1., self.preprocess_steps, device=device).unsqueeze(1)
            dt = timesteps[1] - timesteps[0]

            pbar = tqdm(range(0, timesteps.shape[0]-2), leave=False)
            for i in pbar:
                t = torch.ones(dims, device=device) * timesteps[i]
                z = torch.randn(x.shape, device=device) # R^4 Gaussian

                coeff = self.scheduler.drift_coeff(t)
                arccos_x0_mask = x[...,0].clip(min=-1, max=1).arccos()
                sin_arccos_x0_mask = (1 - x[...,0]**2).clip(min=0).sqrt()
                arccos_x0_unif = x[...,2].clip(min=-1, max=1).arccos()
                sin_arccos_x0_unif = (1 - x[...,2]**2).clip(min=0).sqrt()

                drift = torch.stack([
                    coeff * arccos_x0_mask * sin_arccos_x0_mask,
                    -coeff * x[...,0] * x[...,1] * arccos_x0_mask.clip(min=self.eps) / sin_arccos_x0_mask.clip(min=self.eps),
                    coeff * arccos_x0_unif * sin_arccos_x0_unif,
                    coeff * (self.inner_prod - x[...,2] * x[...,3]) * arccos_x0_unif.clip(min=self.eps) / sin_arccos_x0_unif.clip(min=self.eps)
                ], dim=-1) + self.coord_laplacian(x, t)

                diffusion = torch.einsum("...,...i->...i", self.scheduler.beta(t), 1 - x**2).clip(min=0).sqrt()

                x = x + drift * dt + diffusion * z * dt.abs().sqrt()
                
                proj_f[0, i+1] = x[...,0].mean()
                proj_0[0, i+1] = x[...,1].mean()
                proj_f[1, i+1] = x[...,2].mean()
                proj_0[1, i+1] = x[...,3].mean()

        alphas, rhos = self.moment_matching(proj_f, proj_0)
        
        print(f"Preprocess Done.")
        return alphas, rhos

    # Eq.(25) in the paper
    def moment_matching(self, proj_f, proj_0):
        rtheta = proj_0[0] / proj_f[0].clip(min=self.eps)
        alphas_mask = 1 / (1 + rtheta**2).sqrt()
        cos_norm_mask = proj_f[0].clip(min=self.eps) / alphas_mask

        _rtheta = proj_f[1] / proj_0[1]
        _rtheta = (_rtheta - self.inner_prod)**2
        alphas_unif = (_rtheta / (1 - self.inner_prod**2 + _rtheta)).sqrt()

        cos_norm_start = proj_0[1].clip(min=self.eps) / (1 - alphas_unif**2).sqrt().clip(min=self.eps)
        cos_norm_end = proj_f[1] / (self.proj_norm * alphas_unif + self.inner_prod * (1 - alphas_unif**2).sqrt())
        cos_norm_unif = torch.cat([cos_norm_start[:len(alphas_unif)//2], cos_norm_end[len(alphas_unif)//2:]], dim=0)

        rhos_mask = [0.]
        for i in tqdm(range(1,len(cos_norm_mask)), leave=False):
            init = 1e-4 if i==1 else rhos_mask[-1]
            rho = self.solve_rho(cos_norm_mask[i].item(), init)
            rhos_mask.append(np.abs(rho).item())
        rhos_mask = torch.tensor(rhos_mask, device=self.device)

        rhos_unif = [0.]
        for i in tqdm(range(1,len(cos_norm_unif)), leave=False):
            init = 1e-4 if i==1 else rhos_unif[-1]
            rho = self.solve_rho(cos_norm_unif[i].item(), init)
            rhos_unif.append(np.abs(rho).item())
        rhos_unif = torch.tensor(rhos_unif, device=self.device)

        alphas = torch.stack([alphas_mask, alphas_unif], dim=0)
        rhos = torch.stack([rhos_mask, rhos_unif], dim=0)
        rhos = rhos * self.rho_scale # Calibrate rho for low-dimension

        return alphas, rhos

    def interpolant(self, start, end, t):
        ts = torch.linspace(0., 1., self.preprocess_steps, device=t.device)
        idx = torch.searchsorted(ts, t) - 1
        idx = idx.clip(max=len(ts)-2)
        r = (t - ts[idx]) / (ts[idx+1] - ts[idx])

        alphas_mask = self.alphas[0, idx] * (1-r) + self.alphas[0, idx+1] * r
        diff_mask = self.rhos[0, idx] * (1-r) + self.rhos[0, idx+1] * r

        alphas_unif = self.alphas[1, idx] * (1-r) + self.alphas[1, idx+1] * r
        diff_unif = self.rhos[1, idx] * (1-r) + self.rhos[1, idx+1] * r
        
        mu_mask = torch.einsum("...,...ij->...ij", alphas_mask, end) \
            + torch.einsum("...,...ij->...ij", (1 - alphas_mask**2).sqrt(), start)
        
        mu_unif = torch.einsum("...,...ij->...ij", alphas_unif / self.proj_norm, end) \
            + torch.einsum(
                "...,...ij->...ij", 
                ((1 - alphas_unif**2).sqrt() - alphas_unif * self.inner_prod / self.proj_norm), 
                start
            )

        mask_flag = (start[...,self.prior_dist.mask_idx] == 1)
        mu = torch.where(mask_flag.unsqueeze(-1), mu_mask, mu_unif)

        z = torch.randn_like(start)
        noise_mask = torch.einsum("...,...ij->...ij", diff_mask, z)
        noise_unif = torch.einsum("...,...ij->...ij", diff_unif, z)

        normal = mu + torch.where(mask_flag.unsqueeze(-1), noise_mask, noise_unif)
        normal = self.manifold.exp(normal, mu)

        return normal
    

# SDE for Masked diffusion and Uniform diffusion
class LogBridge_Init(LogBridge):
    def __init__(self, manifold, scheduler, prior_dist, **kwargs):
        super().__init__(manifold, scheduler, prior_dist, **kwargs)
        assert isinstance(self.prior_dist, distribution.Initial)
        self.inner_prod = self.prior_dist.init_val
        self.proj_norm = np.sqrt(1 - self.inner_prod**2)

        if kwargs.get("preprocessed", None) is None:
            self.device = kwargs.get("device")
            self.alphas, self.rhos = self.preprocess(kwargs.get("dims", 2**14))
        else:
            self.alphas, self.rhos = kwargs.get("preprocessed")
            self.preprocess_steps = len(self.alphas)
            self.device = self.alphas.device

        self.adjusted_alphas = self.alphas * self.proj_norm + \
            (1 - self.alphas**2).sqrt() * self.inner_prod

    def preprocess(self, dims):
        device = self.device
        print(f"Preprocess LogBridge...")
        with torch.no_grad():
            proj_f = torch.ones(self.preprocess_steps, device=device) * self.inner_prod
            proj_f[-1] = 1.

            proj_0 = torch.ones(self.preprocess_steps, device=device) * self.inner_prod
            proj_0[0] = 1.

            x = torch.stack([
                torch.ones(dims, device=device) * self.inner_prod,
                torch.ones(dims, device=device),
            ], dim=-1)
            timesteps = torch.linspace(0., 1., self.preprocess_steps, device=device).unsqueeze(1)
            dt = timesteps[1] - timesteps[0]

            pbar = tqdm(range(0, timesteps.shape[0]-2), leave=False)
            for i in pbar:
                t = torch.ones(dims, device=device) * timesteps[i]
                z = torch.randn(x.shape, device=device) # R^2 brownian motion

                laplacian_term = self.coord_laplacian(x, t)
                coeff = self.scheduler.drift_coeff(t)
                arccos_x0 = x[...,0].clip(min=-1, max=1).arccos()
                sin_arccos_x0 = (1 - x[...,0]**2).clip(min=0).sqrt() # arccos_x0.sin()
                drift = torch.stack([
                    coeff * arccos_x0 * sin_arccos_x0,
                    coeff * (self.inner_prod - x[...,0] * x[...,1]) * arccos_x0.clip(min=self.eps) / sin_arccos_x0.clip(min=self.eps)
                ], dim=-1) + laplacian_term

                diffusion = torch.einsum("...,...i->...i", self.scheduler.beta(t), 1 - x**2).clip(min=0).sqrt()

                x = x + drift * dt + diffusion * z * dt.abs().sqrt()
                
                proj_f[i+1] = x[...,0].mean()
                proj_0[i+1] = x[...,1].mean()

        alphas, rhos = self.moment_matching(proj_f, proj_0)
        
        print(f"Preprocess Done.")
        return alphas, rhos
    
    # Eq.(25) in the paper
    def moment_matching(self, proj_f, proj_0):
        _rtheta = proj_f / proj_0
        _rtheta = (_rtheta - self.inner_prod)**2
        alphas = (_rtheta / (1 - self.inner_prod**2 + _rtheta)).sqrt()

        cos_norm_start = proj_0.clip(min=self.eps) / (1 - alphas**2).sqrt().clip(min=self.eps)
        cos_norm_end = proj_f / (self.proj_norm * alphas + self.inner_prod * (1 - alphas**2).sqrt())
        cos_norm = torch.cat([cos_norm_start[:len(alphas)//2], cos_norm_end[len(alphas)//2:]], dim=0)

        rhos = [0.]
        for i in tqdm(range(1,len(cos_norm)), leave=False):
            init = 1e-4 if i==1 else rhos[-1]
            rho = self.solve_rho(cos_norm[i].item(), init)
            rhos.append(np.abs(rho).item())
        rhos = torch.tensor(rhos, device=self.device)
        rhos = rhos * self.rho_scale # Calibrate rho for low-dimension

        return alphas, rhos

    def interpolant(self, start, end, t):
        ts = torch.linspace(0., 1., self.preprocess_steps, device=t.device)
        idx = torch.searchsorted(ts, t) - 1
        idx = idx.clip(min=0, max=len(ts)-2)

        r = (t - ts[idx]) / (ts[idx+1] - ts[idx])
        alphas = self.alphas[idx] * (1-r) + self.alphas[idx+1] * r
        diff = self.rhos[idx] * (1-r) + self.rhos[idx+1] * r

        mu = torch.einsum("...,...ij->...ij", alphas / self.proj_norm, end) \
            + torch.einsum("...,...ij->...ij", ((1 - alphas**2).sqrt() - alphas * self.inner_prod / self.proj_norm), start)
        z = torch.randn_like(start)
        normal = mu + torch.einsum("...,...ij->...ij", diff, z)

        normal = self.manifold.exp(normal, mu)

        return normal
    

class ContinuousSDE:
    def __init__(self, schedule="vp", sigma=2.5):
        """
        PyTorch port of JAX GaussianConditionalPath.
        """
        self.schedule = schedule
        self.sigma_val = sigma

    def alpha(self, t):
        # LinearAlpha: alpha_t = t
        return t

    def alpha_dt(self, t):
        # d/dt alpha_t = 1
        return torch.ones_like(t)

    def beta(self, t):
        # BetaVP: beta_t = sqrt(1 - t^2)
        # Added clamp/epsilon for stability at t=1
        return torch.sqrt(1.0 - t**2 + 1e-8)

    def beta_dt(self, t):
        # d/dt sqrt(1 - t^2) = -t / sqrt(1 - t^2)
        eps = 1e-12
        return -t / (torch.sqrt(1.0 - t**2) + eps)

    def sigma(self, t):
        return torch.tensor(self.sigma_val, device=t.device)

    def marginal_prob(self, z0, t):
        """
        x_t = alpha(t) * z0 + beta(t) * eps
        Returns: (mean, std)
        """
        # Ensure t is broadcastable [B, 1, 1...]
        view_shape = [t.shape[0]] + [1] * (z0.ndim - 1)
        t_b = t.view(*view_shape)
        
        return self.alpha(t_b) * z0, self.beta(t_b)

    def get_drift(self, x_t, t, eps_pred, t_eps=1e-4, alpha_min=1e-12, beta_min=1e-6):
        """
        drift = f(x,t) - g(t)^2 * score
        """
        # Broadcast t to [B, 1, 1...]
        view_shape = [t.shape[0]] + [1] * (x_t.ndim - 1)
        t_b = t.view(*view_shape)
        
        # Clamp t for general calculation
        t_clamped = t_b.clamp(min=t_eps, max=1.0 - t_eps)

        # Evaluate Schedules
        a = self.alpha(t_clamped)
        b = self.beta(t_clamped)
        a_dt = self.alpha_dt(t_clamped)
        b_dt = self.beta_dt(t_clamped)
        sigma_t = self.sigma(t_b)

        # Safe denominators
        a_safe = torch.maximum(a, torch.tensor(alpha_min, device=a.device))
        b_safe = torch.maximum(b, torch.tensor(beta_min, device=b.device))

        # Score = -eps / beta
        score = -eps_pred / b_safe

        # --- General Drift Term ---
        # r = alpha' / alpha
        r = a_dt / a_safe
        
        # drift_general = ((b^2 * r - b*b' + 0.5*sigma^2) * score + r * x_t)
        term1 = (b**2 * r - b * b_dt + 0.5 * sigma_t**2) * score
        term2 = r * x_t
        drift_general = term1 + term2

        # --- Exact t->0 Drift Term ---
        # At t=0: drift0 = (-beta'(0) - 0.5*sigma^2) * x_t
        # For BetaVP: beta'(0) = 0
        b_dt_at_0 = self.beta_dt(torch.zeros_like(t_b))
        drift_t0 = (-b_dt_at_0 - 0.5 * sigma_t**2) * x_t

        # --- Blend ---
        # If t <= t_eps, use t0 limit, otherwise general
        mask = (t_b <= t_eps).float()
        drift = mask * drift_t0 + (1.0 - mask) * drift_general

        return drift