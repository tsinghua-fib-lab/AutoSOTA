import math
import torch
from scipy import integrate
import torch.nn.functional as F
from torchdiffeq import odeint

class FlowMatching:
    def __init__(
        self,
        model,
        num_steps: int = 10,
        sampling: str = "deterministic",
        sigma_min: float = 0.1,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        device: str = "cpu"
    ):
        """
        model      – your torch.nn.Module
        num_steps  – number of integration steps if you do ODE sampling
        sampling   – one of:
                      Euclidean: "deterministic", "improved_fm", "schrodinger", "vp_diffusion"
                      Geodesic:  "geo_det", "geo_amb_const", "geo_tan_const",
                                 "geo_amb_sb", "geo_tan_sb"
                      (kept: "hyperspherical_fm" == geo_det)
        sigma_min  – σ for improved_fm and SB schedule base
        beta_min   – β_min for VP diffusion
        beta_max   – β_max for VP diffusion
        device     – cpu / cuda
        """
        self.model       = model
        self.N           = num_steps
        self.sampling    = sampling
        self.sigma_min   = sigma_min
        self.beta_min    = beta_min
        self.beta_max    = beta_max
        self.device      = device

        self.model = self.model.to(self.device)

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def _renorm(x, eps=1e-12):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @staticmethod
    def _proj_tangent(phi, u):
        # project u onto tangent space at phi: u - <u,phi>phi
        return u - (u * phi).sum(dim=-1, keepdim=True) * phi

    @staticmethod
    def _sb_schedule(t, sigma):
        # Schrödinger bridge schedule σ√(t(1−t))
        return sigma * torch.sqrt(t * (1.0 - t)).clamp_min(1e-8)

    @staticmethod
    def logmap_sphere(p, q):
        """
        Logarithm map on the hypersphere: tangent vector v at p pointing to q
        """
        dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        v = q - dot * p
        v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
        return theta * v / v_norm

    @staticmethod
    def expmap_sphere(p, v):
        """
        Exponential map on the hypersphere: move from p along tangent v
        """
        norm_v = torch.norm(v, dim=-1, keepdim=True) + 1e-8
        return torch.cos(norm_v) * p + torch.sin(norm_v) * (v / norm_v)

    @staticmethod
    def geodesic_velocity_at_t(p, v, t, eps=1e-8):
        """u_true = d/dt exp_p(t v) evaluated at time t; lives in T_{z_t} S."""
        alpha = v.norm(dim=-1, keepdim=True).clamp_min(eps)
        return -alpha * torch.sin(alpha * t) * p + torch.cos(alpha * t) * v

    # ----------------------------- training tuple -----------------------------

    def get_train_tuple(self, z0, z1):
        B = z0.shape[0]
        z0, z1 = z0.to(self.device), z1.to(self.device)

        # draw t ∈ (0,1)
        eps = 1e-4
        t = torch.rand((B,1), device=self.device) * (1 - 2*eps) + eps

        # ------------------ Geodesic variants (on S^{d-1}) ------------------
        if self.sampling in {"geo_det", "geo_amb_const", "geo_tan_const",
                            "geo_amb_sb", "geo_tan_sb"}:
            # ensure inputs lie on sphere
            z0 = self._renorm(z0)
            z1 = self._renorm(z1)

            # mean (geodesic) and drift target (tangent at mean)
            v = self.logmap_sphere(z0, z1)          # tangent at z0
            mean = self.expmap_sphere(z0, t * v)    # point at time t
            u_true = self.geodesic_velocity_at_t(z0, v, t)  # tangent @ mean

            if self.sampling == "geo_det":
                z_t = mean

            elif self.sampling == "geo_amb_const":
                # ambient constant noise (no correction term)
                z_t = mean + self.sigma_min * torch.randn_like(mean)

            elif self.sampling == "geo_tan_const":
                # tangent constant noise via Exp(mean, σ ξ_tan) (no correction term)
                xi = torch.randn_like(mean)
                xi_tan = self._proj_tangent(mean, xi)
                z_t = self.expmap_sphere(mean, self.sigma_min * xi_tan)

            elif self.sampling == "geo_amb_sb":
                # ambient SB: γ(t)=σ√(t(1−t))  + SB correction
                gamma = self._sb_schedule(t, self.sigma_min)
                noise = torch.randn_like(mean)
                z_t = mean + gamma * noise

                denom = (2.0 * t * (1.0 - t)).clamp(min=1e-6)
                corr_amb = (z_t - mean) / denom                         # ambient vector
                corr_tan = self._proj_tangent(mean, corr_amb)           # live in T_mean S^{d-1}
                u_true = u_true + (1.0 - 2.0 * t) * corr_tan            # corrected drift (tangent)

            elif self.sampling == "geo_tan_sb":
                # tangent SB via Exp(mean, γ ξ_tan)  + SB correction in tangent
                gamma = self._sb_schedule(t, self.sigma_min)
                xi = torch.randn_like(mean)
                xi_tan = self._proj_tangent(mean, xi)
                delta = gamma * xi_tan                                  # tangent vector at mean
                z_t = self.expmap_sphere(mean, delta)

                denom = (2.0 * t * (1.0 - t)).clamp(min=1e-6)
                corr_tan = delta / denom                                # already tangent @ mean
                u_true = u_true + (1.0 - 2.0 * t) * corr_tan            # corrected drift

            return z_t, t, u_true

        # ------------------ Euclidean variants (original) ------------------
        mean = t * z1 + (1.0 - t) * z0

        if self.sampling == "euc_det":
            z_t    = mean
            u_true = (z1 - z0)

        elif self.sampling == "euc_ot":
            z_t    = torch.normal(mean, self.sigma_min)
            u_true = (z1 - z0)

        elif self.sampling == "euc_sb":
            var    = t * (1.0 - t) * (self.sigma_min ** 2)
            var    = var.clamp(min=1e-6)
            std    = torch.sqrt(var)
            z_t    = torch.normal(mean, std)
            denom  = (2.0 * t * (1.0 - t)).clamp(min=1e-6)
            corr   = (z_t - mean) / denom
            u_true = (1.0 - 2.0 * t) * corr + (z1 - z0)

        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling!r}")

        return z_t, t, u_true

    # ----------------------------- ODE sampling -----------------------------

    @torch.no_grad()
    def sample_ode(
        self,
        z_init,
        N: int,
        reverse: bool = False,
        use_sphere: bool = False,
        t0: torch.Tensor = None,      # optional starting time in [0,1]
    ): 
        """
        ODE sampling. If geodesic (use_sphere=True or geodesic sampling mode),
        integrate intrinsically: z_{k+1} = Exp_{z_k}(Δt * Π_{T_{z_k}} v_θ).
        Else: Euclidean Euler step + renorm for stability.
        """
        if N is None:
            N = self.N

        # 1) initialize point
        z = z_init.detach().clone().to(self.device)

        # 2) determine integration interval
        if t0 is None:
            if reverse:
                start, end = 1.0, 0.0
            else:
                start, end = 0.0, 1.0
        else:
            start = float(t0.mean().item()) if isinstance(t0, torch.Tensor) else float(t0)
            end   = 1.0

        # 3) build time grid
        dt = (end - start) / N
        time_steps = torch.linspace(start, end, N + 1, device=self.device)

        # geodesic modes toggle
        geo_modes = {"geo_det","geo_amb_const","geo_tan_const","geo_amb_sb","geo_tan_sb"}
        do_geo = use_sphere or (self.sampling in geo_modes)

        # 4) integrate
        traj = [z.clone()]
        for i in range(N):
            t = time_steps[i]
            t_vec = t.expand(z.shape[0], 1)  # (B,1)

            out = self.model(z, t_vec)
            v = out[0] if isinstance(out, tuple) else out

            if do_geo:
                # project to tangent and step via Exp
                v = self._proj_tangent(z, v)
                z = self.expmap_sphere(z, v * dt)

            else:
                # Euclidean step then renorm for stability
                z = z + v * dt
                # z = self._renorm(z)

            traj.append(z.clone())

        
        return traj

    # ----------------------------- SDE sampling (Euclidean) -----------------------------

    # @torch.no_grad()
    # def sample_sde(self, z_init, N=50, eps_fn=lambda t: 1e-3):
    #     """
    #     Euler–Maruyama SDE sampling (Euclidean formulations):
    #         dX_t = b(t,X_t) dt - (eps/gamma(t))*eta(t,X_t) dt + sqrt(2 eps) dW_t
    #     Requires that self.denoiser_model has been set.
    #     """
    #     assert hasattr(self, "denoiser_model") and self.denoiser_model is not None, \
    #         "Attach a denoiser_model before calling sample_sde()."

    #     if N <= 1:
    #         return z_init.unsqueeze(0)  # [1, B, D]

    #     eps = 1e-3
    #     ts  = torch.linspace(eps, 1.0, N, device=self.device)
    #     dt  = ts[1] - ts[0]

    #     z    = z_init.to(self.device)
    #     traj = [z]

    #     for i in range(N - 1):
    #         t     = ts[i].view(1,1).expand(z.size(0),1)  # (B,1)
    #         out   = self.model(z, t)
    #         drift = out[0] if isinstance(out, tuple) else out

    #         # compute gamma(t)
    #         if self.sampling == "improved_fm":
    #             gamma = torch.full_like(t, self.sigma_min)
    #         elif self.sampling == "schrodinger":
    #             gamma = (torch.sqrt(t * (1 - t)) * self.sigma_min).clamp(min=1e-6)
    #         elif self.sampling == "vp_diffusion":
    #             T_t   = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
    #             alpha = torch.exp(-0.5 * T_t)
    #             gamma = torch.sqrt(1 - alpha**2).clamp(min=1e-6)
    #         else:
    #             raise RuntimeError(f"SDE sampling not supported for mode={self.sampling}")

    #         eta    = self.denoiser_model(z, t)
    #         eps_val = eps_fn(t)
    #         noise  = torch.randn_like(z) * torch.sqrt(2 * eps_val * dt)

    #         z = z + (drift - (eps_val / gamma) * eta) * dt + noise
    #         z = self._renorm(z)
    #         traj.append(z)

    #     return torch.stack(traj, dim=0)  # [N, B, D]