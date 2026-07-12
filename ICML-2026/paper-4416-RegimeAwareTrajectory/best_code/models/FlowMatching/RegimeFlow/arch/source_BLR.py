import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


class PatternType:
    DIRECTLY_STABLE = 0
    INC_STABLE = 1
    DEC_STABLE = 2
    OSCILLATION = 3
    INCREASING = 4
    DECREASING = 5


@dataclass
class BLRPriorConfig:
    alpha: float = 1.0              # Prior precision
    beta: float = 20.0              # Noise precision  
    noise_scale: float = 0.1        # Sampling noise
    num_harmonics: int = 4          # Fourier harmonics for oscillation
    saturation_scales: Tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0)  # Multi-scale λ
    poly_degree: int = 2            # Polynomial degree for monotonic
    min_variance: float = 1e-6

    saturation_rate: float = 3.0
    slope_window: int = 10


class BasisLibrary:
    @staticmethod
    def constant(t: torch.Tensor) -> torch.Tensor:
        """φ(t) = [1]"""
        B, L = t.shape
        return torch.ones(B, L, 1, device=t.device)
    
    @staticmethod
    def polynomial(t: torch.Tensor, t_ref: torch.Tensor, degree: int = 2) -> torch.Tensor:
        """
        φ(t) = [1, Δt, Δt², ...]
        """
        B, L = t.shape
        device = t.device
        
        # Normalized Δt for numerical stability
        dt = t - t_ref.unsqueeze(1)
        dt_scale = dt.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-6)
        dt_norm = dt / dt_scale
        
        feats = [torch.ones(B, L, 1, device=device)]
        for d in range(1, degree + 1):
            feats.append((dt_norm ** d).unsqueeze(-1))
        
        return torch.cat(feats, dim=-1)
    
    @staticmethod
    def saturation(t: torch.Tensor, t_ref: torch.Tensor, 
                   base_rate: torch.Tensor,
                   scales: Tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0)) -> torch.Tensor:
        """
        φ(t) = [1, 1-e^(-λ₁Δt), 1-e^(-λ₂Δt), ...]
        
        Multi-scale saturation basis for INC_STABLE, DEC_STABLE
        """
        B, L = t.shape
        device = t.device
        
        # Δt from reference (boundary)
        dt = t - t_ref.unsqueeze(1)  # [B, L], can be negative for past
        
        feats = [torch.ones(B, L, 1, device=device)]
        
        for scale in scales:
            rate = base_rate * scale  # [B]
            # f(Δt) = 1 - exp(-λΔt)
            sat_term = 1.0 - torch.exp(-rate.unsqueeze(1) * dt)
            feats.append(sat_term.unsqueeze(-1))
        
        return torch.cat(feats, dim=-1)
    
    @staticmethod
    def fourier(t: torch.Tensor, omega: torch.Tensor, 
                num_harmonics: int = 4) -> torch.Tensor:
        """
        φ(t) = [1, sin(ωt), cos(ωt), sin(2ωt), cos(2ωt), ...]
        
        Fourier basis for OSCILLATION
        """
        B, L = t.shape
        device = t.device
        
        feats = [torch.ones(B, L, 1, device=device)]
        
        wt = omega.unsqueeze(1) * t  # [B, L]
        
        for h in range(1, num_harmonics + 1):
            feats.append(torch.sin(h * wt).unsqueeze(-1))
            feats.append(torch.cos(h * wt).unsqueeze(-1))
        
        return torch.cat(feats, dim=-1)

class BLRSolver:
    
    @staticmethod
    def solve(Phi_past: torch.Tensor, y_past: torch.Tensor,
              Phi_future: torch.Tensor,
              alpha: float, beta: float,
              min_var: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit BLR on past data and predict future.
        
        Posterior: p(w|y) = N(m_N, S_N)
            S_N = (αI + βΦᵀΦ)⁻¹
            m_N = βS_N Φᵀy
        
        Predictive: p(y*|x*) = N(μ*, σ²*)
            μ* = Φ* m_N
            σ²* = 1/β + diag(Φ* S_N Φ*ᵀ)
        
        Args:
            Phi_past: [B, N, K] - basis matrix for past
            y_past: [B, N] - past observations
            Phi_future: [B, F, K] - basis matrix for future
            
        Returns:
            mu: [B, F] - predictive mean
            var: [B, F] - predictive variance
        """
        B, N, K = Phi_past.shape
        device = Phi_past.device
        
        # Posterior covariance
        PhiTPhi = torch.bmm(Phi_past.transpose(1, 2), Phi_past)  # [B, K, K]
        S_inv = alpha * torch.eye(K, device=device).unsqueeze(0) + beta * PhiTPhi
        
        # Stable inversion
        try:
            L = torch.linalg.cholesky(S_inv)
            S = torch.cholesky_inverse(L)
        except:
            S = torch.linalg.pinv(S_inv)
        
        # Posterior mean
        PhiTy = torch.bmm(Phi_past.transpose(1, 2), y_past.unsqueeze(-1))  # [B, K, 1]
        m = beta * torch.bmm(S, PhiTy).squeeze(-1)  # [B, K]
        
        # Predictive distribution
        mu = torch.bmm(Phi_future, m.unsqueeze(-1)).squeeze(-1)  # [B, F]
        
        # Variance (diagonal only for efficiency)
        Phi_S = torch.bmm(Phi_future, S)  # [B, F, K]
        var = 1.0/beta + (Phi_S * Phi_future).sum(dim=-1)  # [B, F]
        var = var.clamp(min=min_var)
        
        return mu, var


class BLRTemplatePriorGenerator(nn.Module):
    
    def __init__(self, config: Optional[BLRPriorConfig] = None):
        super().__init__()
        self.config = config or BLRPriorConfig()
        
        # Learnable noise scale
        # self.log_noise_scale = nn.Parameter(
        #     torch.tensor(np.log(self.config.noise_scale))
        # )
        self.noise_scale = self.config.noise_scale
    
    # @property
    # def noise_scale(self) -> torch.Tensor:
    #     return torch.exp(self.log_noise_scale)
    
    def _extract_reference(self, past_times: torch.Tensor, 
                           past_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N = past_values.shape
        
        return {
            't_ref': past_times[:, -1],
            'y_ref': past_values[:, -1],
            't_range': past_times[:, -1] - past_times[:, 0],
        }
    
    def _compute_omega(self, period: torch.Tensor, 
                       t_range: torch.Tensor, 
                       n_points: int) -> torch.Tensor:
        dt = t_range / (n_points - 1)
        period_time = period * dt
        
        period_time = period_time.clamp(min=t_range * 0.05)
        
        omega = 2 * np.pi / period_time
        return omega
    
    def _get_saturation_rate(self, t_range: torch.Tensor) -> torch.Tensor:
        # return self.config.saturation_scales[2] / (t_range + 1e-6)
        return 1.0 / (t_range + 1e-6)
    
    def _build_basis(self, times: torch.Tensor, 
                     pattern_id: int,
                     ref: Dict[str, torch.Tensor],
                     omega: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_ref = ref['t_ref']
        t_range = ref['t_range']
        
        if pattern_id == PatternType.DIRECTLY_STABLE:
            return BasisLibrary.constant(times)
        
        elif pattern_id in [PatternType.INC_STABLE, PatternType.DEC_STABLE]:
            base_rate = self._get_saturation_rate(t_range)
            return BasisLibrary.saturation(
                times, t_ref, base_rate, self.config.saturation_scales
            )
        
        elif pattern_id == PatternType.OSCILLATION:
            assert omega is not None
            return BasisLibrary.fourier(times, omega, self.config.num_harmonics)
        
        elif pattern_id in [PatternType.INCREASING, PatternType.DECREASING]:
            return BasisLibrary.polynomial(times, t_ref, self.config.poly_degree)
        
        else:
            return BasisLibrary.constant(times)
    
    def _fit_and_predict(self, 
                         past_times: torch.Tensor,
                         past_values: torch.Tensor,
                         future_times: torch.Tensor,
                         pattern_id: int,
                         ref: Dict[str, torch.Tensor],
                         omega: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi_past = self._build_basis(past_times, pattern_id, ref, omega)
        Phi_future = self._build_basis(future_times, pattern_id, ref, omega)
        
        mu, var = BLRSolver.solve(
            Phi_past, past_values, Phi_future,
            self.config.alpha, self.config.beta, self.config.min_variance
        )
        
        return mu, var
    
    def _apply_continuity_constraint(self, 
                                      mu: torch.Tensor, 
                                      y_ref: torch.Tensor) -> torch.Tensor:
        offset = y_ref - mu[:, 0]
        return mu + offset.unsqueeze(1)
    
    def forward(self,
                past_times: torch.Tensor,      # [B, N]
                past_values: torch.Tensor,     # [B, N]
                future_times: torch.Tensor,    # [B, F]
                pattern_ids: torch.Tensor,     # [B]
                period: torch.Tensor          # [B]
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        B, N = past_values.shape
        F = future_times.shape[1]
        device = past_values.device

        ref = self._extract_reference(past_times, past_values)
        
        omega = self._compute_omega(period, ref['t_range'], N)
        
        mu = torch.zeros(B, F, device=device)
        var = torch.ones(B, F, device=device) * self.config.min_variance
        
        for pid in range(6):  # 0-5
            mask = (pattern_ids == pid)
            if not mask.any():
                continue
            
            ref_masked = {k: v[mask] for k, v in ref.items()}
            omega_masked = omega[mask] if pid == PatternType.OSCILLATION else None
            
            mu_p, var_p = self._fit_and_predict(
                past_times[mask],
                past_values[mask],
                future_times[mask],
                pid,
                ref_masked,
                omega_masked
            )
            
            mu[mask] = mu_p
            var[mask] = var_p
        
        mu = self._apply_continuity_constraint(mu, ref['y_ref'])
        
        # x0 = μ + σ * ε
        # std = torch.sqrt(var) * self.noise_scale
        # epsilon = torch.randn_like(mu)
        # x0 = mu + std * epsilon
        # assume the covariance is same as the normal distribution * scale
        # SIGMA_0 = 0.33852607011795044

        SIGMA_0 = self.noise_scale # best for specific settings
        epsilon = torch.randn_like(mu)
        x0 = mu + epsilon * SIGMA_0

        info = {
            'var': var,
            'std': torch.sqrt(var) * self.noise_scale,
            'epsilon': epsilon,
            'ref': ref,
            'omega': omega,
        }
        
        return x0, mu, info
