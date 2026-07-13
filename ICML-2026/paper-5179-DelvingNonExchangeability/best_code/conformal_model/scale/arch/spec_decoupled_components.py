"""Spectral-decoupled scale components.

This module contains the model-side building blocks:
- SpectralInputDecoupler: SGWT analysis that produces node-domain coefficients.
- SpectralExchangeableQuantileNet: low-frequency non-exchangeable backbone +
  high-frequency exchangeable statistics + low-conditioned gating.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from conformal_model.scale.arch.config import QuantileModelConfig
from conformal_model.scale.arch.spectral.sgwt import GraphWaveletKernelFactory, SpectralGraphWaveletTransform
from conformal_model.scale.arch.spectral.auto_tune import auto_tune_high_freq_cutoff
from conformal_model.scale.arch.backbones import LowFreqIdentityBackbone as LFBackbone


def _require_tod_dow(context_exog: torch.Tensor | None) -> torch.Tensor:
    """Return the last-two exogenous channels (tod/dow) with shape (B, T, 2)."""
    if context_exog is None or context_exog.dim() != 3 or context_exog.shape[-1] < 2:
        raise ValueError(
            "Backbone requires `context_exog` to include tod/dow scalars in the last 2 dims. "
            f"Got {None if context_exog is None else tuple(context_exog.shape)}."
        )
    return context_exog[..., -2:].contiguous()


def _inject_tod_dow(low_in: torch.Tensor, tod_dow: torch.Tensor) -> torch.Tensor:
    """Inject raw tod/dow as channels 1 and 2 expected by the backbone."""
    batch, steps, nodes, _ = low_in.shape
    tod = tod_dow[..., 0].unsqueeze(2).expand(batch, steps, nodes).unsqueeze(-1)
    dow = tod_dow[..., 1].unsqueeze(2).expand(batch, steps, nodes).unsqueeze(-1)

    low_aug = torch.cat([low_in, tod, dow], dim=-1)

    low_dim = int(low_in.shape[-1])
    perm = [0, low_dim, low_dim + 1] + list(range(1, low_dim))
    return low_aug[..., perm].contiguous()


def _high_frequency_stats(high_wave: torch.Tensor) -> torch.Tensor:
    """Compute time-exchangeable stats from high-frequency coefficients.

    Args:
        high_wave: (B, T, N, Fh)

    Returns:
        hf_stats: (B, 2*Fh, N, 1) where concat=[std, rms] across time dimension.
    """
    hf_std = torch.std(high_wave, dim=1, keepdim=True)
    hf_rms = torch.sqrt(torch.mean(high_wave**2, dim=1, keepdim=True))
    hf_stats = torch.cat([hf_std, hf_rms], dim=-1)
    return hf_stats.squeeze(1).permute(0, 2, 1).contiguous().unsqueeze(-1)


def _maybe_log_sgwt_energy(
    module: nn.Module,
    scaling: torch.Tensor,
    low_wave: torch.Tensor,
    high_wave: torch.Tensor,
    tag: str,
) -> None:
    """Print one-time SGWT energy diagnostics per module instance."""
    if getattr(module, "_sgwt_energy_logged", False):
        return
    def _energy_sum(x: torch.Tensor) -> float:
        if x is None or x.numel() == 0:
            return 0.0
        return float(torch.sum(x**2).item())

    scaling_energy = _energy_sum(scaling)
    low_band_energy = _energy_sum(low_wave)
    high_band_energy = _energy_sum(high_wave)
    low_total = scaling_energy + low_band_energy
    total = low_total + high_band_energy
    low_ratio = (low_total / total) if total > 0 else 0.0
    high_ratio = (high_band_energy / total) if total > 0 else 0.0
    print(
        f"[SGWT:{tag}] scaling_energy={scaling_energy:.6f} "
        f"low_band_energy={low_band_energy:.6f} high_band_energy={high_band_energy:.6f} "
        f"low_ratio={low_ratio:.6f} high_ratio={high_ratio:.6f}"
    )
    setattr(module, "_sgwt_energy_logged", True)


class SpectralInputDecoupler(nn.Module):
    """SGWT coefficient extractor (analysis)."""

    def __init__(
        self,
        adj_matrix: np.ndarray,
        device: torch.device,
        *,
        n_scales: int = 4,
        kernel_type: str = "meyer",
        n_high_scales: int | None = None,
        auto_tune_signal: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()

        sgwt = SpectralGraphWaveletTransform(adj_matrix)
        factory = GraphWaveletKernelFactory(sgwt.lmax)
        g_func_np, h_func_np, scales = factory.get_kernels(str(kernel_type), int(n_scales))

        evals = np.asarray(sgwt.evals, dtype=float)
        evecs = np.asarray(sgwt.evecs, dtype=float)

        h_resp = np.asarray(h_func_np(evals), dtype=float)
        g_resp = np.stack(
            [np.asarray(g_func_np(float(s), evals), dtype=float) for s in scales], axis=0
        )

        num_scales = int(g_resp.shape[0])
        if n_high_scales is None:
            if auto_tune_signal is not None:
                try:
                    tuned = auto_tune_high_freq_cutoff(
                        adj_matrix=np.asarray(adj_matrix),
                        signal_data=auto_tune_signal,
                        max_scales=num_scales,
                        kernel_type=str(kernel_type),
                        verbose=False,
                    )
                    n_high_scales = int(tuned.get("suggested_n_high_scales", num_scales // 2))
                except Exception:
                    n_high_scales = num_scales // 2
            else:
                n_high_scales = num_scales // 2
        n_high_scales = int(max(0, min(num_scales, int(n_high_scales))))

        self._high_scale_idx: tuple[int, ...] = tuple(range(0, n_high_scales))
        self._low_scale_idx: tuple[int, ...] = tuple(range(n_high_scales, num_scales))

        self.register_buffer("evecs", torch.from_numpy(evecs).float().to(device))
        self.register_buffer("h_resp", torch.from_numpy(h_resp).float().to(device))
        self.register_buffer("g_resp", torch.from_numpy(g_resp).float().to(device))

        self.kernel_type = str(kernel_type)
        self.scales = [float(s) for s in list(scales)]
        self.n_scales = int(num_scales)

    @property
    def low_scale_idx(self) -> tuple[int, ...]:
        return self._low_scale_idx

    @property
    def high_scale_idx(self) -> tuple[int, ...]:
        return self._high_scale_idx

    def forward_coeffs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute SGWT scaling + per-scale wavelet coefficients.

        Args:
            x: (B, T, N) or (B, T, N, C)

        Returns:
            scaling:  (B, T, N, C)
            wavelets: (B, T, N, C, M)
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        batch, steps, nodes, channels = x.shape
        if nodes != int(self.evecs.shape[0]):
            raise ValueError(
                f"Input num_nodes={nodes} does not match graph size {int(self.evecs.shape[0])}"
            )

        x_perm = x.permute(0, 1, 3, 2)
        x_hat = torch.matmul(x_perm, self.evecs)

        x_hat_scaling = x_hat * self.h_resp
        scaling_ctn = torch.matmul(x_hat_scaling, self.evecs.t())

        wavelet_hat = x_hat.unsqueeze(3) * self.g_resp.view(1, 1, 1, self.n_scales, nodes)
        wavelets_ctmn = torch.matmul(wavelet_hat, self.evecs.t())

        scaling = scaling_ctn.permute(0, 1, 3, 2).contiguous()
        wavelets = wavelets_ctmn.permute(0, 1, 4, 2, 3).contiguous()
        return scaling, wavelets


class SpectralExchangeableQuantileNet(nn.Module):
    """Residual-quantile predictor with SGWT low/high decoupling."""

    def __init__(
        self,
        *,
        cfg: QuantileModelConfig,
        adjacency: np.ndarray,
        gso: torch.Tensor,
        n_scales: int = 4,
        kernel_type: str = "meyer",
        n_high_scales: int | None = None,
        auto_tune_signal: torch.Tensor | np.ndarray | None = None,
        backbone_out_dim: int = 128,
        backbone_kwargs: dict | None = None,
        enable_gating: bool = True,
        high_stats_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.horizon = int(cfg.horizon)
        self.num_nodes = int(cfg.num_nodes)
        self.q_dim = len(cfg.quantiles)
        self.enable_gating = bool(enable_gating)

        backbone_kwargs = dict(backbone_kwargs or {})

        backbone_time_dim = 2
        c_in = int(cfg.context_channels)

        self.decoupler = SpectralInputDecoupler(
            adjacency,
            device=gso.device,
            n_scales=n_scales,
            kernel_type=kernel_type,
            n_high_scales=n_high_scales,
            auto_tune_signal=auto_tune_signal,
        )

        n_low = len(self.decoupler.low_scale_idx)
        n_high = len(self.decoupler.high_scale_idx)
        self.n_low_scales = int(n_low)
        self.n_high_scales = int(n_high)

        low_c_in = c_in * (1 + n_low)
        model_args = {
            "num_nodes": int(cfg.num_nodes),
            "node_dim": int(backbone_kwargs.get("node_dim", 128)),
            "input_len": int(cfg.input_length),
            "input_dim": int(low_c_in + backbone_time_dim),
            "embed_dim": int(backbone_kwargs.get("embed_dim", 64)),
            "output_len": int(backbone_out_dim),
            "num_layer": int(backbone_kwargs.get("num_layer", 6)),
            "temp_dim_tid": int(backbone_kwargs.get("temp_dim_tid", 128)),
            "temp_dim_diw": int(backbone_kwargs.get("temp_dim_diw", 128)),
            "time_of_day_size": int(backbone_kwargs.get("time_of_day_size", 288)),
            "day_of_week_size": int(backbone_kwargs.get("day_of_week_size", 7)),
        }
        self.backbone_low = LFBackbone(**model_args)

        self.high_feat_dim = int(c_in * max(1, n_high))
        self.high_stat_dim = int(self.high_feat_dim * 2)

        self.low_out_proj = nn.Conv2d(
            in_channels=int(backbone_out_dim),
            out_channels=self.horizon * self.q_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        hidden_dim = int(high_stats_hidden_dim or self.high_stat_dim)
        self.high_stats_mlp = nn.Sequential(
            nn.Conv2d(self.high_stat_dim, hidden_dim, kernel_size=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.horizon * self.q_dim, kernel_size=(1, 1), bias=True),
        )

        if self.enable_gating:
            self.gate_proj = nn.Conv2d(
                in_channels=int(backbone_out_dim),
                out_channels=self.horizon * self.q_dim,
                kernel_size=(1, 1),
                bias=True,
            )
        else:
            self.gate_proj = None

    def forward(self, context: torch.Tensor, context_exog: torch.Tensor) -> torch.Tensor:
        """Predict residual quantiles.

        Args:
            context: (B, T, N) or (B, T, N, C)
            context_exog: (B, T, exog_dim) where last two channels are tod/dow.
        """
        if context.dim() == 3:
            context = context.unsqueeze(-1)
        if context.dim() != 4:
            raise ValueError(f"Expected context (B, T, N, C), got shape {tuple(context.shape)}")

        batch, steps, nodes, _ = context.shape
        tod_dow = _require_tod_dow(context_exog)

        scaling, wavelets = self.decoupler.forward_coeffs(context)

        low_idx = self.decoupler.low_scale_idx
        high_idx = self.decoupler.high_scale_idx

        if len(low_idx) > 0:
            low_wave = wavelets[:, :, :, :, low_idx].reshape(batch, steps, nodes, -1)
            low_in = torch.cat([scaling, low_wave], dim=-1)
        else:
            low_wave = torch.zeros(
                batch, steps, nodes, 0, device=wavelets.device, dtype=wavelets.dtype
            )
            low_in = scaling

        low_in = _inject_tod_dow(low_in, tod_dow)

        low_features = self.backbone_low(
            history_data=low_in,
            future_data=None,
            batch_seen=0,
            epoch=0,
            train=bool(self.training),
        )

        if low_features.dim() == 4:
            low_features = low_features[..., -1:].contiguous()

        low_out = self.low_out_proj(low_features).squeeze(-1)

        if len(high_idx) > 0:
            high_wave = wavelets[:, :, :, :, high_idx].reshape(batch, steps, nodes, -1)
        else:
            high_wave = torch.zeros(
                batch, steps, nodes, self.high_feat_dim, device=wavelets.device, dtype=wavelets.dtype
            )

        _maybe_log_sgwt_energy(self, scaling, low_wave, high_wave, "scale")

        hf_stats = _high_frequency_stats(high_wave)
        high_out = self.high_stats_mlp(hf_stats).squeeze(-1)

        if self.enable_gating and self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(low_features).squeeze(-1))
            fused = low_out + gate * high_out
        else:
            fused = low_out + high_out

        out = fused.view(batch, self.horizon, self.q_dim, nodes).permute(0, 1, 3, 2).contiguous()
        return out


class SpectralAllNNQuantileNet(nn.Module):
    """SGWT decoupling with both low/high branches modeled by backbones."""

    def __init__(
        self,
        *,
        cfg: QuantileModelConfig,
        adjacency: np.ndarray,
        gso: torch.Tensor,
        n_scales: int = 4,
        kernel_type: str = "meyer",
        n_high_scales: int | None = None,
        auto_tune_signal: torch.Tensor | np.ndarray | None = None,
        backbone_out_dim: int = 128,
        backbone_kwargs: dict | None = None,
        enable_gating: bool = True,
        high_stats_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.horizon = int(cfg.horizon)
        self.num_nodes = int(cfg.num_nodes)
        self.q_dim = len(cfg.quantiles)
        self.enable_gating = bool(enable_gating)

        backbone_kwargs = dict(backbone_kwargs or {})

        backbone_time_dim = 2
        c_in = int(cfg.context_channels)

        self.decoupler = SpectralInputDecoupler(
            adjacency,
            device=gso.device,
            n_scales=n_scales,
            kernel_type=kernel_type,
            n_high_scales=n_high_scales,
            auto_tune_signal=auto_tune_signal,
        )

        n_low = len(self.decoupler.low_scale_idx)
        n_high = len(self.decoupler.high_scale_idx)
        self.n_low_scales = int(n_low)
        self.n_high_scales = int(n_high)

        low_c_in = c_in * (1 + n_low)
        high_c_in = c_in * max(1, n_high)

        model_args = {
            "num_nodes": int(cfg.num_nodes),
            "node_dim": int(backbone_kwargs.get("node_dim", 128)),
            "input_len": int(cfg.input_length),
            "input_dim": int(low_c_in + backbone_time_dim),
            "embed_dim": int(backbone_kwargs.get("embed_dim", 64)),
            "output_len": int(backbone_out_dim),
            "num_layer": int(backbone_kwargs.get("num_layer", 6)),
            "temp_dim_tid": int(backbone_kwargs.get("temp_dim_tid", 128)),
            "temp_dim_diw": int(backbone_kwargs.get("temp_dim_diw", 128)),
            "time_of_day_size": int(backbone_kwargs.get("time_of_day_size", 288)),
            "day_of_week_size": int(backbone_kwargs.get("day_of_week_size", 7)),
        }
        self.backbone_low = LFBackbone(**model_args)

        high_model_args = dict(model_args)
        high_model_args["input_dim"] = int(high_c_in + backbone_time_dim)
        self.backbone_high = LFBackbone(**high_model_args)

        self.high_feat_dim = int(high_c_in)

        self.low_out_proj = nn.Conv2d(
            in_channels=int(backbone_out_dim),
            out_channels=self.horizon * self.q_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.high_out_proj = nn.Conv2d(
            in_channels=int(backbone_out_dim),
            out_channels=self.horizon * self.q_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        if self.enable_gating:
            self.gate_proj = nn.Conv2d(
                in_channels=int(backbone_out_dim),
                out_channels=self.horizon * self.q_dim,
                kernel_size=(1, 1),
                bias=True,
            )
        else:
            self.gate_proj = None

    def forward(self, context: torch.Tensor, context_exog: torch.Tensor) -> torch.Tensor:
        if context.dim() == 3:
            context = context.unsqueeze(-1)
        if context.dim() != 4:
            raise ValueError(f"Expected context (B, T, N, C), got shape {tuple(context.shape)}")

        batch, steps, nodes, _ = context.shape
        _ = _require_tod_dow(context_exog)
        tod_dow = _require_tod_dow(context_exog)

        scaling, wavelets = self.decoupler.forward_coeffs(context)

        low_idx = self.decoupler.low_scale_idx
        high_idx = self.decoupler.high_scale_idx

        if len(low_idx) > 0:
            low_wave = wavelets[:, :, :, :, low_idx].reshape(batch, steps, nodes, -1)
            low_in = torch.cat([scaling, low_wave], dim=-1)
        else:
            low_wave = torch.zeros(
                batch, steps, nodes, 0, device=wavelets.device, dtype=wavelets.dtype
            )
            low_in = scaling

        low_in = _inject_tod_dow(low_in, tod_dow)

        low_features = self.backbone_low(
            history_data=low_in,
            future_data=None,
            batch_seen=0,
            epoch=0,
            train=bool(self.training),
        )

        if low_features.dim() == 4:
            low_features = low_features[..., -1:].contiguous()

        low_out = self.low_out_proj(low_features).squeeze(-1)

        if len(high_idx) > 0:
            high_wave = wavelets[:, :, :, :, high_idx].reshape(batch, steps, nodes, -1)
        else:
            high_wave = torch.zeros(
                batch, steps, nodes, self.high_feat_dim, device=wavelets.device, dtype=wavelets.dtype
            )

        _maybe_log_sgwt_energy(self, scaling, low_wave, high_wave, "scale_all_nn")

        high_in = _inject_tod_dow(high_wave, tod_dow)

        high_features = self.backbone_high(
            history_data=high_in,
            future_data=None,
            batch_seen=0,
            epoch=0,
            train=bool(self.training),
        )

        if high_features.dim() == 4:
            high_features = high_features[..., -1:].contiguous()

        high_out = self.high_out_proj(high_features).squeeze(-1)

        if self.enable_gating and self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(low_features).squeeze(-1))
            fused = low_out + gate * high_out
        else:
            fused = low_out + high_out

        out = fused.view(batch, self.horizon, self.q_dim, nodes).permute(0, 1, 3, 2).contiguous()
        return out


class SpectralAllStatsQuantileNet(nn.Module):
    """SGWT decoupling with high-frequency stats only (no low conditioning)."""

    def __init__(
        self,
        *,
        cfg: QuantileModelConfig,
        adjacency: np.ndarray,
        gso: torch.Tensor,
        n_scales: int = 4,
        kernel_type: str = "meyer",
        n_high_scales: int | None = None,
        auto_tune_signal: torch.Tensor | np.ndarray | None = None,
        backbone_out_dim: int = 128,
        backbone_kwargs: dict | None = None,
        enable_gating: bool = True,
        high_stats_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.horizon = int(cfg.horizon)
        self.num_nodes = int(cfg.num_nodes)
        self.q_dim = len(cfg.quantiles)
        self.enable_gating = bool(enable_gating)

        backbone_kwargs = dict(backbone_kwargs or {})

        c_in = int(cfg.context_channels)

        self.decoupler = SpectralInputDecoupler(
            adjacency,
            device=gso.device,
            n_scales=n_scales,
            kernel_type=kernel_type,
            n_high_scales=n_high_scales,
            auto_tune_signal=auto_tune_signal,
        )

        n_low = len(self.decoupler.low_scale_idx)
        n_high = len(self.decoupler.high_scale_idx)
        self.n_low_scales = int(n_low)
        self.n_high_scales = int(n_high)

        high_c_in = c_in * max(1, n_high)
        self.high_feat_dim = int(high_c_in)
        self.high_stat_dim = int(self.high_feat_dim * 2)

        hidden_dim = int(high_stats_hidden_dim or self.high_stat_dim)
        self.high_stats_mlp = nn.Sequential(
            nn.Conv2d(self.high_stat_dim, hidden_dim, kernel_size=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.horizon * self.q_dim, kernel_size=(1, 1), bias=True),
        )

    def forward(self, context: torch.Tensor, context_exog: torch.Tensor) -> torch.Tensor:
        if context.dim() == 3:
            context = context.unsqueeze(-1)
        if context.dim() != 4:
            raise ValueError(f"Expected context (B, T, N, C), got shape {tuple(context.shape)}")

        batch, steps, nodes, _ = context.shape

        scaling, wavelets = self.decoupler.forward_coeffs(context)

        low_idx = self.decoupler.low_scale_idx
        high_idx = self.decoupler.high_scale_idx

        if len(low_idx) > 0:
            low_wave = wavelets[:, :, :, :, low_idx].reshape(batch, steps, nodes, -1)
            low_in = torch.cat([scaling, low_wave], dim=-1)
        else:
            low_wave = torch.zeros(
                batch, steps, nodes, 0, device=wavelets.device, dtype=wavelets.dtype
            )
            low_in = scaling

        if len(high_idx) > 0:
            high_wave = wavelets[:, :, :, :, high_idx].reshape(batch, steps, nodes, -1)
        else:
            high_wave = torch.zeros(
                batch, steps, nodes, self.high_feat_dim, device=wavelets.device, dtype=wavelets.dtype
            )

        _maybe_log_sgwt_energy(self, scaling, low_wave, high_wave, "scale_wo_low")

        high_stats = _high_frequency_stats(high_wave)

        high_out = self.high_stats_mlp(high_stats).squeeze(-1)
        out = high_out.view(batch, self.horizon, self.q_dim, nodes).permute(0, 1, 3, 2).contiguous()
        return out


class SpectralLowOnlyQuantileNet(nn.Module):
    """SGWT decoupling with low-frequency backbone only (no high branch)."""

    def __init__(
        self,
        *,
        cfg: QuantileModelConfig,
        adjacency: np.ndarray,
        gso: torch.Tensor,
        n_scales: int = 4,
        kernel_type: str = "meyer",
        n_high_scales: int | None = None,
        auto_tune_signal: torch.Tensor | np.ndarray | None = None,
        backbone_out_dim: int = 128,
        backbone_kwargs: dict | None = None,
        enable_gating: bool = False,
        high_stats_hidden_dim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()

        self.horizon = int(cfg.horizon)
        self.num_nodes = int(cfg.num_nodes)
        self.q_dim = len(cfg.quantiles)
        _ = bool(enable_gating)

        backbone_kwargs = dict(backbone_kwargs or {})

        backbone_time_dim = 2
        c_in = int(cfg.context_channels)

        self.decoupler = SpectralInputDecoupler(
            adjacency,
            device=gso.device,
            n_scales=n_scales,
            kernel_type=kernel_type,
            n_high_scales=n_high_scales,
            auto_tune_signal=auto_tune_signal,
        )

        n_low = len(self.decoupler.low_scale_idx)
        self.n_low_scales = int(n_low)
        n_high = len(self.decoupler.high_scale_idx)
        self.n_high_scales = int(n_high)
        self.high_feat_dim = int(c_in * max(1, n_high))

        low_c_in = c_in * (1 + n_low)
        model_args = {
            "num_nodes": int(cfg.num_nodes),
            "node_dim": int(backbone_kwargs.get("node_dim", 128)),
            "input_len": int(cfg.input_length),
            "input_dim": int(low_c_in + backbone_time_dim),
            "embed_dim": int(backbone_kwargs.get("embed_dim", 64)),
            "output_len": int(backbone_out_dim),
            "num_layer": int(backbone_kwargs.get("num_layer", 6)),
            "temp_dim_tid": int(backbone_kwargs.get("temp_dim_tid", 128)),
            "temp_dim_diw": int(backbone_kwargs.get("temp_dim_diw", 128)),
            "time_of_day_size": int(backbone_kwargs.get("time_of_day_size", 288)),
            "day_of_week_size": int(backbone_kwargs.get("day_of_week_size", 7)),
        }
        self.backbone_low = LFBackbone(**model_args)

        self.low_out_proj = nn.Conv2d(
            in_channels=int(backbone_out_dim),
            out_channels=self.horizon * self.q_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, context: torch.Tensor, context_exog: torch.Tensor) -> torch.Tensor:
        if context.dim() == 3:
            context = context.unsqueeze(-1)
        if context.dim() != 4:
            raise ValueError(f"Expected context (B, T, N, C), got shape {tuple(context.shape)}")

        batch, steps, nodes, _ = context.shape
        tod_dow = _require_tod_dow(context_exog)

        scaling, wavelets = self.decoupler.forward_coeffs(context)
        low_idx = self.decoupler.low_scale_idx
        high_idx = self.decoupler.high_scale_idx

        if len(low_idx) > 0:
            low_wave = wavelets[:, :, :, :, low_idx].reshape(batch, steps, nodes, -1)
            low_in = torch.cat([scaling, low_wave], dim=-1)
        else:
            low_wave = torch.zeros(
                batch, steps, nodes, 0, device=wavelets.device, dtype=wavelets.dtype
            )
            low_in = scaling

        low_in = _inject_tod_dow(low_in, tod_dow)

        if len(high_idx) > 0:
            high_wave = wavelets[:, :, :, :, high_idx].reshape(batch, steps, nodes, -1)
        else:
            high_wave = torch.zeros(
                batch, steps, nodes, self.high_feat_dim, device=wavelets.device, dtype=wavelets.dtype
            )

        _maybe_log_sgwt_energy(self, scaling, low_wave, high_wave, "scale_wo_high")

        low_features = self.backbone_low(
            history_data=low_in,
            future_data=None,
            batch_seen=0,
            epoch=0,
            train=bool(self.training),
        )

        if low_features.dim() == 4:
            low_features = low_features[..., -1:].contiguous()

        low_out = self.low_out_proj(low_features).squeeze(-1)
        out = low_out.view(batch, self.horizon, self.q_dim, nodes).permute(0, 1, 3, 2).contiguous()
        return out


class NoSpectralExchangeableQuantileNet(nn.Module):
    """No-SGWT baseline with low/backbone + high/stats branches."""

    def __init__(
        self,
        *,
        cfg: QuantileModelConfig,
        adjacency: np.ndarray,
        gso: torch.Tensor | None = None,
        n_scales: int = 4,
        kernel_type: str = "meyer",
        n_high_scales: int | None = None,
        backbone_out_dim: int = 128,
        backbone_kwargs: dict | None = None,
        enable_gating: bool = True,
        **_: object,
    ) -> None:
        super().__init__()

        self.horizon = int(cfg.horizon)
        self.num_nodes = int(cfg.num_nodes)
        self.q_dim = len(cfg.quantiles)
        self.enable_gating = bool(enable_gating)

        backbone_kwargs = dict(backbone_kwargs or {})

        backbone_time_dim = 2
        c_in = int(cfg.context_channels)

        model_args = {
            "num_nodes": int(cfg.num_nodes),
            "node_dim": int(backbone_kwargs.get("node_dim", 128)),
            "input_len": int(cfg.input_length),
            "input_dim": int(c_in + backbone_time_dim),
            "embed_dim": int(backbone_kwargs.get("embed_dim", 64)),
            "output_len": int(backbone_out_dim),
            "num_layer": int(backbone_kwargs.get("num_layer", 6)),
            "temp_dim_tid": int(backbone_kwargs.get("temp_dim_tid", 128)),
            "temp_dim_diw": int(backbone_kwargs.get("temp_dim_diw", 128)),
            "time_of_day_size": int(backbone_kwargs.get("time_of_day_size", 288)),
            "day_of_week_size": int(backbone_kwargs.get("day_of_week_size", 7)),
        }
        self.backbone_low = LFBackbone(**model_args)

        self.high_feat_dim = int(c_in)
        self.high_stat_dim = int(self.high_feat_dim * 2)

        self.low_out_proj = nn.Conv2d(
            in_channels=int(backbone_out_dim),
            out_channels=self.horizon * self.q_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.high_out_proj = nn.Conv2d(
            in_channels=self.high_stat_dim,
            out_channels=self.horizon * self.q_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        if self.enable_gating:
            self.gate_proj = nn.Conv2d(
                in_channels=int(backbone_out_dim),
                out_channels=self.horizon * self.q_dim,
                kernel_size=(1, 1),
                bias=True,
            )
        else:
            self.gate_proj = None

    def forward(self, context: torch.Tensor, context_exog: torch.Tensor) -> torch.Tensor:
        if context.dim() == 3:
            context = context.unsqueeze(-1)
        if context.dim() != 4:
            raise ValueError(f"Expected context (B, T, N, C), got shape {tuple(context.shape)}")

        batch, steps, nodes, _ = context.shape
        tod_dow = _require_tod_dow(context_exog)

        low_in = _inject_tod_dow(context, tod_dow)

        low_features = self.backbone_low(
            history_data=low_in,
            future_data=None,
            batch_seen=0,
            epoch=0,
            train=bool(self.training),
        )

        if low_features.dim() == 4:
            low_features = low_features[..., -1:].contiguous()

        low_out = self.low_out_proj(low_features).squeeze(-1)

        high_wave = context
        hf_stats = _high_frequency_stats(high_wave)
        high_out = self.high_out_proj(hf_stats).squeeze(-1)

        if self.enable_gating and self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(low_features).squeeze(-1))
            fused = low_out + gate * high_out
        else:
            fused = low_out + high_out

        out = fused.view(batch, self.horizon, self.q_dim, nodes).permute(0, 1, 3, 2).contiguous()
        return out


_MODEL_REGISTRY = {
    "scale": SpectralExchangeableQuantileNet,
    "scaleo3": SpectralExchangeableQuantileNet,
    "scale_wo_high": SpectralLowOnlyQuantileNet,
    "scale_wo_low": SpectralAllStatsQuantileNet,
    "scale_no_sgwt": NoSpectralExchangeableQuantileNet,
    "scale_nosgwt": NoSpectralExchangeableQuantileNet,
}


def resolve_scale_model(name: str | None):
    if not name:
        return SpectralExchangeableQuantileNet
    return _MODEL_REGISTRY.get(str(name).lower(), SpectralExchangeableQuantileNet)
