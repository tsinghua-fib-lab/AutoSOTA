from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch as PyGBatch
except Exception:
    PyGBatch = None

from augmentation_gc import BernoulliEdgeAugmentor
from rade_convs_gc import (
    BatchedGraphCache,
    _delta_E_inv_caseA,
    _delta_E_inv_caseB,
    _delta_E_inv_sqrt_caseA,
    _delta_E_inv_sqrt_caseB,
    _scatter_add_rows,
)


P_PROBABILITY_UPPER: float = 1.0 - 1e-6
Q_PROBABILITY_UPPER: float = 1.0


def _grad_norm(grads) -> float:
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total += float((grad * grad).sum().item())
    return math.sqrt(max(total, 0.0))


def _dot(grads_a, grads_b) -> float:
    total = 0.0
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        total += float((grad_a * grad_b).sum().item())
    return total


def _freeze_batchnorm_buffers(model: nn.Module) -> List[Tuple[nn.Module, bool]]:
    states = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            states.append((module, module.training))
            module.eval()
    return states


def _restore_batchnorm_buffers(states: List[Tuple[nn.Module, bool]]) -> None:
    for module, was_training in states:
        module.train(was_training)


@dataclass
class PQTunerGCConfig:
    expected_add_ratio_scale: float = 0.0
    deletion_penalty_lambda: float = 0.0
    densification_penalty_lambda: float = 1.0
    p_max: float = P_PROBABILITY_UPPER
    optimize_rho: bool = False
    eps: float = 1e-12
    seed: int = 0
    data_anchor: str = "clean"
    adam_lr: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    gat_denom_penalty_lambda: float = 1.0
    gat_denom_cv2_threshold: float = 1.0
    gat_corr_penalty_lambda: float = 50.0
    gat_corr_threshold: float = 2.0


class PQGradNormTunerGC:
    def __init__(self, *, gnn: str, rade_variant: str, pooling: str, cfg: PQTunerGCConfig):
        self.cfg = cfg
        self.gnn = str(gnn).lower().strip()
        self.rade_variant = str(rade_variant).lower().strip()
        self.pooling = str(pooling).lower().strip()
        self.data_anchor = str(cfg.data_anchor).lower().strip()
        if self.gnn not in {"gin", "gcn", "gat"}:
            raise ValueError("gnn must be 'gin', 'gcn', or 'gat'")
        if self.rade_variant not in {"rade-of", "rade-ofs"}:
            raise ValueError("rade_variant must be 'rade-of' or 'rade-ofs'")
        if self.pooling not in {"sum", "mean"}:
            raise ValueError("pooling must be 'sum' or 'mean'")
        if self.data_anchor not in {"clean", "aug"}:
            raise ValueError("data_anchor must be 'clean' or 'aug'")
        self.optimize_rho = bool(getattr(cfg, "optimize_rho", False))
        self.expected_add_ratio_scale = max(0.0, float(getattr(cfg, "expected_add_ratio_scale", 0.0)))
        self.p_upper = min(float(getattr(cfg, "p_max", P_PROBABILITY_UPPER)), P_PROBABILITY_UPPER)
        if not (0.0 <= self.p_upper < 1.0):
            raise ValueError(f"p_max must be in [0, 1). Got {self.p_upper}")
        self.gat_denom_penalty_lambda = float(getattr(cfg, "gat_denom_penalty_lambda", 0.0))
        if self.gat_denom_penalty_lambda < 0.0:
            raise ValueError(f"gat_denom_penalty_lambda must be >= 0. Got {self.gat_denom_penalty_lambda}")
        self.gat_denom_cv2_threshold = float(getattr(cfg, "gat_denom_cv2_threshold", 1.0))
        if self.gat_denom_cv2_threshold < 0.0:
            raise ValueError(f"gat_denom_cv2_threshold must be >= 0. Got {self.gat_denom_cv2_threshold}")
        self.gat_corr_penalty_lambda = float(getattr(cfg, "gat_corr_penalty_lambda", 0.0))
        if self.gat_corr_penalty_lambda < 0.0:
            raise ValueError(f"gat_corr_penalty_lambda must be >= 0. Got {self.gat_corr_penalty_lambda}")
        self.gat_corr_threshold = float(getattr(cfg, "gat_corr_threshold", 1.0))
        if self.gat_corr_threshold < 1.0:
            raise ValueError(f"gat_corr_threshold must be >= 1. Got {self.gat_corr_threshold}")

        self._adam_p: Optional[nn.Parameter] = None
        self._adam_q: Optional[nn.Parameter] = None
        self._adam_opt: Optional[torch.optim.Optimizer] = None
        self._adam_device: Optional[torch.device] = None

    def _rho_from_q_float(self, q: float) -> float:
        return float(q) * float(self.expected_add_ratio_scale)

    def _rho_from_q_tensor(self, q: torch.Tensor) -> torch.Tensor:
        return q * q.new_tensor(float(self.expected_add_ratio_scale))

    def _q_from_rho_float(self, rho: float) -> float:
        scale = float(self.expected_add_ratio_scale)
        if scale <= 0.0:
            return 0.0
        return float(rho) / scale

    def _q_from_rho_tensor(self, rho: torch.Tensor) -> torch.Tensor:
        scale = float(self.expected_add_ratio_scale)
        if scale <= 0.0:
            return rho.new_zeros(())
        return rho / rho.new_tensor(scale)

    def _adam_add_parameter_upper(self) -> float:
        if self.optimize_rho:
            return float(self._rho_from_q_float(Q_PROBABILITY_UPPER))
        return Q_PROBABILITY_UPPER

    def _add_parameter_name(self) -> str:
        return "rho" if self.optimize_rho else "q"

    def _add_parameter_from_q_float(self, q: float) -> float:
        if self.optimize_rho:
            return self._rho_from_q_float(q)
        return float(q)

    def _q_from_add_parameter_tensor(self, add_param: torch.Tensor) -> torch.Tensor:
        if self.optimize_rho:
            return self._q_from_rho_tensor(add_param)
        return add_param

    def _densification_penalty_float(self, rho: float) -> float:
        lam = float(getattr(self.cfg, "densification_penalty_lambda", 0.0))
        if lam <= 0.0:
            return 0.0
        rho = float(rho)
        return lam * rho * rho

    def _densification_penalty_tensor(self, rho: torch.Tensor) -> torch.Tensor:
        lam = float(getattr(self.cfg, "densification_penalty_lambda", 0.0))
        if lam <= 0.0:
            return rho.new_zeros(())
        return rho.new_tensor(lam) * rho * rho

    def _deletion_penalty_float(self, p: float) -> float:
        lam = float(getattr(self.cfg, "deletion_penalty_lambda", 0.0))
        if lam <= 0.0:
            return 0.0
        p = float(p)
        return lam * p * p

    def _deletion_penalty_tensor(self, p: torch.Tensor) -> torch.Tensor:
        lam = float(getattr(self.cfg, "deletion_penalty_lambda", 0.0))
        if lam <= 0.0:
            return p.new_zeros(())
        return p.new_tensor(lam) * p * p

    def _gat_denominator_cv2_values(self, p, q) -> List[torch.Tensor]:
        if self.gnn != "gat" or self.gat_denom_penalty_lambda <= 0.0:
            return []
        modules = getattr(self, "_gat_modules", None) or []
        values = []
        for module in modules:
            proxy = getattr(module, "gat_denominator_cv2_proxy", None)
            if not callable(proxy):
                continue
            cv2 = proxy(p=p, q=q, eps=float(self.cfg.eps))
            if cv2.numel() == 0:
                continue
            values.append(torch.nan_to_num(cv2.reshape(-1), nan=0.0, posinf=1.0e6, neginf=0.0))
        return values

    def _gat_denominator_penalty_float(self, p: float, q: float) -> float:
        values = self._gat_denominator_cv2_values(float(p), float(q))
        if not values:
            return 0.0
        cv2 = torch.cat([value.detach() for value in values], dim=0)
        excess = torch.clamp_min(cv2 - float(self.gat_denom_cv2_threshold), 0.0)
        return float(self.gat_denom_penalty_lambda) * float(excess.square().mean().item())

    def _gat_denominator_penalty_tensor(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        values = self._gat_denominator_cv2_values(p, q)
        if not values:
            return p.new_zeros(())
        penalties = []
        threshold = p.new_tensor(float(self.gat_denom_cv2_threshold))
        for cv2 in values:
            cv2 = cv2.to(device=p.device, dtype=p.dtype)
            excess = torch.clamp_min(cv2 - threshold, 0.0)
            penalties.append(excess.square().mean())
        return p.new_tensor(float(self.gat_denom_penalty_lambda)) * torch.stack(penalties).mean()

    def _gat_denominator_stats_float(self, p: float, q: float) -> Tuple[float, float, float]:
        values = self._gat_denominator_cv2_values(float(p), float(q))
        if not values:
            return 0.0, 0.0, 0.0
        cv2 = torch.cat([value.detach() for value in values], dim=0)
        excess = torch.clamp_min(cv2 - float(self.gat_denom_cv2_threshold), 0.0)
        penalty = float(self.gat_denom_penalty_lambda) * float(excess.square().mean().item())
        return penalty, float(cv2.mean().item()), float(cv2.max().item())

    def _gat_correction_gain_values(self, p, q) -> List[torch.Tensor]:
        if self.gnn != "gat" or self.gat_corr_penalty_lambda <= 0.0:
            return []
        modules = getattr(self, "_gat_modules", None) or []
        values = []
        for module in modules:
            proxy = getattr(module, "gat_correction_gain_proxy", None)
            if not callable(proxy):
                continue
            gain = proxy(p=p, q=q, eps=float(self.cfg.eps))
            if gain.numel() == 0:
                continue
            values.append(torch.nan_to_num(gain.reshape(-1), nan=0.0, posinf=1.0e6, neginf=0.0))
        return values

    def _gat_correction_penalty_from_excess_tensor(self, excess: torch.Tensor) -> torch.Tensor:
        if excess.numel() == 0:
            return excess.new_zeros(())
        mean_part = excess.square().mean()
        max_part = excess.max().square()
        return excess.new_tensor(float(self.gat_corr_penalty_lambda)) * (mean_part + max_part)

    def _gat_correction_penalty_float(self, p: float, q: float) -> float:
        values = self._gat_correction_gain_values(float(p), float(q))
        if not values:
            return 0.0
        gain = torch.cat([value.detach() for value in values], dim=0)
        excess = torch.clamp_min(gain - float(self.gat_corr_threshold), 0.0)
        return float(self._gat_correction_penalty_from_excess_tensor(excess).item())

    def _gat_correction_penalty_tensor(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        values = self._gat_correction_gain_values(p, q)
        if not values:
            return p.new_zeros(())
        penalties = []
        threshold = p.new_tensor(float(self.gat_corr_threshold))
        for gain in values:
            gain = gain.to(device=p.device, dtype=p.dtype)
            excess = torch.clamp_min(gain - threshold, 0.0)
            penalties.append(self._gat_correction_penalty_from_excess_tensor(excess))
        return torch.stack(penalties).mean()

    def _gat_correction_stats_float(self, p: float, q: float) -> Tuple[float, float, float]:
        values = self._gat_correction_gain_values(float(p), float(q))
        if not values:
            return 0.0, 0.0, 0.0
        gain = torch.cat([value.detach() for value in values], dim=0)
        excess = torch.clamp_min(gain - float(self.gat_corr_threshold), 0.0)
        penalty = float(self._gat_correction_penalty_from_excess_tensor(excess).item())
        return penalty, float(gain.mean().item()), float(gain.max().item())

    def _objective_penalty_float(self, p: float, q: float) -> float:
        return (
            self._deletion_penalty_float(p)
            + self._densification_penalty_float(self._rho_from_q_float(q))
            + self._gat_denominator_penalty_float(p, q)
            + self._gat_correction_penalty_float(p, q)
        )

    def _objective_penalty_tensor(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (
            self._deletion_penalty_tensor(p)
            + self._densification_penalty_tensor(self._rho_from_q_tensor(q))
            + self._gat_denominator_penalty_tensor(p, q)
            + self._gat_correction_penalty_tensor(p, q)
        )

    @torch.no_grad()
    def _messages_detached(self, emb: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = emb.detach()
        w = weight.detach()
        msg = h @ w.t()
        return msg, msg * msg

    def _loss_and_w(self, logits: torch.Tensor, y: torch.Tensor, loss_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        loss_type = str(loss_type).lower().strip()
        if loss_type == "ce":
            if y.dim() == 2 and y.size(1) == 1:
                y = y.squeeze(1)
            y = y.view(-1).to(torch.long)
            loss = F.cross_entropy(logits, y)
            prob = torch.softmax(logits, dim=-1)
            return loss, prob * (1.0 - prob)
        if loss_type == "bce":
            if y.dim() == 1:
                y = y.view(-1, 1)
            y_f = y.to(torch.float32)
            mask = ((y == 0) | (y == 1)).to(torch.float32)
            loss_el = F.binary_cross_entropy_with_logits(logits, y_f, reduction="none")
            loss = (loss_el * mask).sum() / mask.sum().clamp(min=1.0)
            z = torch.sigmoid(logits)
            return loss, z * (1.0 - z) * mask
        raise ValueError(f"Unsupported loss_type={loss_type}.")

    def _pool_w2(self, batch_vec: torch.Tensor, num_graphs: int, num_nodes: int, device) -> torch.Tensor:
        if self.pooling == "sum":
            return torch.ones((num_nodes,), device=device, dtype=torch.float32)
        ones = torch.ones((num_nodes,), device=device, dtype=torch.float32)
        n_per_graph = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        n_per_graph.index_add_(0, batch_vec, ones)
        return 1.0 / n_per_graph[batch_vec].clamp(min=1.0).pow(2.0)

    def _var_gin_bases(self, cache: BatchedGraphCache, msg: torch.Tensor, msg_sq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = cache.edge_index_clean_dir
        neigh_sum_sq = _scatter_add_rows(msg_sq[row], col, cache.num_nodes)
        comp_sum_sq = cache.complement_sum_unweighted(msg_sq)
        if self.rade_variant == "rade-of":
            comp_sum_msg = cache.complement_sum_unweighted(msg)
            comp_size = cache.comp_size.to(msg.dtype).clamp_min(1.0).unsqueeze(1)
            mu = comp_sum_msg / comp_size
            add_centered = comp_sum_sq - 2.0 * mu * comp_sum_msg + comp_size * (mu * mu)
            return neigh_sum_sq, add_centered
        return neigh_sum_sq, comp_sum_sq

    @torch.no_grad()
    def _var_gcn(self, cache: BatchedGraphCache, p: float, q: float, msg: torch.Tensor, msg_sq: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = cache.edge_index_clean_dir
        deg = cache.deg_clean.to(torch.float32)
        comp = cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        t_a = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        u_a = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)
        e = (1.0 - p) * (t_a[row] * t_a[col])
        e2 = (1.0 - p) * (u_a[row] * u_a[col])
        var_g = (e2 - e * e).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])
        factor = alpha_sq * (var_g / (e * e + eps))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * msg_sq[row], col, cache.num_nodes)

        t_b = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        u_b = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        t_b2 = t_b * t_b

        if self.rade_variant == "rade-of":
            mu = cache.complement_mean_weighted(msg, t_b)
            ones = torch.ones((cache.num_nodes, 1), device=msg.device, dtype=msg.dtype)
            a_u = cache.complement_sum_weighted(msg_sq, u_b)
            b_u = cache.complement_sum_weighted(msg, u_b)
            c_u = cache.complement_sum_weighted(ones, u_b)
            s1 = a_u - 2.0 * mu * b_u + (mu * mu) * c_u
            a_t = cache.complement_sum_weighted(msg_sq, t_b2)
            b_t = cache.complement_sum_weighted(msg, t_b2)
            c_t = cache.complement_sum_weighted(ones, t_b2)
            s2 = a_t - 2.0 * mu * b_t + (mu * mu) * c_t
        else:
            s1 = cache.complement_sum_weighted(msg_sq, u_b)
            s2 = cache.complement_sum_weighted(msg_sq, t_b2)

        add_var = (q * u_b.unsqueeze(1) * s1) - ((q * q) * t_b2.unsqueeze(1) * s2)
        return neigh_var + add_var

    def _var_gcn_tensor(
        self,
        cache: BatchedGraphCache,
        p: torch.Tensor,
        q: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = cache.edge_index_clean_dir
        deg = cache.deg_clean.to(torch.float32)
        comp = cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        t_a = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        u_a = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)
        e = (1.0 - p) * (t_a[row] * t_a[col])
        e2 = (1.0 - p) * (u_a[row] * u_a[col])
        var_g = (e2 - e * e).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])
        factor = alpha_sq * (var_g / (e * e + eps))
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * msg_sq[row], col, cache.num_nodes)

        t_b = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        u_b = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        t_b2 = t_b * t_b

        if self.rade_variant == "rade-of":
            mu = cache.complement_mean_weighted(msg, t_b)
            ones = torch.ones((cache.num_nodes, 1), device=msg.device, dtype=msg.dtype)
            a_u = cache.complement_sum_weighted(msg_sq, u_b)
            b_u = cache.complement_sum_weighted(msg, u_b)
            c_u = cache.complement_sum_weighted(ones, u_b)
            s1 = a_u - 2.0 * mu * b_u + (mu * mu) * c_u
            a_t = cache.complement_sum_weighted(msg_sq, t_b2)
            b_t = cache.complement_sum_weighted(msg, t_b2)
            c_t = cache.complement_sum_weighted(ones, t_b2)
            s2 = a_t - 2.0 * mu * b_t + (mu * mu) * c_t
        else:
            s1 = cache.complement_sum_weighted(msg_sq, u_b)
            s2 = cache.complement_sum_weighted(msg_sq, t_b2)

        add_var = (q * u_b.unsqueeze(1) * s1) - ((q * q) * t_b2.unsqueeze(1) * s2)
        return neigh_var + add_var

    @staticmethod
    def _collect_gat_modules(model: nn.Module) -> List[nn.Module]:
        modules = [
            module
            for module in model.modules()
            if callable(getattr(module, "gat_variance_proxy", None))
        ]
        return modules[-1:] if modules else []

    def _var_gat(self, cache: BatchedGraphCache, p: float, q: float, msg: torch.Tensor, msg_sq: torch.Tensor) -> torch.Tensor:
        _ = cache
        modules = getattr(self, "_gat_modules", None)
        if not modules:
            return torch.zeros_like(msg_sq)
        estimates = [
            module.gat_variance_proxy(msg, msg_sq, p=float(p), q=float(q), eps=float(self.cfg.eps))
            for module in modules
        ]
        return estimates[0] if len(estimates) == 1 else torch.stack(estimates, dim=0).mean(dim=0)

    def _var_gat_tensor(
        self,
        cache: BatchedGraphCache,
        p: torch.Tensor,
        q: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
    ) -> torch.Tensor:
        _ = cache
        modules = getattr(self, "_gat_modules", None)
        if not modules:
            return torch.zeros_like(msg_sq) + (p * 0.0 + q * 0.0)
        estimates = [
            module.gat_variance_proxy(msg, msg_sq, p=p, q=q, eps=float(self.cfg.eps))
            for module in modules
        ]
        value = estimates[0] if len(estimates) == 1 else torch.stack(estimates, dim=0).mean(dim=0)
        return value + (p * 0.0 + q * 0.0)

    def _ensure_adam_state(self, *, current_p: float, current_q: float, device: torch.device) -> None:
        if self._adam_opt is not None and self._adam_device == device and self._adam_p is not None and self._adam_q is not None:
            return
        add_upper = max(float(self._adam_add_parameter_upper()), 0.0)
        add0 = float(max(0.0, min(self._add_parameter_from_q_float(float(current_q)), add_upper)))
        p0 = float(max(0.0, min(float(current_p), self.p_upper)))
        q0 = add0
        self._adam_p = nn.Parameter(torch.tensor(p0, device=device, dtype=torch.float32))
        self._adam_q = nn.Parameter(torch.tensor(q0, device=device, dtype=torch.float32))
        self._adam_opt = torch.optim.Adam(
            [self._adam_p, self._adam_q],
            lr=float(self.cfg.adam_lr),
            betas=(float(self.cfg.adam_beta1), float(self.cfg.adam_beta2)),
            eps=float(self.cfg.adam_eps),
        )
        self._adam_device = device

    def _adam_rates_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._adam_p is None or self._adam_q is None:
            raise RuntimeError("Adam pq controller is not initialized.")
        p = torch.clamp(self._adam_p, 0.0, float(self.p_upper))
        add_param = torch.clamp(self._adam_q, 0.0, float(self._adam_add_parameter_upper()))
        q = torch.clamp(self._q_from_add_parameter_tensor(add_param), 0.0, Q_PROBABILITY_UPPER)
        return p, q

    @torch.no_grad()
    def _project_adam_state_(self) -> None:
        if self._adam_p is None or self._adam_q is None:
            raise RuntimeError("Adam pq controller is not initialized.")
        self._adam_p.clamp_(0.0, float(self.p_upper))
        self._adam_q.clamp_(0.0, float(self._adam_add_parameter_upper()))

    @torch.no_grad()
    def _adam_rates_float(self, aug_mode: str) -> Tuple[float, float]:
        p_t, q_t = self._adam_rates_tensor()
        p = float(p_t.item())
        q = float(q_t.item())
        if aug_mode == "drop":
            q = 0.0
        elif aug_mode == "add":
            p = 0.0
        elif aug_mode == "none":
            p, q = 0.0, 0.0
        p = float(max(0.0, min(p, float(self.p_upper))))
        q = float(max(0.0, min(q, Q_PROBABILITY_UPPER)))
        return p, q

    def _augment_anchor_batch(
        self,
        *,
        batch,
        p: float,
        q: float,
        aug_mode: str,
        seed: int,
        device: torch.device,
        mask_sharing: str,
        num_layers: int,
    ):
        if PyGBatch is None:
            raise RuntimeError("torch_geometric.data.Batch is required for PQ augmentation anchors.")
        if aug_mode == "none" or (p <= 0.0 and q <= 0.0):
            return batch, None, None

        data_list = batch.to_data_list()
        keep_layers = [[] for _ in range(int(num_layers))] if mask_sharing == "layerwise" else None
        add_layers = [[] for _ in range(int(num_layers))] if mask_sharing == "layerwise" else None
        keep_list = [] if mask_sharing == "shared" else None
        add_list = [] if mask_sharing == "shared" else None
        node_offset = 0

        for graph_idx, data in enumerate(data_list):
            augmentor = BernoulliEdgeAugmentor.from_graph(data.edge_index, int(data.num_nodes), edge_attr=None)
            if mask_sharing == "shared":
                _, edge_index_keep, edge_index_add, _ = augmentor.augment_edge_indices(
                    p=float(p),
                    q=float(q),
                    mode=aug_mode,
                    seed=int(seed) + 97 * int(graph_idx),
                    device=torch.device("cpu"),
                )
                if edge_index_keep.numel() > 0:
                    keep_list.append(edge_index_keep + int(node_offset))
                if edge_index_add.numel() > 0:
                    add_list.append(edge_index_add + int(node_offset))
            else:
                for layer_idx in range(int(num_layers)):
                    _, edge_index_keep, edge_index_add, _ = augmentor.augment_edge_indices(
                        p=float(p),
                        q=float(q),
                        mode=aug_mode,
                        seed=int(seed) + 97 * int(graph_idx) + 1_000_000 * int(layer_idx),
                        device=torch.device("cpu"),
                    )
                    if edge_index_keep.numel() > 0:
                        keep_layers[layer_idx].append(edge_index_keep + int(node_offset))
                    if edge_index_add.numel() > 0:
                        add_layers[layer_idx].append(edge_index_add + int(node_offset))
            node_offset += int(data.num_nodes)

        batch_out = PyGBatch.from_data_list(data_list).to(device)
        if mask_sharing == "shared":
            keep_obj = torch.cat(keep_list, dim=1).to(device) if keep_list else torch.empty((2, 0), dtype=torch.long, device=device)
            add_obj = torch.cat(add_list, dim=1).to(device) if add_list else torch.empty((2, 0), dtype=torch.long, device=device)
            return batch_out, keep_obj, add_obj
        keep_obj = [
            torch.cat(layer_edges, dim=1).to(device) if layer_edges else torch.empty((2, 0), dtype=torch.long, device=device)
            for layer_edges in keep_layers
        ]
        add_obj = [
            torch.cat(layer_edges, dim=1).to(device) if layer_edges else torch.empty((2, 0), dtype=torch.long, device=device)
            for layer_edges in add_layers
        ]
        return batch_out, keep_obj, add_obj

    def suggest_pq_batch(
        self,
        *,
        model: nn.Module,
        batch,
        loss_type: str,
        aug_mode: str,
        current_p: float,
        current_q: float,
        seed: int,
        mask_sharing: str,
        num_layers: int,
    ) -> Tuple[float, float, Dict[str, float]]:
        cfg = self.cfg
        eps = float(cfg.eps)
        aug_mode = str(aug_mode).lower().strip()
        mask_sharing = str(mask_sharing).lower().strip()

        x = batch.x
        edge_index = batch.edge_index
        batch_vec = batch.batch
        num_graphs = int(batch.num_graphs)
        edge_attr = getattr(batch, "edge_attr", None)
        node_ids = getattr(batch, "n_id", None)

        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)
        device = x.device

        devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(seed) + int(cfg.seed))
            if devices:
                torch.cuda.set_device(devices[0])
                torch.cuda.manual_seed(int(seed) + int(cfg.seed))

            model.zero_grad(set_to_none=True)

            if self.data_anchor == "aug":
                batch_anchor, edge_index_keep0, edge_index_add0 = self._augment_anchor_batch(
                    batch=batch,
                    p=float(current_p),
                    q=float(current_q),
                    aug_mode=aug_mode,
                    seed=int(seed) + 31_337,
                    device=device,
                    mask_sharing=mask_sharing,
                    num_layers=int(num_layers),
                )
                logits_anchor, _ = model(
                    batch_anchor.x,
                    batch_anchor.edge_index,
                    batch_anchor.batch,
                    edge_attr=getattr(batch_anchor, "edge_attr", None),
                    edge_index_keep=edge_index_keep0,
                    edge_index_add=edge_index_add0,
                    p=float(current_p),
                    q=float(current_q),
                    node_ids=getattr(batch_anchor, "n_id", None),
                    return_embeddings=True,
                )
            else:
                logits_anchor, _ = model(
                    x,
                    edge_index,
                    batch_vec,
                    edge_attr=edge_attr,
                    node_ids=node_ids,
                    return_embeddings=True,
                )

            loss_data, w_graph = self._loss_and_w(logits_anchor, batch.y, loss_type=loss_type)
            params = [param for param in model.parameters() if param.requires_grad]
            g_data = torch.autograd.grad(loss_data, params, retain_graph=False, create_graph=False, allow_unused=True)
            g_data_norm = _grad_norm(g_data)
            log_gd = math.log(g_data_norm + eps)

            model.zero_grad(set_to_none=True)
            logits, node_emb = model(
                x,
                edge_index,
                batch_vec,
                edge_attr=edge_attr,
                node_ids=node_ids,
                return_embeddings=True,
            )
            _, w_graph = self._loss_and_w(logits, batch.y, loss_type=loss_type)
            msg, msg_sq = self._messages_detached(node_emb, model.pred.weight)
            self._gat_modules = self._collect_gat_modules(model) if self.gnn == "gat" else []
            cache = BatchedGraphCache.from_edge_index(edge_index, batch_vec, num_graphs=num_graphs, node_ids=node_ids)
            pool_w2 = self._pool_w2(batch_vec, num_graphs, cache.num_nodes, device=msg.device).to(msg.dtype).unsqueeze(1)
            batch_size = max(1, num_graphs)

            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                if self.gnn == "gin":
                    v_drop_node, v_add_node = self._var_gin_bases(cache, msg, msg_sq)
                    v_drop_graph = torch.zeros((num_graphs, v_drop_node.size(1)), device=msg.device, dtype=msg.dtype)
                    v_add_graph = torch.zeros_like(v_drop_graph)
                    v_drop_graph.index_add_(0, batch_vec, v_drop_node * pool_w2)
                    v_add_graph.index_add_(0, batch_vec, v_add_node * pool_w2)
                    r_drop = 0.5 * (w_graph * v_drop_graph).sum() / batch_size
                    r_add = 0.5 * (w_graph * v_add_graph).sum() / batch_size
                    g_drop = torch.autograd.grad(r_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
                    g_add = torch.autograd.grad(r_add, params, retain_graph=True, create_graph=False, allow_unused=True)
                    s = float(p_try) / max(1.0 - float(p_try), eps)
                    t = float(q_try) * (1.0 - float(q_try))
                    a = _dot(g_drop, g_drop)
                    b = _dot(g_add, g_add)
                    c = _dot(g_drop, g_add)
                    g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                    g_reg = math.sqrt(max(g2, 0.0))
                    obj = (math.log(g_reg + eps) - log_gd) ** 2 + self._objective_penalty_float(
                        float(p_try), float(q_try)
                    )
                    return float(obj), float(g_reg)

                if self.gnn == "gcn":
                    v_node = self._var_gcn(cache, float(p_try), float(q_try), msg, msg_sq)
                elif self.gnn == "gat":
                    v_node = self._var_gat(cache, float(p_try), float(q_try), msg, msg_sq)
                else:
                    raise ValueError(f"Unsupported gnn={self.gnn}.")
                v_graph = torch.zeros((num_graphs, v_node.size(1)), device=msg.device, dtype=msg.dtype)
                v_graph.index_add_(0, batch_vec, v_node * pool_w2)
                regularizer = 0.5 * (w_graph * v_graph).sum() / batch_size
                g_reg = torch.autograd.grad(regularizer, params, retain_graph=True, create_graph=False, allow_unused=True)
                greg = _grad_norm(g_reg)
                obj = (math.log(greg + eps) - log_gd) ** 2 + self._objective_penalty_float(
                    float(p_try), float(q_try)
                )
                return float(obj), float(greg)

            def objective_tensor(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                log_gd_t = w_graph.new_tensor(float(log_gd))
                eps_t = w_graph.new_tensor(float(eps))
                if self.gnn == "gin":
                    v_drop_node, v_add_node = self._var_gin_bases(cache, msg, msg_sq)
                    v_drop_graph = torch.zeros((num_graphs, v_drop_node.size(1)), device=msg.device, dtype=msg.dtype)
                    v_add_graph = torch.zeros_like(v_drop_graph)
                    v_drop_graph.index_add_(0, batch_vec, v_drop_node * pool_w2)
                    v_add_graph.index_add_(0, batch_vec, v_add_node * pool_w2)
                    r_drop = 0.5 * (w_graph * v_drop_graph).sum() / batch_size
                    r_add = 0.5 * (w_graph * v_add_graph).sum() / batch_size
                    g_drop = torch.autograd.grad(r_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
                    g_add = torch.autograd.grad(r_add, params, retain_graph=True, create_graph=False, allow_unused=True)
                    a = w_graph.new_tensor(_dot(g_drop, g_drop))
                    b = w_graph.new_tensor(_dot(g_add, g_add))
                    c = w_graph.new_tensor(_dot(g_drop, g_add))
                    one = torch.ones_like(p_try)
                    s = p_try / (one - p_try).clamp_min(float(eps))
                    t = q_try * (one - q_try)
                    g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                    g_reg = torch.sqrt(torch.clamp_min(g2, 0.0))
                    obj = (torch.log(g_reg + eps_t) - log_gd_t) ** 2 + self._objective_penalty_tensor(
                        p_try, q_try
                    )
                    return obj, g_reg

                if self.gnn == "gcn":
                    v_node = self._var_gcn_tensor(cache, p_try, q_try, msg, msg_sq)
                elif self.gnn == "gat":
                    v_node = self._var_gat_tensor(cache, p_try, q_try, msg, msg_sq)
                else:
                    raise ValueError(f"Unsupported gnn={self.gnn}.")
                v_graph = torch.zeros((num_graphs, v_node.size(1)), device=msg.device, dtype=msg.dtype)
                v_graph.index_add_(0, batch_vec, v_node * pool_w2)
                regularizer = 0.5 * (w_graph * v_graph).sum() / batch_size
                g_reg = torch.autograd.grad(regularizer, params, retain_graph=True, create_graph=True, allow_unused=True)
                g2 = w_graph.new_zeros(())
                for grad in g_reg:
                    if grad is None:
                        continue
                    g2 = g2 + (grad * grad).sum()
                greg = torch.sqrt(torch.clamp_min(g2, 0.0))
                obj = (torch.log(greg + eps_t) - log_gd_t) ** 2 + self._objective_penalty_tensor(
                    p_try, q_try
                )
                return obj, greg

            self._ensure_adam_state(current_p=float(current_p), current_q=float(current_q), device=device)
            p_before, q_before = self._adam_rates_tensor()
            if aug_mode == "drop":
                q_before = q_before * 0.0
            elif aug_mode == "add":
                p_before = p_before * 0.0
            elif aug_mode == "none":
                p_before = p_before * 0.0
                q_before = q_before * 0.0

            self._adam_opt.zero_grad(set_to_none=True)
            obj_t, greg_t = objective_tensor(p_before, q_before)
            obj_t.backward(retain_graph=True)
            self._adam_opt.step()
            self._project_adam_state_()

            p_new, q_new = self._adam_rates_float(aug_mode)
            obj_new, greg_new = objective(float(p_new), float(q_new))
            p_best = float(p_before.detach().item())
            q_best = float(q_before.detach().item())
            rho_best = float(self._rho_from_q_float(q_best))
            rho_new = float(self._rho_from_q_float(float(q_new)))
            gat_denom_penalty_best, gat_denom_cv2_mean_best, gat_denom_cv2_max_best = (
                self._gat_denominator_stats_float(p_best, q_best)
            )
            gat_denom_penalty_new, gat_denom_cv2_mean_new, gat_denom_cv2_max_new = self._gat_denominator_stats_float(
                p_new,
                q_new,
            )
            gat_corr_penalty_best, gat_corr_gain_mean_best, gat_corr_gain_max_best = self._gat_correction_stats_float(
                p_best,
                q_best,
            )
            gat_corr_penalty_new, gat_corr_gain_mean_new, gat_corr_gain_max_new = self._gat_correction_stats_float(
                p_new,
                q_new,
            )

            info = {
                "G_data": float(g_data_norm),
                "G_reg_best": float(greg_t.detach().item()),
                "p_best": p_best,
                "q_best": q_best,
                "rho_best": rho_best,
                "p_new": float(p_new),
                "q_new": float(q_new),
                "rho_new": rho_new,
                "obj": float(obj_t.detach().item()),
                "G_reg_new": float(greg_new),
                "obj_new": float(obj_new),
                "deletion_penalty_best": float(self._deletion_penalty_float(p_best)),
                "densification_penalty_best": float(self._densification_penalty_float(rho_best)),
                "rho_penalty_best": float(self._densification_penalty_float(rho_best)),
                "deletion_penalty_new": float(self._deletion_penalty_float(p_new)),
                "densification_penalty_new": float(self._densification_penalty_float(rho_new)),
                "rho_penalty_new": float(self._densification_penalty_float(rho_new)),
                "gat_denom_penalty_best": float(gat_denom_penalty_best),
                "gat_denom_penalty_new": float(gat_denom_penalty_new),
                "gat_denom_cv2_mean_best": float(gat_denom_cv2_mean_best),
                "gat_denom_cv2_mean_new": float(gat_denom_cv2_mean_new),
                "gat_denom_cv2_max_best": float(gat_denom_cv2_max_best),
                "gat_denom_cv2_max_new": float(gat_denom_cv2_max_new),
                "gat_corr_penalty_best": float(gat_corr_penalty_best),
                "gat_corr_penalty_new": float(gat_corr_penalty_new),
                "gat_corr_gain_mean_best": float(gat_corr_gain_mean_best),
                "gat_corr_gain_mean_new": float(gat_corr_gain_mean_new),
                "gat_corr_gain_max_best": float(gat_corr_gain_max_best),
                "gat_corr_gain_max_new": float(gat_corr_gain_max_new),
                "optimize_rho": bool(self.optimize_rho),
                "expected_add_ratio_scale": float(self.expected_add_ratio_scale),
                "deletion_penalty_lambda": float(getattr(cfg, "deletion_penalty_lambda", 0.0)),
                "densification_penalty_lambda": float(getattr(cfg, "densification_penalty_lambda", 0.0)),
                "gat_denom_penalty_lambda": float(self.gat_denom_penalty_lambda),
                "gat_denom_cv2_threshold": float(self.gat_denom_cv2_threshold),
                "gat_corr_penalty_lambda": float(self.gat_corr_penalty_lambda),
                "gat_corr_threshold": float(self.gat_corr_threshold),
                "p_max": float(self.p_upper),
                "adam_lr": float(cfg.adam_lr),
                "optimization_variable": self._add_parameter_name(),
                "n_eval": 2.0,
                "solver_success": 1.0,
            }

        _restore_batchnorm_buffers(bn_states)
        if not was_training:
            model.eval()
        return float(p_new), float(q_new), info
