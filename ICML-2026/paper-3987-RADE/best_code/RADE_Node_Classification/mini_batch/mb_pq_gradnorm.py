from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .mb_rade_convs import (
    BatchGraphCache,
    _delta_E_inv_caseA,
    _delta_E_inv_caseB,
    _delta_E_inv_sqrt_caseA,
    _delta_E_inv_sqrt_caseB,
    _scatter_add_rows,
)


P_PROBABILITY_UPPER: float = 1.0 - 1e-6
Q_PROBABILITY_UPPER: float = 1.0


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}")
    return rade_variant


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


def _pick_subset(idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    if k <= 0 or k >= idx.numel():
        return idx
    gen = torch.Generator(device=idx.device)
    gen.manual_seed(int(seed))
    perm = torch.randperm(idx.numel(), generator=gen, device=idx.device)
    return idx[perm[:k]]


def _w_multiclass(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    return prob * (1.0 - prob)


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
class PQTunerMBConfig:
    expected_add_ratio_scale: float = 0.0
    deletion_penalty_lambda: float = 0.0
    densification_penalty_lambda: float = 1.0
    optimize_rho: bool = False
    cap_p: bool = False
    p_cap_max: float = 0.9
    cap_rho: bool = False
    rho_max: float = 0.2
    eps: float = 1e-12
    subset_nodes: int = 0
    seed: int = 0
    adam_lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


class PQGradNormTunerMB:
    def __init__(self, gnn: str, rade_variant: str, cfg: PQTunerMBConfig):
        self.cfg = cfg
        self.gnn = str(gnn).lower().strip()
        self.rade_variant = _validate_rade_variant(rade_variant)
        if self.gnn not in {"gin", "gcn", "gat"}:
            raise ValueError("gnn must be 'gin', 'gcn', or 'gat'")

        self.optimize_rho = bool(getattr(cfg, "optimize_rho", False))
        self.expected_add_ratio_scale = max(0.0, float(getattr(cfg, "expected_add_ratio_scale", 0.0)))

        self.cap_p = bool(getattr(cfg, "cap_p", False))
        p_cap_max = float(getattr(cfg, "p_cap_max", 0.9))
        if not (0.0 <= p_cap_max < 1.0):
            raise ValueError(f"p_cap_max must be in [0, 1). Got {p_cap_max}")
        self.p_probability_upper = min(P_PROBABILITY_UPPER, p_cap_max) if self.cap_p else P_PROBABILITY_UPPER

        self.rho_probability_upper = float(self.expected_add_ratio_scale) * float(Q_PROBABILITY_UPPER)
        self.cap_rho = bool(getattr(cfg, "cap_rho", False))
        rho_max = float(getattr(cfg, "rho_max", 0.2))
        if rho_max < 0.0:
            raise ValueError(f"rho_max must be >= 0. Got {rho_max}")
        self.rho_upper = min(self.rho_probability_upper, rho_max) if self.cap_rho else self.rho_probability_upper
        self.q_probability_upper = (
            min(Q_PROBABILITY_UPPER, self._q_from_rho_float(self.rho_upper))
            if self.cap_rho
            else Q_PROBABILITY_UPPER
        )

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
            return float(self.rho_upper)
        return float(self.q_probability_upper)

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

    def _objective_penalty_float(self, p: float, q: float) -> float:
        return self._deletion_penalty_float(p) + self._densification_penalty_float(self._rho_from_q_float(q))

    def _objective_penalty_tensor(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self._deletion_penalty_tensor(p) + self._densification_penalty_tensor(self._rho_from_q_tensor(q))

    @torch.no_grad()
    def _messages_detached(self, emb: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = emb.detach()
        w = weight.detach()
        msg = h @ w.t()
        return msg, msg * msg

    def _var_gin_bases(
        self,
        cache: BatchGraphCache,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        batch_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = cache.edge_index_clean_dir
        neigh_sum_sq = _scatter_add_rows(msg_sq[row], col, batch_nodes)
        comp_sum_sq = cache.complement_sum_unweighted(msg_sq)

        if self.rade_variant == "rade-of":
            comp_sum_m = cache.complement_sum_unweighted(msg)
            comp_size = cache.comp_size.to(msg.dtype).clamp_min(1.0).unsqueeze(1)
            mu = comp_sum_m / comp_size
            add_centered = comp_sum_sq - 2.0 * mu * comp_sum_m + comp_size * (mu * mu)
            return neigh_sum_sq, add_centered

        return neigh_sum_sq, comp_sum_sq

    @torch.no_grad()
    def _var_gcn(
        self,
        cache: BatchGraphCache,
        p: float,
        q: float,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        batch_nodes: int,
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
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * msg_sq[row], col, batch_nodes)

        t_b = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        u_b = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        t_b2 = t_b * t_b

        if self.rade_variant == "rade-of":
            mu = cache.complement_mean_weighted(msg, t_b)
            ones = torch.ones((batch_nodes, 1), device=msg.device, dtype=msg.dtype)

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
        cache: BatchGraphCache,
        p: torch.Tensor,
        q: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        batch_nodes: int,
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
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * msg_sq[row], col, batch_nodes)

        t_b = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        u_b = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        t_b2 = t_b * t_b

        if self.rade_variant == "rade-of":
            mu = cache.complement_mean_weighted(msg, t_b)
            ones = torch.ones((batch_nodes, 1), device=msg.device, dtype=msg.dtype)

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

    def _var_gat(
        self,
        cache: BatchGraphCache,
        p: float,
        q: float,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
    ) -> torch.Tensor:
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
        cache: BatchGraphCache,
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

    @staticmethod
    def _safe_logit_from_bounded(x: float, x_max: float, eps: float = 1e-6) -> float:
        if x_max <= 0.0:
            return 0.0
        ratio = float(x) / float(x_max)
        ratio = min(max(ratio, eps), 1.0 - eps)
        return math.log(ratio / (1.0 - ratio))

    def _ensure_adam_state(
        self,
        *,
        current_p: float,
        current_q: float,
        device: torch.device,
    ) -> None:
        if self._adam_opt is not None and self._adam_device == device and self._adam_p is not None and self._adam_q is not None:
            return

        cfg = self.cfg
        p0 = float(max(0.0, min(current_p, self.p_probability_upper)))
        q0 = float(max(0.0, min(current_q, self.q_probability_upper)))
        add_upper = max(float(self._adam_add_parameter_upper()), 0.0)
        add0 = float(max(0.0, min(self._add_parameter_from_q_float(q0), add_upper)))
        self._adam_p = nn.Parameter(torch.tensor(p0, device=device, dtype=torch.float32))
        self._adam_q = nn.Parameter(torch.tensor(add0, device=device, dtype=torch.float32))
        self._adam_opt = torch.optim.Adam(
            [self._adam_p, self._adam_q],
            lr=float(cfg.adam_lr),
            betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
            eps=float(cfg.adam_eps),
        )
        self._adam_device = device

    def _adam_rates_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._adam_p is None or self._adam_q is None:
            raise RuntimeError("Adam pq controller is not initialized.")

        p = torch.clamp(self._adam_p, 0.0, self.p_probability_upper)
        add_param = torch.clamp(self._adam_q, 0.0, float(self._adam_add_parameter_upper()))
        q = torch.clamp(self._q_from_add_parameter_tensor(add_param), 0.0, self.q_probability_upper)
        return p, q

    @torch.no_grad()
    def _project_adam_state_(self) -> None:
        if self._adam_p is None or self._adam_q is None:
            raise RuntimeError("Adam pq controller is not initialized.")
        self._adam_p.copy_(
            torch.nan_to_num(
                self._adam_p,
                nan=0.0,
                posinf=self.p_probability_upper,
                neginf=0.0,
            )
        )
        self._adam_q.copy_(
            torch.nan_to_num(
                self._adam_q,
                nan=0.0,
                posinf=float(self._adam_add_parameter_upper()),
                neginf=0.0,
            )
        )
        if self._adam_opt is not None:
            for state in self._adam_opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.copy_(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        self._adam_p.clamp_(0.0, self.p_probability_upper)
        self._adam_q.clamp_(0.0, float(self._adam_add_parameter_upper()))

    @torch.no_grad()
    def _adam_rates_float(self, aug_mode: str) -> Tuple[float, float]:
        p_t, q_t = self._adam_rates_tensor()
        p = float(p_t.item())
        q = float(q_t.item())

        aug_mode = str(aug_mode).lower().strip()
        if aug_mode == "drop":
            q = 0.0
        elif aug_mode == "add":
            p = 0.0
        elif aug_mode == "none":
            p, q = 0.0, 0.0

        p = float(max(0.0, min(p, self.p_probability_upper)))
        q = float(max(0.0, min(q, self.q_probability_upper)))
        return p, q

    def _make_objective(
        self,
        *,
        cache: BatchGraphCache,
        params,
        idx: torch.Tensor,
        weight: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        log_gd: float,
        eps: float,
        batch_nodes: int,
    ) -> Callable[[float, float], Tuple[float, float]]:
        if self.gnn == "gin":
            v_drop, v_add = self._var_gin_bases(cache, msg, msg_sq, batch_nodes)
            r_drop = 0.5 * (weight[idx] * v_drop[idx]).sum() / max(int(idx.numel()), 1)
            r_add = 0.5 * (weight[idx] * v_add[idx]).sum() / max(int(idx.numel()), 1)

            g_drop = torch.autograd.grad(r_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
            g_add = torch.autograd.grad(r_add, params, retain_graph=True, create_graph=False, allow_unused=True)

            a = _dot(g_drop, g_drop)
            b = _dot(g_add, g_add)
            c = _dot(g_drop, g_add)

            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                s = p_try / max(1.0 - p_try, eps)
                t = q_try * (1.0 - q_try)
                g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                g_reg = math.sqrt(max(g2, 0.0))
                obj = (math.log(g_reg + eps) - log_gd) ** 2 + self._objective_penalty_float(
                    float(p_try), float(q_try)
                )
                return float(obj), float(g_reg)

            return objective

        if self.gnn in {"gcn", "gat"}:
            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                if self.gnn == "gcn":
                    v = self._var_gcn(cache, float(p_try), float(q_try), msg, msg_sq, batch_nodes)
                else:
                    v = self._var_gat(cache, float(p_try), float(q_try), msg, msg_sq)
                regularizer = 0.5 * (weight[idx] * v[idx]).sum() / max(int(idx.numel()), 1)
                g_reg = torch.autograd.grad(
                    regularizer,
                    params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                greg = _grad_norm(g_reg)
                obj = (math.log(greg + eps) - log_gd) ** 2 + self._objective_penalty_float(
                    float(p_try), float(q_try)
                )
                return float(obj), float(greg)

            return objective

        raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin', 'gcn', or 'gat'.")

    def _make_objective_tensor(
        self,
        *,
        cache: BatchGraphCache,
        params,
        idx: torch.Tensor,
        weight: torch.Tensor,
        msg: torch.Tensor,
        msg_sq: torch.Tensor,
        log_gd: float,
        eps: float,
        batch_nodes: int,
    ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        log_gd_t = weight.new_tensor(float(log_gd))
        eps_t = weight.new_tensor(float(eps))

        if self.gnn == "gin":
            v_drop, v_add = self._var_gin_bases(cache, msg, msg_sq, batch_nodes)
            r_drop = 0.5 * (weight[idx] * v_drop[idx]).sum() / max(int(idx.numel()), 1)
            r_add = 0.5 * (weight[idx] * v_add[idx]).sum() / max(int(idx.numel()), 1)

            g_drop = torch.autograd.grad(r_drop, params, retain_graph=True, create_graph=False, allow_unused=True)
            g_add = torch.autograd.grad(r_add, params, retain_graph=True, create_graph=False, allow_unused=True)

            a = weight.new_tensor(_dot(g_drop, g_drop))
            b = weight.new_tensor(_dot(g_add, g_add))
            c = weight.new_tensor(_dot(g_drop, g_add))

            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                one = torch.ones_like(p_try)
                s = p_try / (one - p_try).clamp_min(float(eps))
                t = q_try * (one - q_try)
                g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                g_reg = torch.sqrt(torch.clamp_min(g2, 0.0))
                obj = (torch.log(g_reg + eps_t) - log_gd_t) ** 2 + self._objective_penalty_tensor(
                    p_try, q_try
                )
                return obj, g_reg

            return objective

        if self.gnn in {"gcn", "gat"}:
            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if self.gnn == "gcn":
                    v = self._var_gcn_tensor(cache, p_try, q_try, msg, msg_sq, batch_nodes)
                else:
                    v = self._var_gat_tensor(cache, p_try, q_try, msg, msg_sq)
                regularizer = 0.5 * (weight[idx] * v[idx]).sum() / max(int(idx.numel()), 1)
                g_reg = torch.autograd.grad(
                    regularizer,
                    params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )
                g2 = weight.new_zeros(())
                for grad in g_reg:
                    if grad is None:
                        continue
                    g2 = g2 + (grad * grad).sum()
                greg = torch.sqrt(torch.clamp_min(g2, 0.0))
                obj = (torch.log(greg + eps_t) - log_gd_t) ** 2 + self._objective_penalty_tensor(
                    p_try, q_try
                )
                return obj, greg

            return objective

        raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin', 'gcn', or 'gat'.")

    def _prepare_batch_inputs(
        self,
        model: nn.Module,
        x_b: torch.Tensor,
        edge_index_b_clean: torch.Tensor,
        y_b: torch.Tensor,
        train_mask_b: torch.Tensor,
        epoch_seed: int,
        batch_id: int,
        *,
        node_ids_b: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        device = next(model.parameters()).device

        x_b = x_b.to(device)
        edge_index_b_clean = edge_index_b_clean.to(device=device, dtype=torch.long)
        y_b = y_b.to(device)
        if y_b.dim() == 2 and y_b.size(1) == 1:
            y_b = y_b.squeeze(1)
        y_b = y_b.view(-1).to(torch.long)
        train_mask_b = train_mask_b.to(device=device, dtype=torch.bool)
        node_ids_b = None if node_ids_b is None else node_ids_b.to(device=device, dtype=torch.long)

        idx_all = torch.where(train_mask_b)[0]
        if idx_all.numel() == 0:
            return None

        idx = _pick_subset(idx_all, cfg.subset_nodes, seed=int(epoch_seed) + int(batch_id) + int(cfg.seed))
        devices = []
        if x_b.is_cuda and x_b.device.index is not None:
            devices = [x_b.device.index]

        return {
            "x_b": x_b,
            "edge_index_b_clean": edge_index_b_clean,
            "y_b": y_b,
            "train_mask_b": train_mask_b,
            "node_ids_b": node_ids_b,
            "idx_all": idx_all,
            "idx": idx,
            "batch_nodes": int(x_b.size(0)),
            "device": device,
            "devices": devices,
        }

    def _build_batch_snapshot(
        self,
        model: nn.Module,
        x_b: torch.Tensor,
        edge_index_b_clean: torch.Tensor,
        y_b: torch.Tensor,
        train_mask_b: torch.Tensor,
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
        batch_id: int,
        *,
        node_ids_b: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        eps = cfg.eps
        prep = self._prepare_batch_inputs(
            model,
            x_b,
            edge_index_b_clean,
            y_b,
            train_mask_b,
            epoch_seed,
            batch_id,
            node_ids_b=node_ids_b,
        )
        if prep is None:
            return None

        x_b = prep["x_b"]
        edge_index_b_clean = prep["edge_index_b_clean"]
        y_b = prep["y_b"]
        node_ids_b = prep["node_ids_b"]
        idx_all = prep["idx_all"]
        idx = prep["idx"]
        batch_nodes = int(prep["batch_nodes"])
        device = prep["device"]
        devices = prep["devices"]

        cache = BatchGraphCache.from_edge_index(
            edge_index_b_clean,
            batch_nodes,
            node_ids=node_ids_b,
            global_num_nodes_total=getattr(model, "global_num_nodes", batch_nodes),
        )

        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        with torch.random.fork_rng(devices=devices, enabled=True):
            seed = int(epoch_seed) + 1_000_003 * int(batch_id) + int(cfg.seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.set_device(device)
                torch.cuda.manual_seed(seed)

            model.zero_grad(set_to_none=True)
            logits_anchor = model(x_b, edge_index_b_clean, p=0.0, q=0.0, node_ids=node_ids_b)
            loss_data = criterion(logits_anchor[idx], y_b[idx])
            params = [param for param in model.parameters() if param.requires_grad]
            g_data = torch.autograd.grad(loss_data, params, retain_graph=False, create_graph=False, allow_unused=True)
            g_data_norm = _grad_norm(g_data)
            log_gd = math.log(g_data_norm + eps)

            model.zero_grad(set_to_none=True)
            logits, emb = model(
                x_b,
                edge_index_b_clean,
                p=0.0,
                q=0.0,
                node_ids=node_ids_b,
                return_embeddings=True,
            )

            weight = _w_multiclass(logits)
            msg, msg_sq = self._messages_detached(emb, model.pred_local.weight)
            self._gat_modules = self._collect_gat_modules(model) if self.gnn == "gat" else []

            aug_mode_norm = str(aug_mode).lower().strip()
            p_max = self.p_probability_upper if aug_mode_norm in ("both", "drop") else 0.0
            q_max = self.q_probability_upper if aug_mode_norm in ("both", "add") else 0.0

        _restore_batchnorm_buffers(bn_states)
        if not was_training:
            model.eval()

        return {
            "cache": cache,
            "params": params,
            "idx": idx,
            "weight": weight,
            "msg": msg,
            "msg_sq": msg_sq,
            "log_gd": float(log_gd),
            "eps": float(eps),
            "batch_nodes": int(batch_nodes),
            "batch_train_nodes": float(idx_all.numel()),
            "batch_train_used": float(idx.numel()),
            "G_data": float(g_data_norm),
            "p_max": float(p_max),
            "q_max": float(q_max),
        }

    def _build_snapshot_objectives(
        self,
        snapshot: Dict[str, Any],
    ) -> Tuple[
        Callable[[float, float], Tuple[float, float]],
        Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ]:
        objective = self._make_objective(
            cache=snapshot["cache"],
            params=snapshot["params"],
            idx=snapshot["idx"],
            weight=snapshot["weight"],
            msg=snapshot["msg"],
            msg_sq=snapshot["msg_sq"],
            log_gd=snapshot["log_gd"],
            eps=snapshot["eps"],
            batch_nodes=snapshot["batch_nodes"],
        )

        objective_tensor = self._make_objective_tensor(
            cache=snapshot["cache"],
            params=snapshot["params"],
            idx=snapshot["idx"],
            weight=snapshot["weight"],
            msg=snapshot["msg"],
            msg_sq=snapshot["msg_sq"],
            log_gd=snapshot["log_gd"],
            eps=snapshot["eps"],
            batch_nodes=snapshot["batch_nodes"],
        )

        return objective, objective_tensor

    def _solve_snapshot(
        self,
        snapshot: Dict[str, Any],
        *,
        current_p: float,
        current_q: float,
        aug_mode: str,
        device: torch.device,
    ) -> Tuple[float, float, Dict[str, Any]]:
        cfg = self.cfg
        aug_mode = str(aug_mode).lower().strip()
        g_data_norm = float(snapshot["G_data"])
        objective, objective_tensor = self._build_snapshot_objectives(snapshot)

        with torch.random.fork_rng(
            devices=[device.index] if device.type == "cuda" and device.index is not None else [],
            enabled=True,
        ):
            self._ensure_adam_state(
                current_p=float(current_p),
                current_q=float(current_q),
                device=device,
            )

            p_before, q_before = self._adam_rates_tensor()
            if aug_mode == "drop":
                q_before = q_before * 0.0
            elif aug_mode == "add":
                p_before = p_before * 0.0
            elif aug_mode == "none":
                p_before = p_before * 0.0
                q_before = q_before * 0.0

            if self._adam_opt is None:
                raise RuntimeError("Adam pq controller is not initialized.")
            self._adam_opt.zero_grad(set_to_none=True)
            obj_t, greg_t = objective_tensor(p_before, q_before)
            obj_t.backward(retain_graph=True)
            self._adam_opt.step()
            self._project_adam_state_()

            p_new, q_new = self._adam_rates_float(aug_mode)
            obj_new, greg_new = objective(float(p_new), float(q_new))

        best_p = float(p_before.detach().item())
        best_q = float(q_before.detach().item())
        best_greg = float(greg_t.detach().item())
        best_obj = float(obj_t.detach().item())

        info: Dict[str, Any] = {
            "G_data": float(g_data_norm),
            "G_reg_best": float(best_greg),
            "p_best": float(best_p),
            "q_best": float(best_q),
            "rho_best": float(self._rho_from_q_float(best_q)),
            "p_new": float(p_new),
            "q_new": float(q_new),
            "rho_new": float(self._rho_from_q_float(q_new)),
            "obj": float(best_obj),
            "G_reg_new": float(greg_new),
            "obj_new": float(obj_new),
            "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
            "densification_penalty_best": float(
                self._densification_penalty_float(self._rho_from_q_float(best_q))
            ),
            "rho_penalty_best": float(self._densification_penalty_float(self._rho_from_q_float(best_q))),
            "deletion_penalty_new": float(self._deletion_penalty_float(p_new)),
            "densification_penalty_new": float(
                self._densification_penalty_float(self._rho_from_q_float(q_new))
            ),
            "rho_penalty_new": float(self._densification_penalty_float(self._rho_from_q_float(q_new))),
            "batch_train_nodes": float(snapshot["batch_train_nodes"]),
            "batch_train_used": float(snapshot["batch_train_used"]),
            "optimize_rho": bool(self.optimize_rho),
            "optimization_variable": self._add_parameter_name(),
            "deletion_penalty_lambda": float(getattr(cfg, "deletion_penalty_lambda", 0.0)),
            "densification_penalty_lambda": float(getattr(cfg, "densification_penalty_lambda", 0.0)),
            "n_eval": 2.0,
            "solver_success": 1.0,
            "adam_lr": float(cfg.adam_lr),
        }
        return float(p_new), float(q_new), info

    def suggest_pq_for_batch(
        self,
        model: nn.Module,
        x_b: torch.Tensor,
        edge_index_b_clean: torch.Tensor,
        y_b: torch.Tensor,
        train_mask_b: torch.Tensor,
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
        batch_id: int,
        *,
        node_ids_b: Optional[torch.Tensor] = None,
        mask_sharing: str = "shared",
        num_layers: int = 1,
    ) -> Tuple[float, float, Dict[str, Any]]:
        _ = mask_sharing, num_layers
        device = next(model.parameters()).device
        snapshot = self._build_batch_snapshot(
            model=model,
            x_b=x_b,
            edge_index_b_clean=edge_index_b_clean,
            y_b=y_b,
            train_mask_b=train_mask_b,
            criterion=criterion,
            current_p=current_p,
            current_q=current_q,
            aug_mode=aug_mode,
            epoch_seed=epoch_seed,
            batch_id=batch_id,
            node_ids_b=node_ids_b,
        )
        if snapshot is None:
            return 0.0, 0.0, {"skip": 1.0}
        return self._solve_snapshot(
            snapshot,
            current_p=current_p,
            current_q=current_q,
            aug_mode=aug_mode,
            device=device,
        )
