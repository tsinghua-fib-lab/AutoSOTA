from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from augmentation import BernoulliEdgeAugmentor
from rade_convs import (
    GraphCache,
    _delta_E_inv_caseA,
    _delta_E_inv_caseB,
    _delta_E_inv_sqrt_caseA,
    _delta_E_inv_sqrt_caseB,
    _scatter_add_rows,
)


def _validate_rade_variant(rade_variant: str) -> str:
    rade_variant = str(rade_variant).lower().strip()
    if rade_variant not in {"rade-of", "rade-ofs"}:
        raise ValueError(
            f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}"
        )
    return rade_variant


# Keep p strictly below 1 because the GradNorm objective uses p / (1 - p).
P_PROBABILITY_UPPER: float = 1.0 - 1e-6
Q_PROBABILITY_UPPER: float = 1.0


def _grad_norm(grads) -> float:
    s = 0.0
    for g in grads:
        if g is None:
            continue
        s += float((g * g).sum().item())
    return math.sqrt(max(s, 0.0))


def _dot(grads_a, grads_b) -> float:
    s = 0.0
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            continue
        s += float((ga * gb).sum().item())
    return s


def _pick_subset(idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    # k == 0 means: use full idx (full-batch setting)
    if k <= 0 or k >= idx.numel():
        return idx
    g = torch.Generator(device=idx.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(idx.numel(), generator=g, device=idx.device)
    return idx[perm[:k]]


def _w_multiclass(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    return prob * (1.0 - prob)


def _freeze_batchnorm_buffers(model: nn.Module) -> List[Tuple[nn.Module, bool]]:
    """
    Put BN modules into eval() so running_mean/var buffers are not updated during tuning,
    while the overall model remains in train() (so dropout stays on if you use it).
    Returns list of (bn_module, was_training) to restore later.
    """
    states = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            states.append((m, m.training))
            m.eval()
    return states


def _restore_batchnorm_buffers(states: List[Tuple[nn.Module, bool]]) -> None:
    for m, was_training in states:
        m.train(was_training)


@dataclass
class PQTunerConfig:
    eps: float = 1e-12
    subset_nodes: int = 0   # 0 => full train_idx
    seed: int = 0
    deletion_penalty_lambda: float = 0.0
    densification_penalty_lambda: float = 1.0
    # If False, Adam optimizes the Bernoulli add probability q.
    # If True, Adam optimizes rho = q / d and converts rho -> q before evaluating G_reg.
    optimize_rho: bool = False
    # Adam optimizes p and one add-side variable:
    # q by default, or rho when optimize_rho=True.
    cap_p: bool = False
    p_max: float = 0.9
    cap_rho: bool = False
    rho_max: float = 0.2
    adam_lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


class PQGradNormTuner:
    """
    Epoch-wise GradNorm matcher for RADE-OF / RADE-OFS.

    Search option:
    - adam: one stateful Adam step on projected p and the add-side variable

    Optimization variables:
    - By default, the online controllers optimize (p, q).
    - With cfg.optimize_rho=True, they optimize (p, rho), where rho = q / d.
      The graph perturbation model is still parameterized by q, so G_reg is always
      evaluated at q = rho * d. The densification penalty is evaluated on rho.

    Evaluation paths:
    - GIN: fast path via two base gradients
    - GCN/GAT: each candidate/objective evaluation does one autograd.grad on the detached proxy

    GAT uses the sampled-index moment estimator.
    """

    def __init__(
        self,
        edge_index_clean: torch.Tensor,
        num_nodes: int,
        gnn: str,
        rade_variant: str,
        cfg: PQTunerConfig,
    ):
        self.cfg = cfg
        self.gnn = str(gnn).lower().strip()                # "gin", "gcn", or "gat"
        self.rade_variant = _validate_rade_variant(rade_variant)
        if self.gnn not in {"gin", "gcn", "gat"}:
            raise ValueError(f"gnn must be one of {{'gin', 'gcn', 'gat'}}. Got {self.gnn}")
        self.optimize_rho = bool(getattr(cfg, "optimize_rho", False))
        self.cap_p = bool(getattr(cfg, "cap_p", False))
        p_max_cfg = float(getattr(cfg, "p_max", 0.9))
        if not (0.0 <= p_max_cfg < 1.0):
            raise ValueError(f"PQTunerConfig.p_max must be in [0, 1). Got {p_max_cfg}")
        self.p_probability_upper = min(P_PROBABILITY_UPPER, p_max_cfg) if self.cap_p else P_PROBABILITY_UPPER
        self.cache = GraphCache.from_edge_index(edge_index_clean, int(num_nodes))
        self.N = int(num_nodes)

        self.edge_index_clean = edge_index_clean
        self.augmentor = BernoulliEdgeAugmentor.from_edge_index(
            edge_index_clean, num_nodes=self.N
        )

        self.clean_edges_undirected = self._num_clean_edges_undirected()
        self.non_edges_undirected = self._num_non_edges_undirected()
        total_pairs = max(float(self.clean_edges_undirected + self.non_edges_undirected), 1.0)
        self.graph_density = float(self.clean_edges_undirected) / total_pairs
        if self.graph_density <= 0.0:
            self.density_ratio_scale = 0.0
        else:
            self.density_ratio_scale = 1.0 / float(self.graph_density)

        # We use the sparse-graph approximation rho = q / d, where d is graph density.
        # This scale is the only conversion factor between the optimization budget
        # variable rho and the actual Bernoulli add probability q.
        self.expected_add_ratio_scale = float(self.density_ratio_scale)

        self.rho_probability_upper = float(self.expected_add_ratio_scale) * float(Q_PROBABILITY_UPPER)
        self.cap_rho = bool(getattr(cfg, "cap_rho", False))
        rho_max_cfg = float(getattr(cfg, "rho_max", 0.2))
        if rho_max_cfg < 0.0:
            raise ValueError(f"PQTunerConfig.rho_max must be >= 0. Got {rho_max_cfg}")
        self.rho_upper = min(self.rho_probability_upper, rho_max_cfg) if self.cap_rho else self.rho_probability_upper
        self.q_probability_upper = (
            min(Q_PROBABILITY_UPPER, self._q_from_rho_float(self.rho_upper))
            if self.cap_rho
            else Q_PROBABILITY_UPPER
        )

        # Stateful Adam controller. The add-side
        # parameter is either raw q or rho, controlled by cfg.optimize_rho.
        self._adam_p: Optional[nn.Parameter] = None
        self._adam_add_param: Optional[nn.Parameter] = None
        self._adam_opt: Optional[torch.optim.Optimizer] = None
        self._adam_device: Optional[torch.device] = None

    def _num_non_edges_undirected(self) -> int:
        # Assumes edge_index_clean_dir contains both directions for each undirected edge.
        m_dir = int(self.cache.edge_index_clean_dir.size(1))
        m_undir = m_dir // 2
        n = int(self.N)
        return n * (n - 1) // 2 - m_undir

    def _num_clean_edges_undirected(self) -> int: # |E|
        return int(self.cache.edge_index_clean_dir.size(1)) // 2

    def _rho_from_q_float(self, q: float) -> float:
        return float(q) * float(self.expected_add_ratio_scale) # rho = q / d

    def _rho_from_q_tensor(self, q: torch.Tensor) -> torch.Tensor:
        return q * q.new_tensor(float(self.expected_add_ratio_scale))

    def _q_from_rho_float(self, rho: float) -> float: # q = rho.d
        scale = float(self.expected_add_ratio_scale)
        if scale <= 0.0:
            return 0.0
        return float(rho) / scale

    def _q_from_rho_tensor(self, rho: torch.Tensor) -> torch.Tensor: # q = rho * d
        scale = float(self.expected_add_ratio_scale)
        if scale <= 0.0:
            return rho.new_zeros(())
        return rho / rho.new_tensor(scale)

    def _add_parameter_name(self) -> str:
        # Name of the variable the online optimizer stores and updates.
        # This is only an optimizer parameterization choice; RADE sampling and
        # G_reg still use q.
        return "rho" if self.optimize_rho else "q"

    def _add_parameter_upper(self) -> float:
        # q is a probability in [0, 1]. If optimizing rho, the equivalent upper
        # bound is rho(q=1) = 1 / d.
        return float(self.rho_upper if self.optimize_rho else self.q_probability_upper)

    def _add_parameter_from_q_float(self, q: float) -> float:
        # Initialize the online state from the externally visible q.
        if self.optimize_rho:
            return self._rho_from_q_float(q)
        return float(q)

    def _q_from_add_parameter_float(self, add_param: float) -> float:
        # Convert the optimizer's add-side variable into the Bernoulli probability
        # needed by augmentation statistics and G_reg.
        if self.optimize_rho:
            return self._q_from_rho_float(add_param)
        return float(add_param)

    def _q_from_add_parameter_tensor(self, add_param: torch.Tensor) -> torch.Tensor:
        # When optimizing
        # rho, gradients flow through q = rho * d into the G_reg term, while the
        # penalty term is still computed from rho after q is mapped back to rho.
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
        return self._deletion_penalty_float(p) + self._densification_penalty_float(
            self._rho_from_q_float(q)
        )

    def _objective_penalty_tensor(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self._deletion_penalty_tensor(p) + self._densification_penalty_tensor(
            self._rho_from_q_tensor(q)
        )

    @torch.no_grad()
    def _messages_detached(
        self, emb: torch.Tensor, W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = emb.detach()
        W = W.detach()
        m = h @ W.t()      # [N, C]
        return m, m * m

    def _var_gin_bases(
        self, m: torch.Tensor, m_sq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = self.cache.edge_index_clean_dir
        neigh_sum_sq = _scatter_add_rows(m_sq[row], col, self.N)  # [N, C]
        comp_sum_sq = self.cache.complement_sum_unweighted(m_sq)  # [N, C]

        if self.rade_variant == "rade-of":
            comp_sum_m = self.cache.complement_sum_unweighted(m)  # [N, C]
            comp_size = self.cache.comp_size.to(m.dtype).clamp_min(1.0).unsqueeze(1)
            mu = comp_sum_m / comp_size
            add_centered = comp_sum_sq - 2.0 * mu * comp_sum_m + comp_size * (mu * mu)
            return neigh_sum_sq, add_centered

        # RADE-OFS
        return neigh_sum_sq, comp_sum_sq

    @torch.no_grad()
    def _var_gcn(self, p: float, q: float, m: torch.Tensor, m_sq: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.eps
        row, col = self.cache.edge_index_clean_dir

        deg = self.cache.deg_clean.to(torch.float32)
        comp = self.cache.comp_size.to(torch.float32)
        d_hat = (deg + 1.0).clamp_min(1.0)

        # --- retained clean edges: Case A ---
        tA = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        uA = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)

        E = (1.0 - p) * (tA[row] * tA[col])
        E2 = (1.0 - p) * (uA[row] * uA[col])
        VarG = (E2 - E * E).clamp_min(0.0)

        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])  # clean alpha^2
        factor = alpha_sq * (VarG / (E * E + eps))  # per-directed-edge
        neigh_var = _scatter_add_rows(factor.unsqueeze(1) * m_sq[row], col, self.N)

        # --- clean non-edges: Case B ---
        tB = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        uB = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        tB2 = tB * tB

        if self.rade_variant == "rade-of":
            # weighted centering with weights proportional to tB[j]
            mu = self.cache.complement_mean_weighted(m, tB)

            ones = torch.ones((self.N, 1), device=m.device, dtype=m.dtype)

            # S1 = sum uB[j] (m_j - mu_i)^2
            A_u = self.cache.complement_sum_weighted(m_sq, uB)
            B_u = self.cache.complement_sum_weighted(m, uB)
            C_u = self.cache.complement_sum_weighted(ones, uB)
            S1 = A_u - 2.0 * mu * B_u + (mu * mu) * C_u

            # S2 = sum tB[j]^2 (m_j - mu_i)^2
            A_t = self.cache.complement_sum_weighted(m_sq, tB2)
            B_t = self.cache.complement_sum_weighted(m, tB2)
            C_t = self.cache.complement_sum_weighted(ones, tB2)
            S2 = A_t - 2.0 * mu * B_t + (mu * mu) * C_t
        else:
            # RADE-OFS: uncentered non-edge contributions
            S1 = self.cache.complement_sum_weighted(m_sq, uB)
            S2 = self.cache.complement_sum_weighted(m_sq, tB2)

        add_var = (q * uB.unsqueeze(1) * S1) - ((q * q) * tB2.unsqueeze(1) * S2)

        return neigh_var + add_var

    def _var_gcn_tensor(
            self,
            p: torch.Tensor,
            q: torch.Tensor,
            m: torch.Tensor,
            m_sq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate Var(delta_i) for GCN under RADE.

        Output:
            V: [N, C], where V[i, c] approximates Var(delta_{i,c}). # c is classes.
            I remove it in other comments for readability.

        Main idea:
            For GCN, alpha_{ij}^{G'} = A'_{ij} / sqrt(d'_i d'_j).
            Thus, Var(alpha_{ij}^{G'}) depends on the random perturbed degrees.
            Unlike GIN, we cannot write the variance as a fixed statistic times
            p/(1-p) or q(1-q). We recompute the GCN variance approximation for
            each candidate (p, q).
        """
        eps = self.cfg.eps
        row, col = self.cache.edge_index_clean_dir  # row = source j, col = target i

        deg = self.cache.deg_clean.to(torch.float32)
        comp = self.cache.comp_size.to(torch.float32)

        # GCN uses self-loop degree. For an edge j -> i,
        # (alpha_{ij}^G)^2 = 1 / (d_hat[j] d_hat[i]).
        d_hat = (deg + 1.0).clamp_min(1.0)

        # ------------------------------------------------------------------
        # Existing-edge / drop side: A_ij = 1
        # ------------------------------------------------------------------
        # For an edge j -> i,
        #
        #   alpha_{ij}^{G'} = A'_{ij} / sqrt(d'_j d'_i),
        #
        # where A'_{ij} is kept with probability (1-p), and d'_j,d'_i
        # are perturbed degrees. Case A gives delta-method approximations
        # for the degree moments on existing edges.
        #
        # tA[k] approx E[(d'_k)^(-1/2)] under the existing-edge case.
        # uA[k] approx E[(d'_k)^(-1)]   under the existing-edge case.
        tA = _delta_E_inv_sqrt_caseA(deg, comp, p=p, q=q)
        uA = _delta_E_inv_caseA(deg, comp, p=p, q=q, eps=eps)

        # First and second moments of the perturbed GCN coefficient
        # on each clean edge j -> i:
        #
        #   E  is E[alpha_{ij}^{G'}]
        #      = (1-p) tA[j] tA[i]
        #
        #   E2 is E[(alpha_{ij}^{G'})^2]
        #      = (1-p) uA[j] uA[i]
        #
        #   VarG is Var(alpha_{ij}^{G'})
        #        = E[(alpha_{ij}^{G'})^2] - E[alpha_{ij}^{G'}]^2.
        E = (1.0 - p) * (tA[row] * tA[col])
        E2 = (1.0 - p) * (uA[row] * uA[col])
        VarG = (E2 - E * E).clamp_min(0.0)

        # RADE-OF/RADE-OFS corrects existing-edge messages by
        #
        #   alpha_{ij}^G / E[alpha_{ij}^{G'}].
        #
        # Thus, the variance contribution from an edge j -> i is
        #
        #   m_j^2 (alpha_{ij}^G)^2
        #   Var(alpha_{ij}^{G'}) / E[alpha_{ij}^{G'}]^2.
        #
        # Here factor is the coefficient multiplying m_j^2.
        alpha_sq = 1.0 / (d_hat[row] * d_hat[col])
        factor = alpha_sq * (VarG / (E * E + eps))

        # Sum the edge/drop contributions over all neighbors j of i:
        #
        #   neigh_var[i]
        #   = sum_{j:A_ij=1} m_j^2 (alpha_{ij}^G)^2
        #     Var(alpha_{ij}^{G'}) / E[alpha_{ij}^{G'}]^2.
        neigh_var = _scatter_add_rows(
            factor.unsqueeze(1) * m_sq[row],
            col,
            self.N,
        )

        # ------------------------------------------------------------------
        # Non-edge / add side: A_ij = 0
        # ------------------------------------------------------------------
        # For a non-edge j -> i,
        #
        #   alpha_{ij}^{G'} = A'_{ij} / sqrt(d'_j d'_i),
        #
        # where A'_{ij} is added with probability q.
        # Case B gives delta-method approximations for degree moments
        # on candidate added non-edges.
        #
        # tB[k] approx E[(d'_k)^(-1/2)] under the non-edge case.
        # uB[k] approx E[(d'_k)^(-1)]   under the non-edge case.
        tB = _delta_E_inv_sqrt_caseB(deg, comp, p=p, q=q)
        uB = _delta_E_inv_caseB(deg, comp, p=p, q=q, eps=eps)
        tB2 = tB * tB

        if self.rade_variant == "rade-of":
            # RADE-OF centers added non-edge messages.
            #
            # For GCN, E[alpha_{ij}^{G'}] approx q tB[i] tB[j].
            # For fixed target node i, q and tB[i] are constant across j.
            # Thus, the weighted non-neighbor mean is
            #
            #   mu_i = sum_{j:A_ij=0} tB[j] m_j
            #          / sum_{j:A_ij=0} tB[j].
            #
            # This is the GCN version of the RADE-OF centering term.
            mu = self.cache.complement_mean_weighted(m, tB)

            ones = torch.ones((self.N, 1), device=m.device, dtype=m.dtype)

            # We need the add-side variance
            #
            #   sum_{j:A_ij=0} Var(alpha_{ij}^{G'})
            #       (m_j - mu_i)^2.
            #
            # For non-edges,
            #
            #   Var(alpha_{ij}^{G'})
            #   approx q uB[i] uB[j] - q^2 tB[i]^2 tB[j]^2.
            #
            # So we compute two weighted complement sums:
            #
            #   S1[i] = sum_{j:A_ij=0} uB[j]  (m_j - mu_i)^2
            #   S2[i] = sum_{j:A_ij=0} tB[j]^2(m_j - mu_i)^2.
            #
            # The code uses the expansion
            #
            #   sum_j w_j (m_j - mu_i)^2
            #   = sum_j w_j m_j^2
            #     - 2 mu_i sum_j w_j m_j
            #     + mu_i^2 sum_j w_j.
            #
            # First weighted sum: weights w_j = uB[j].
            A_u = self.cache.complement_sum_weighted(m_sq, uB)
            B_u = self.cache.complement_sum_weighted(m, uB)
            C_u = self.cache.complement_sum_weighted(ones, uB)
            S1 = A_u - 2.0 * mu * B_u + (mu * mu) * C_u

            # Second weighted sum: weights w_j = tB[j]^2.
            A_t = self.cache.complement_sum_weighted(m_sq, tB2)
            B_t = self.cache.complement_sum_weighted(m, tB2)
            C_t = self.cache.complement_sum_weighted(ones, tB2)
            S2 = A_t - 2.0 * mu * B_t + (mu * mu) * C_t

        else:
            # RADE-OFS does not center added non-edge messages.
            # The non-edge contribution is retained, so we use m_j^2 directly:
            #
            #   S1[i] = sum_{j:A_ij=0} uB[j]   m_j^2
            #   S2[i] = sum_{j:A_ij=0} tB[j]^2 m_j^2.
            S1 = self.cache.complement_sum_weighted(m_sq, uB)
            S2 = self.cache.complement_sum_weighted(m_sq, tB2)

        # Combine the two non-edge moments:
        #
        #   add_var[i]
        #   = q uB[i] S1[i] - q^2 tB[i]^2 S2[i]
        #
        # which is equivalent to
        #
        #   sum_{j:A_ij=0}
        #   [q uB[i]uB[j] - q^2 tB[i]^2tB[j]^2]
        #   * corrected_message_{ij}^2.
        #
        # For RADE-OF:
        #   corrected_message_{ij} = m_j - mu_i.
        #
        # For RADE-OFS:
        #   corrected_message_{ij} = m_j.
        add_var = (q * uB.unsqueeze(1) * S1) - ((q * q) * tB2.unsqueeze(1) * S2)

        # Final GCN approximation:
        #
        #   Var(delta_{i,c}) = existing-edge variance + non-edge variance.
        return neigh_var + add_var

    @staticmethod
    def _collect_gat_modules(model: nn.Module) -> List[nn.Module]:
        modules = [
            module
            for module in model.modules()
            if callable(getattr(module, "gat_variance_proxy", None))
        ]
        return modules[-1:] if modules else []

    def _var_gat(self, p: float, q: float, m: torch.Tensor, m_sq: torch.Tensor) -> torch.Tensor:
        modules = getattr(self, "_gat_modules", None)
        if not modules:
            return torch.zeros_like(m_sq)
        estimates = [
            module.gat_variance_proxy(m, m_sq, p=float(p), q=float(q), eps=float(self.cfg.eps))
            for module in modules
        ]
        if len(estimates) == 1:
            return estimates[0]
        return torch.stack(estimates, dim=0).mean(dim=0)

    def _var_gat_tensor(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        m: torch.Tensor,
        m_sq: torch.Tensor,
    ) -> torch.Tensor:
        modules = getattr(self, "_gat_modules", None)
        if not modules:
            return torch.zeros_like(m_sq) + (p * 0.0 + q * 0.0)
        estimates = [
            module.gat_variance_proxy(m, m_sq, p=p, q=q, eps=float(self.cfg.eps))
            for module in modules
        ]
        out = estimates[0] if len(estimates) == 1 else torch.stack(estimates, dim=0).mean(dim=0)
        return out + (p * 0.0 + q * 0.0)

    def _make_objective(
        self,
        *,
        params,
        idx: torch.Tensor,
        w: torch.Tensor,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        logGd: float,
        eps: float,
    ) -> Callable[[float, float], Tuple[float, float]]:
        if self.gnn == "gin":
            V_drop, V_add = self._var_gin_bases(m, m_sq)

            R_drop = 0.5 * (w[idx] * V_drop[idx]).sum() / max(int(idx.numel()), 1)
            R_add = 0.5 * (w[idx] * V_add[idx]).sum() / max(int(idx.numel()), 1)

            g_drop = torch.autograd.grad(
                R_drop, params, retain_graph=True, create_graph=False, allow_unused=True
            )
            g_add = torch.autograd.grad(
                R_add, params, retain_graph=True, create_graph=False, allow_unused=True
            )

            a = _dot(g_drop, g_drop)
            b = _dot(g_add, g_add)
            c = _dot(g_drop, g_add)

            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                s = p_try / max(1.0 - p_try, eps)   # p / (1-p)
                t = q_try * (1.0 - q_try)           # q(1-q)
                g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                Greg = math.sqrt(max(g2, 0.0))
                obj = (math.log(Greg + eps) - logGd) ** 2 + self._objective_penalty_float(
                    float(p_try), float(q_try)
                )
                return float(obj), float(Greg)

            return objective

        if self.gnn == "gcn":
            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                V = self._var_gcn(float(p_try), float(q_try), m, m_sq)
                R = 0.5 * (w[idx] * V[idx]).sum() / max(int(idx.numel()), 1)

                g_reg = torch.autograd.grad(
                    R, params, retain_graph=True, create_graph=False, allow_unused=True
                )
                Greg = _grad_norm(g_reg)
                obj = (math.log(Greg + eps) - logGd) ** 2 + self._objective_penalty_float(
                    float(p_try), float(q_try)
                )
                return float(obj), float(Greg)

            return objective

        if self.gnn == "gat":
            def objective(p_try: float, q_try: float) -> Tuple[float, float]:
                V = self._var_gat(float(p_try), float(q_try), m, m_sq)
                R = 0.5 * (w[idx] * V[idx]).sum() / max(int(idx.numel()), 1)

                g_reg = torch.autograd.grad(
                    R, params, retain_graph=True, create_graph=False, allow_unused=True
                )
                Greg = _grad_norm(g_reg)
                obj = (math.log(Greg + eps) - logGd) ** 2 + self._objective_penalty_float(
                    float(p_try), float(q_try)
                )
                return float(obj), float(Greg)

            return objective

        raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin', 'gcn', or 'gat'.")

    def _make_objective_tensor(
        self,
        *,
        params,
        idx: torch.Tensor,
        w: torch.Tensor,
        m: torch.Tensor,
        m_sq: torch.Tensor,
        logGd: float,
        eps: float,
    ) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        logGd_t = w.new_tensor(float(logGd))
        eps_t = w.new_tensor(float(eps))

        if self.gnn == "gin":
            V_drop, V_add = self._var_gin_bases(m, m_sq)

            R_drop = 0.5 * (w[idx] * V_drop[idx]).sum() / max(int(idx.numel()), 1)
            R_add = 0.5 * (w[idx] * V_add[idx]).sum() / max(int(idx.numel()), 1)

            g_drop = torch.autograd.grad(
                R_drop, params, retain_graph=True, create_graph=False, allow_unused=True
            )
            g_add = torch.autograd.grad(
                R_add, params, retain_graph=True, create_graph=False, allow_unused=True
            )

            a = w.new_tensor(_dot(g_drop, g_drop))
            b = w.new_tensor(_dot(g_add, g_add))
            c = w.new_tensor(_dot(g_drop, g_add))

            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                one = torch.ones_like(p_try)
                s = p_try / (one - p_try).clamp_min(float(eps))
                t = q_try * (one - q_try)
                g2 = (s * s) * a + (t * t) * b + 2.0 * s * t * c
                Greg = torch.sqrt(torch.clamp_min(g2, eps_t * eps_t))
                obj = (torch.log(Greg + eps_t) - logGd_t) ** 2 + self._objective_penalty_tensor(
                    p_try, q_try
                )
                return obj, Greg

            return objective

        if self.gnn == "gcn":
            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                V = self._var_gcn_tensor(p_try, q_try, m, m_sq)
                R = 0.5 * (w[idx] * V[idx]).sum() / max(int(idx.numel()), 1)

                g_reg = torch.autograd.grad(
                    R, params, retain_graph=True, create_graph=True, allow_unused=True
                )
                g2 = w.new_zeros(())
                for g in g_reg:
                    if g is None:
                        continue
                    g2 = g2 + (g * g).sum()
                Greg = torch.sqrt(torch.clamp_min(g2, eps_t * eps_t))
                obj = (torch.log(Greg + eps_t) - logGd_t) ** 2 + self._objective_penalty_tensor(
                    p_try, q_try
                )
                return obj, Greg

            return objective

        if self.gnn == "gat":
            def objective(p_try: torch.Tensor, q_try: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                V = self._var_gat_tensor(p_try, q_try, m, m_sq)
                R = 0.5 * (w[idx] * V[idx]).sum() / max(int(idx.numel()), 1)

                g_reg = torch.autograd.grad(
                    R, params, retain_graph=True, create_graph=True, allow_unused=True
                )
                g2 = w.new_zeros(())
                for g in g_reg:
                    if g is None:
                        continue
                    g2 = g2 + (g * g).sum()
                Greg = torch.sqrt(torch.clamp_min(g2, eps_t * eps_t))
                obj = (torch.log(Greg + eps_t) - logGd_t) ** 2 + self._objective_penalty_tensor(
                    p_try, q_try
                )
                return obj, Greg

            return objective

        raise ValueError(f"Unsupported gnn={self.gnn}. Expected 'gin', 'gcn', or 'gat'.")

    def _ensure_adam_state(
        self,
        *,
        current_p: float,
        current_q: float,
        device: torch.device,
    ) -> None:
        if (
            self._adam_opt is not None
            and self._adam_device == device
            and self._adam_p is not None
            and self._adam_add_param is not None
        ):
            return

        cfg = self.cfg
        p0 = float(max(0.0, min(current_p, self.p_probability_upper)))
        q0 = float(max(0.0, min(current_q, self.q_probability_upper)))
        # The training loop carries q because q is the actual Bernoulli
        # add probability. The optimizer state may instead store rho; add0 is
        # the selected add-side optimization variable.
        add0 = float(max(0.0, min(self._add_parameter_from_q_float(q0), self._add_parameter_upper())))
        self._adam_p = nn.Parameter(torch.tensor(p0, device=device, dtype=torch.float32))
        self._adam_add_param = nn.Parameter(torch.tensor(add0, device=device, dtype=torch.float32))

        self._adam_opt = torch.optim.Adam(
            [self._adam_p, self._adam_add_param],
            lr=float(cfg.adam_lr),
            betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
            eps=float(cfg.adam_eps),
        )
        self._adam_device = device

    def _adam_rates_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._adam_p is None or self._adam_add_param is None:
            raise RuntimeError("Adam p/add controller is not initialized.")
        p = torch.clamp(self._adam_p, 0.0, self.p_probability_upper)
        add_param = torch.clamp(self._adam_add_param, 0.0, self._add_parameter_upper())
        # Return q even when Adam optimizes rho. Downstream objective functions
        # are written in terms of the physical perturbation probability q.
        q = torch.clamp(self._q_from_add_parameter_tensor(add_param), 0.0, self.q_probability_upper)
        return p, q

    @torch.no_grad()
    def _project_adam_state_(self) -> None:
        if self._adam_p is None or self._adam_add_param is None:
            raise RuntimeError("Adam p/add controller is not initialized.")
        self._adam_p.copy_(
            torch.nan_to_num(
                self._adam_p,
                nan=0.0,
                posinf=self.p_probability_upper,
                neginf=0.0,
            )
        )
        self._adam_add_param.copy_(
            torch.nan_to_num(
                self._adam_add_param,
                nan=0.0,
                posinf=self._add_parameter_upper(),
                neginf=0.0,
            )
        )
        if self._adam_opt is not None:
            for state in self._adam_opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.copy_(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        self._adam_p.clamp_(0.0, self.p_probability_upper)
        # Project the stored optimization variable. This is q in the default
        # mode and rho when --pq_optimize_rho is enabled.
        self._adam_add_param.clamp_(0.0, self._add_parameter_upper())

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

    def suggest_pq(
        self,
        model,
        data,
        train_idx: torch.Tensor,
        criterion,
        current_p: float,
        current_q: float,
        aug_mode: str,
        epoch_seed: int,
        mask_sharing: str = "shared",
        num_layers: int = 1,
    ) -> Tuple[float, float, Dict[str, Any]]:
        cfg = self.cfg
        eps = cfg.eps

        idx = _pick_subset(train_idx, cfg.subset_nodes, seed=epoch_seed)

        was_training = model.training
        model.train()
        bn_states = _freeze_batchnorm_buffers(model)

        devices = []
        if data.x.is_cuda and data.x.device.index is not None:
            devices = [data.x.device.index]

        try:
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.manual_seed(int(epoch_seed) + int(cfg.seed))
                if devices:
                    torch.cuda.set_device(devices[0])
                    torch.cuda.manual_seed(int(epoch_seed) + int(cfg.seed))

                model.zero_grad(set_to_none=True)

                # Clean-only anchor for the data gradient norm.
                logits_anchor = model(data.x, data.edge_index, p=0.0, q=0.0)
                loss_data = criterion(logits_anchor[idx], data.y[idx])
                params = [param for param in model.parameters() if param.requires_grad]
                g_data = torch.autograd.grad(
                    loss_data, params, retain_graph=False, create_graph=False, allow_unused=True
                )
                G_data = _grad_norm(g_data)
                logGd = math.log(G_data + eps)

                model.zero_grad(set_to_none=True)
                logits, emb = model(data.x, data.edge_index, p=0.0, q=0.0, return_embeddings=True)

                W = model.pred_local.weight
                m, m_sq = self._messages_detached(emb, W)
                self._gat_modules = self._collect_gat_modules(model) if self.gnn == "gat" else []

                w = _w_multiclass(logits)

                aug_mode = str(aug_mode).lower().strip()
                p_upper = self.p_probability_upper if aug_mode in ("both", "drop") else 0.0
                q_upper = self.q_probability_upper if aug_mode in ("both", "add") else 0.0

                objective = self._make_objective(
                    params=params,
                    idx=idx,
                    w=w,
                    m=m,
                    m_sq=m_sq,
                    logGd=logGd,
                    eps=eps,
                )
                objective_tensor = self._make_objective_tensor(
                    params=params,
                    idx=idx,
                    w=w,
                    m=m,
                    m_sq=m_sq,
                    logGd=logGd,
                    eps=eps,
                )

                self._ensure_adam_state(
                    current_p=float(current_p),
                    current_q=float(current_q),
                    device=data.x.device,
                )

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
                rho_new = float(self._rho_from_q_float(q_new))

                best_p = float(p_before.detach().item())
                best_q = float(q_before.detach().item())
                best_rho = float(self._rho_from_q_float(best_q))
                best_Greg = float(greg_t.detach().item())
                best_obj = float(obj_t.detach().item())
                eps_floor = float(cfg.eps) * (1.0 + 1e-6)

                info = {
                    "G_data": float(G_data),
                    "G_reg_best": float(best_Greg),
                    "p_best": float(best_p),
                    "rho_best": float(best_rho),
                    "q_best": float(best_q),
                    "expected_add_ratio_best": float(best_rho),
                    "deletion_penalty_best": float(self._deletion_penalty_float(best_p)),
                    "densification_penalty_best": float(self._densification_penalty_float(best_rho)),
                    "rho_penalty_best": float(self._densification_penalty_float(best_rho)),
                    "p_new": float(p_new),
                    "rho_new": float(rho_new),
                    "q_new": float(q_new),
                    "expected_add_ratio_new": float(rho_new),
                    "deletion_penalty_new": float(self._deletion_penalty_float(p_new)),
                    "densification_penalty_new": float(self._densification_penalty_float(rho_new)),
                    "rho_penalty_new": float(self._densification_penalty_float(rho_new)),
                    "obj": float(best_obj),
                    "G_reg_new": float(greg_new),
                    "obj_new": float(obj_new),
                    "optimization_variable": self._add_parameter_name(),
                    "optimize_rho": bool(self.optimize_rho),
                    "deletion_penalty_lambda": float(cfg.deletion_penalty_lambda),
                    "densification_penalty_lambda": float(cfg.densification_penalty_lambda),
                    "graph_density": float(self.graph_density),
                    "density_ratio_scale": float(self.density_ratio_scale),
                    "p_cap_enabled": bool(self.cap_p),
                    "p_probability_upper": float(self.p_probability_upper),
                    "rho_cap_enabled": bool(self.cap_rho),
                    "rho_probability_upper": float(self.rho_upper),
                    "q_probability_upper": float(self.q_probability_upper),
                    "n_eval": 2.0,
                    "solver_success": 1.0,
                    "adam_lr": float(cfg.adam_lr),
                    "pq_eps": float(cfg.eps),
                    "G_reg_best_at_eps_floor": bool(best_Greg <= eps_floor),
                    "G_reg_new_at_eps_floor": bool(float(greg_new) <= eps_floor),
                }
                return p_new, q_new, info
        finally:
            _restore_batchnorm_buffers(bn_states)
            if not was_training:
                model.eval()
