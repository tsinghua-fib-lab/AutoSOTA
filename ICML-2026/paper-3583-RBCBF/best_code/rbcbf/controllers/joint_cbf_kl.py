"""Joint KL-CBF controller used in Stage-2 experiments."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from rbcbf.detectors.base import StepObservation
from rbcbf.scorers import MultiRuleScorer

from .base import ControlAction, Controller

LOGGER = logging.getLogger(__name__)


class JointCBFKLController(Controller):
    """
    Constructs joint linear constraints from the multi-label scorer and
    solves a KL-projection inside the candidate set.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        scorer: MultiRuleScorer,
        labels: Sequence[str],
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.labels = list(labels)
        cfg = dict(config or {})
        self.top_k = int(cfg.get("top_k", 32))
        self.support_topk = max(1, int(cfg.get("support_topk", self.top_k)))
        self.temperature = float(cfg.get("temperature", 1.0))
        self.constraint_mode = str(cfg.get("constraint_mode", "cbf_increment"))
        self.eps_shift = float(cfg.get("eps_shift", 0.0))
        self.cbf_gamma = float(cfg.get("cbf_gamma", 1.0))
        self.target_margin = float(cfg.get("target_margin", 0.0))
        self.margin_buffer = float(cfg.get("margin_buffer", 0.0))
        self.dual_max_iters = int(cfg.get("dual_max_iters", cfg.get("max_projection_steps", 200)))
        self.dual_lr = float(cfg.get("dual_lr", 0.5))
        self.constraint_tol = float(cfg.get("dual_tol", cfg.get("constraint_tol", 1e-4)))
        self.lambda_max = float(cfg.get("lambda_max", 100.0))
        backoff_cfg = dict(cfg.get("delta_backoff") or {})
        self.delta_backoff_enabled = bool(backoff_cfg.get("enabled", False))
        self.delta_backoff_max_iter = int(backoff_cfg.get("max_iter", 6))
        self.delta_backoff_min_scale = float(backoff_cfg.get("min_scale", 1 / (2 ** self.delta_backoff_max_iter)))
        self.exp_clip = float(cfg.get("exp_clip", 20.0))
        adaptive_cfg = cfg.get("adaptive_topk", {}) or {}
        self.adaptive_topk_enabled = bool(adaptive_cfg.get("enabled", False))
        self.adaptive_topk_value = int(adaptive_cfg.get("expand_once", self.top_k))
        self.best_effort_objective = str(cfg.get("best_effort_objective", "max_min_next_margin"))
        self.min_prob = float(cfg.get("min_prob", 1e-8))
        self.enable_best_effort = bool(cfg.get("enable_best_effort", False))
        raw_fallback = list(cfg.get("infeasible_fallback", []))
        if not raw_fallback:
            raw_fallback = ["expand_topk", "original_token"]
        if not self.enable_best_effort:
            raw_fallback = [step for step in raw_fallback if step != "best_effort_token"]
        else:
            filtered = [step for step in raw_fallback if step != "best_effort_token"]
            filtered.append("best_effort_token")
            raw_fallback = filtered
        self.fallback_order = raw_fallback
        self.activation_mode = str(cfg.get("activation_mode", "boundary_only"))
        self.candidate_u_threshold = float(cfg.get("candidate_u_threshold", 0.0))
        self.candidate_rule_topk = int(cfg.get("candidate_rule_topk", 0))
        self.candidate_target_margin = float(cfg.get("candidate_target_margin", 0.0))
        self.candidate_target_kappa = float(cfg.get("candidate_target_kappa", 0.0))
        self.candidate_target_schedule = cfg.get("candidate_target_schedule")
        self.enable_target_clipping = bool(cfg.get("enable_target_clipping", False))
        self.clipping_slack = float(cfg.get("clipping_slack", 1e-3))
        self.lp_check_steps = int(cfg.get("lp_check_steps", 150))
        self.lp_check_lr = float(cfg.get("lp_check_lr", 0.05))
        self.active_rules_mode = str(cfg.get("active_rules_mode", "kbd_only"))
        self.bd_threshold = float(cfg.get("bd_threshold", 0.0))
        self.target_mode = str(cfg.get("target_mode", "constant_margin"))
        self.target_delta = float(cfg.get("target_delta", 0.0))
        self.target_delta_cap = float(cfg.get("target_delta_cap", 1.0))
        self.use_argmax = bool(cfg.get("use_argmax", False))
        self._eps = 1e-8

    @staticmethod
    def _decode_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
        return tokenizer.decode([token_id], skip_special_tokens=False)

    def _shift_margins(self, margins: Sequence[float]) -> List[float]:
        return [float(val) + self.eps_shift for val in margins]

    def compute_shifted_margin_matrix(
        self,
        candidate_infos: Sequence[Dict[str, object]],
        active_rules: Sequence[int],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        matrix = torch.zeros(
            len(active_rules),
            len(candidate_infos),
            dtype=torch.float32,
            device=device,
        )
        for r_idx, rule_idx in enumerate(active_rules):
            for c_idx, info in enumerate(candidate_infos):
                shifted = info.get("shifted_margins") or []
                if 0 <= rule_idx < len(shifted):
                    matrix[r_idx, c_idx] = float(shifted[rule_idx])
        return matrix

    def _select_active_rules(
        self,
        base_indices: Sequence[int],
        candidate_context: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        boundary_set = {idx for idx in base_indices if 0 <= idx < len(self.labels)}
        active = set(boundary_set)
        meta = {
            "indices": active,
            "reason": "kbd" if self.active_rules_mode == "kbd_only" else ("boundary" if active else "none"),
            "candidate_used": False,
            "u_total": None,
            "boundary_rules_size": len(boundary_set),
            "candidate_rules_size": 0,
            "candidate_rule_topk_used": self.candidate_rule_topk,
        }
        if self.active_rules_mode == "kbd_only":
            return meta
        if not candidate_context:
            return meta
        is_candidate = bool(candidate_context.get("is_candidate"))
        if not is_candidate:
            return meta
        u_total = float(candidate_context.get("u_total", 0.0))
        meta["u_total"] = u_total
        mode = self.activation_mode
        candidate_ok = mode in {"boundary_or_candidate", "boundary_or_candidate_or_ut"}
        if candidate_ok and mode == "boundary_or_candidate_or_ut":
            candidate_ok = u_total >= self.candidate_u_threshold
        if not candidate_ok:
            return meta
        u_per_rule = candidate_context.get("u_per_rule") or []
        candidates: List[int] = []
        if self.candidate_rule_topk > 0 and u_per_rule:
            ranked = sorted(
                [(idx, float(val)) for idx, val in enumerate(u_per_rule) if val > 0.0],
                key=lambda item: item[1],
                reverse=True,
            )
            candidates = [idx for idx, _ in ranked[: self.candidate_rule_topk]]
        elif u_per_rule:
            candidates = [idx for idx, val in enumerate(u_per_rule) if val > 0.0]
        if not candidates and u_per_rule:
            candidates = list(range(len(u_per_rule)))
        filtered_candidates = [idx for idx in candidates if 0 <= idx < len(self.labels)]
        active.update(filtered_candidates)
        meta.update(
            {
                "indices": active,
                "reason": "candidate" if filtered_candidates else meta["reason"],
                "candidate_used": bool(filtered_candidates),
                "candidate_rules_size": len(filtered_candidates),
            }
        )
        return meta

    def build_constraints(
        self,
        obs: StepObservation,
        prompt_text: str,
        prefix_text: str,
        candidate_tokens: Sequence[int],
        active_indices: Sequence[int],
        candidate_context: Optional[Dict[str, object]] = None,
        forced_active_indices: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, object]:
        if not candidate_tokens:
            return {}
        prefix_shifted = self._shift_margins(obs.h)
        if forced_active_indices:
            forced = [
                idx
                for idx in forced_active_indices
                if isinstance(idx, int) and 0 <= idx < len(self.labels)
            ]
            activation_meta = {
                "indices": set(forced),
                "reason": "forced_kterm",
                "candidate_used": False,
                "u_total": None,
                "boundary_rules_size": len(forced),
                "candidate_rules_size": 0,
                "candidate_rule_topk_used": 0,
            }
        else:
            activation_meta = self._select_active_rules(active_indices, candidate_context)
        active_rules = sorted(activation_meta["indices"])
        if not active_rules:
            return {}

        token_ids = list(candidate_tokens)
        candidate_infos: List[Dict[str, object]] = []
        token_strs = [self._decode_token(self.tokenizer, int(token_id)) for token_id in token_ids]
        candidate_texts = [prefix_text + token_str for token_str in token_strs]
        batch_scores = None
        score_prefix_batch = getattr(self.scorer, "score_prefix_batch", None)
        if callable(score_prefix_batch):
            try:
                batch_scores = score_prefix_batch(prompt_text, candidate_texts)
            except Exception as exc:  # pragma: no cover - safety net
                LOGGER.warning("Failed batch-scoring %s candidate tokens: %s", len(token_ids), exc)
        for idx, token_id in enumerate(token_ids):
            token_str = token_strs[idx]
            try:
                if batch_scores is not None and idx < len(batch_scores):
                    scores = batch_scores[idx]
                else:
                    scores = self.scorer.score_prefix(prompt_text, candidate_texts[idx])
                margins = [float(scores.margins.get(label, 0.0)) for label in self.labels]
                probabilities = [float(scores.probabilities.get(label, 0.0)) for label in self.labels]
            except Exception as exc:  # pragma: no cover - safety net
                LOGGER.warning("Failed to score candidate token %s: %s", token_id, exc)
                margins = list(obs.h)
                probabilities = [0.0 for _ in self.labels]
            candidate_infos.append(
                {
                    "token_id": int(token_id),
                    "token_str": token_str,
                    "margins": margins,
                    "shifted_margins": self._shift_margins(margins),
                    "probabilities": probabilities,
                }
            )

        if not candidate_infos:
            return {}

        candidate_target = None
        if candidate_context and candidate_context.get("is_candidate"):
            candidate_target = self._compute_candidate_target(candidate_context)

        increments: List[List[float]] = []
        bounds: List[float] = []
        row_rule_indices: List[int] = []

        if self.constraint_mode == "cbf_increment":
            for label_idx in active_rules:
                base_shifted = float(prefix_shifted[label_idx])
                if candidate_target is not None:
                    bound = candidate_target - base_shifted
                else:
                    bound = -self.cbf_gamma * base_shifted + self.margin_buffer
                if bound <= 0.0:
                    continue
                row: List[float] = []
                for info in candidate_infos:
                    cand_shifted = float(info["shifted_margins"][label_idx])
                    row.append(cand_shifted - base_shifted)
                increments.append(row)
                bounds.append(bound)
                row_rule_indices.append(int(label_idx))
        else:
            min_target = max(0.0, candidate_target if candidate_target is not None else self.margin_buffer)
            for label_idx in active_rules:
                row = []
                for info in candidate_infos:
                    shifted = info.get("shifted_margins") or []
                    val = float(shifted[label_idx]) if 0 <= label_idx < len(shifted) else 0.0
                    row.append(val)
                increments.append(row)
                bounds.append(min_target)
                row_rule_indices.append(int(label_idx))

        if not increments:
            return {}

        base_shifted_per_rule = [float(prefix_shifted[label_idx]) for label_idx in active_rules]
        meta_info = {
            "active_rules": active_rules,
            "active_rules_rows": row_rule_indices,
            "activation_reason": activation_meta["reason"],
            "candidate_used": activation_meta["candidate_used"],
            "u_total": activation_meta.get("u_total"),
            "target_margin_desired": candidate_target,
            "target_margin_used": candidate_target,
            "target_schedule": (self.candidate_target_schedule or {}).get("type"),
            "base_shifted": base_shifted_per_rule,
            "boundary_rules_size": activation_meta.get("boundary_rules_size", 0),
            "candidate_rules_size": activation_meta.get("candidate_rules_size", 0),
            "candidate_rule_topk_used": activation_meta.get("candidate_rule_topk_used"),
            "delta_per_rule_raw": {},
            "base_bounds_tensor": torch.tensor(bounds, dtype=torch.float32, device=device) if bounds else None,
        }

        inc_tensor = torch.tensor(increments, dtype=torch.float32, device=device)
        bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)
        return {
            "increments": inc_tensor,
            "bounds": bounds_tensor,
            "candidates": candidate_infos,
            "mode": self.constraint_mode,
            "prefix_shifted": prefix_shifted,
            "meta": meta_info,
        }

    def _compute_candidate_target(self, candidate_context: Dict[str, object]) -> Optional[float]:
        target = self.candidate_target_margin
        if self.candidate_target_kappa:
            target += self.candidate_target_kappa * float(candidate_context.get("u_total", 0.0))
        schedule = self.candidate_target_schedule or {}
        schedule_type = schedule.get("type")
        if schedule_type == "constant":
            target = float(schedule.get("value", target))
        elif schedule_type == "linear_u":
            base = float(schedule.get("base", 0.0))
            kappa = float(schedule.get("kappa", 0.0))
            norm = float(schedule.get("norm", 1.0)) or 1.0
            u_total = float(candidate_context.get("u_total", 0.0))
            u_norm = max(0.0, min(1.0, u_total / norm))
            target = base + kappa * u_norm
        return target if target != 0.0 else None

    def analyze_feasibility(self, constraints: Dict[str, object]) -> Dict[str, object]:
        meta = constraints.get("meta") or {}
        active_rules = meta.get("active_rules") or []
        candidates = constraints.get("candidates") or []
        bounds = constraints.get("bounds")
        result = {
            "active_rules_size": len(active_rules),
            "boundary_rules_size": meta.get("boundary_rules_size", 0),
            "candidate_rules_size": meta.get("candidate_rules_size", 0),
            "candidate_rule_topk_used": meta.get("candidate_rule_topk_used"),
            "max_next_per_rule": [],
            "max_next_per_rule_min": None,
            "headroom_tokenwise": None,
            "rule_upper_bound_fail": False,
            "lp_feasible": None,
            "lp_max_violation": None,
            "tau_per_rule": [],
        }
        if not active_rules or not candidates or bounds is None or bounds.numel() == 0:
            return result
        num_rules = len(active_rules)
        num_candidates = len(candidates)
        device = bounds.device if isinstance(bounds, torch.Tensor) else None
        matrix = self.compute_shifted_margin_matrix(candidates, active_rules, device=device)
        max_per_rule = torch.max(matrix, dim=1).values
        result["max_next_per_rule"] = max_per_rule.tolist()
        if max_per_rule.numel() > 0:
            result["max_next_per_rule_min"] = float(torch.min(max_per_rule).item())
        headroom = torch.min(matrix, dim=0).values
        if headroom.numel() > 0:
            result["headroom_tokenwise"] = float(torch.max(headroom).item())
        mode = constraints.get("mode", "cbf_increment")
        tau_per_rule: List[float] = []
        bounds_len = bounds.numel() if bounds is not None else 0
        if mode == "cbf_increment":
            base_shifted = meta.get("base_shifted") or [0.0] * num_rules
            for idx in range(num_rules):
                base_val = float(base_shifted[idx]) if idx < len(base_shifted) else 0.0
                bound_val = float(bounds[idx].item()) if idx < bounds_len else 0.0
                tau = base_val + bound_val
                tau_per_rule.append(tau)
        else:
            for idx in range(num_rules):
                bound_val = float(bounds[idx].item()) if idx < bounds_len else 0.0
                tau_per_rule.append(bound_val)
        result["tau_per_rule"] = tau_per_rule
        rule_upper_fail = False
        for idx, tau in enumerate(tau_per_rule):
            if idx < len(max_per_rule) and tau > float(max_per_rule[idx]) + self.constraint_tol:
                rule_upper_fail = True
                break
        result["rule_upper_bound_fail"] = rule_upper_fail
        lp_feasible = None
        lp_max_violation = None
        if num_candidates > 0:
            feasible, max_violation = self._approximate_lp_feasible(
                matrix,
                torch.tensor(tau_per_rule, dtype=torch.float32, device=matrix.device),
            )
            lp_feasible = feasible
            lp_max_violation = max_violation
        result["lp_feasible"] = lp_feasible
        result["lp_max_violation"] = lp_max_violation
        return result

    def _approximate_lp_feasible(self, matrix: torch.Tensor, targets: torch.Tensor) -> tuple[bool, float]:
        num_candidates = matrix.shape[1]
        if num_candidates == 0:
            return False, float("inf")
        device = matrix.device
        q = torch.full((num_candidates,), 1.0 / num_candidates, dtype=torch.float32, device=device)
        targets = targets.to(device=device, dtype=torch.float32)
        best_violation = float("inf")
        for _ in range(self.lp_check_steps):
            lhs = matrix.matmul(q)
            violation = targets - lhs
            max_violation = float(torch.max(violation).item())
            best_violation = min(best_violation, max_violation)
            if max_violation <= self.constraint_tol:
                return True, max_violation
            grad = -matrix.t().matmul(torch.clamp(violation, min=0.0))
            q = q + self.lp_check_lr * grad
            q = self._project_simplex(q)
        lhs = matrix.matmul(q)
        violation = targets - lhs
        max_violation = float(torch.max(violation).item())
        return False, max_violation

    @staticmethod
    def _project_simplex(vec: torch.Tensor) -> torch.Tensor:
        if vec.numel() == 0:
            return vec
        v = vec.clone()
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0)
        rho = torch.nonzero(u * torch.arange(1, len(u) + 1, device=v.device) > (cssv - 1), as_tuple=False)
        if rho.numel() == 0:
            return torch.full_like(v, 1.0 / len(v))
        rho = rho[-1, 0] + 1
        theta = (cssv[rho - 1] - 1) / rho
        w = torch.clamp(v - theta, min=0.0)
        denom = torch.sum(w)
        if denom <= 0:
            w = torch.full_like(v, 1.0 / len(v))
        else:
            w = w / denom
        return w

    def summarize_projection(
        self,
        constraints: Dict[str, object],
        base_probs: torch.Tensor,
        solution: Optional[Dict[str, object]] = None,
        analysis: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {}
        meta = constraints.get("meta") or {}
        active_rules = meta.get("active_rules") or []
        candidates = constraints.get("candidates") or []
        if not active_rules or not candidates or base_probs.numel() == 0:
            return summary
        probs = torch.clamp(base_probs, min=self._eps)
        probs = probs / probs.sum()
        matrix = self.compute_shifted_margin_matrix(candidates, active_rules, device=base_probs.device)
        if matrix.numel() == 0:
            return summary
        exp_vals = matrix.matmul(probs)
        max_vals = torch.max(matrix, dim=1).values
        tau_list = (analysis or {}).get("tau_per_rule") or []
        tau = torch.zeros(len(active_rules), dtype=torch.float32, device=matrix.device)
        for idx in range(len(active_rules)):
            tau[idx] = float(tau_list[idx]) if idx < len(tau_list) else 0.0
        slack_p = exp_vals - tau
        binding_under_p = bool(slack_p.numel() > 0 and torch.min(slack_p).item() < -self.constraint_tol)
        headroom_per_rule = max_vals - tau
        headroom_per_rule_min = (
            float(torch.min(headroom_per_rule).item()) if headroom_per_rule.numel() > 0 else None
        )
        headroom_tokenwise = (analysis or {}).get("headroom_tokenwise")
        entropy_p = float(-(probs * torch.log(torch.clamp(probs, min=self._eps))).sum().item())
        top1_p = float(torch.max(probs).item())
        diag: Dict[str, object] = {
            "ep_m": exp_vals.tolist(),
            "max_m": max_vals.tolist(),
            "gk": (max_vals - exp_vals).tolist(),
            "slack_p": slack_p.tolist(),
            "slack_p_min": float(torch.min(slack_p).item()) if slack_p.numel() > 0 else None,
            "binding_under_p": binding_under_p,
            "headroom_per_rule": headroom_per_rule.tolist(),
            "headroom_per_rule_min": headroom_per_rule_min,
            "headroom_tokenwise": headroom_tokenwise,
            "entropy_p": entropy_p,
            "top1_prob_p": top1_p,
        }
        if solution and "distribution" in solution:
            q = solution["distribution"]
            if isinstance(q, list):
                q_vec = torch.tensor(q, dtype=torch.float32, device=matrix.device)
            else:
                q_vec = q.clone().float().to(matrix.device)
            q_vec = torch.clamp(q_vec, min=self._eps)
            q_vec = q_vec / q_vec.sum()
            tv = 0.5 * torch.sum(torch.abs(q_vec - probs)).item()
            entropy_q = float(-(q_vec * torch.log(torch.clamp(q_vec, min=self._eps))).sum().item())
            diag["tv_q_p"] = float(tv)
            diag["entropy_q"] = entropy_q
            diag["eq_m"] = matrix.matmul(q_vec).tolist()
            diag["eq_minus_target"] = (matrix.matmul(q_vec) - tau).tolist()
        lambdas = solution.get("lambdas") if solution else None
        if lambdas:
            lam_tensor = torch.tensor(lambdas, dtype=torch.float32, device=matrix.device)
            if lam_tensor.numel() > 0:
                frac = float((lam_tensor >= (self.lambda_max - 1e-6)).float().mean().item())
                diag["lambda_clip_fraction"] = frac
            diag["lambda_norm"] = float(torch.norm(lam_tensor).item())
        diag["dual_residual"] = float(solution.get("dual_residual", 0.0) if solution else 0.0)
        diag["dual_iters"] = int(solution.get("dual_iters", 0) if solution else 0)
        summary.update(diag)
        return summary

    def apply_delta_scale(
        self,
        constraints: Dict[str, object],
        scale: float,
        policy_context: Optional[Dict[str, object]] = None,
    ) -> None:
        meta = constraints.get("meta") or {}
        raw_map = meta.get("delta_per_rule_raw")
        bounds = constraints.get("bounds")
        active_rules = meta.get("active_rules") or []
        if raw_map is None or bounds is None or not active_rules:
            return
        details = meta.get("delta_details") or {}
        for row_idx, rule_idx in enumerate(active_rules):
            raw_val = float(raw_map.get(rule_idx, 0.0))
            scaled = max(0.0, scale * raw_val)
            if row_idx < bounds.shape[0]:
                bounds[row_idx] = scaled
            meta.setdefault("delta_per_rule", {})[rule_idx] = scaled
            if rule_idx in details:
                details[rule_idx]["delta"] = scaled

    def classify_kl_reason(
        self,
        status: str,
        diagnostics: Optional[Dict[str, object]],
        solution: Optional[Dict[str, object]],
    ) -> str:
        if status != "ok":
            return "infeasible"
        if not diagnostics:
            return "unknown"
        tv = float(diagnostics.get("tv_q_p", 0.0) or 0.0)
        binding = bool(diagnostics.get("binding_under_p"))
        dual_res = float(diagnostics.get("dual_residual", 0.0) if diagnostics else 0.0)
        lambda_clip = float(diagnostics.get("lambda_clip_fraction", 0.0) or 0.0)
        if not binding:
            return "non_binding_p"
        if tv <= 1e-6:
            if dual_res > max(self.constraint_tol * 5.0, 1e-6) or lambda_clip > 0.2:
                return "solver_not_converged_or_clipped"
            headroom_min = diagnostics.get("headroom_per_rule_min")
            if headroom_min is not None and headroom_min < 1e-3:
                return "binding_but_mild_authority_limited"
            headroom_tok = diagnostics.get("headroom_tokenwise")
            if headroom_tok is not None and float(headroom_tok) < 1e-3:
                return "binding_but_mild_authority_limited"
            return "binding_but_mild_authority_limited"
        return "strong_projection"

    def clip_target_bounds(self, constraints: Dict[str, object], new_target: float) -> None:
        meta = constraints.get("meta") or {}
        meta["target_margin_used"] = new_target
        bounds = constraints.get("bounds")
        if bounds is None:
            return
        mode = constraints.get("mode", "cbf_increment")
        if mode == "cbf_increment":
            base_shifted = meta.get("base_shifted") or []
            for idx in range(bounds.shape[0]):
                base_val = float(base_shifted[idx]) if idx < len(base_shifted) else 0.0
                bounds[idx] = max(0.0, new_target - base_val)
        else:
            bounds.fill_(new_target)

    def apply_target_mode(self, constraints: Dict[str, object], base_probs: torch.Tensor) -> None:
        if self.target_mode != "ep_plus_delta":
            return
        delta = max(0.0, float(self.target_delta))
        if delta <= 0:
            return
        meta = constraints.get("meta") or {}
        active_rules = meta.get("active_rules") or []
        candidates = constraints.get("candidates") or []
        if not active_rules or not candidates:
            return
        matrix = self.compute_shifted_margin_matrix(candidates, active_rules, device=base_probs.device)
        if matrix.numel() == 0:
            return
        probs = torch.clamp(base_probs, min=self._eps)
        probs = probs / probs.sum()
        exp_vals = matrix.matmul(probs)
        max_vals = torch.max(matrix, dim=1).values
        delta_vec = torch.full_like(exp_vals, delta)
        if self.target_delta_cap > 0:
            headroom = torch.clamp(max_vals - exp_vals, min=0.0)
            capped = headroom * float(self.target_delta_cap)
            delta_vec = torch.minimum(delta_vec, capped)
        tau = exp_vals + delta_vec
        mode = constraints.get("mode", "cbf_increment")
        if mode == "cbf_increment":
            base_shifted = meta.get("base_shifted") or []
            for idx in range(len(active_rules)):
                base_val = float(base_shifted[idx]) if idx < len(base_shifted) else 0.0
                if idx < constraints["bounds"].shape[0]:
                    constraints["bounds"][idx] = max(0.0, float(tau[idx].item()) - base_val)
        else:
            for idx in range(len(active_rules)):
                if idx < constraints["bounds"].shape[0]:
                    constraints["bounds"][idx] = float(tau[idx].item())
        meta["target_margin_used"] = float(torch.max(tau).item())
        schedule = meta.get("target_mode_details", {})
        schedule.update({"mode": "ep_plus_delta", "delta": delta, "delta_cap": self.target_delta_cap})
        meta["target_mode_details"] = schedule
    def solve_projection(
        self,
        base_probs: torch.Tensor,
        increments: torch.Tensor,
        bounds: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Solve the KL projection with linear inequality constraints.
        Returns a dict containing the new distribution, dual variables and feasibility flag.
        """
        if increments.numel() == 0 or bounds.numel() == 0:
            probs = base_probs / base_probs.sum()
            return {"distribution": probs, "lambdas": [], "feasible": True, "dual_iters": 0, "dual_residual": 0.0}

        base = torch.clamp(base_probs, min=self.min_prob).float()
        base = base / base.sum()
        lambdas = torch.zeros(bounds.shape[0], dtype=torch.float32, device=bounds.device)
        q = base.clone()

        for step_idx in range(1, self.dual_max_iters + 1):
            proj_term = increments.t().matmul(lambdas)
            if self.exp_clip:
                proj_term = torch.clamp(proj_term, -self.exp_clip, self.exp_clip)
            logits = torch.log(base + self._eps) + proj_term
            q = torch.softmax(logits, dim=-1)
            lhs = increments.matmul(q)
            violations = bounds - lhs
            max_violation = float(torch.max(violations))
            if max_violation <= self.constraint_tol:
                return {
                    "distribution": q,
                    "lambdas": lambdas.tolist(),
                    "feasible": True,
                    "dual_iters": step_idx,
                    "dual_residual": max_violation,
                }
            step = torch.clamp(violations, min=0.0)
            if torch.all(step <= self.constraint_tol):
                break
            lambdas = torch.clamp(lambdas + self.dual_lr * step, min=0.0, max=self.lambda_max)

        lhs = increments.matmul(q)
        violations = bounds - lhs
        max_violation = float(torch.max(violations))
        feasible = bool(max_violation <= self.constraint_tol)
        return {
            "distribution": q,
            "lambdas": lambdas.tolist(),
            "feasible": feasible,
            "dual_iters": self.dual_max_iters,
            "dual_residual": max_violation,
        }

    def sample(self, probs: torch.Tensor) -> int:
        if self.use_argmax:
            return int(torch.argmax(probs).item())
        probs = torch.clamp(probs, min=self.min_prob)
        probs = probs / probs.sum()
        if self.temperature != 1.0:
            logits = torch.log(probs)
            logits = logits / self.temperature
            probs = torch.softmax(logits, dim=-1)
        probs = probs / probs.sum()
        return int(torch.multinomial(probs, num_samples=1).item())

    def decide(self, state: Dict[str, object]) -> ControlAction:  # pragma: no cover - unused hook
        return ControlAction(action="noop")


__all__ = ["JointCBFKLController"]
