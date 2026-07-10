from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from ...io.utils import atomic_write_json
from ..utils.geometry import normalized_subspace_similarity
from .core_space import build_lora_groups
from .grassmann_utils import (
    incremental_grassmann_mean,
    orth_from_factor,
    orthonormal_basis,
    resolve_nonnegative_weights,
)
from .registry import register


@dataclass(frozen=True)
class GeodesicCorePrepared:
    variant: str
    mean_weighting: str
    mask_support_mode: str
    merge_weight_override: tuple[float, ...] | None
    tasks: tuple[str, ...]
    bases: dict[str, dict[str, torch.Tensor]]
    task_masks: dict[str, dict[str, torch.Tensor]] | None = None
    posterior_projectors: dict[str, dict[str, tuple[torch.Tensor, ...]]] | None = None
    similarity_artifact_path: str | None = None


@dataclass(frozen=True)
class GeodesicCoreSpace:
    """
    Core-space LoRA merging with shared Grassmann geodesic mean bases.
    """

    name: str = "geo_core"

    @staticmethod
    def _resolve_variant(method_params: dict[str, Any] | None) -> str:
        params = method_params or {}
        variant = str(params.get("geo_core_variant", "geodesic_mean")).strip().lower()
        aliases = {
            "geodesic_mean": "geodesic_mean",
            "geodesic-mean": "geodesic_mean",
            "mean": "geodesic_mean",
            "core_referenced_tangent": "core_referenced_tangent",
            "core-referenced-tangent": "core_referenced_tangent",
            "crt": "core_referenced_tangent",
            "core_similarity_weights": "core_similarity_weights",
            "core-similarity-weights": "core_similarity_weights",
            "similarity_weights": "core_similarity_weights",
            "similarity-weights": "core_similarity_weights",
            "core_similarity_mask": "core_similarity_mask",
            "core-similarity-mask": "core_similarity_mask",
            "similarity_mask": "core_similarity_mask",
            "similarity-mask": "core_similarity_mask",
            "core_posterior": "core_posterior",
            "core-posterior": "core_posterior",
            "posterior": "core_posterior",
        }
        try:
            return aliases[variant]
        except KeyError as exc:
            raise ValueError(
                "geo_core_variant must be one of: "
                "'geodesic_mean', 'core_referenced_tangent', 'core_similarity_weights', "
                "'core_similarity_mask', 'core_posterior' "
                f"(got {variant!r})."
            ) from exc

    @staticmethod
    def _normalize_weights(weights: Sequence[float], *, context_name: str) -> tuple[float, ...]:
        total = float(sum(float(weight) for weight in weights))
        if total <= 0.0:
            raise ValueError(f"{context_name} requires a positive total weight.")
        return tuple(float(weight) / total for weight in weights)

    @staticmethod
    def _resolve_mean_weighting(method_params: dict[str, Any] | None) -> str:
        params = method_params or {}
        weighting = str(params.get("geo_mean_weighting", "equal")).strip().lower()
        aliases = {
            "equal": "equal",
            "uniform": "equal",
            "merge_weights": "merge_weights",
            "merge-weights": "merge_weights",
            "weighted": "merge_weights",
        }
        try:
            return aliases[weighting]
        except KeyError as exc:
            raise ValueError(
                f"geo_mean_weighting must be one of: 'equal', 'merge_weights' (got {weighting!r})."
            ) from exc

    @staticmethod
    def _resolve_geo_weight_lambda(method_params: dict[str, Any] | None) -> float:
        params = method_params or {}
        lam = params.get("geo_weight_lambda", params.get("geo_similarity_lambda", 1.0))
        lam_f = float(lam)
        if not (0.0 <= lam_f <= 1.0):
            raise ValueError(f"geo_weight_lambda must be in [0, 1] (got {lam!r}).")
        return lam_f

    @staticmethod
    def _resolve_geo_mask_lambda(method_params: dict[str, Any] | None) -> float:
        params = method_params or {}
        lam = params.get("geo_mask_lambda", 1.0)
        lam_f = float(lam)
        if not (0.0 <= lam_f <= 1.0):
            raise ValueError(f"geo_mask_lambda must be in [0, 1] (got {lam!r}).")
        return lam_f

    @staticmethod
    def _resolve_posterior_tau(method_params: dict[str, Any] | None) -> float:
        params = method_params or {}
        tau = float(params.get("geo_posterior_tau", 1.0))
        if tau <= 0.0:
            raise ValueError(f"geo_posterior_tau must be > 0 (got {tau!r}).")
        return tau

    @staticmethod
    def _resolve_posterior_max_iter(method_params: dict[str, Any] | None) -> int:
        params = method_params or {}
        max_iter = int(params.get("geo_posterior_max_iter", 100))
        if max_iter <= 0:
            raise ValueError(f"geo_posterior_max_iter must be > 0 (got {max_iter!r}).")
        return max_iter

    @staticmethod
    def _resolve_posterior_tol(method_params: dict[str, Any] | None) -> float:
        params = method_params or {}
        tol = float(params.get("geo_posterior_tol", 1e-6))
        if tol <= 0.0:
            raise ValueError(f"geo_posterior_tol must be > 0 (got {tol!r}).")
        return tol

    @staticmethod
    def _resolve_mask_support_mode(method_params: dict[str, Any] | None) -> str:
        params = method_params or {}
        raw = str(params.get("geo_mask_support", "subspace")).strip().lower()
        aliases = {
            "subspace": "subspace",
            "orth": "subspace",
            "basis": "subspace",
            "factor": "factor",
            "raw_factor": "factor",
            "magnitude": "factor",
            "magnitude_aware": "factor",
            "magnitude-aware": "factor",
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise ValueError(
                "geo_mask_support must be one of: 'subspace', 'factor' "
                f"(got {raw!r})."
            ) from exc

    @staticmethod
    def _similarity_artifact_path(*, artifact_dir: str | Path | None) -> Path | None:
        if artifact_dir is None:
            return None
        return Path(artifact_dir) / "geo_core_similarity.json"

    @staticmethod
    def _save_similarity_artifact(
        *,
        artifact_path: Path,
        tasks: Sequence[str],
        mask_support_mode: str,
        layer_similarity: dict[str, dict[str, list[list[float]]]],
        task_masks: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> None:
        if not layer_similarity:
            return

        def _mean_matrix(metric_key: str) -> list[list[float]]:
            n = len(tasks)
            acc = [[0.0 for _ in range(n)] for _ in range(n)]
            for layer_payload in layer_similarity.values():
                matrix = layer_payload[metric_key]
                for i in range(n):
                    for j in range(n):
                        acc[i][j] += float(matrix[i][j])
            scale = 1.0 / max(1, len(layer_similarity))
            for i in range(n):
                for j in range(n):
                    acc[i][j] *= scale
            return acc

        layers_payload = {layer_key: dict(payload) for layer_key, payload in layer_similarity.items()}
        if task_masks is not None:
            for layer_key in layers_payload:
                layer_masks: dict[str, list[list[float]]] = {}
                for task in tasks:
                    task_layer_masks = task_masks.get(task, {})
                    mask = task_layer_masks.get(layer_key, None)
                    if mask is not None:
                        layer_masks[task] = mask.to(dtype=torch.float32, device="cpu").tolist()
                layers_payload[layer_key]["task_masks"] = layer_masks

        artifact = {
            "tasks": list(tasks),
            "num_layers": len(layer_similarity),
            "mask_support_mode": mask_support_mode,
            "mean_u_similarity": _mean_matrix("u_similarity"),
            "mean_v_similarity": _mean_matrix("v_similarity"),
            "mean_joint_similarity": _mean_matrix("joint_similarity"),
            "layers": layers_payload,
        }
        atomic_write_json(str(artifact_path), artifact)

    @staticmethod
    def _frobenius_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * y)

    @classmethod
    def _solve_posterior_system(
        cls,
        *,
        prior_core: torch.Tensor,
        task_cores: Sequence[torch.Tensor],
        omegas: Sequence[torch.Tensor],
        gammas: Sequence[torch.Tensor],
        weights: Sequence[float],
        tau: float,
        tol: float,
        max_iter: int,
    ) -> torch.Tensor:
        if len(task_cores) != len(omegas) or len(task_cores) != len(gammas) or len(task_cores) != len(weights):
            raise ValueError("Posterior refinement requires matching numbers of task cores, projectors, and weights.")

        prior64 = prior_core.to(dtype=torch.float64)
        omegas64 = [omega.to(dtype=torch.float64) for omega in omegas]
        gammas64 = [gamma.to(dtype=torch.float64) for gamma in gammas]
        task_cores64 = [core.to(dtype=torch.float64) for core in task_cores]

        def apply_operator(x: torch.Tensor) -> torch.Tensor:
            out = float(tau) * x
            for weight, omega, gamma in zip(weights, omegas64, gammas64, strict=True):
                out = out + float(weight) * (omega @ x @ gamma)
            return out

        rhs = float(tau) * prior64
        for weight, omega, task_core, gamma in zip(weights, omegas64, task_cores64, gammas64, strict=True):
            rhs = rhs + float(weight) * (omega @ task_core @ gamma)

        x = prior64.clone()
        residual = rhs - apply_operator(x)
        rhs_norm = float(torch.linalg.norm(rhs).item())
        threshold = float(tol) * max(1.0, rhs_norm)
        residual_norm = float(torch.linalg.norm(residual).item())
        if residual_norm <= threshold:
            return x.contiguous()

        direction = residual.clone()
        delta_old = cls._frobenius_inner(residual, residual)
        for _ in range(int(max_iter)):
            applied = apply_operator(direction)
            denom = cls._frobenius_inner(direction, applied)
            denom_value = float(denom.item())
            if abs(denom_value) <= 1e-20:
                break
            step = delta_old / denom
            x = x + step * direction
            residual = residual - step * applied
            residual_norm = float(torch.linalg.norm(residual).item())
            if residual_norm <= threshold:
                break
            delta_new = cls._frobenius_inner(residual, residual)
            beta = delta_new / delta_old.clamp_min(1e-20)
            direction = residual + beta * direction
            delta_old = delta_new
        return x.contiguous()

    def prepare(
        self,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
        method_params: dict[str, Any] | None = None,
        weights: Sequence[float] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> GeodesicCorePrepared:
        if not lora_by_task:
            raise ValueError("lora_by_task is empty.")

        _ = peft_cfg
        variant = self._resolve_variant(method_params)
        mean_weighting = self._resolve_mean_weighting(method_params)
        geo_weight_lambda = self._resolve_geo_weight_lambda(method_params)
        geo_mask_lambda = self._resolve_geo_mask_lambda(method_params)
        mask_support_mode = self._resolve_mask_support_mode(method_params)
        tasks = list(lora_by_task.keys())
        layer_groups = {task: build_lora_groups(lora_by_task[task]) for task in tasks}
        if not layer_groups[tasks[0]]:
            raise ValueError("No LoRA layers found in peft_state_dict.")

        bases: dict[str, dict[str, torch.Tensor]] = {}
        merge_weight_override: tuple[float, ...] | None = None
        task_masks: dict[str, dict[str, torch.Tensor]] | None = {} if variant == "core_similarity_mask" else None
        posterior_projectors: dict[str, dict[str, tuple[torch.Tensor, ...]]] | None = (
            {} if variant == "core_posterior" else None
        )
        similarity_artifact_path = self._similarity_artifact_path(artifact_dir=artifact_dir)
        similarity_by_layer: dict[str, dict[str, list[list[float]]]] = {}
        ref_layers = layer_groups[tasks[0]]
        for layer_key in tqdm(ref_layers, desc="Preparing geo_core bases", unit="layer"):
            u_bases: list[torch.Tensor] = []
            v_bases: list[torch.Tensor] = []

            for task in tasks:
                layer = layer_groups[task].get(layer_key, None)
                if layer is None:
                    raise ValueError(f"Missing LoRA layer '{layer_key}' for task '{task}'.")
                u_bases.append(orth_from_factor(layer.b, side="B"))
                v_bases.append(orth_from_factor(layer.a, side="A"))

            if variant == "geodesic_mean":
                U_geo = incremental_grassmann_mean(
                    u_bases,
                    weighting=mean_weighting,
                    weights=weights,
                    context_name="geo_core",
                )
                V_geo = incremental_grassmann_mean(
                    v_bases,
                    weighting=mean_weighting,
                    weights=weights,
                    context_name="geo_core",
                )
                bases[layer_key] = {
                    "U": U_geo.to(dtype=torch.float32).contiguous(),
                    "V": V_geo.to(dtype=torch.float32).contiguous(),
                }
                continue

            if variant == "core_referenced_tangent":
                layer_weights = resolve_nonnegative_weights(
                    num_bases=len(tasks),
                    weights=weights,
                    context_name="geo_core core_referenced_tangent",
                )
                merge_rank = int(u_bases[0].shape[1])
                for u_basis, v_basis in zip(u_bases[1:], v_bases[1:], strict=True):
                    if int(u_basis.shape[1]) != merge_rank or int(v_basis.shape[1]) != merge_rank:
                        raise ValueError(
                            "geo_core core_referenced_tangent requires the same LoRA rank across tasks for each layer."
                        )

                b_concat = torch.cat(
                    [layer_groups[task][layer_key].b.to(dtype=torch.float32) for task in tasks],
                    dim=1,
                )
                a_concat_t = torch.cat(
                    [layer_groups[task][layer_key].a.T.to(dtype=torch.float32) for task in tasks],
                    dim=1,
                )
                U_core = orthonormal_basis(b_concat)
                V_core = orthonormal_basis(a_concat_t)

                p_u_bar = torch.zeros((U_core.shape[1], U_core.shape[1]), dtype=torch.float64)
                p_v_bar = torch.zeros((V_core.shape[1], V_core.shape[1]), dtype=torch.float64)
                for weight, U_t, V_t in zip(layer_weights, u_bases, v_bases, strict=True):
                    u_coords = U_core.T @ U_t.to(dtype=U_core.dtype, device=U_core.device)
                    v_coords = V_core.T @ V_t.to(dtype=V_core.dtype, device=V_core.device)
                    p_u_bar = p_u_bar + float(weight) * (u_coords @ u_coords.T)
                    p_v_bar = p_v_bar + float(weight) * (v_coords @ v_coords.T)

                eigvals_u, eigvecs_u = torch.linalg.eigh(p_u_bar)
                eigvals_v, eigvecs_v = torch.linalg.eigh(p_v_bar)
                del eigvals_u, eigvals_v
                R_u = eigvecs_u[:, -merge_rank:].contiguous()
                R_v = eigvecs_v[:, -merge_rank:].contiguous()
                U_merge = U_core @ R_u
                V_merge = V_core @ R_v

                bases[layer_key] = {
                    "U_core": U_core.to(dtype=torch.float32).contiguous(),
                    "V_core": V_core.to(dtype=torch.float32).contiguous(),
                    "R_U": R_u.to(dtype=torch.float32).contiguous(),
                    "R_V": R_v.to(dtype=torch.float32).contiguous(),
                    "U": U_merge.to(dtype=torch.float32).contiguous(),
                    "V": V_merge.to(dtype=torch.float32).contiguous(),
                }
                continue

            if variant == "core_similarity_weights":
                layer_weights = resolve_nonnegative_weights(
                    num_bases=len(tasks),
                    weights=weights,
                    context_name="geo_core core_similarity_weights",
                )
                b_concat = torch.cat(
                    [layer_groups[task][layer_key].b.to(dtype=torch.float32) for task in tasks],
                    dim=1,
                )
                a_concat_t = torch.cat(
                    [layer_groups[task][layer_key].a.T.to(dtype=torch.float32) for task in tasks],
                    dim=1,
                )
                U_core = orthonormal_basis(b_concat)
                V_core = orthonormal_basis(a_concat_t)
                bases[layer_key] = {
                    "U": U_core.to(dtype=torch.float32).contiguous(),
                    "V": V_core.to(dtype=torch.float32).contiguous(),
                }

                if merge_weight_override is None:
                    if len(tasks) == 1:
                        merge_weight_override = (1.0,)
                    else:
                        geometry_scores: list[float] = []
                        for idx, (U_t, V_t) in enumerate(zip(u_bases, v_bases, strict=True)):
                            sim_sum = 0.0
                            for jdx, (U_j, V_j) in enumerate(zip(u_bases, v_bases, strict=True)):
                                if idx == jdx:
                                    continue
                                s_u = normalized_subspace_similarity(U_t, U_j)
                                s_v = normalized_subspace_similarity(V_t, V_j)
                                sim_sum += s_u * s_v
                            geometry_scores.append(sim_sum / float(len(tasks) - 1))

                        weighted_scores = [
                            float(alpha_t) * float(g_t)
                            for alpha_t, g_t in zip(layer_weights, geometry_scores, strict=True)
                        ]
                        core_weights = self._normalize_weights(
                            layer_weights,
                            context_name="geo_core core_similarity_weights",
                        )
                        if sum(weighted_scores) <= 0.0:
                            # If geometry gives no signal, fall back to the original normalized merge weights.
                            merge_weight_override = core_weights
                        else:
                            geo_weights = self._normalize_weights(
                                weighted_scores,
                                context_name="geo_core core_similarity_weights",
                            )
                            merge_weight_override = tuple(
                                ((1.0 - geo_weight_lambda) * float(core_w)) + (geo_weight_lambda * float(geo_w))
                                for core_w, geo_w in zip(core_weights, geo_weights, strict=True)
                            )
                continue

            if variant == "core_similarity_mask":
                b_concat = torch.cat(
                    [layer_groups[task][layer_key].b.to(dtype=torch.float32) for task in tasks],
                    dim=1,
                )
                a_concat_t = torch.cat(
                    [layer_groups[task][layer_key].a.T.to(dtype=torch.float32) for task in tasks],
                    dim=1,
                )
                U_core = orthonormal_basis(b_concat)
                V_core = orthonormal_basis(a_concat_t)
                bases[layer_key] = {
                    "U": U_core.to(dtype=torch.float32).contiguous(),
                    "V": V_core.to(dtype=torch.float32).contiguous(),
                }

                assert task_masks is not None
                for task in tasks:
                    task_masks.setdefault(task, {})

                if len(tasks) == 1:
                    similarity_by_layer[layer_key] = {
                        "u_similarity": [[1.0]],
                        "v_similarity": [[1.0]],
                        "joint_similarity": [[1.0]],
                    }
                    task_masks[tasks[0]][layer_key] = torch.ones(
                        (U_core.shape[1], V_core.shape[1]),
                        dtype=torch.float32,
                    )
                    continue

                row_support_by_task: dict[str, torch.Tensor] = {}
                col_support_by_task: dict[str, torch.Tensor] = {}
                for task, U_t, V_t in zip(tasks, u_bases, v_bases, strict=True):
                    layer = layer_groups[task][layer_key]
                    if mask_support_mode == "subspace":
                        u_support_coords = U_core.T @ U_t.to(dtype=U_core.dtype, device=U_core.device)
                        v_support_coords = V_core.T @ V_t.to(dtype=V_core.dtype, device=V_core.device)
                    elif mask_support_mode == "factor":
                        u_support_coords = U_core.T @ layer.b.to(dtype=U_core.dtype, device=U_core.device)
                        v_support_coords = V_core.T @ layer.a.T.to(dtype=V_core.dtype, device=V_core.device)
                    else:
                        raise AssertionError(f"Unhandled geo_mask_support mode: {mask_support_mode}")
                    row_support_by_task[task] = torch.sum(u_support_coords * u_support_coords, dim=1)
                    col_support_by_task[task] = torch.sum(v_support_coords * v_support_coords, dim=1)

                u_similarity: list[list[float]] = []
                v_similarity: list[list[float]] = []
                joint_similarity: list[list[float]] = []
                for U_t, V_t in zip(u_bases, v_bases, strict=True):
                    u_row: list[float] = []
                    v_row: list[float] = []
                    joint_row: list[float] = []
                    for U_j, V_j in zip(u_bases, v_bases, strict=True):
                        s_u = normalized_subspace_similarity(U_t, U_j)
                        s_v = normalized_subspace_similarity(V_t, V_j)
                        u_row.append(float(s_u))
                        v_row.append(float(s_v))
                        joint_row.append(float(s_u * s_v))
                    u_similarity.append(u_row)
                    v_similarity.append(v_row)
                    joint_similarity.append(joint_row)
                similarity_by_layer[layer_key] = {
                    "u_similarity": u_similarity,
                    "v_similarity": v_similarity,
                    "joint_similarity": joint_similarity,
                }

                eps = 1e-12
                for task, U_t, V_t in zip(tasks, u_bases, v_bases, strict=True):
                    row_agg = torch.zeros(U_core.shape[1], dtype=torch.float64)
                    col_agg = torch.zeros(V_core.shape[1], dtype=torch.float64)
                    row_sim_total = 0.0
                    col_sim_total = 0.0
                    for other_task, U_j, V_j in zip(tasks, u_bases, v_bases, strict=True):
                        if other_task == task:
                            continue
                        s_u = normalized_subspace_similarity(U_t, U_j)
                        s_v = normalized_subspace_similarity(V_t, V_j)
                        row_agg = row_agg + float(s_u) * row_support_by_task[other_task].to(dtype=torch.float64)
                        col_agg = col_agg + float(s_v) * col_support_by_task[other_task].to(dtype=torch.float64)
                        row_sim_total += float(s_u)
                        col_sim_total += float(s_v)

                    if row_sim_total > 0.0:
                        row_agg = row_agg / row_sim_total
                    else:
                        row_agg = torch.zeros_like(row_agg)
                    if col_sim_total > 0.0:
                        col_agg = col_agg / col_sim_total
                    else:
                        col_agg = torch.zeros_like(col_agg)

                    self_row = row_support_by_task[task].to(dtype=torch.float64)
                    self_col = col_support_by_task[task].to(dtype=torch.float64)
                    row_gate = torch.where(
                        self_row > eps,
                        (row_agg / self_row.clamp_min(eps)).clamp(min=0.0, max=1.0),
                        torch.ones_like(self_row),
                    )
                    col_gate = torch.where(
                        self_col > eps,
                        (col_agg / self_col.clamp_min(eps)).clamp(min=0.0, max=1.0),
                        torch.ones_like(self_col),
                    )
                    mask = row_gate[:, None] * col_gate[None, :]
                    blended_mask = ((1.0 - geo_mask_lambda) + (geo_mask_lambda * mask)).to(dtype=torch.float32)
                    task_masks[task][layer_key] = blended_mask.contiguous()
                continue

            if variant == "core_posterior":
                self._resolve_posterior_tau(method_params)
                self._resolve_posterior_max_iter(method_params)
                self._resolve_posterior_tol(method_params)
                resolve_nonnegative_weights(
                    num_bases=len(tasks),
                    weights=weights,
                    context_name="geo_core core_posterior",
                )
                b_concat = torch.cat(
                    [layer_groups[task][layer_key].b.to(dtype=torch.float32) for task in tasks],
                    dim=1,
                )
                a_concat_t = torch.cat(
                    [layer_groups[task][layer_key].a.T.to(dtype=torch.float32) for task in tasks],
                    dim=1,
                )
                U_core = orthonormal_basis(b_concat)
                V_core = orthonormal_basis(a_concat_t)
                bases[layer_key] = {
                    "U": U_core.to(dtype=torch.float32).contiguous(),
                    "V": V_core.to(dtype=torch.float32).contiguous(),
                }

                assert posterior_projectors is not None
                omegas: list[torch.Tensor] = []
                gammas: list[torch.Tensor] = []
                for U_t, V_t in zip(u_bases, v_bases, strict=True):
                    u_coords = U_core.T @ U_t.to(dtype=U_core.dtype, device=U_core.device)
                    v_coords = V_core.T @ V_t.to(dtype=V_core.dtype, device=V_core.device)
                    omegas.append((u_coords @ u_coords.T).to(dtype=torch.float32).contiguous())
                    gammas.append((v_coords @ v_coords.T).to(dtype=torch.float32).contiguous())
                posterior_projectors[layer_key] = {
                    "omega": tuple(omegas),
                    "gamma": tuple(gammas),
                }
                continue

            raise AssertionError(f"Unhandled geo_core variant: {variant}")

        if variant == "core_similarity_mask" and similarity_artifact_path is not None:
            self._save_similarity_artifact(
                artifact_path=similarity_artifact_path,
                tasks=tasks,
                mask_support_mode=mask_support_mode,
                layer_similarity=similarity_by_layer,
                task_masks=task_masks,
            )

        return GeodesicCorePrepared(
            variant=variant,
            mean_weighting=mean_weighting,
            mask_support_mode=mask_support_mode,
            merge_weight_override=merge_weight_override,
            tasks=tuple(tasks),
            bases=bases,
            task_masks=task_masks,
            posterior_projectors=posterior_projectors,
            similarity_artifact_path=(str(similarity_artifact_path) if similarity_artifact_path is not None else None),
        )

    def project(
        self,
        prepared: GeodesicCorePrepared,
        *,
        lora_by_task: dict[str, dict[str, torch.Tensor]],
        peft_cfg: dict[str, Any],
    ) -> dict[str, dict[str, torch.Tensor]]:
        _ = peft_cfg
        core_by_task: dict[str, dict[str, torch.Tensor]] = {}
        for task, peft_state in lora_by_task.items():
            layers = build_lora_groups(peft_state)
            out_layers: dict[str, torch.Tensor] = {}
            for layer_key, layer in layers.items():
                if layer_key not in prepared.bases:
                    continue
                basis = prepared.bases[layer_key]
                if prepared.variant in {
                    "geodesic_mean",
                    "core_similarity_weights",
                    "core_similarity_mask",
                    "core_posterior",
                }:
                    U = basis["U"]
                    V = basis["V"]
                    b = layer.b.to(dtype=U.dtype, device=U.device)
                    a = layer.a.to(dtype=U.dtype, device=U.device)
                    core = U.T @ b @ a @ V
                    if prepared.variant == "core_similarity_mask":
                        if prepared.task_masks is None or task not in prepared.task_masks:
                            raise RuntimeError("geo_core core_similarity_mask requires prepared task masks.")
                        mask = prepared.task_masks[task].get(layer_key, None)
                        if mask is None:
                            raise RuntimeError(f"Missing geo_core mask for task '{task}' layer '{layer_key}'.")
                        core = mask.to(dtype=core.dtype, device=core.device) * core
                elif prepared.variant == "core_referenced_tangent":
                    U_core = basis["U_core"]
                    V_core = basis["V_core"]
                    R_u = basis["R_U"]
                    R_v = basis["R_V"]
                    b = layer.b.to(dtype=U_core.dtype, device=U_core.device)
                    a = layer.a.to(dtype=U_core.dtype, device=U_core.device)
                    core = R_u.T @ (U_core.T @ b @ a @ V_core) @ R_v
                else:
                    raise AssertionError(f"Unhandled geo_core variant: {prepared.variant}")
                out_layers[layer_key] = core.to(dtype=torch.float32).contiguous()
            core_by_task[task] = out_layers
        return core_by_task

    def refine_merged_core(
        self,
        prepared: GeodesicCorePrepared,
        *,
        merged_core: dict[str, torch.Tensor],
        tuned_cores: Sequence[Mapping[str, torch.Tensor]],
        weights: Sequence[float] | None,
        method_params: dict[str, Any] | None = None,
        tasks: Sequence[str] | None = None,
        peft_cfg: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = peft_cfg
        if prepared.variant != "core_posterior":
            return merged_core
        if prepared.posterior_projectors is None:
            raise RuntimeError("geo_core core_posterior requires stored posterior projectors.")

        task_order = tuple(tasks) if tasks is not None else prepared.tasks
        if task_order != prepared.tasks:
            raise ValueError(
                "geo_core core_posterior requires tuned cores in the same task order used during prepare()."
            )

        resolved_weights = resolve_nonnegative_weights(
            num_bases=len(tuned_cores),
            weights=weights,
            context_name="geo_core core_posterior",
        )
        normalized_weights = self._normalize_weights(
            resolved_weights,
            context_name="geo_core core_posterior",
        )
        tau = self._resolve_posterior_tau(method_params)
        max_iter = self._resolve_posterior_max_iter(method_params)
        tol = self._resolve_posterior_tol(method_params)

        refined_core = dict(merged_core)
        for layer_key, merged_layer_core in merged_core.items():
            layer_projectors = prepared.posterior_projectors.get(layer_key, None)
            if layer_projectors is None:
                continue

            task_layer_cores: list[torch.Tensor] = []
            for tuned_core in tuned_cores:
                layer_core = tuned_core.get(layer_key, None)
                if layer_core is None:
                    raise ValueError(f"Missing projected core for layer '{layer_key}' during geo_core refinement.")
                task_layer_cores.append(layer_core)

            solved = self._solve_posterior_system(
                prior_core=merged_layer_core,
                task_cores=task_layer_cores,
                omegas=layer_projectors["omega"],
                gammas=layer_projectors["gamma"],
                weights=normalized_weights,
                tau=tau,
                tol=tol,
                max_iter=max_iter,
            )
            refined_core[layer_key] = solved.to(dtype=merged_layer_core.dtype, device=merged_layer_core.device)

        return refined_core

    def lift(
        self,
        prepared: GeodesicCorePrepared,
        *,
        merged_core: dict[str, torch.Tensor],
        lora_template: dict[str, torch.Tensor],
        peft_cfg: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        _ = peft_cfg
        template_layers = build_lora_groups(lora_template)
        out: dict[str, torch.Tensor] = {}

        for layer_key, tpl in template_layers.items():
            if layer_key not in prepared.bases:
                continue
            core = merged_core.get(layer_key, None)
            if core is None:
                delta = torch.zeros_like(tpl.b @ tpl.a)
            else:
                U = prepared.bases[layer_key]["U"]
                V = prepared.bases[layer_key]["V"]
                c = core.to(dtype=U.dtype, device=U.device)
                delta = (U @ c @ V.T).to(dtype=torch.float32)
            out[f"{layer_key}.weight"] = delta.to(dtype=tpl.b.dtype)

        return out


register(GeodesicCoreSpace())
