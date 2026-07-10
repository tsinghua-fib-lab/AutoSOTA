from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class IsoCTSMerge:
    name: str = "isocts_merge"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        w = default_weights(len(tuned), weights)

        method_params = get_method_params(kwargs)

        common_space_fraction = float(method_params.get("common_space_fraction", 0.8))
        vector_1d_merge = str(method_params.get("vector_1d_merge", "zero")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("isocts_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]

        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction: TensorDict = {}
        for k in keys:
            b = base[k]
            if b.ndim == 2 and "text_projection" not in k:
                direction[k] = self._isocts_delta(
                    [d[k] for d in deltas], w=w, common_space_fraction=common_space_fraction
                ).to(dtype=b.dtype, device=b.device)
            elif b.ndim == 1 and vector_1d_merge == "average":
                denom = float(w.sum().clamp_min(1e-12).item())
                acc = torch.zeros_like(b)
                for wi, d in zip(w, deltas, strict=True):
                    acc = acc + float(wi) * d[k].to(dtype=acc.dtype, device=acc.device)
                direction[k] = (acc / denom).to(dtype=b.dtype, device=b.device)
            else:
                direction[k] = torch.zeros_like(b)
        return base, direction

    def apply(self, prepared: tuple[TensorDict, TensorDict], *, alpha: float, **kwargs) -> TensorDict:
        base, direction = prepared
        return axpy_state_dict(base, direction, alpha=float(alpha))

    def merge(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        alpha: float = 1.0,
        strict: bool = False,
        **kwargs,
    ) -> TensorDict:
        prepared = self.prepare(
            base=base,
            tuned=tuned,
            weights=weights,
            strict=strict,
            **kwargs,
        )
        return self.apply(prepared, alpha=float(alpha))

    @staticmethod
    def _isocts_delta(mats: list[torch.Tensor], w: torch.Tensor, common_space_fraction: float) -> torch.Tensor:
        # Keep signature for API compatibility; current algorithm ignores merge weights.
        _ = w

        out_dtype = mats[0].dtype
        out_device = mats[0].device
        work_dtype = torch.float64

        combined_w = sum(mat.to(dtype=work_dtype) for mat in mats)
        n_tasks = len(mats)
        min_dim = min(combined_w.shape)

        # Compute common-space size while keeping the remaining task-specific
        # dimensions divisible across tasks.
        common_space_dim = int(min_dim * common_space_fraction)
        task_specific_total_dim = round((min_dim - common_space_dim) / n_tasks) * n_tasks
        common_space_dim = min_dim - task_specific_total_dim
        task_dims_per_task = int((min_dim - common_space_dim) / n_tasks)

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_u = u[:, :common_space_dim]
        common_s = s[:common_space_dim]
        common_v = v[:common_space_dim, :]

        combined_space_u = torch.zeros_like(u, device=combined_w.device)
        combined_space_s = torch.zeros_like(s, device=combined_w.device)
        combined_space_v = torch.zeros_like(v, device=combined_w.device)

        # Remove common-space components and keep each task's top singular subspace.
        common_proj = common_u @ common_u.T
        for task_idx, mat in enumerate(mats):
            mat_work = mat.to(dtype=work_dtype)
            mat_task_space = mat_work - common_proj @ mat_work
            u_ts, s_ts, v_ts = torch.linalg.svd(mat_task_space, full_matrices=False)

            start = task_idx * task_dims_per_task
            end = (task_idx + 1) * task_dims_per_task
            combined_space_u[:, start:end] = u_ts[:, :task_dims_per_task]
            combined_space_s[start:end] = s_ts[:task_dims_per_task]
            combined_space_v[start:end, :] = v_ts[:task_dims_per_task, :]

        common_start = n_tasks * task_dims_per_task
        common_end = common_start + common_space_dim
        combined_space_u[:, common_start:common_end] = common_u
        combined_space_s[common_start:common_end] = common_s
        combined_space_v[common_start:common_end, :] = common_v

        # Orthogonalize to enforce valid bases before isotropization.
        u_u, _, v_u = torch.linalg.svd(combined_space_u, full_matrices=False)
        u_v, _, v_v = torch.linalg.svd(combined_space_v, full_matrices=False)
        combined_space_u = u_u @ v_u
        combined_space_v = u_v @ v_v

        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
        out = combined_space_u @ torch.diag(combined_space_s) @ combined_space_v
        return out.to(dtype=out_dtype, device=out_device)


register(IsoCTSMerge())
