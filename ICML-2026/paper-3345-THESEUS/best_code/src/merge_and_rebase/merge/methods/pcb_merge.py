from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class PCBMerge:
    """
    Simplified PCB merge on flattened task vectors [N, D].

    Defaults follow the reference implementation, while keeping the code compact and robust:
      - clamp absolute deltas by per-task rank ratios
      - build a balancing mask from intra/inter signals
      - aggregate with per-task lambda scaling
    """

    name: str = "pcb"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        if len(tuned) == 0:
            raise ValueError("tuned must be non-empty")

        method_params = get_method_params(kwargs)
        clamp_min_ratio = float(method_params.get("clamp_min_ratio", 0.01))
        clamp_max_ratio = float(method_params.get("clamp_max_ratio", 0.01))
        att_ratio = float(method_params.get("att_ratio", 0.05))
        lam = float(method_params.get("lam", 1.2))

        self._validate_ratios(
            clamp_min_ratio=clamp_min_ratio,
            clamp_max_ratio=clamp_max_ratio,
            att_ratio=att_ratio,
        )

        w = default_weights(len(tuned), weights)
        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]
        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        direction = self._pcb_direction(
            base=base,
            deltas=deltas,
            keys=keys,
            w=w,
            clamp_min_ratio=clamp_min_ratio,
            clamp_max_ratio=clamp_max_ratio,
            att_ratio=att_ratio,
            lam=lam,
        )
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
    def _validate_ratios(*, clamp_min_ratio: float, clamp_max_ratio: float, att_ratio: float) -> None:
        if not (0.0 <= clamp_min_ratio < 1.0):
            raise ValueError("clamp_min_ratio must be in [0, 1).")
        if not (0.0 <= clamp_max_ratio < 1.0):
            raise ValueError("clamp_max_ratio must be in [0, 1).")
        if clamp_min_ratio + clamp_max_ratio >= 1.0:
            raise ValueError("clamp_min_ratio + clamp_max_ratio must be < 1.")
        if not (0.0 < att_ratio <= 1.0):
            raise ValueError("att_ratio must be in (0, 1].")

    @staticmethod
    def _normalize_minmax(x: torch.Tensor, *, dim: int, eps: float = 1e-12) -> torch.Tensor:
        min_values = x.amin(dim=dim, keepdim=True)
        max_values = x.amax(dim=dim, keepdim=True)
        denom = (max_values - min_values).clamp_min(eps)
        return (x - min_values) / denom

    @staticmethod
    def _clamp_by_ratio(x: torch.Tensor, *, min_ratio: float, max_ratio: float) -> torch.Tensor:
        if x.ndim == 1:
            d = x.shape[0]
            sorted_x, _ = torch.sort(x)
            lo_idx = int(d * min_ratio)
            hi_idx = int(d * (1.0 - max_ratio) - 1)
            hi_idx = max(lo_idx, hi_idx)
            min_v = sorted_x[lo_idx]
            max_v = sorted_x[hi_idx]
            return torch.clamp(x, min=min_v, max=max_v)

        if x.ndim == 2:
            d = x.shape[1]
            sorted_x, _ = torch.sort(x, dim=1)
            lo_idx = int(d * min_ratio)
            hi_idx = int(d * (1.0 - max_ratio) - 1)
            hi_idx = max(lo_idx, hi_idx)
            min_v = sorted_x[:, lo_idx].unsqueeze(1)
            max_v = sorted_x[:, hi_idx].unsqueeze(1)
            return torch.clamp(x, min=min_v, max=max_v)

        raise ValueError(f"Expected x to be 1D or 2D, got shape {tuple(x.shape)}")

    @classmethod
    def _pcb_delta(
        cls,
        M: torch.Tensor,
        *,
        w: torch.Tensor,
        clamp_min_ratio: float,
        clamp_max_ratio: float,
        att_ratio: float,
        lam: float,
    ) -> torch.Tensor:
        if M.ndim != 2:
            raise ValueError(f"Expected M to have shape [N, D], got {tuple(M.shape)}")
        if M.shape[0] == 0:
            return torch.empty((0,), dtype=M.dtype, device=M.device)
        if w.numel() != M.shape[0]:
            raise ValueError(f"weights length must match row count in M. got {w.numel()} vs {M.shape[0]}")

        abs_M = M.abs()
        abs_clamped = cls._clamp_by_ratio(abs_M, min_ratio=clamp_min_ratio, max_ratio=clamp_max_ratio)
        clamped_M = M.sign() * abs_clamped

        norm_abs = cls._normalize_minmax(abs_clamped, dim=1)
        intra = torch.exp(float(M.shape[0]) * norm_abs.square())
        signed_norm = M.sign() * norm_abs
        inter = torch.tanh(M * signed_norm.sum(dim=0))
        balancing = intra * inter

        # Keep strongest attention fraction per task before normalizing to [0, 1].
        scale_seed = cls._clamp_by_ratio(balancing, min_ratio=1.0 - att_ratio, max_ratio=0.0)
        scale = cls._normalize_minmax(scale_seed, dim=1)

        lams = (float(lam) * w.to(device=M.device, dtype=M.dtype)).view(-1, 1)
        num = (clamped_M * lams * scale).sum(dim=0)
        den = scale.sum(dim=0).clamp_min(1e-12)
        return num / den

    @staticmethod
    def _ratio_indices(length: int, *, min_ratio: float, max_ratio: float) -> tuple[int, int]:
        lo_idx = int(length * min_ratio)
        hi_idx = int(length * (1.0 - max_ratio) - 1)
        hi_idx = max(lo_idx, hi_idx)
        return lo_idx, hi_idx

    @classmethod
    def _row_clamp_bounds(
        cls,
        deltas: Sequence[TensorDict],
        keys: Sequence[str],
        *,
        min_ratio: float,
        max_ratio: float,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower: list[torch.Tensor] = []
        upper: list[torch.Tensor] = []
        for delta in deltas:
            flat_abs = TaskVector.flatten_dict(delta, keys, dtype=dtype).abs()
            if flat_abs.numel() == 0:
                lower.append(torch.tensor(0.0, dtype=dtype))
                upper.append(torch.tensor(0.0, dtype=dtype))
                continue
            lo_idx, hi_idx = cls._ratio_indices(flat_abs.numel(), min_ratio=min_ratio, max_ratio=max_ratio)
            lower.append(flat_abs.kthvalue(lo_idx + 1).values.detach())
            upper.append(flat_abs.kthvalue(hi_idx + 1).values.detach())
        return torch.stack(lower), torch.stack(upper)

    @classmethod
    def _pcb_direction(
        cls,
        *,
        base: TensorDict,
        deltas: Sequence[TensorDict],
        keys: Sequence[str],
        w: torch.Tensor,
        clamp_min_ratio: float,
        clamp_max_ratio: float,
        att_ratio: float,
        lam: float,
        eps: float = 1e-12,
    ) -> TensorDict:
        if len(deltas) == 0:
            return {}
        work_dtype = torch.float32
        row_min, row_max = cls._row_clamp_bounds(
            deltas,
            keys,
            min_ratio=clamp_min_ratio,
            max_ratio=clamp_max_ratio,
            dtype=work_dtype,
        )
        row_denom = (row_max - row_min).clamp_min(eps)
        n_tasks = len(deltas)

        balance_chunks: list[list[torch.Tensor]] = [[] for _ in deltas]
        balance_max = torch.full((n_tasks,), float("-inf"), dtype=work_dtype)
        total_params = 0
        for key in keys:
            chunk = torch.stack([delta[key].reshape(-1).to(dtype=work_dtype) for delta in deltas], dim=0)
            total_params += int(chunk.shape[1])
            abs_clamped = torch.clamp(chunk.abs(), min=row_min[:, None], max=row_max[:, None])
            norm_abs = (abs_clamped - row_min[:, None]) / row_denom[:, None]
            signed_norm = chunk.sign() * norm_abs
            inter = torch.tanh(chunk * signed_norm.sum(dim=0, keepdim=True))
            intra = torch.exp(float(n_tasks) * norm_abs.square())
            balancing = intra * inter
            balance_max = torch.maximum(balance_max, balancing.amax(dim=1))
            for idx in range(n_tasks):
                balance_chunks[idx].append(balancing[idx].detach())

        scale_min: list[torch.Tensor] = []
        for idx in range(n_tasks):
            flat_balancing = torch.cat(balance_chunks[idx], dim=0)
            lo_idx, _ = cls._ratio_indices(flat_balancing.numel(), min_ratio=1.0 - att_ratio, max_ratio=0.0)
            scale_min.append(flat_balancing.kthvalue(lo_idx + 1).values.detach())
        scale_min_t = torch.stack(scale_min)
        scale_denom = (balance_max - scale_min_t).clamp_min(eps)
        lams = (float(lam) * w.to(dtype=work_dtype)).view(-1, 1)

        direction: TensorDict = {}
        for key in keys:
            chunk = torch.stack([delta[key].reshape(-1).to(dtype=work_dtype) for delta in deltas], dim=0)
            abs_clamped = torch.clamp(chunk.abs(), min=row_min[:, None], max=row_max[:, None])
            clamped_chunk = chunk.sign() * abs_clamped
            norm_abs = (abs_clamped - row_min[:, None]) / row_denom[:, None]
            signed_norm = chunk.sign() * norm_abs
            inter = torch.tanh(chunk * signed_norm.sum(dim=0, keepdim=True))
            intra = torch.exp(float(n_tasks) * norm_abs.square())
            balancing = intra * inter
            scale_seed = torch.clamp(balancing, min=scale_min_t[:, None])
            scale = (scale_seed - scale_min_t[:, None]) / scale_denom[:, None]
            num = (clamped_chunk * lams * scale).sum(dim=0)
            den = scale.sum(dim=0).clamp_min(eps)
            merged = (num / den).view_as(base[key]).to(dtype=base[key].dtype, device=base[key].device)
            direction[key] = merged

        return direction


register(PCBMerge())
register(PCBMerge(name="pcb_merge"))
