from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector, assert_compatible, default_key_filter, filter_state_dict, intersect_keys
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class TIESMerge:
    """
    TIES-style merge:
      - prune each task vector to topK magnitude per-row (here: global vector topK fraction)
      - resolve sign per-dimension (majority)
      - disjoint merge by sign (mean by default)
    """

    name: str = "ties_merge"

    def prepare(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> tuple[TensorDict, TensorDict]:
        method_params = get_method_params(kwargs)
        merging_type = str(method_params.get("merging_type", "mean"))  # mean | sum | max
        topk = float(method_params.get("topk", 1.0))  # topK fraction per row for pruning (in [0,1] or [0,100])
        low_memory = bool(method_params.get("low_memory", False))

        w = default_weights(len(tuned), weights)

        if low_memory:
            return self._prepare_low_memory(
                base=base,
                tuned=tuned,
                weights=w,
                topk=topk,
                merging_type=merging_type,
                strict=strict,
            )

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]
        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)

        flat = TaskVector.stack_flattened(deltas, keys, dtype=torch.float32)

        pruned, _mask = self._topk_mask(flat, topk=topk)
        sign = self._resolve_sign(pruned)
        merged_flat = self._disjoint_merge(pruned, sign, w=w, merge=merging_type)

        direction: TensorDict = TaskVector.unflatten_like(merged_flat, like=base, keys=keys)
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
        keep_ratio: float | None = None,
        strict: bool = False,
        **kwargs,
    ) -> TensorDict:
        prepared = self.prepare(
            base=base,
            tuned=tuned,
            weights=weights,
            keep_ratio=keep_ratio,
            strict=strict,
            **kwargs,
        )
        return self.apply(prepared, alpha=float(alpha))

    @staticmethod
    def _topk_mask(M: torch.Tensor, topk: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Keep top |x| fraction per row (topk in [0,1] or [0,100]).
        Returns pruned M and boolean mask of kept entries.
        """
        if topk > 1.0:
            topk = topk / 100.0
        topk = float(topk)

        if topk >= 1.0:
            mask = torch.ones_like(M, dtype=torch.bool)
            return M, mask

        _, d = M.shape
        k = max(1, int(d * topk))
        vals, _ = torch.topk(M.abs(), k=k, dim=1, largest=True, sorted=False)
        thr = vals.min(dim=1, keepdim=True).values
        mask = M.abs() >= thr
        return M * mask, mask

    @staticmethod
    def _resolve_sign(M: torch.Tensor) -> torch.Tensor:
        """
        Majority sign per dimension after pruning.
        Zeros get filled with global majority.
        """
        if torch.all(M == 0):
            return torch.ones(M.shape[1], device=M.device, dtype=torch.float32)
        s = torch.sign(M.sum(dim=0))
        global_majority = torch.sign(s.sum())
        global_majority = global_majority if global_majority != 0 else torch.tensor(1.0, device=s.device)
        s[s == 0] = global_majority
        return s

    @staticmethod
    def _disjoint_merge(M: torch.Tensor, ref_sign: torch.Tensor, *, w: torch.Tensor, merge: str) -> torch.Tensor:
        """
        Select entries agreeing with ref_sign and aggregate across rows with weights.
        """
        keep = torch.where(ref_sign.unsqueeze(0) > 0, M > 0, M < 0)
        selected = M * keep

        w_row = w.to(selected.device, selected.dtype).view(-1, 1)
        selected = selected * w_row

        if merge == "mean":
            denom = (keep.to(selected.dtype) * w_row).sum(dim=0).clamp_min(1e-12)
            return selected.sum(dim=0) / denom
        if merge == "sum":
            return selected.sum(dim=0)
        if merge == "max":
            vals, _ = selected.abs().max(dim=0)
            return vals * ref_sign.to(vals.dtype)
        raise ValueError(f"Unknown TIES merge type '{merge}'")

    @staticmethod
    def _normalize_topk(topk: float) -> float:
        if topk > 1.0:
            topk = topk / 100.0
        return float(topk)

    @staticmethod
    def _shared_filtered_keys(
        base: TensorDict,
        tuned: Sequence[TensorDict],
        *,
        strict: bool,
    ) -> tuple[TensorDict, list[TensorDict], list[str]]:
        b = filter_state_dict(base, default_key_filter)
        tuned_filtered = [filter_state_dict(t, default_key_filter) for t in tuned]
        if not tuned_filtered:
            return b, [], []

        keyset = set(b.keys())
        for t in tuned_filtered:
            keys = intersect_keys(b, t)
            if strict:
                missing_in_tuned = sorted(set(b.keys()) - set(t.keys()))
                missing_in_base = sorted(set(t.keys()) - set(b.keys()))
                if missing_in_tuned or missing_in_base:
                    raise KeyError(
                        "Checkpoint keys mismatch.\n"
                        f"Missing in tuned: {missing_in_tuned[:10]}{' ...' if len(missing_in_tuned) > 10 else ''}\n"
                        f"Missing in base: {missing_in_base[:10]}{' ...' if len(missing_in_base) > 10 else ''}"
                    )
            assert_compatible(b, t, keys)
            keyset &= set(t.keys())

        return b, tuned_filtered, sorted(keyset)

    @staticmethod
    def _task_topk_threshold(
        *,
        base: TensorDict,
        tuned: TensorDict,
        keys: Sequence[str],
        k: int,
        total_numel: int,
    ) -> torch.Tensor | None:
        if k >= total_numel:
            return None

        # Keeping the smaller side bounds peak memory for both tiny and large pruning fractions.
        smallest_count = total_numel - k + 1
        keep_largest = k <= smallest_count
        heap_size = k if keep_largest else smallest_count
        retained: torch.Tensor | None = None

        for key in keys:
            mags = (tuned[key] - base[key]).detach().abs().reshape(-1).to(dtype=torch.float32)
            if mags.numel() == 0:
                continue
            candidate = mags if retained is None else torch.cat((retained, mags), dim=0)
            if candidate.numel() > heap_size:
                retained = torch.topk(candidate, k=heap_size, largest=keep_largest, sorted=False).values
            else:
                retained = candidate

        if retained is None or retained.numel() == 0:
            return None
        if keep_largest:
            return retained.min()
        return retained.max()

    def _prepare_low_memory(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: torch.Tensor,
        topk: float,
        merging_type: str,
        strict: bool,
    ) -> tuple[TensorDict, TensorDict]:
        b, tuned_filtered, keys = self._shared_filtered_keys(base, tuned, strict=strict)
        if not tuned_filtered or not keys:
            return base, {}

        topk = self._normalize_topk(topk)
        if not (0.0 < topk):
            raise ValueError("TIES method_params['topk'] must be > 0.")

        total_numel = sum(int(b[key].numel()) for key in keys)
        if total_numel == 0:
            return base, {}
        k = max(1, int(total_numel * topk))

        thresholds: list[torch.Tensor | None]
        if topk >= 1.0:
            thresholds = [None for _ in tuned_filtered]
        else:
            thresholds = [
                self._task_topk_threshold(base=b, tuned=t, keys=keys, k=k, total_numel=total_numel)
                for t in tuned_filtered
            ]

        majority_sum = 0.0
        for key in keys:
            ref = b[key]
            signed_sum = torch.zeros_like(ref, dtype=torch.float32)
            for task_idx, tuned_sd in enumerate(tuned_filtered):
                delta = (tuned_sd[key] - ref).detach().to(dtype=torch.float32)
                threshold = thresholds[task_idx]
                if threshold is not None:
                    delta = delta * (delta.abs() >= threshold).to(dtype=delta.dtype)
                signed_sum = signed_sum + delta
            majority_sum += float(torch.sign(signed_sum).sum().item())

        global_majority = 1.0 if majority_sum == 0.0 else (1.0 if majority_sum > 0.0 else -1.0)

        direction: TensorDict = {}
        for key in keys:
            ref = b[key]
            signed_sum = torch.zeros_like(ref, dtype=torch.float32)
            pruned_deltas: list[torch.Tensor] = []
            for task_idx, tuned_sd in enumerate(tuned_filtered):
                delta = (tuned_sd[key] - ref).detach().to(dtype=torch.float32)
                threshold = thresholds[task_idx]
                if threshold is not None:
                    delta = delta * (delta.abs() >= threshold).to(dtype=delta.dtype)
                pruned_deltas.append(delta)
                signed_sum = signed_sum + delta

            ref_sign = torch.sign(signed_sum)
            ref_sign[ref_sign == 0] = global_majority

            if merging_type == "mean":
                numerator = torch.zeros_like(ref, dtype=torch.float32)
                denom = torch.zeros_like(ref, dtype=torch.float32)
                for task_idx, delta in enumerate(pruned_deltas):
                    keep = torch.where(ref_sign > 0, delta > 0, delta < 0)
                    w_i = float(weights[task_idx])
                    numerator = numerator + delta * keep.to(dtype=delta.dtype) * w_i
                    denom = denom + keep.to(dtype=denom.dtype) * w_i
                direction[key] = numerator / denom.clamp_min(1e-12)
            elif merging_type == "sum":
                acc = torch.zeros_like(ref, dtype=torch.float32)
                for task_idx, delta in enumerate(pruned_deltas):
                    keep = torch.where(ref_sign > 0, delta > 0, delta < 0)
                    acc = acc + delta * keep.to(dtype=delta.dtype) * float(weights[task_idx])
                direction[key] = acc
            elif merging_type == "max":
                max_abs = torch.zeros_like(ref, dtype=torch.float32)
                for task_idx, delta in enumerate(pruned_deltas):
                    keep = torch.where(ref_sign > 0, delta > 0, delta < 0)
                    selected_abs = (delta * keep.to(dtype=delta.dtype) * float(weights[task_idx])).abs()
                    max_abs = torch.maximum(max_abs, selected_abs)
                direction[key] = max_abs * ref_sign.to(dtype=max_abs.dtype)
            else:
                raise ValueError(f"Unknown TIES merge type '{merging_type}'")

        return base, direction


register(TIESMerge())
