from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import TensorDict
from ..registry import register
from ..task_vectors import TaskVector, assert_compatible, default_key_filter, filter_state_dict, intersect_keys
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class DAREMerge:
    name: str = "dare_merge"

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

        if "drop_rate" in method_params:
            drop_rate = float(method_params["drop_rate"])
        elif "p" in method_params:
            drop_rate = float(method_params["p"])
        elif "keep_ratio" in method_params:
            drop_rate = 1.0 - float(method_params["keep_ratio"])
        else:
            drop_rate = 0.9

        seed_val = method_params.get("seed", None)
        seed = None if seed_val is None else int(seed_val)
        rescale = bool(method_params.get("rescale", True))
        low_memory = bool(method_params.get("low_memory", False))

        if low_memory:
            return self._prepare_low_memory(
                base=base,
                tuned=tuned,
                weights=w,
                drop_rate=drop_rate,
                rescale=rescale,
                seed=seed,
                strict=strict,
            )

        tvs = [TaskVector.from_checkpoints(base, t, strict=strict) for t in tuned]

        deltas = [tv.delta for tv in tvs]
        keys = TaskVector.common_keys(base, deltas)
        flat = TaskVector.stack_flattened(deltas, keys, dtype=torch.float32)
        merged_flat = self._dare_delta(flat, w=w, drop_rate=drop_rate, rescale=rescale, seed=seed)
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
    def _dare_delta(
        M: torch.Tensor, *, w: torch.Tensor, drop_rate: float, rescale: bool, seed: int | None
    ) -> torch.Tensor:
        """
        DARE on flattened task vectors [N, D]:
          1) random unstructured sparsification per task
          2) optional scaling by 1/(1-p) to preserve expectation
          3) weighted aggregation across tasks
        """
        if M.ndim != 2:
            raise ValueError(f"Expected M to have shape [N, D], got {tuple(M.shape)}")

        if M.shape[0] == 0:
            return torch.empty((0,), dtype=M.dtype, device=M.device)

        if not (0.0 <= drop_rate < 1.0):
            raise ValueError("drop_rate must satisfy 0 <= drop_rate < 1.")

        if w.numel() != M.shape[0]:
            raise ValueError(f"weights length must match row count in M. got {w.numel()} vs {M.shape[0]}")

        keep_prob = 1.0 - float(drop_rate)
        if keep_prob == 1.0:
            sparse = M
        else:
            gen = None
            if seed is not None:
                gen = torch.Generator(device=M.device)
                gen.manual_seed(seed)
            mask = (torch.rand(M.shape, device=M.device, generator=gen) < keep_prob).to(M.dtype)
            sparse = M * mask
            if rescale:
                sparse = sparse / keep_prob

        return (sparse * w.to(device=M.device, dtype=M.dtype).view(-1, 1)).sum(dim=0)

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

    def _prepare_low_memory(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: torch.Tensor,
        drop_rate: float,
        rescale: bool,
        seed: int | None,
        strict: bool,
    ) -> tuple[TensorDict, TensorDict]:
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError("drop_rate must satisfy 0 <= drop_rate < 1.")

        b, tuned_filtered, keys = self._shared_filtered_keys(base, tuned, strict=strict)
        if not tuned_filtered or not keys:
            return base, {}
        if weights.numel() != len(tuned_filtered):
            raise ValueError(
                f"weights length must match tuned checkpoint count. got {weights.numel()} vs {len(tuned_filtered)}"
            )

        keep_prob = 1.0 - float(drop_rate)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=b[keys[0]].device)
            gen.manual_seed(seed)

        direction: TensorDict = {key: torch.zeros_like(b[key], dtype=torch.float32) for key in keys}
        for task_idx, tuned_sd in enumerate(tuned_filtered):
            w_i = float(weights[task_idx])
            for key in keys:
                delta = (tuned_sd[key] - b[key]).detach().to(dtype=torch.float32)
                if keep_prob != 1.0:
                    mask = torch.rand(delta.shape, device=delta.device, generator=gen) < keep_prob
                    delta = delta * mask.to(dtype=delta.dtype)
                    if rescale:
                        delta = delta / keep_prob
                direction[key] = direction[key] + delta * w_i

        return base, direction


register(DAREMerge())
