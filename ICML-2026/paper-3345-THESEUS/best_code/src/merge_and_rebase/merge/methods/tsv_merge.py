from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tqdm import tqdm

from ..base import TensorDict
from ..registry import register
from ._common import axpy_state_dict, default_weights, get_method_params


@dataclass(frozen=True)
class TSVMerge:
    name: str = "tsv_merge"

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
        direction = self._build_direction(
            base=base,
            tuned=tuned,
            weights=weights,
            strict=strict,
            method_params=method_params,
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
    def _parse_dtype(name: str | torch.dtype) -> torch.dtype:
        if isinstance(name, torch.dtype):
            return name
        key = str(name).strip().lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        if key not in mapping:
            raise ValueError("Unknown dtype for TSV merge. Use one of: fp16, bf16, fp32, fp64 (or float16/32/64).")
        return mapping[key]

    @staticmethod
    def _rank_from_singular_values(
        num_singular_values: int,
        *,
        sv_reduction: float,
        max_rank: int | None,
    ) -> int:
        r = max(1, int(num_singular_values * float(sv_reduction)))
        if max_rank is not None:
            r = min(r, int(max_rank))
        return max(1, int(r))

    def _build_direction(
        self,
        *,
        base: TensorDict,
        tuned: Sequence[TensorDict],
        weights: Sequence[float] | None = None,
        strict: bool = False,
        method_params: dict | None = None,
    ) -> TensorDict:
        if not tuned:
            raise ValueError("tsv_merge requires at least one tuned checkpoint.")

        params = dict(method_params or {})
        w = default_weights(len(tuned), weights)
        vector_1d_merge = str(params.get("vector_1d_merge", "zero")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("tsv_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")

        sv_reduction = float(params.get("sv_reduction", 1.0 / max(1, len(tuned))))
        if not (0.0 < sv_reduction <= 1.0):
            raise ValueError("tsv_merge method_params['sv_reduction'] must be in (0, 1].")

        max_rank = params.get("max_rank", None)

        if max_rank is not None:
            max_rank = int(max_rank)
            if max_rank <= 0:
                raise ValueError("tsv_merge method_params['max_rank'] must be > 0.")

        svd_dtype = self._parse_dtype(params.get("svd_dtype", "float64"))
        if svd_dtype not in {torch.float32, torch.float64}:
            raise ValueError("tsv_merge method_params['svd_dtype'] must be float32/fp32 or float64/fp64.")
        accum_dtype = self._parse_dtype(str(params.get("accum_dtype", "float32")))
        # Backward-compat params that no longer affect execution:
        # TSV now always processes one key at a time to keep memory bounded.
        _ = bool(params.get("low_memory", False))
        key_batch_size_raw = params.get("key_batch_size", None)
        if key_batch_size_raw is not None and int(key_batch_size_raw) < 0:
            raise ValueError("tsv_merge method_params['key_batch_size'] must be >= 0.")
        progress = bool(params.get("progress", True))

        base_candidates: set[str] = {
            k
            for k, b in base.items()
            if isinstance(b, torch.Tensor) and b.ndim == 2 and "text_projection" not in k and b.is_floating_point()
        }

        common_keys: set[str] | None = None
        for i in range(len(tuned)):
            sd = tuned[i]
            current: set[str] = set()
            for k, t in sd.items():
                if k not in base_candidates:
                    continue
                b = base[k]
                if not isinstance(t, torch.Tensor):
                    continue
                if t.shape != b.shape:
                    continue
                if not t.is_floating_point():
                    continue
                current.add(k)
            common_keys = current if common_keys is None else common_keys.intersection(current)
            del sd

        keys = sorted(common_keys or set())

        if strict:
            expected = set(base_candidates)
            if set(keys) != expected:
                missing = sorted(expected - set(keys))
                raise ValueError(
                    "Strict mode: tuned checkpoints do not match base 2D keyspace for TSV merge. "
                    f"Missing keys sample: {missing[:10]}"
                )

        n_tuned = int(len(tuned))
        direction: TensorDict = {}

        key_iter = tqdm(
            keys,
            desc=f"Computing TSV directions - SVD Precision: {svd_dtype}",
            disable=(not progress) or (len(keys) <= 1),
        )
        for k in key_iter:
            b = base[k]
            k_min = min(int(b.shape[0]), int(b.shape[1]))
            r = self._rank_from_singular_values(
                k_min,
                sv_reduction=sv_reduction,
                max_rank=max_rank,
            )
            rt = int(r) * n_tuned
            sum_u = torch.zeros((int(b.shape[0]), rt), dtype=accum_dtype, device="cpu")
            sum_s = torch.zeros((rt,), dtype=accum_dtype, device="cpu")
            sum_v = torch.zeros((rt, int(b.shape[1])), dtype=accum_dtype, device="cpu")

            b_cpu = b.detach().to(device="cpu", dtype=svd_dtype)
            for i in range(n_tuned):
                sd = tuned[i]
                wi = float(w[i])
                t = sd[k].detach().to(device="cpu", dtype=svd_dtype)
                mat = t - b_cpu
                u, s, vh = torch.linalg.svd(mat, full_matrices=False)

                lo = int(i) * int(r)
                hi = lo + int(r)
                sum_u[:, lo:hi] = u[:, :r].to(dtype=accum_dtype, device="cpu")
                sum_s[lo:hi] = (s[:r] * wi).to(dtype=accum_dtype, device="cpu")
                sum_v[lo:hi, :] = vh[:r, :].to(dtype=accum_dtype, device="cpu")
                del sd

            u_u, _, vh_u = torch.linalg.svd(sum_u.to(dtype=svd_dtype), full_matrices=False)
            u_v, _, vh_v = torch.linalg.svd(sum_v.to(dtype=svd_dtype), full_matrices=False)
            merged = torch.linalg.multi_dot((u_u, vh_u, torch.diag(sum_s.to(dtype=svd_dtype)), u_v, vh_v))
            direction[k] = merged.to(dtype=b.dtype, device=b.device)

        if vector_1d_merge == "average":
            denom = float(w.sum().clamp_min(1e-12).item())
            one_d_candidates: set[str] = {
                k
                for k, b in base.items()
                if isinstance(b, torch.Tensor) and b.ndim == 1 and b.is_floating_point()
            }
            common_1d_keys: set[str] | None = None
            for sd in tuned:
                current: set[str] = set()
                for k, t in sd.items():
                    if k not in one_d_candidates:
                        continue
                    b = base[k]
                    if not isinstance(t, torch.Tensor):
                        continue
                    if t.shape != b.shape:
                        continue
                    if not t.is_floating_point():
                        continue
                    current.add(k)
                common_1d_keys = current if common_1d_keys is None else common_1d_keys.intersection(current)

            for k in sorted(common_1d_keys or set()):
                b = base[k]
                acc = torch.zeros_like(b)
                for wi, sd in zip(w, tuned, strict=True):
                    acc = acc + float(wi) * (sd[k].to(dtype=acc.dtype, device=acc.device) - b)
                direction[k] = acc / denom

        return direction


register(TSVMerge())
