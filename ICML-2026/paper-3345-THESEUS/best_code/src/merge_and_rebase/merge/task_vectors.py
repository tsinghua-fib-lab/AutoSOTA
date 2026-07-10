from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from .base import TensorDict

KeyFilter = Callable[[str, torch.Tensor], bool]


def _is_float_tensor(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and t.is_floating_point()


def default_key_filter(k: str, v: torch.Tensor) -> bool:
    """
    Keep only floating-point tensors.
    Exclude common non-mergeable buffers by name heuristics if needed.
    """
    if not _is_float_tensor(v):
        return False
    # common exclusions (safe defaults, can be overridden)
    if k.endswith("num_batches_tracked"):
        return False
    if "position_ids" in k:
        return False
    return True


def filter_state_dict(sd: Mapping[str, torch.Tensor], key_filter: KeyFilter) -> TensorDict:
    return {k: v for k, v in sd.items() if key_filter(k, v)}


def intersect_keys(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> Sequence[str]:
    return sorted(set(a.keys()).intersection(b.keys()))


def assert_compatible(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor], keys: Sequence[str]) -> None:
    for k in keys:
        ta = a[k]
        tb = b[k]
        if ta.shape != tb.shape:
            raise ValueError(f"Shape mismatch for key '{k}': {tuple(ta.shape)} vs {tuple(tb.shape)}")
        if ta.dtype != tb.dtype:
            # dtype mismatch is allowed if you handle casts, but it's usually a bug
            raise ValueError(f"Dtype mismatch for key '{k}': {ta.dtype} vs {tb.dtype}")


@dataclass(frozen=True)
class TaskVector:
    """
    Represents a task vector Δ over parameters.
    """

    delta: TensorDict

    @staticmethod
    def from_checkpoints(
        base: Mapping[str, torch.Tensor],
        tuned: Mapping[str, torch.Tensor],
        *,
        key_filter: KeyFilter = default_key_filter,
        strict: bool = True,
    ) -> TaskVector:
        b = filter_state_dict(base, key_filter)
        t = filter_state_dict(tuned, key_filter)

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

        d: TensorDict = {}
        for k in keys:
            d[k] = (t[k] - b[k]).detach()
        return TaskVector(delta=d)

    def keys(self) -> Sequence[str]:
        return sorted(self.delta.keys())

    @staticmethod
    def common_keys(
        base: Mapping[str, torch.Tensor],
        deltas: Sequence[Mapping[str, torch.Tensor]],
        *,
        keys: Sequence[str] | None = None,
        sort: bool = True,
    ) -> list[str]:
        """
        Keys shared by base and all deltas. Optionally restrict to a provided key list.
        """
        if keys is not None:
            kset = set(keys)
            kset &= set(base.keys())
            for d in deltas:
                kset &= set(d.keys())
            out = list(kset)
            return sorted(out) if sort else out

        kset = set(base.keys())
        for d in deltas:
            kset &= set(d.keys())
        out = list(kset)
        return sorted(out) if sort else out

    @staticmethod
    def flatten_dict(
        d: Mapping[str, torch.Tensor],
        keys: Sequence[str],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Deterministic flatten: concatenates tensors in `keys` order.
        """
        parts: list[torch.Tensor] = []
        for k in keys:
            t = d[k]
            if device is not None:
                t = t.to(device)
            parts.append(t.reshape(-1).to(dtype=dtype))
        if not parts:
            return torch.empty(0, dtype=dtype, device=device)
        return torch.cat(parts, dim=0)

    @staticmethod
    def unflatten_like(
        flat: torch.Tensor,
        like: Mapping[str, torch.Tensor],
        keys: Sequence[str],
        *,
        out_dtype: torch.dtype | None = None,
        out_device: torch.device | None = None,
    ) -> TensorDict:
        """
        Inverse of flatten_dict using shapes from `like`.
        Returns a new dict {k: tensor} for the provided keys.
        """
        out: TensorDict = {}
        offset = 0
        for k in keys:
            ref = like[k]
            n = ref.numel()
            chunk = flat[offset : offset + n].view_as(ref)
            if out_dtype is not None:
                chunk = chunk.to(dtype=out_dtype)
            if out_device is not None:
                chunk = chunk.to(device=out_device)
            out[k] = chunk
            offset += n
        return out

    @staticmethod
    def stack_flattened(
        deltas: Sequence[Mapping[str, torch.Tensor]],
        keys: Sequence[str],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Returns [N, D] stacked flattened deltas in deterministic key order.
        """
        rows = [TaskVector.flatten_dict(d, keys, dtype=dtype, device=device) for d in deltas]
        if not rows:
            return torch.empty((0, 0), dtype=dtype, device=device)
        return torch.stack(rows, dim=0)

    def to(self, *, device: str | None = None, dtype: torch.dtype | None = None) -> TaskVector:
        out: TensorDict = {}
        for k, v in self.delta.items():
            vv = v
            if device is not None:
                vv = vv.to(device)
            if dtype is not None:
                vv = vv.to(dtype=dtype)
            out[k] = vv
        return TaskVector(delta=out)

    def clone(self) -> TaskVector:
        return TaskVector(delta={k: v.clone() for k, v in self.delta.items()})

    # -------------------
    # Arithmetic operators
    # -------------------

    def scaled(self, alpha: float) -> TaskVector:
        a = float(alpha)
        return TaskVector(delta={k: v * a for k, v in self.delta.items()})

    def add(self, other: TaskVector, *, strict: bool = True) -> TaskVector:
        if strict and set(self.delta.keys()) != set(other.delta.keys()):
            raise KeyError("TaskVector key sets differ (strict=True).")
        keys = intersect_keys(self.delta, other.delta)
        out = {k: self.delta[k] + other.delta[k] for k in keys}
        return TaskVector(delta=out)

    def sub(self, other: TaskVector, *, strict: bool = True) -> TaskVector:
        if strict and set(self.delta.keys()) != set(other.delta.keys()):
            raise KeyError("TaskVector key sets differ (strict=True).")
        keys = intersect_keys(self.delta, other.delta)
        out = {k: self.delta[k] - other.delta[k] for k in keys}
        return TaskVector(delta=out)

    def l2_norm(self) -> float:
        s = 0.0
        for v in self.delta.values():
            s += float((v.float() ** 2).sum().item())
        return float(s**0.5)

    def normalized(self, eps: float = 1e-12) -> TaskVector:
        n = self.l2_norm()
        if n < eps:
            return self
        inv = 1.0 / n
        return self.scaled(inv)

    def clamp_(self, min_value: float, max_value: float) -> TaskVector:
        out: TensorDict = {}
        mn, mx = float(min_value), float(max_value)
        for k, v in self.delta.items():
            out[k] = v.clamp(mn, mx)
        return TaskVector(delta=out)

    # -------------------
    # Masking and shaping
    # -------------------

    def mask_by_magnitude(self, keep_ratio: float) -> TaskVector:
        """
        Keep only the largest |Δ| entries globally (unstructured pruning).
        """
        r = float(keep_ratio)
        if not (0.0 < r <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")

        # flatten magnitudes
        mags = torch.cat([self.delta[k].abs().flatten().float() for k in self.keys()])
        if mags.numel() == 0:
            return self

        k = max(1, int(round(r * mags.numel())))
        thresh = torch.topk(mags, k=k, largest=True).values.min()

        out: TensorDict = {}
        for name, v in self.delta.items():
            m = (v.abs().float() >= thresh).to(v.dtype)
            out[name] = v * m
        return TaskVector(delta=out)

    def mask_by_sign_agreement(self, other: TaskVector, *, strict: bool = True) -> TaskVector:
        """
        Keep entries where sign(Δ_self) == sign(Δ_other).
        Useful for "consensus" masking across tasks.
        """
        if strict and set(self.delta.keys()) != set(other.delta.keys()):
            raise KeyError("TaskVector key sets differ (strict=True).")
        keys = intersect_keys(self.delta, other.delta)
        out: TensorDict = {}
        for k in keys:
            a = self.delta[k]
            b = other.delta[k]
            m = (torch.sign(a) == torch.sign(b)).to(a.dtype)
            out[k] = a * m
        return TaskVector(delta=out)


def apply_task_vector(
    target: Mapping[str, torch.Tensor],
    tv: TaskVector,
    *,
    alpha: float = 1.0,
    key_filter: KeyFilter = default_key_filter,
    strict: bool = False,
) -> TensorDict:
    """
    Returns a new state_dict: target + alpha * tv
    Only applies to keys present in both and passing key_filter.
    """
    tgt = filter_state_dict(target, key_filter)
    keys = intersect_keys(tgt, tv.delta)

    if strict and set(keys) != set(tv.delta.keys()):
        missing = sorted(set(tv.delta.keys()) - set(keys))
        raise KeyError(f"Target missing {len(missing)} keys from task vector. Example: {missing[:10]}")

    out: TensorDict = dict(target)  # preserve original mapping entries too
    a = float(alpha)
    for k in keys:
        out[k] = tgt[k] + a * tv.delta[k].to(dtype=tgt[k].dtype, device=tgt[k].device)
    return out


def compose_task_vectors(
    base: Mapping[str, torch.Tensor],
    vectors: Sequence[TaskVector],
    weights: Sequence[float] | None = None,
    *,
    key_filter: KeyFilter = default_key_filter,
    strict: bool = False,
) -> TensorDict:
    """
    θ = θ_base + Σ_i w_i Δ_i
    Returns a new state_dict.
    """
    if weights is None:
        weights = [1.0] * len(vectors)
    if len(weights) != len(vectors):
        raise ValueError("weights and vectors must have same length")

    base_f = filter_state_dict(base, key_filter)
    out: TensorDict = dict(base)

    # determine keys that exist in base and every vector
    keys = set(base_f.keys())
    for tv in vectors:
        keys = keys.intersection(tv.delta.keys())
    keys = sorted(keys)

    if strict:
        for i, tv in enumerate(vectors):
            missing = sorted(set(tv.delta.keys()) - set(keys))
            if missing:
                raise KeyError(f"Vector {i} has keys not in common intersection. Example: {missing[:10]}")

    # apply composition
    for k in keys:
        acc = base_f[k]
        for w, tv in zip(weights, vectors, strict=False):
            acc = acc + float(w) * tv.delta[k].to(dtype=acc.dtype, device=acc.device)
        out[k] = acc

    return out
