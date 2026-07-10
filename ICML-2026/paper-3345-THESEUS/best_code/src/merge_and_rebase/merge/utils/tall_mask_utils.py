from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..task_vectors import TaskVector

TensorDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class TallMaskCache:
    tasks: Sequence[str]
    keys: Sequence[str]
    base_flat: torch.Tensor  # θ0 (float32, CPU)
    tuned_flat: dict[str, torch.Tensor]  # θt per task (float32, CPU)
    diff_pt: dict[str, torch.Tensor]  # |θ0 - θt| per task (float32, CPU)
    tvsum_flat: torch.Tensor  # Σ w_i * Δ_i (float32, CPU) (optionally topk-pruned per task)


def build_tall_mask_cache(
    *,
    base: Mapping[str, torch.Tensor],
    tuned_by_task: Mapping[str, Mapping[str, torch.Tensor]],
    weights_by_task: Mapping[str, float] | None = None,
    keep_ratio: float | None = None,
    strict: bool = False,
) -> TallMaskCache:
    """
    tuned_by_task must contain FULL tuned weights θ_t, not deltas.

    IMPORTANT: we compute keys only from TaskVector deltas, so we only operate on
    the float parameters that TaskVector itself considers mergeable.
    """
    tasks = sorted(tuned_by_task.keys())
    if not tasks:
        raise ValueError("tuned_by_task is empty")

    # Build task vectors once (these already apply key filtering inside TaskVector.from_checkpoints)
    tvs: dict[str, TaskVector] = {}
    for t in tasks:
        tv = TaskVector.from_checkpoints(base, tuned_by_task[t], strict=strict)
        if keep_ratio is not None:
            tv = tv.mask_by_magnitude(float(keep_ratio))
        tvs[t] = tv

    # Keys are the intersection of base and all deltas (safe, float-only)
    keys = TaskVector.common_keys(base, [tvs[t].delta for t in tasks])

    # Flatten θ0
    base_cpu: TensorDict = {k: base[k].detach().cpu() for k in keys}
    base_flat = TaskVector.flatten_dict(base_cpu, keys, dtype=torch.float32)

    # Flatten θt and precompute diff_pt = |θ0 - θt|
    tuned_flat: dict[str, torch.Tensor] = {}
    diff_pt: dict[str, torch.Tensor] = {}
    for t in tasks:
        tuned_cpu: TensorDict = {k: tuned_by_task[t][k].detach().cpu() for k in keys}
        ft = TaskVector.flatten_dict(tuned_cpu, keys, dtype=torch.float32)
        tuned_flat[t] = ft
        diff_pt[t] = (base_flat - ft).abs()

    # Sum (optionally pruned) deltas, then flatten once
    tvsum_sd: TensorDict = {k: torch.zeros_like(base_cpu[k], dtype=torch.float32) for k in keys}
    for t in tasks:
        w = 1.0 if weights_by_task is None else float(weights_by_task.get(t, 1.0))
        d = tvs[t].delta
        for k in keys:
            tvsum_sd[k] = tvsum_sd[k] + float(w) * d[k].detach().cpu().to(dtype=torch.float32)

    tvsum_flat = TaskVector.flatten_dict(tvsum_sd, keys, dtype=torch.float32)

    return TallMaskCache(
        tasks=tasks,
        keys=keys,
        base_flat=base_flat,
        tuned_flat=tuned_flat,
        diff_pt=diff_pt,
        tvsum_flat=tvsum_flat,
    )


def tall_masks(
    cache: TallMaskCache,
    *,
    lambdas: Sequence[float],
    alpha: float = 1.0,
    return_state_dict: bool = True,
    like: Mapping[str, torch.Tensor] | None = None,
) -> dict[float, dict[str, TensorDict | torch.Tensor]]:
    """
    mask_t = |θ0 - θt| > lambda * |θmulti - θt|
    θmulti = θ0 + alpha * tvsum
    """
    theta_multi = cache.base_flat + float(alpha) * cache.tvsum_flat

    # For this alpha, diff_multi depends only on theta_multi (so compute once per task)
    diff_multi: dict[str, torch.Tensor] = {}
    for t in cache.tasks:
        ft = cache.tuned_flat[t]
        diff_multi[t] = (theta_multi - ft).abs()

    out: dict[float, dict[str, TensorDict | torch.Tensor]] = {}
    for lam in lambdas:
        lam_f = float(lam)
        per_task: dict[str, TensorDict | torch.Tensor] = {}

        for t in cache.tasks:
            m_flat = cache.diff_pt[t] > (diff_multi[t] * lam_f)  # bool [D]

            if return_state_dict:
                if like is None:
                    raise ValueError("return_state_dict=True requires `like` for unflatten shapes")
                per_task[t] = TaskVector.unflatten_like(
                    m_flat,
                    like=like,
                    keys=cache.keys,
                    out_dtype=torch.bool,
                    out_device=None,
                )
            else:
                per_task[t] = m_flat

        out[lam_f] = per_task

    return out


def construct_consensus_mask(
    *,
    base_sd: TensorDict,
    tall_masks: dict[str, TensorDict],  # task -> mask dict (bool / 0-1)
    prun_thre_k: int,
    keys: list[str] | None = None,  # optional fixed key order
) -> TensorDict:
    """
    Returns consensus mask as a TensorDict (same shapes as base_sd) with bool tensors.
    A parameter is kept if it is activated in >= prun_thre_k tasks.
    """
    if prun_thre_k < 0:
        raise ValueError("prun_thre_k must be >= 0")
    if len(tall_masks) == 0:
        raise ValueError("tall_masks must be non-empty")

    mask_dicts = list(tall_masks.values())

    # Restrict to keys shared by base and all task masks
    shared_keys = TaskVector.common_keys(base_sd, mask_dicts, keys=keys)

    # Flatten each task mask to a boolean vector in a deterministic order
    flats = []
    for md in mask_dicts:
        flat = TaskVector.flatten_dict(md, shared_keys, dtype=torch.float32)
        flats.append(flat)

    M = torch.stack(flats, dim=0)  # [T, D], float32 0/1 or bool casted

    # Count number of tasks that activate each parameter
    counts = (M > 0).sum(dim=0)  # [D], int64
    consensus_flat = counts >= int(prun_thre_k)  # [D], bool

    # Unflatten back to dict (bool tensors, same shapes as base)
    consensus = TaskVector.unflatten_like(
        consensus_flat.to(torch.bool),
        like=base_sd,
        keys=shared_keys,
        out_dtype=torch.bool,
    )
    return consensus
