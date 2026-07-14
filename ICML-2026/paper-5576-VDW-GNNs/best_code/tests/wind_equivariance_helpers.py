from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch, Data


@dataclass
class VectorCheckResult:
    name: str
    shape: Tuple[int, ...]
    vector_axis: int
    rel_mean: float
    rel_max: float
    cos_err_mean: float
    coverage: float
    ok: bool


def maybe_get_vector_axis(
    x: torch.Tensor,
    *,
    d: int = 3,
) -> Optional[int]:
    """
    Heuristic: return the last axis with size==d, else None.
    """
    if not isinstance(x, torch.Tensor):
        return None
    axes = [i for i, s in enumerate(x.shape) if int(s) == int(d)]
    if len(axes) == 0:
        return None
    return int(axes[-1])


def rotate_along_axis(
    x: torch.Tensor,
    R: torch.Tensor,
    *,
    axis: int,
) -> torch.Tensor:
    """
    Apply y = x @ R^T along the specified vector axis.
    """
    if x.shape[axis] != R.shape[0]:
        raise ValueError("rotate_along_axis: axis size must match R dimension.")
    if R.shape[0] != R.shape[1]:
        raise ValueError("rotate_along_axis: R must be square.")

    # Move vector axis to last, apply matmul, then move back.
    x_perm = torch.movedim(x, axis, -1)  # (..., d)
    y_perm = x_perm @ R.T
    y = torch.movedim(y_perm, -1, axis)
    return y


def vector_equivariance_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    R: torch.Tensor,
    *,
    axis: int,
    node_mask: Optional[torch.Tensor] = None,
    node_axis: Optional[int] = None,
    eps_abs: float = 1e-8,
    eps_rel: float = 1e-3,
) -> Tuple[float, float, float, float]:
    """
    Compute equivariance errors for vector-valued tensors:
      expected_b = rotate(a)
      compare b vs expected_b along vector axis.
    """
    if (node_mask is not None) and (node_axis is not None):
        node_mask = node_mask.to(device=a.device).bool()
        idx = torch.nonzero(node_mask, as_tuple=False).squeeze(1)
        a = torch.index_select(a, dim=int(node_axis), index=idx)
        b = torch.index_select(b, dim=int(node_axis), index=idx)

    expected = rotate_along_axis(a, R, axis=axis)
    diff = b - expected

    # Flatten all but vector axis into a batch of vectors
    expected_vec = torch.movedim(expected, axis, -1).reshape(-1, 3)
    b_vec = torch.movedim(b, axis, -1).reshape(-1, 3)
    diff_vec = torch.movedim(diff, axis, -1).reshape(-1, 3)

    den = torch.linalg.norm(expected_vec, dim=-1)
    num = torch.linalg.norm(diff_vec, dim=-1)

    med = torch.median(den)
    thresh = torch.maximum(
        torch.full_like(den, eps_abs),
        eps_rel * med,
    )
    mask = den >= thresh

    if mask.any():
        rel = (num[mask] / torch.clamp(den[mask], min=1e-12))
        rel_mean = float(rel.mean().item())
        rel_max = float(rel.max().item())
    else:
        rel = (num / torch.clamp(den, min=1e-12))
        rel_mean = float(rel.mean().item())
        rel_max = float(rel.max().item())

    # Cosine error on masked entries
    dot = (b_vec * expected_vec).sum(dim=-1)
    cos = dot / (
        torch.clamp(torch.linalg.norm(b_vec, dim=-1), min=1e-12)
        * torch.clamp(torch.linalg.norm(expected_vec, dim=-1), min=1e-12)
    )
    cos = torch.clamp(cos, -1.0, 1.0)
    if mask.any():
        cos_err_mean = float((1.0 - cos[mask]).mean().item())
    else:
        cos_err_mean = float((1.0 - cos).mean().item())

    coverage = float(mask.float().mean().item())
    return rel_mean, rel_max, cos_err_mean, coverage


def collect_module_outputs(
    model: torch.nn.Module,
    batch: Data,
    *,
    device: torch.device,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, Any]]:
    """
    Run model(batch) with forward hooks on all submodules and collect tensor outputs.

    Returns:
    - per_module: dict name -> list[tensor outputs] in call order
    - top_out: the model's own output dict (or raw output)
    """
    per_module: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def _register(name: str, mod: torch.nn.Module) -> None:
        def hook(_mod, _inp, out):
            def _add_tensor(key: str, t: torch.Tensor) -> None:
                if key not in per_module:
                    per_module[key] = []
                per_module[key].append(t.detach())

            if isinstance(out, torch.Tensor):
                _add_tensor(name, out)
            elif isinstance(out, (list, tuple)):
                for i, item in enumerate(out):
                    if isinstance(item, torch.Tensor):
                        _add_tensor(f"{name}[{i}]", item)
            elif isinstance(out, dict):
                # Some modules may emit dicts; store any tensor leaves
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        _add_tensor(f"{name}.{k}", v)

        handles.append(mod.register_forward_hook(hook))

    for name, mod in model.named_modules():
        _register(name if name != "" else "<model>", mod)

    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        top_out = model(batch)

    for h in handles:
        h.remove()

    return per_module, top_out


def masked_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Mean squared error over masked rows, averaged over all elements.
    """
    mask = mask.bool()
    if mask.numel() == 0:
        return float("nan")
    if mask.sum().item() == 0:
        return float("nan")
    p = preds[mask]
    t = target[mask]
    return float(torch.mean((p - t) ** 2).item())


def masked_vector_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Mean squared error over masked rows, summing over vector coordinates.

    This matches the "vector" metric used in MultiTargetMSE(mode='vector'):
    sum across the last dimension (coordinates) per node, then mean over nodes.
    """
    mask = mask.bool()
    if mask.numel() == 0:
        return float("nan")
    if mask.sum().item() == 0:
        return float("nan")
    p = preds[mask]
    t = target[mask]
    if p.shape != t.shape:
        raise ValueError("masked_vector_mse expects preds and target with same shape.")
    if p.ndim < 2:
        raise ValueError("masked_vector_mse expects a vector dimension on last axis.")
    return float(((p - t) ** 2).sum(dim=-1).mean().item())


def generate_equivariance_report(
    model: torch.nn.Module,
    data_a: Data,
    data_b: Data,
    rot_mat: torch.Tensor,
    *,
    device: torch.device,
    target_key: str,
    rotation_mse_check: Optional[float] = None,
    tol_rel: float = 1e-2,
    tol_cos: float = 1e-2,
    top_k: int = 30,
    rot_deg: Optional[float] = None,
    rotation_seed: Optional[int] = None,
    extra_lines: Optional[Sequence[str]] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> str:
    """
    Generate a text report for layer-by-layer equivariance checks.
    """
    if not hasattr(data_a, target_key) or not hasattr(data_b, target_key):
        raise RuntimeError(f"Expected data to have target_key '{target_key}'.")
    target_a_frozen = getattr(data_a, target_key).detach().clone()
    target_b_frozen = getattr(data_b, target_key).detach().clone()

    mod_a, out_a = collect_module_outputs(model, data_a, device=device)
    mod_b, out_b = collect_module_outputs(model, data_b, device=device)

    results: List[VectorCheckResult] = []

    if valid_mask is None:
        if not hasattr(data_a, "valid_mask"):
            raise RuntimeError("Expected data to have 'valid_mask' for masked equivariance reporting.")
        valid_mask = getattr(data_a, "valid_mask").detach().cpu()
    valid_mask = valid_mask.detach().cpu()
    N_mask = int(valid_mask.shape[0])
    N_valid = int(valid_mask.bool().sum().item())
    frac_valid = float(valid_mask.float().mean().item())

    common_keys = set(mod_a.keys()).intersection(set(mod_b.keys()))
    for key in sorted(common_keys):
        outs_a = mod_a.get(key)
        outs_b = mod_b.get(key)
        if outs_a is None or outs_b is None:
            continue
        n = min(len(outs_a), len(outs_b))
        for i in range(n):
            ta = outs_a[i]
            tb = outs_b[i]
            if not (isinstance(ta, torch.Tensor) and isinstance(tb, torch.Tensor)):
                continue
            if ta.shape != tb.shape:
                continue
            axis = maybe_get_vector_axis(ta, d=3)
            if axis is None:
                continue
            # Only report tensors where we can apply the node-level valid_mask
            node_axis = None
            for ax, s in enumerate(ta.shape):
                if int(s) == N_mask:
                    node_axis = int(ax)
                    break
            if node_axis is None:
                continue

            rel_mean, rel_max, cos_err_mean, coverage = vector_equivariance_metrics(
                ta.to(device),
                tb.to(device),
                rot_mat.to(device=device, dtype=ta.dtype),
                axis=axis,
                node_mask=valid_mask,
                node_axis=node_axis,
            )
            ok = (rel_mean <= tol_rel) and (cos_err_mean <= tol_cos)
            results.append(
                VectorCheckResult(
                    name=f"{key}#{i}",
                    shape=tuple(int(s) for s in ta.shape),
                    vector_axis=int(axis),
                    rel_mean=float(rel_mean),
                    rel_max=float(rel_max),
                    cos_err_mean=float(cos_err_mean),
                    coverage=float(coverage),
                    ok=bool(ok),
                )
            )

    # Final preds check (explicit)
    preds_a = out_a.get("preds") if isinstance(out_a, dict) else None
    preds_b = out_b.get("preds") if isinstance(out_b, dict) else None
    if isinstance(preds_a, torch.Tensor) and isinstance(preds_b, torch.Tensor):
        pa = preds_a
        pb = preds_b
        # VDWModular node-vector preds are often (N,1,3); squeeze channel dim
        if pa.dim() == 3 and int(pa.shape[1]) == 1:
            pa = pa.squeeze(1)
        if pb.dim() == 3 and int(pb.shape[1]) == 1:
            pb = pb.squeeze(1)
        axis = maybe_get_vector_axis(pa, d=3)
        if axis is not None and pa.shape == pb.shape:
            node_axis = None
            for ax, s in enumerate(pa.shape):
                if int(s) == N_mask:
                    node_axis = int(ax)
                    break
            if node_axis is not None:
                rel_mean, rel_max, cos_err_mean, coverage = vector_equivariance_metrics(
                    pa.to(device),
                    pb.to(device),
                    rot_mat.to(device=device, dtype=pa.dtype),
                    axis=axis,
                    node_mask=valid_mask,
                    node_axis=node_axis,
                )
                ok = (rel_mean <= tol_rel) and (cos_err_mean <= tol_cos)
                results.append(
                    VectorCheckResult(
                        name="__FINAL__.preds",
                        shape=tuple(int(s) for s in pa.shape),
                        vector_axis=int(axis),
                        rel_mean=float(rel_mean),
                        rel_max=float(rel_max),
                        cos_err_mean=float(cos_err_mean),
                        coverage=float(coverage),
                        ok=bool(ok),
                    )
                )

    results_sorted = sorted(results, key=lambda r: (r.rel_mean, r.cos_err_mean), reverse=True)
    top_k = max(1, int(top_k))

    # Final validation MSEs
    raw_mse_a = float("nan")
    raw_mse_b = float("nan")
    raw_vec_mse_a = float("nan")
    raw_vec_mse_b = float("nan")
    norm_elem_mse_a = float("nan")
    norm_elem_mse_b = float("nan")
    norm_vec_mse_a = float("nan")
    norm_vec_mse_b = float("nan")
    norm_enabled = False
    train_mse_a = float("nan")
    train_mse_b = float("nan")
    train_vec_mse_a = float("nan")
    train_vec_mse_b = float("nan")
    preds_shape = None
    target_shape = None
    with torch.no_grad():
        if hasattr(data_a, "batch"):
            data_a_dev = data_a.to(device)
        else:
            data_a_dev = Batch.from_data_list([data_a]).to(device)
        if hasattr(data_b, "batch"):
            data_b_dev = data_b.to(device)
        else:
            data_b_dev = Batch.from_data_list([data_b]).to(device)
        outA = model(data_a_dev)
        outB = model(data_b_dev)
        pA = outA["preds"] if isinstance(outA, dict) else outA
        pB = outB["preds"] if isinstance(outB, dict) else outB
        tA_raw = data_a_dev[target_key]
        tB_raw = data_b_dev[target_key]
        if isinstance(pA, torch.Tensor):
            preds_shape = tuple(int(s) for s in pA.shape)
        if isinstance(tA_raw, torch.Tensor):
            target_shape = tuple(int(s) for s in tA_raw.shape)
        if isinstance(pA, torch.Tensor) and isinstance(tA_raw, torch.Tensor):
            raw_mse_a = masked_mse(pA, tA_raw, data_a_dev.valid_mask.bool())
            if pA.shape == tA_raw.shape:
                raw_vec_mse_a = masked_vector_mse(pA, tA_raw, data_a_dev.valid_mask.bool())
        if isinstance(pB, torch.Tensor) and isinstance(tB_raw, torch.Tensor):
            raw_mse_b = masked_mse(pB, tB_raw, data_a_dev.valid_mask.bool())
            if pB.shape == tB_raw.shape:
                raw_vec_mse_b = masked_vector_mse(pB, tB_raw, data_a_dev.valid_mask.bool())
        norm_enabled = bool(getattr(model, "has_normalized_train_targets", False))
        if norm_enabled and hasattr(model, "get_denormalized"):
            try:
                pA_denorm = model.get_denormalized(pA) if torch.is_tensor(pA) else None
                pB_denorm = model.get_denormalized(pB) if torch.is_tensor(pB) else None
                if torch.is_tensor(pA_denorm):
                    if pA_denorm.dim() == 3 and int(pA_denorm.shape[1]) == 1:
                        pA_denorm = pA_denorm.squeeze(1)
                    if tA_raw.dim() == 3 and int(tA_raw.shape[1]) == 1:
                        tA_cmp = tA_raw.squeeze(1)
                    else:
                        tA_cmp = tA_raw
                    norm_elem_mse_a = masked_mse(pA_denorm, tA_cmp, data_a_dev.valid_mask.bool())
                    if pA_denorm.shape == tA_cmp.shape:
                        norm_vec_mse_a = masked_vector_mse(pA_denorm, tA_cmp, data_a_dev.valid_mask.bool())
                if torch.is_tensor(pB_denorm):
                    if pB_denorm.dim() == 3 and int(pB_denorm.shape[1]) == 1:
                        pB_denorm = pB_denorm.squeeze(1)
                    if tB_raw.dim() == 3 and int(tB_raw.shape[1]) == 1:
                        tB_cmp = tB_raw.squeeze(1)
                    else:
                        tB_cmp = tB_raw
                    norm_elem_mse_b = masked_mse(pB_denorm, tB_cmp, data_a_dev.valid_mask.bool())
                    if pB_denorm.shape == tB_cmp.shape:
                        norm_vec_mse_b = masked_vector_mse(pB_denorm, tB_cmp, data_a_dev.valid_mask.bool())
            except Exception:
                pass
        if hasattr(data_a_dev, "train_mask") and isinstance(pA, torch.Tensor) and isinstance(tA_raw, torch.Tensor):
            train_mse_a = masked_mse(pA, tA_raw, data_a_dev.train_mask.bool())
            if pA.shape == tA_raw.shape:
                train_vec_mse_a = masked_vector_mse(pA, tA_raw, data_a_dev.train_mask.bool())
        if hasattr(data_b_dev, "train_mask") and isinstance(pB, torch.Tensor) and isinstance(tB_raw, torch.Tensor):
            train_mse_b = masked_mse(pB, tB_raw, data_b_dev.train_mask.bool())
            if pB.shape == tB_raw.shape:
                train_vec_mse_b = masked_vector_mse(pB, tB_raw, data_b_dev.train_mask.bool())
        # Normalize shapes to (N, d)
        if isinstance(pA, torch.Tensor) and pA.dim() == 3 and int(pA.shape[1]) == 1:
            pA = pA.squeeze(1)
        if isinstance(pB, torch.Tensor) and pB.dim() == 3 and int(pB.shape[1]) == 1:
            pB = pB.squeeze(1)
        tA = data_a_dev[target_key]
        tB = data_b_dev[target_key]
        frozen_tA = target_a_frozen.to(device)
        frozen_tB = target_b_frozen.to(device)
        if isinstance(tA, torch.Tensor) and tA.dim() == 3 and int(tA.shape[1]) == 1:
            tA = tA.squeeze(1)
        if isinstance(tB, torch.Tensor) and tB.dim() == 3 and int(tB.shape[1]) == 1:
            tB = tB.squeeze(1)
        if isinstance(frozen_tA, torch.Tensor) and frozen_tA.dim() == 3 and int(frozen_tA.shape[1]) == 1:
            frozen_tA = frozen_tA.squeeze(1)
        if isinstance(frozen_tB, torch.Tensor) and frozen_tB.dim() == 3 and int(frozen_tB.shape[1]) == 1:
            frozen_tB = frozen_tB.squeeze(1)
        m = data_a_dev.valid_mask.bool()
        mse_a = masked_mse(pA, tA, m)
        mse_b = masked_mse(pB, tB, m)
        mse_a_frozen = masked_mse(pA, frozen_tA, m)
        mse_b_frozen = masked_mse(pB, frozen_tB, m)
        vec_mse_a = masked_vector_mse(pA, tA, m)
        vec_mse_b = masked_vector_mse(pB, tB, m)
        vec_mse_a_frozen = masked_vector_mse(pA, frozen_tA, m)
        vec_mse_b_frozen = masked_vector_mse(pB, frozen_tB, m)
        ratio = float(mse_b / mse_a) if (mse_a == mse_a and mse_a != 0.0) else float("nan")
        vec_ratio = float(vec_mse_b / vec_mse_a) if (vec_mse_a == vec_mse_a and vec_mse_a != 0.0) else float("nan")

    target_a_after = getattr(data_a, target_key)
    target_b_after = getattr(data_b, target_key)
    target_a_after = target_a_after.to(device=target_a_frozen.device)
    target_b_after = target_b_after.to(device=target_b_frozen.device)
    target_a_mutated = not torch.equal(target_a_after, target_a_frozen)
    target_b_mutated = not torch.equal(target_b_after, target_b_frozen)

    lines: List[str] = ["=== Wind VDWModular equivariance probe ==="]
    if extra_lines:
        lines.extend(list(extra_lines))
    if rot_deg is not None:
        lines.append(f"rotation_deg: {rot_deg:.4f}")
    if rotation_seed is not None:
        lines.append(f"rotation_seed_used: {rotation_seed}")
    lines.append(f"tol_rel: {tol_rel:.3e}  tol_cos: {tol_cos:.3e}")
    lines.append(f"valid_mask_fraction: {frac_valid:.3f}")
    lines.append(f"equivariance_eval_nodes: {N_valid}/{N_mask}")
    lines.append(f"targets_mutated: A={target_a_mutated}  B={target_b_mutated}")
    if preds_shape is not None and target_shape is not None:
        lines.append(f"preds_shape: {preds_shape}  target_shape: {target_shape}")
    lines.append(f"valid_mse_A: {mse_a:.6e}  valid_mse_B: {mse_b:.6e}  ratio(B/A): {ratio:.3f}")
    lines.append(
        "valid_mse_frozen_A: "
        f"{mse_a_frozen:.6e}  valid_mse_frozen_B: {mse_b_frozen:.6e}"
    )
    lines.append(
        "valid_vec_mse_A: "
        f"{vec_mse_a:.6e}  valid_vec_mse_B: {vec_mse_b:.6e}  ratio(B/A): {vec_ratio:.3f}"
    )
    lines.append(
        "valid_vec_mse_frozen_A: "
        f"{vec_mse_a_frozen:.6e}  valid_vec_mse_frozen_B: {vec_mse_b_frozen:.6e}"
    )
    lines.append(
        "raw_pred_mse_A: "
        f"{raw_mse_a:.6e}  raw_pred_mse_B: {raw_mse_b:.6e}"
    )
    lines.append(
        "raw_pred_vec_mse_A: "
        f"{raw_vec_mse_a:.6e}  raw_pred_vec_mse_B: {raw_vec_mse_b:.6e}"
    )
    lines.append(f"has_normalized_train_targets: {norm_enabled}")
    if norm_enabled:
        lines.append(
            "denorm_elem_mse_A: "
            f"{norm_elem_mse_a:.6e}  denorm_elem_mse_B: {norm_elem_mse_b:.6e}"
        )
        lines.append(
            "denorm_vec_mse_A: "
            f"{norm_vec_mse_a:.6e}  denorm_vec_mse_B: {norm_vec_mse_b:.6e}"
        )
    if train_mse_a == train_mse_a or train_mse_b == train_mse_b:
        lines.append(
            "train_mask_mse_A: "
            f"{train_mse_a:.6e}  train_mask_mse_B: {train_mse_b:.6e}"
        )
        lines.append(
            "train_mask_vec_mse_A: "
            f"{train_vec_mse_a:.6e}  train_mask_vec_mse_B: {train_vec_mse_b:.6e}"
        )
    if rotation_mse_check is not None:
        lines.append(f"rotation_mse_check: {rotation_mse_check:.6e}")
    lines.append(f"num_vector_checks: {len(results_sorted)}")

    if len(results_sorted) == 0:
        lines.append("No vector-like intermediate tensors detected (axis size 3).")
        return "\n".join(lines)

    lines.append("")
    lines.append("Top offenders (largest mean relative error):")
    for r in results_sorted[:top_k]:
        status = "OK" if r.ok else "FAIL"
        lines.append(
            f"- {status}  {r.name}  axis={r.vector_axis}  "
            f"rel_mean={r.rel_mean:.3e}  rel_max={r.rel_max:.3e}  "
            f"cos_err_mean={r.cos_err_mean:.3e}  cov={r.coverage:.3f}"
        )

    worst = results_sorted[0]
    lines.append("")
    lines.append("Worst offender summary:")
    lines.append(
        f"{worst.name}  rel_mean={worst.rel_mean:.3e}  rel_max={worst.rel_max:.3e}  "
        f"cos_err_mean={worst.cos_err_mean:.3e}  cov={worst.coverage:.3f}"
    )

    n_fail = sum(1 for r in results_sorted if not r.ok)
    lines.append("")
    lines.append(f"Failures above tolerance: {n_fail}/{len(results_sorted)}")

    return "\n".join(lines)
