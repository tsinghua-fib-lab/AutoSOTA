# scripts/bench_gepc_sar.py
# -*- coding: utf-8 -*-
"""
Benchmark GEPC on SAR ImageFolder datasets (user-provided).

- Fits calibration on ID-train (ID-only).
- Reports AUROC (OOD-high) for each OOD set (also AUPR and FPR@95TPR).
- Optionally exports qualitative examples (raw / heatmap / overlay + .npy maps).

Expected SAR folder format (torchvision ImageFolder):
  <split_root>/0/*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ---------------------------- robust imports (repo layout changes) ----------------------------
try:
    # preferred (matches your bench_gepc_images.py)
    from gepc.adapters.improved import ImprovedDiffusionAdapter
except Exception:  # pragma: no cover
    # fallback if re-exported in __init__.py
    from gepc.adapters import ImprovedDiffusionAdapter  # type: ignore


try:
    from gepc.methods.gepc import GEPC
except Exception as e:  # pragma: no cover
    raise ImportError("Could not import GEPC from gepc.methods.gepc. Check your repo layout.") from e


# Metrics: prefer repo utilities if present, else fallback to local sklearn versions
try:
    from gepc.utils.metrics import auroc_ood_high as _auroc_ood_high  # type: ignore
except Exception:  # pragma: no cover
    _auroc_ood_high = None


# ---------------------------- plotting defaults ----------------------------
PAPER_RC = {
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 16,
}
plt.rcParams.update(PAPER_RC)


# ---------------------------- determinism helpers ----------------------------
def set_global_determinism(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------- small utils ----------------------------
def to_numpy1d(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1).astype(np.float32)
    return x


def _sanitize_tag(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s if s else "ood"


# ---------------------------- data helpers ----------------------------
def _ensure_chw3(x: torch.Tensor) -> torch.Tensor:
    # x: [C,H,W] in [0,1]
    if x.dim() != 3:
        return x
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


def make_sar_loader(
    root: str,
    *,
    batch_size: int,
    image_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    """
    Loads SAR chips from an ImageFolder root.

    NOTE: we keep tensors in [0,1]. GEPC converts to [-1,1] internally (via to_minus1_1).
    """
    tfm = transforms.Compose([
        transforms.Resize((int(image_size), int(image_size))),
        transforms.CenterCrop(int(image_size)),
        transforms.ToTensor(),
        transforms.Lambda(_ensure_chw3),
    ])

    ds = ImageFolder(root=str(root), transform=tfm)

    gen = torch.Generator()
    gen.manual_seed(int(seed))

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=gen,
    )


def clamp_loader(loader: DataLoader, limit: Optional[int]) -> DataLoader:
    if limit is None or int(limit) <= 0:
        return loader
    N = min(int(limit), len(loader.dataset))
    subset = Subset(loader.dataset, np.arange(N))
    return DataLoader(
        subset,
        batch_size=int(loader.batch_size),
        shuffle=False,
        num_workers=0,  # strongest determinism
        pin_memory=True,
        drop_last=False,
    )


def dataloader_info(loader: DataLoader) -> Tuple[int, int, int]:
    n_items = len(loader.dataset)
    n_batches = len(loader)
    bsz = int(getattr(loader, "batch_size", 1))
    return n_items, n_batches, bsz


def collect_inputs_in_order(loader: DataLoader) -> np.ndarray:
    xs: List[np.ndarray] = []
    for x, _ in loader:
        xs.append(x.detach().cpu().numpy())
    if not xs:
        return np.zeros((0, 3, 1, 1), dtype=np.float32)
    return np.concatenate(xs, axis=0)


# ---------------------------- metrics (OOD-high) ----------------------------
def auroc_ood_high(s_id: np.ndarray, s_ood: np.ndarray) -> float:
    if _auroc_ood_high is not None:
        return float(_auroc_ood_high(s_id, s_ood))
    y = np.concatenate([np.zeros_like(s_id, dtype=np.int32), np.ones_like(s_ood, dtype=np.int32)])
    s = np.concatenate([s_id, s_ood], axis=0)
    return float(roc_auc_score(y, s))


def aupr_ood_high(s_id: np.ndarray, s_ood: np.ndarray) -> float:
    y = np.concatenate([np.zeros_like(s_id, dtype=np.int32), np.ones_like(s_ood, dtype=np.int32)])
    s = np.concatenate([s_id, s_ood], axis=0)
    return float(average_precision_score(y, s))


def fpr_at_tpr(s_id: np.ndarray, s_ood: np.ndarray, target_tpr: float = 0.95) -> float:
    y = np.concatenate([np.zeros_like(s_id, dtype=np.int32), np.ones_like(s_ood, dtype=np.int32)])
    s = np.concatenate([s_id, s_ood], axis=0)
    fpr, tpr, _ = roc_curve(y, s)
    mask = tpr >= float(target_tpr)
    if not np.any(mask):
        return 1.0
    return float(np.min(fpr[mask]))


# ---------------------------- qualitative export ----------------------------
def to_display01(x_chw: np.ndarray) -> np.ndarray:
    # x in [0,1] or [-1,1] -> grayscale [0,1]
    if x_chw.ndim == 3:
        img = x_chw.mean(axis=0)
    else:
        img = x_chw
    if img.min() < -1e-6:
        img = 0.5 * (img + 1.0)
    img01 = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img01.astype(np.float32)


def save_triplet(
    raw01_hw: np.ndarray,
    map_hw: np.ndarray,
    out_prefix: Path,
    *,
    cmap: str = "magma",
    add_colorbar: bool = True,
    save_raw_map: bool = True,
    global_v: Optional[float] = None,
) -> float:
    """
    Saves:
      - *_raw.png
      - *_gepc.png
      - *_overlay.png
      - optionally *_gepc_cb.png, *_overlay_cb.png
      - optionally *_map.npy
    Returns the scale used for normalization (v_used).
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # raw
    plt.figure(figsize=(3.2, 3.2))
    plt.imshow(raw01_hw, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_prefix) + "_raw.png", dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close()

    # normalize map magnitude
    a = np.abs(map_hw).astype(np.float32)
    v_img = float(np.quantile(a, 0.99)) + 1e-8
    v_used = float(global_v) if (global_v is not None and global_v > 0) else v_img
    hm = np.clip(a / (v_used + 1e-8), 0.0, 1.0)

    # heatmap
    plt.figure(figsize=(3.2, 3.2))
    plt.imshow(hm, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_prefix) + "_gepc.png", dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close()

    # overlay
    plt.figure(figsize=(3.2, 3.2))
    plt.imshow(raw01_hw, cmap="gray")
    plt.imshow(hm, cmap=cmap, alpha=0.55, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_prefix) + "_overlay.png", dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close()

    if add_colorbar:
        fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
        im = ax.imshow(hm, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=12)
        fig.savefig(str(out_prefix) + "_gepc_cb.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
        ax.imshow(raw01_hw, cmap="gray")
        im = ax.imshow(hm, cmap=cmap, alpha=0.55, vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=12)
        fig.savefig(str(out_prefix) + "_overlay_cb.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    if save_raw_map:
        np.save(str(out_prefix) + "_map.npy", map_hw.astype(np.float32))

    return v_used


def compute_v_global_from_id_pool(id_maps: List[np.ndarray], q: float = 0.99) -> Optional[float]:
    if not id_maps:
        return None
    vals = [float(np.quantile(np.abs(m), q)) for m in id_maps]
    v = float(np.median(vals))
    return v if v > 0 else None


def _get_timesteps_for_maps(m: Any, adapter: Any, *, avg_over_t: bool, t_debug: Optional[int]) -> List[int]:
    """
    Best-effort timestep selection:
    - If avg_over_t: use kept timesteps if available, else all final timesteps.
    - Else: use t_debug if provided, else the middle timestep from final list.
    """
    if hasattr(m, "_build_t_list"):
        try:
            m._build_t_list(adapter)
        except Exception:
            pass

    t_final = getattr(m, "_t_final", None) or getattr(m, "t_list", None)
    if t_final is None or len(t_final) == 0:
        return [int(t_debug) if t_debug is not None else 0]

    t_final = [int(t) for t in t_final]
    t_kept = getattr(m, "_t_kept", None)

    if avg_over_t:
        if isinstance(t_kept, (set, list, tuple)) and len(t_kept) > 0:
            return [t for t in t_final if t in set(t_kept)]
        return t_final

    if t_debug is not None:
        return [int(t_debug)]
    return [t_final[len(t_final) // 2]]


def _import_gepc_internals_for_maps():
    """
    Import GEPC internal helpers used to compute invariance maps.
    We keep them in GEPC (single source of truth) and only rely on them here.
    """
    try:
        from gepc.methods.gepc import (  # type: ignore
            to_minus1_1,
            _build_group_ops,
            _forward_noisy,
            _pred_raw,
            _score_from_eps,
            _split_eps_var,
            _pool_spatial,
        )
        return to_minus1_1, _build_group_ops, _forward_noisy, _pred_raw, _score_from_eps, _split_eps_var, _pool_spatial
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Qualitative export requires internal helpers from gepc.methods.gepc:\n"
            "  to_minus1_1, _build_group_ops, _forward_noisy, _pred_raw, _score_from_eps, _split_eps_var, _pool_spatial\n"
            "They were not found / not importable. Either expose a public map API on GEPC or keep these helpers."
        ) from e


def _compute_gepc_map_single(
    m: Any,
    adapter: Any,
    x0_bchw: torch.Tensor,
    *,
    avg_over_t: bool,
    t_debug: Optional[int],
) -> np.ndarray:
    """
    Computes a per-sample invariance map [H,W] for a single image (batch=1).

    Priority:
      1) if GEPC exposes a public method hook, use it
      2) else fallback to GEPC internal helpers (imported with try/except)
    """
    # (1) public hook if you add it in GEPC later
    for hook_name in ("invariance_map_single", "compute_invariance_map_single", "collect_invariance_map_single"):
        if hasattr(m, hook_name):
            fn = getattr(m, hook_name)
            try:
                out = fn(adapter, x0_bchw, avg_over_t=avg_over_t, t_debug=t_debug)
                out = np.asarray(out).astype(np.float32)
                if out.ndim == 2:
                    return out
            except Exception:
                pass  # fallback below

    # (2) fallback via GEPC internals
    (
        to_minus1_1,
        _build_group_ops,
        _forward_noisy,
        _pred_raw,
        _score_from_eps,
        _split_eps_var,
        _pool_spatial,
    ) = _import_gepc_internals_for_maps()

    dev = next(adapter.model.parameters()).device
    x0 = to_minus1_1(x0_bchw.to(dev)).float()  # [1,C,H,W]

    tau = float(getattr(m, "tau", 1e-8))
    amp = str(getattr(m, "amp", "fp16"))
    clamp_x = bool(getattr(m, "clamp_x", False))

    group_shifts = bool(getattr(m, "group_shifts", False))
    shift_px = int(getattr(m, "shift_px", 1))
    spatial_pool = str(getattr(m, "spatial_pool", "mean"))
    topk_rho = float(getattr(m, "topk_rho", 0.10))

    # generator (if exists)
    gen = getattr(m, "_gen", None)
    if gen is None and hasattr(m, "_ensure_generator"):
        try:
            m._ensure_generator(dev)
            gen = getattr(m, "_gen", None)
        except Exception:
            gen = None

    t_used = _get_timesteps_for_maps(m, adapter, avg_over_t=avg_over_t, t_debug=t_debug)

    maps_acc = []
    for t_idx in t_used:
        xt = _forward_noisy(adapter, x0, int(t_idx), gen=gen).float()
        if clamp_x:
            xt = xt.clamp(-1, 1)

        B, C, H, W = xt.shape
        Gops = _build_group_ops(H, W, use_shifts=group_shifts, s=shift_px)

        y0 = _pred_raw(adapter, xt, int(t_idx), amp=amp)
        eps0, _ = _split_eps_var(y0, C)
        s0 = _score_from_eps(adapter, eps0, int(t_idx))  # [1,C,H,W]

        base_scalar = _pool_spatial(s0.square(), spatial_pool, topk_rho)  # [1]
        denom = base_scalar.view(B, 1, 1) + tau

        xs, invs = [], []
        for (g, ginv) in Gops:
            xs.append(g(xt))
            invs.append(ginv)
        xg = torch.cat(xs, dim=0)

        yg = _pred_raw(adapter, xg, int(t_idx), amp=amp)
        epsg, _ = _split_eps_var(yg, C)
        sg = _score_from_eps(adapter, epsg, int(t_idx))  # [nG*B,C,H,W]

        sg_back = []
        for gi, ginv in enumerate(invs):
            sg_i = sg[gi * B:(gi + 1) * B]
            sg_back.append(ginv(sg_i))
        sg_back = torch.stack(sg_back, 0)  # [nG,B,C,H,W]

        diff = (sg_back - s0.unsqueeze(0)).square().mean(dim=2)  # [nG,B,H,W]
        diff = diff / denom.view(1, B, 1, 1)
        m_bhw = diff.mean(dim=0)  # [B,H,W]
        maps_acc.append(m_bhw)

    m_bhw = torch.stack(maps_acc, 0).mean(dim=0) if len(maps_acc) else torch.zeros((1, 1, 1), device=dev)
    return m_bhw[0].detach().cpu().numpy().astype(np.float32)


def export_qualitative(
    *,
    adapter: Any,
    m: Any,
    id_loader: DataLoader,
    od_loader: DataLoader,
    sid: np.ndarray,
    sod: np.ndarray,
    out_dir: Path,
    ood_tag: str,
    qual_k_id: int,
    qual_k_ood: int,
    map_avg_over_t: bool,
    map_t_debug: Optional[int],
) -> None:
    """
    Exports K ID (lowest score) and K OOD (highest score) examples.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    xid = collect_inputs_in_order(id_loader)  # [N,3,H,W]
    xod = collect_inputs_in_order(od_loader)  # [M,3,H,W]

    raw_id = [to_display01(x) for x in xid]
    raw_od = [to_display01(x) for x in xod]

    id_idx = np.argsort(sid)[: max(1, int(qual_k_id))] if sid.size > 0 else np.array([], dtype=int)
    od_idx = np.argsort(-sod)[: max(1, int(qual_k_ood))] if sod.size > 0 else np.array([], dtype=int)

    id_maps_pool: List[np.ndarray] = []
    for idx in id_idx.tolist():
        x0 = torch.from_numpy(xid[int(idx)]).unsqueeze(0)
        mp = _compute_gepc_map_single(m, adapter, x0, avg_over_t=map_avg_over_t, t_debug=map_t_debug)
        id_maps_pool.append(mp)

    v_global = compute_v_global_from_id_pool(id_maps_pool, q=0.99)

    meta: Dict[str, Any] = {
        "ood_tag": str(ood_tag),
        "map_avg_over_t": bool(map_avg_over_t),
        "map_t_debug": None if map_avg_over_t else (None if map_t_debug is None else int(map_t_debug)),
        "v_global": None if v_global is None else float(v_global),
        "v_global_def": "median over selected-ID q99(|map|)",
        "id": [],
        "ood": [],
    }

    # OOD
    for j, idx in enumerate(od_idx.tolist()):
        x0 = torch.from_numpy(xod[int(idx)]).unsqueeze(0)
        mp = _compute_gepc_map_single(m, adapter, x0, avg_over_t=map_avg_over_t, t_debug=map_t_debug)

        prefix = out_dir / f"ood_{j:02d}"
        v_img = save_triplet(raw_od[int(idx)], mp, prefix, add_colorbar=True, save_raw_map=True, global_v=None)
        if v_global is not None:
            save_triplet(
                raw_od[int(idx)], mp, Path(str(prefix) + "_global"),
                add_colorbar=True, save_raw_map=False, global_v=v_global
            )
        meta["ood"].append({"index": int(idx), "score": float(sod[int(idx)]), "v_used_img": float(v_img)})

    # ID
    for j, idx in enumerate(id_idx.tolist()):
        mp = id_maps_pool[j]
        prefix = out_dir / f"id_{j:02d}"
        v_img = save_triplet(raw_id[int(idx)], mp, prefix, add_colorbar=True, save_raw_map=True, global_v=None)
        if v_global is not None:
            save_triplet(
                raw_id[int(idx)], mp, Path(str(prefix) + "_global"),
                add_colorbar=True, save_raw_map=False, global_v=v_global
            )
        meta["id"].append({"index": int(idx), "score": float(sid[int(idx)]), "v_used_img": float(v_img)})

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------- config helpers ----------------------------
def _normalize_ood_cfg(ood_cfg: Any) -> List[Dict[str, Any]]:
    if ood_cfg is None:
        return []
    if isinstance(ood_cfg, dict):
        return [dict(ood_cfg)]
    if isinstance(ood_cfg, str):
        return [{"name": ood_cfg, "root": ood_cfg, "limit": None}]
    out: List[Dict[str, Any]] = []
    for it in (ood_cfg if isinstance(ood_cfg, (list, tuple)) else [ood_cfg]):
        out.append({"name": it, "root": it, "limit": None} if isinstance(it, str) else dict(it))
    return out


def yaml_safe_load(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return dict(cfg)


# ---------------------------- optional: save scores ----------------------------
def save_scores_npz(
    out_base: Path,
    tag: str,
    sid: np.ndarray,
    sod: np.ndarray,
    auroc_val: float,
    aupr_val: float,
    fpr95_val: float,
) -> Path:
    tag = _sanitize_tag(tag)
    if out_base.suffix.lower() == ".npz":
        out_path = out_base.with_name(out_base.stem + f"_{tag}.npz")
    else:
        out_base.mkdir(parents=True, exist_ok=True)
        out_path = out_base / f"scores_{tag}.npz"

    y = np.concatenate([np.zeros_like(sid, dtype=np.int32), np.ones_like(sod, dtype=np.int32)])
    s = np.concatenate([sid, sod], axis=0)
    fpr, tpr, thr = roc_curve(y, s)

    np.savez(
        out_path,
        sid=sid,
        sod=sod,
        y_true=y,
        scores=s,
        auroc=float(auroc_val),
        aupr=float(aupr_val),
        fpr95=float(fpr95_val),
        roc_fpr=fpr,
        roc_tpr=tpr,
        roc_thr=thr,
    )
    return out_path


# ---------------------------- main ----------------------------
def main() -> None:
    # --- pre-parse config ---
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", required=True)
    args0, remaining = p0.parse_known_args()

    cfg = yaml_safe_load(args0.config)

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=args0.config)

    # basic overrides
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--strict_determinism", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # outputs
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (defaults next to config).")
    p.add_argument("--save_metrics_json", type=str, default=None, help="Optional explicit metrics.json path.")
    p.add_argument("--save_scores_npz", type=str, default=None, help="Optional directory or .npz base path.")

    # qualitative
    p.add_argument("--qual_dir", type=str, default=None, help="Directory to export qualitative examples.")
    p.add_argument("--qual_k_id", type=int, default=1)
    p.add_argument("--qual_k_ood", type=int, default=1)
    p.add_argument("--map_avg_over_t", action="store_true")
    p.add_argument("--map_t_debug", type=int, default=None)

    args = p.parse_args(remaining)

    # YAML defaults
    args.device = int(args.device) if args.device is not None else int(cfg.get("device", 0))
    args.seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 1337))
    if not args.strict_determinism:
        args.strict_determinism = bool(cfg.get("strict_determinism", False))

    set_global_determinism(args.seed, deterministic=bool(args.strict_determinism))

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # --- resolve output root ---
    cfg_path = Path(args0.config)
    run_tag = cfg_path.stem
    default_out_dir = cfg_path.parent / "results_gepc_sar" / run_tag
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # snapshot config
    try:
        import yaml
        with open(out_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:
        pass

    # --- sizes / adapter ---
    image_size = int(cfg.get("image_size", 256))
    data_image_size = int(cfg.get("data_image_size", image_size))

    model_path = cfg.get("model_path", None)
    if not model_path:
        raise ValueError("model_path must be set in the YAML config.")

    adapter_type = str(cfg.get("adapter", "improved")).lower()
    if adapter_type != "improved":
        raise ValueError("bench_gepc_sar.py supports adapter: improved only.")

    # minimal args object for the adapter
    class _Args:
        pass

    a = _Args()
    a.model_path = model_path
    a.device = args.device
    a.image_size = image_size
    a.data_image_size = data_image_size
    a.n_ddim_steps = int(cfg.get("n_ddim_steps", 10))
    a.improved_args = (cfg.get("improved_args", {}) or {})

    t0 = time.time()
    adapter = ImprovedDiffusionAdapter(a)
    logging.info(f"Adapter ready in {time.time() - t0:.2f}s")

    # --- datasets ---
    ev = cfg.get("eval", {}) or {}
    id_tr = ev.get("id_train", {}) or {}
    id_te = ev.get("id_test", {}) or {}
    ood_cfgs = _normalize_ood_cfg(ev.get("ood", []))
    if len(ood_cfgs) == 0:
        raise ValueError("eval.ood is empty: provide a dict or a list of dicts.")

    if "root" not in id_tr or "root" not in id_te:
        raise ValueError("eval.id_train.root and eval.id_test.root must be set (ImageFolder roots).")

    batch_size = int(cfg.get("batch_size", 1))
    num_workers = int(cfg.get("num_workers", 0))

    id_train = make_sar_loader(
        id_tr["root"],
        batch_size=batch_size,
        image_size=data_image_size,
        shuffle=True,
        seed=args.seed,
        num_workers=num_workers,
    )
    id_test = make_sar_loader(
        id_te["root"],
        batch_size=batch_size,
        image_size=data_image_size,
        shuffle=False,
        seed=args.seed,
        num_workers=num_workers,
    )

    id_train = clamp_loader(id_train, id_tr.get("limit", None))
    id_test = clamp_loader(id_test, id_te.get("limit", None))

    ntr, btr, bsz = dataloader_info(id_train)
    nte, bte, _ = dataloader_info(id_test)
    logging.info(f"ID-train: {ntr} items | {btr} batches | batch_size={bsz}")
    logging.info(f"ID-test : {nte} items | {bte} batches")

    # --- method (GEPC) ---
    method_cfg = (cfg.get("gepc") or cfg.get("slidpc") or {})
    method_cfg = dict(method_cfg)
    method_cfg.setdefault("seed", args.seed)
    method_cfg.setdefault("verbose", bool(args.verbose))
    m = GEPC(**method_cfg)

    logging.info(f"Fitting {getattr(m, 'name', 'GEPC')} on ID-train...")
    t1 = time.time()
    m.fit_id_train(adapter, id_train)
    fit_sec = time.time() - t1
    logging.info(f"Fit done in {fit_sec:.2f}s")

    logging.info("Scoring ID-test...")
    t2 = time.time()
    sid = to_numpy1d(m.score_loader(adapter, id_test, tag="ID"))
    id_sec = time.time() - t2
    logging.info(f"ID scoring done in {id_sec:.2f}s")

    results: Dict[str, Any] = {
        "method": getattr(m, "name", "GEPC"),
        "adapter": adapter_type,
        "seed": int(args.seed),
        "strict_determinism": bool(args.strict_determinism),
        "backbone": {
            "model_path": str(model_path),
            "image_size": int(image_size),
            "data_image_size": int(data_image_size),
        },
        "id": {
            "train_root": str(id_tr.get("root", "")),
            "test_root": str(id_te.get("root", "")),
            "n_id_test": int(sid.size),
        },
        "timing": {
            "fit_sec": float(fit_sec),
            "id_score_sec": float(id_sec),
        },
        "oods": [],
    }

    # --- loop OODs ---
    for ocfg in ood_cfgs:
        if "root" not in ocfg:
            raise ValueError(f"Each eval.ood entry must contain 'root'. Got: {ocfg}")

        od_name = ocfg.get("name", None) or Path(str(ocfg.get("root", "ood"))).name
        od_tag = _sanitize_tag(od_name)
        od_root = ocfg["root"]
        od_limit = ocfg.get("limit", None)

        od_test = make_sar_loader(
            od_root,
            batch_size=batch_size,
            image_size=data_image_size,
            shuffle=False,
            seed=args.seed,
            num_workers=num_workers,
        )
        od_test = clamp_loader(od_test, od_limit)

        nod, bod, _ = dataloader_info(od_test)
        logging.info(f"OOD-test[{od_tag}]: {nod} items | {bod} batches")

        logging.info(f"Scoring OOD-test [{od_tag}]...")
        t3 = time.time()
        sod = to_numpy1d(m.score_loader(adapter, od_test, tag=f"OOD:{od_tag}"))
        ood_sec = time.time() - t3
        logging.info(f"OOD scoring [{od_tag}] done in {ood_sec:.2f}s")

        auroc_val = auroc_ood_high(sid, sod)
        aupr_val = aupr_ood_high(sid, sod)
        fpr95_val = fpr_at_tpr(sid, sod, target_tpr=0.95)

        print(f"\n[{getattr(m, 'name', 'GEPC')}] ID vs OOD[{od_tag}]")
        print(f"  AUROC={auroc_val:.4f} | AUPR={aupr_val:.4f} | FPR@95TPR={fpr95_val:.4f}")

        entry = {
            "tag": od_tag,
            "name": str(od_name),
            "root": str(od_root),
            "limit": None if od_limit is None else int(od_limit),
            "n": int(len(od_test.dataset)),
            "auroc": float(auroc_val),
            "aupr": float(aupr_val),
            "fpr95": float(fpr95_val),
            "timing": {
                "ood_score_sec": float(ood_sec),
                "ood_ms_per_img": float(1000.0 * ood_sec / max(1, int(sod.size))),
            },
        }
        results["oods"].append(entry)

        # Save per-OOD scores (optional)
        if args.save_scores_npz:
            save_scores_npz(Path(args.save_scores_npz), od_tag, sid, sod, auroc_val, aupr_val, fpr95_val)

        # Qualitative export (optional)
        qdir = Path(args.qual_dir) if args.qual_dir else None
        if qdir is not None:
            export_qualitative(
                adapter=adapter,
                m=m,
                id_loader=id_test,
                od_loader=od_test,
                sid=sid,
                sod=sod,
                out_dir=qdir / od_tag,
                ood_tag=od_tag,
                qual_k_id=int(args.qual_k_id),
                qual_k_ood=int(args.qual_k_ood),
                map_avg_over_t=bool(args.map_avg_over_t),
                map_t_debug=(None if args.map_avg_over_t else args.map_t_debug),
            )
            print(f"[INFO] Qualitative examples saved to: {qdir / od_tag}")

    # Save metrics JSON (default inside out_dir)
    out_json = Path(args.save_metrics_json) if args.save_metrics_json else (out_dir / "metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
