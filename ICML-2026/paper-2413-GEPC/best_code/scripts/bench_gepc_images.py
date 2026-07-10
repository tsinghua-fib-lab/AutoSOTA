# scripts/bench_gepc_images.py
import argparse
import json
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from gepc.adapters.improved import ImprovedDiffusionAdapter
from gepc.datasets.images import load_data
from gepc.methods.gepc import GEPC
from gepc.utils.metrics import auroc_ood_high


# ----------------------------- reproducibility -----------------------------

def set_global_determinism(seed: int, deterministic: bool = True) -> None:
    """Set as much determinism as reasonably possible (CUDA + CPU)."""
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Avoid TF32 (can change numerics)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Ensure dataloader workers are deterministically seeded."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def rebuild_loader_with_generator(
    loader: DataLoader,
    shuffle: bool,
    seed: int,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Rebuild a DataLoader with an explicit torch.Generator for deterministic shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers if num_workers is None else num_workers,
        pin_memory=getattr(loader, "pin_memory", True),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=gen,
    )


def clamp_loader(loader: DataLoader, limit: Optional[int]) -> DataLoader:
    """Clamp dataset to the first N items for ultra-stable subsets (no randomness)."""
    if not limit or int(limit) <= 0:
        return loader
    N = min(int(limit), len(loader.dataset))
    idx = np.arange(N)
    return DataLoader(
        Subset(loader.dataset, idx),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,  # maximally stable
        pin_memory=getattr(loader, "pin_memory", True),
        drop_last=False,
    )


def _normalize_ood_cfg(ood_cfg):
    """Accept dict / list / string for YAML eval.ood and normalize into a list of dict."""
    if ood_cfg is None:
        return []
    if isinstance(ood_cfg, dict):
        return [ood_cfg]
    if isinstance(ood_cfg, str):
        return [{"name": ood_cfg, "split": "test", "limit": None, "download": True}]
    out = []
    for it in (ood_cfg if isinstance(ood_cfg, (list, tuple)) else [ood_cfg]):
        if isinstance(it, str):
            out.append({"name": it, "split": "test", "limit": None, "download": True})
        else:
            out.append(it)
    return out


# ----------------------------- main -----------------------------

def main() -> None:
    # Pre-parse config path (so --config is mandatory and clean)
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", required=True)
    args0, remaining = p0.parse_known_args()

    cfg = yaml.safe_load(open(args0.config, "r")) or {}

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=args0.config)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--in_dist", default=None)     # override YAML eval.id_*
    p.add_argument("--out_dist", default=None)    # if set, only evaluate this OOD
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--strict_determinism", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(remaining)

    # Resolve seed / determinism
    method_cfg = (cfg.get("gepc") or cfg.get("slidpc") or {})  # backward-compatible
    seed = int(args.seed) if args.seed is not None else int(method_cfg.get("seed", cfg.get("seed", 1337)))
    strict_det = bool(args.strict_determinism) or bool(cfg.get("strict_determinism", False))
    set_global_determinism(seed, deterministic=strict_det)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Backbone config
    backbone_size = int(cfg.get("image_size", 32))
    data_size = int(cfg.get("data_image_size", backbone_size))
    model_path = cfg.get("model_path", None)
    if not model_path:
        raise ValueError("model_path must be set in YAML")

    device = int(args.device) if args.device is not None else int(cfg.get("device", 0))
    batch_size = int(cfg.get("batch_size", 128))
    data_dir = args.data_dir or cfg.get("data_root", "./data")
    improved_args = cfg.get("improved_args", {}) or {}

    # Eval config
    ev = cfg.get("eval", {}) or {}
    idtr_cfg = (ev.get("id_train", {}) or {})
    idte_cfg = (ev.get("id_test", {}) or {})
    ood_list = _normalize_ood_cfg(ev.get("ood", []))

    # Override ID dataset
    if args.in_dist is not None:
        idtr_cfg = dict(idtr_cfg)
        idte_cfg = dict(idte_cfg)
        idtr_cfg["name"] = args.in_dist
        idte_cfg["name"] = args.in_dist

    id_name = idtr_cfg.get("name", None)
    if not id_name:
        raise ValueError("eval.id_train.name must be set (or --in_dist)")

    # Optional filter OODs
    if args.out_dist is not None:
        ood_list = [o for o in ood_list if o.get("name") == args.out_dist]
        if not ood_list:
            raise ValueError(f"--out_dist={args.out_dist} not found in eval.ood list")

    # Adapter args-like object
    class _Args:
        pass

    a = _Args()
    a.model_path = model_path
    a.device = device
    a.image_size = backbone_size
    a.data_image_size = data_size
    a.n_ddim_steps = int(cfg.get("n_ddim_steps", 10))
    a.improved_args = improved_args

    adapter = ImprovedDiffusionAdapter(a)

    # Load ID loaders once
    id_train = load_data(
        name=id_name,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=data_size,
        split=idtr_cfg.get("split", "train"),
        limit=idtr_cfg.get("limit", None),
        download=idtr_cfg.get("download", True),
        shuffle=True,
        model_image_size=backbone_size,
    )
    id_test = load_data(
        name=id_name,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=data_size,
        split=idte_cfg.get("split", "test"),
        limit=idte_cfg.get("limit", None),
        download=idte_cfg.get("download", True),
        shuffle=False,
        model_image_size=backbone_size,
    )

    # Clamp + deterministic rebuild (kept to preserve exact historical AUC behavior)
    id_train = clamp_loader(id_train, idtr_cfg.get("limit", None))
    id_test = clamp_loader(id_test, idte_cfg.get("limit", None))
    id_train = rebuild_loader_with_generator(id_train, shuffle=True, seed=seed)
    id_test = rebuild_loader_with_generator(id_test, shuffle=False, seed=seed)

    # Build method (GEPC)
    method_cfg = dict(method_cfg)
    method_cfg.setdefault("seed", seed)
    m = GEPC(**method_cfg)

    # Fit on ID train
    t0 = time.time()
    m.fit_id_train(adapter, id_train)
    fit_sec = time.time() - t0

    # Score ID test once
    t0 = time.time()
    sid = m.score_loader(adapter, id_test, tag=f"ID_{id_name}")
    id_score_sec = time.time() - t0

    # Output dir
    out_root = os.path.join("results", "gepc", id_name)
    os.makedirs(out_root, exist_ok=True)

    with open(os.path.join(out_root, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    results = {
        "method": m.name,
        "id": id_name,
        "seed": int(seed),
        "strict_determinism": bool(strict_det),
        "backbone": {
            "model_path": model_path,
            "image_size": backbone_size,
            "data_image_size": data_size,
        },
        "method_hparams": method_cfg,
        "timing": {
            "fit_sec": float(fit_sec),
            "id_score_sec": float(id_score_sec),
            "id_n": int(getattr(sid, "size", 0)),
            "id_ms_per_img": float(1000.0 * id_score_sec / max(1, getattr(sid, "size", 0))),
        },
        "pairs": [],
    }

    # Loop OODs
    for ood_cfg in ood_list:
        ood_name = ood_cfg.get("name")
        od_test = load_data(
            name=ood_name,
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=data_size,
            split=ood_cfg.get("split", "test"),
            limit=ood_cfg.get("limit", None),
            download=ood_cfg.get("download", True),
            shuffle=False,
            model_image_size=backbone_size,
        )
        od_test = clamp_loader(od_test, ood_cfg.get("limit", None))
        od_test = rebuild_loader_with_generator(od_test, shuffle=False, seed=seed)

        t0 = time.time()
        sod = m.score_loader(adapter, od_test, tag=f"OOD_{ood_name}")
        ood_score_sec = time.time() - t0

        auroc = auroc_ood_high(sid, sod)
        print(f"[{m.name}] AUROC {id_name} vs {ood_name} = {auroc:.4f}")

        results["pairs"].append({
            "ood": ood_name,
            "auroc": float(auroc),
            "timing": {
                "ood_score_sec": float(ood_score_sec),
                "ood_n": int(getattr(sod, "size", 0)),
                "ood_ms_per_img": float(1000.0 * ood_score_sec / max(1, getattr(sod, "size", 0))),
            }
        })

    with open(os.path.join(out_root, "main_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    flat = [{"id": id_name, "ood": p["ood"], "auroc": p["auroc"], "seed": int(seed)} for p in results["pairs"]]
    with open(os.path.join(out_root, "main_results_flat.json"), "w") as f:
        json.dump(flat, f, indent=2)


if __name__ == "__main__":
    main()
