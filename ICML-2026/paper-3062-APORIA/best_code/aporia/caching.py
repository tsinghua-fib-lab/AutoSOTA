"""
Cache helpers shared by the experiment runners.

All caches follow the same layout:

    <cache_dir>/
        <key>.parquet          # one per (model, prompt), or whatever key
        meta.json              # global run metadata

The structural-analysis cache is a special case (whole-study dump):

    <cache_dir>/
        results_df-<lambda>.parquet
        geometry_store-<lambda>.pkl
        null_store-<lambda>.pkl
        meta-<lambda>.json
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any


# ============================================================
# ========================= GENERIC ==========================
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ============================================================
# ================ STRUCTURAL-ANALYSIS CACHE =================
# ============================================================

def _structural_paths(cache_dir: str | Path, lambda_reg: float) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {
        "results": cache_dir / f"results_df-{lambda_reg}.parquet",
        "geometry": cache_dir / f"geometry_store-{lambda_reg}.pkl",
        "null":     cache_dir / f"null_store-{lambda_reg}.pkl",
        "meta":     cache_dir / f"meta-{lambda_reg}.json",
    }


def structural_cache_exists(cache_dir: str | Path, lambda_reg: float) -> bool:
    return all(p.exists() for p in _structural_paths(cache_dir, lambda_reg).values())


def load_structural_cache(cache_dir: str | Path, lambda_reg: float):
    """Return ``(results_df, geometry_store, null_store)`` from disk."""
    import pandas as pd
    paths = _structural_paths(cache_dir, lambda_reg)
    results_df = pd.read_parquet(paths["results"])
    with open(paths["geometry"], "rb") as f:
        geometry_store = pickle.load(f)
    with open(paths["null"], "rb") as f:
        null_store = pickle.load(f)
    return results_df, geometry_store, null_store


def save_structural_cache(
    cache_dir: str | Path,
    lambda_reg: float,
    results_df,
    geometry_store: dict,
    null_store: dict,
    meta: dict,
) -> None:
    ensure_dir(cache_dir)
    paths = _structural_paths(cache_dir, lambda_reg)

    results_df.to_parquet(paths["results"], index=False)

    with open(paths["geometry"], "wb") as f:
        pickle.dump(geometry_store, f)
    with open(paths["null"], "wb") as f:
        pickle.dump(null_store, f)

    write_json(paths["meta"], meta)


# ============================================================
# ================ PER-PAIR EXPERIMENT CACHE =================
# ============================================================

def per_pair_path(cache_dir: str | Path, **fields: Any) -> Path:
    """Build a cache filename of the form ``key1=val1__key2=val2.parquet``.

    Used by label-propagation and lambda-sensitivity runners, where each
    (model, prompt[, detector]) combination is cached independently.
    """
    parts = [f"{k}={v}" for k, v in fields.items()]
    fname = "__".join(parts) + ".parquet"
    return Path(cache_dir) / fname
