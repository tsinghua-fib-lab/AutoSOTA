#!/usr/bin/env python3
"""
Backfill missing timing metrics in per-fold results and regenerate aggregated k-fold results.

Expected layout under --root:
    day_*/<model_name>/fold_*/metrics/{results.yaml, results.pkl}
    day_*/<model_name>/metrics/kfold_results.yaml (will be rewritten)

Example usage:
python scripts/python/fix_kfold_missing_timings.py \
    --root /path/to/experiments \
    --model vdw_supcon_2 \
    --kfolds-name kfolds_results.yaml \
    --required-keys mean_train_time,mean_infer_time,best_epoch
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml

# Ensure repository root (two levels up) is on sys.path when run from anywhere.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.kfold_results import compute_kfold_results

REQUIRED_KEYS: Tuple[str, ...] = ("mean_train_time", "mean_infer_time", "best_epoch")


def parse_args() -> argparse.Namespace:
    """
    CLI options.
    """
    parser = argparse.ArgumentParser(description="Backfill missing timing metrics for k-fold runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing day_* folders.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory name under each day_* folder.",
    )
    parser.add_argument(
        "--kfolds-name",
        type=str,
        default="kfolds_results.yaml",
        help="Optional extra aggregated filename to mirror kfold_results.yaml (if used).",
    )
    parser.add_argument(
        "--required-keys",
        type=str,
        default=",".join(REQUIRED_KEYS),
        help="Comma-separated metric keys to backfill from results.pkl.",
    )
    return parser.parse_args()


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file, returning an empty dict on failure.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as exc:
        print(f"Failed to load YAML {path}: {exc}")
        return {}


def _safe_load_pickle_dict(path: Path) -> Dict[str, Any]:
    """
    Load a pickle expected to contain a dictionary, returning {} on failure.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as handle:
            data = pickle.load(handle) or {}
        if not isinstance(data, dict):
            print(f"Pickle {path} is not a dict (got {type(data)}); skipping.")
            return {}
        return data
    except Exception as exc:
        print(f"Failed to load pickle {path}: {exc}")
        return {}


def _backfill_fold_metrics(
    fold_dir: Path,
    required_keys: Iterable[str],
) -> bool:
    """
    Update fold-level results.yaml with missing timing metrics from results.pkl.
    """
    metrics_dir = fold_dir / "metrics"
    res_path = metrics_dir / "results.yaml"
    pkl_path = metrics_dir / "results.pkl"

    if not metrics_dir.exists():
        return False

    data = _safe_load_yaml(res_path)
    missing = [key for key in required_keys if key not in data]
    if not missing:
        return False

    pkl_data = _safe_load_pickle_dict(pkl_path)
    if not pkl_data:
        print(f"No usable pickle for {res_path}; still missing {missing}.")
        return False

    added_keys = []
    for key in missing:
        if key in pkl_data:
            data[key] = pkl_data[key]
            added_keys.append(key)

    if not added_keys:
        print(f"Pickle {pkl_path} lacked requested keys for {res_path}: {missing}.")
        return False

    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    print(f"Updated {res_path} with: {', '.join(added_keys)}")
    return True


def _write_aggregates(
    metrics_dir: Path,
    kfolds_name: str,
) -> None:
    """
    Recompute and write aggregated k-fold results.
    """
    aggregate = compute_kfold_results(metrics_dir)
    if not aggregate:
        print(f"Skipped aggregation for {metrics_dir} (insufficient or missing folds).")
        return

    default_path = metrics_dir / "kfold_results.yaml"
    with open(default_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(aggregate, handle)
    print(f"Wrote aggregated results to {default_path}")

    alt_path = metrics_dir / kfolds_name
    if alt_path != default_path:
        with open(alt_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(aggregate, handle)
        print(f"Wrote aggregated results to {alt_path}")


def main() -> None:
    """
    Entry point.
    """
    args = parse_args()
    root = args.root.resolve()
    required_keys = tuple(key.strip() for key in args.required_keys.split(",") if key.strip())

    if not required_keys:
        print("No required keys specified; nothing to do.")
        return

    print(f"Searching under {root} for day_* folders and model '{args.model}'.")
    for day_dir in sorted(path for path in root.glob("day_*") if path.is_dir()):
        model_dir = day_dir / args.model
        if not model_dir.is_dir():
            continue

        print(f"Processing {model_dir}")
        fold_dirs = sorted(path for path in model_dir.glob("fold_*") if path.is_dir())
        updated_any_fold = False
        for fold_dir in fold_dirs:
            updated_any_fold |= _backfill_fold_metrics(fold_dir, required_keys)

        metrics_dir = model_dir / "metrics"
        if metrics_dir.exists():
            if updated_any_fold:
                _write_aggregates(metrics_dir, args.kfolds_name)
            else:
                # Recompute anyway in case previous aggregated files were missing expected keys.
                _write_aggregates(metrics_dir, args.kfolds_name)
        else:
            print(f"No metrics directory found at {metrics_dir}; skipping aggregation.")


if __name__ == "__main__":
    main()
