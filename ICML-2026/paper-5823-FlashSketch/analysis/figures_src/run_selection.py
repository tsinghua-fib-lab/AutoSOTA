from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from pathlib import Path

import polars as pl

import globals as g
from io_utils import read_json
from provenance import get_tree_hashes


def _matches_required_datasets(df, required_datasets):
    """Return True if df contains all required dataset identifiers."""
    if not required_datasets:
        return True

    if "dataset" not in df.columns or "dataset_name" not in df.columns:
        return False

    if "dataset_group" not in df.columns:
        available = {(row["dataset"], row["dataset_name"], None) for row in df.select(["dataset", "dataset_name"]).unique().to_dicts()}
    else:
        available = {
            (row["dataset"], row["dataset_name"], row["dataset_group"])
            for row in df.select(["dataset", "dataset_name", "dataset_group"]).unique().to_dicts()
        }

    for dataset, dataset_name, dataset_group in required_datasets:
        if dataset_group is None:
            if not any(
                item_dataset == dataset and item_name == dataset_name
                for item_dataset, item_name, _ in available
            ):
                return False
        else:
            if (dataset, dataset_name, dataset_group) not in available:
                return False

    return True


def _has_required_tasks(df, required_tasks):
    """Return True if df contains any of the required tasks."""
    if not required_tasks:
        return True
    if "task" not in df.columns:
        return False
    tasks = set(df.select(pl.col("task").unique()).to_series().to_list())
    return bool(tasks.intersection(required_tasks))


TREE_HASH_PATHS = ("sketches", "kernels", "bench", "data")


def _manifest_tree_hashes_match(candidate, current_hashes):
    """Return True if the run manifest matches the current tree hashes."""
    manifest_path = Path(candidate).parent / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("git", {}).get("dirty") is True:
        return False
    manifest_hashes = manifest.get("tree_hashes")
    if not isinstance(manifest_hashes, dict):
        return False
    for path, expected in current_hashes.items():
        if manifest_hashes.get(path) != expected:
            return False
    return True


def find_latest_results(
    required_columns,
    required_datasets=None,
    required_tasks=None,
    extra_filter=None,
    tree_hash_paths=None,
):
    """Return latest results.parquet under file_storage with required data."""
    runs_dir = g.FILE_STORAGE_PATH / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError("No runs directory found under file_storage.")

    candidates = list(runs_dir.rglob("results.parquet"))
    if not candidates:
        raise FileNotFoundError("No results.parquet files found under file_storage/runs.")

    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    current_hashes = get_tree_hashes(tree_hash_paths or TREE_HASH_PATHS)

    required_cols = set(required_columns)
    if required_datasets:
        required_cols.update({"dataset", "dataset_name"})
        if any(dataset_group is not None for _, _, dataset_group in required_datasets):
            required_cols.add("dataset_group")
    if required_tasks:
        required_cols.add("task")

    for candidate in candidates:
        scan = pl.scan_parquet(candidate)
        schema = scan.schema
        if not required_cols.issubset(schema.keys()):
            continue
        if not _manifest_tree_hashes_match(candidate, current_hashes):
            continue
        df = scan.select(sorted(required_cols)).collect()
        if not _has_required_tasks(df, required_tasks):
            continue
        if not _matches_required_datasets(df, required_datasets):
            continue
        if extra_filter is not None and not extra_filter(df):
            continue
        return candidate

    raise FileNotFoundError("No results.parquet contains required columns, tasks, and datasets.")
