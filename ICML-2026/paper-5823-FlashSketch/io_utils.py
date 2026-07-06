from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import json
from pathlib import Path

import polars as pl


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    """Write a JSON file to disk with stable formatting."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path):
    """Read a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_parquet(path, rows):
    """Write a list of dict rows to a Parquet file."""
    ensure_dir(Path(path).parent)
    if not rows:
        pl.DataFrame().write_parquet(path)
        return
    pl.DataFrame(rows).write_parquet(path)


def read_parquet(path):
    """Read a Parquet file into a Polars DataFrame."""
    return pl.read_parquet(path)
