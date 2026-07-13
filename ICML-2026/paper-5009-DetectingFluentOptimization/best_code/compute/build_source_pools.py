#!/usr/bin/env python3
"""
Build prompt pools for benign sampling (TyDiQA + OpenOrca) as CSVs.

These CSVs are intended to be fed into compute_token_stats.py to produce the
token-stat CSVs required by sample_openorca_pp.py.

Notes:
  - This script is optional and requires the `datasets` package and network access.
  - The exact pool size can be large; producing token stats can take many hours and
    generate multi-GB CSVs. See README.md for guidance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _require_datasets():
    try:
        from datasets import load_dataset  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This script requires the `datasets` package (pip install datasets) and "
            "network access to download the datasets."
        ) from exc


def build_tydiqa(out_csv: Path, *, max_rows: int, seed: int) -> None:
    _require_datasets()
    from datasets import load_dataset

    ds = load_dataset("tydiqa", "primary_task", split="train")
    df = ds.shuffle(seed=seed).to_pandas()

    if "question" not in df.columns:
        raise ValueError("Expected TyDiQA to contain a 'question' column")
    if "language" not in df.columns:
        raise ValueError("Expected TyDiQA to contain a 'language' column")

    df = df[["question", "language"]].rename(columns={"question": "full_prompt"})
    df["suffix"] = ""
    df["is_adversarial"] = 0
    df["algorithm"] = "tydiqa"
    df = df.head(int(max_rows)).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def build_openorca(out_csv: Path, *, max_rows: int, seed: int) -> None:
    _require_datasets()
    from datasets import load_dataset

    # OpenOrca is commonly mirrored on HF; field names vary slightly across mirrors.
    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    df = ds.shuffle(seed=seed).to_pandas()

    # Prefer instruction-style fields when present.
    candidates = ["instruction", "question", "prompt", "input"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(f"Could not find any of {candidates} in OpenOrca columns: {list(df.columns)}")

    df = df[[col]].rename(columns={col: "full_prompt"})
    df["suffix"] = ""
    df["is_adversarial"] = 0
    df["algorithm"] = "openorca"
    df = df.head(int(max_rows)).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/pools"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tydiqa-max-rows", type=int, default=100000)
    p.add_argument("--openorca-max-rows", type=int, default=100000)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    build_tydiqa(out_dir / "tydiqa_pool.csv", max_rows=args.tydiqa_max_rows, seed=args.seed)
    build_openorca(out_dir / "openorca_pool.csv", max_rows=args.openorca_max_rows, seed=args.seed)
    print(f"[OK] Wrote pool CSVs to {out_dir}")


if __name__ == "__main__":
    main()

