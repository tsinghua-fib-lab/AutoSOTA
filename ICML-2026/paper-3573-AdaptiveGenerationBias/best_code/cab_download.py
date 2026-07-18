#!/usr/bin/env python3
"""
Download eth-sri/cab from Hugging Face and write JSONL shards:

  out_dir/
    explicit/
      sex.jsonl
      race.jsonl
      religion.jsonl
    implicit/
      sex.jsonl
      race.jsonl
      religion.jsonl

Each output line is a JSON object with exactly:
  {
    "superdomain": ...,
    "domain": ...,
    "topic": ...,
    "example": ...,
    "score": null
  }

Field mapping from HF dataset:
  superdomain  <- superdomain (remapped broad topical area)
  domain       <- domain (remapped specific context)
  topic        <- topic
  example      <- example
  score        <- None (emitted as JSON null)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, TextIO

from huggingface_hub import snapshot_download
from datasets import load_dataset


def _safe_filename(name: str) -> str:
    # conservative filename sanitizer
    s = str(name).strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s) or "unknown"


def project_row(row: dict) -> dict:
    """
    Project a raw dataset row into the exact output schema required.
    """
    return {
        "superdomain": row.get("superdomain", None),
        "domain": row.get("domain", None),
        "topic": row.get("topic", None),
        "example": row.get("example", None),
        "score": None,  # dataset has no score field; required output wants null
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", default="eth-sri/cab", help="HF dataset repo id, i.e., eth-sri/cab"
    )
    parser.add_argument("--out_dir", default="cab_download", help="Output directory")
    parser.add_argument(
        "--revision",
        default=None,
        help="(Not yet required) Optional git revision/branch/tag/commit",
    )
    parser.add_argument("--token", default=None, help="(Not required) HF token")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download the dataset repo snapshot
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            token=args.token,
        )
    )

    # The repo contains cab.json
    cab_json_path = snapshot_path / "cab.json"
    if not cab_json_path.exists():
        raise FileNotFoundError(f"Expected {cab_json_path} in the snapshot, but it was not found.")

    # 2) Load cab.json via 🤗 Datasets
    ds = load_dataset(
        "json",
        data_files={"train": str(cab_json_path)},
        split="train",
    )

    # 3) Writers keyed by expl_impl -> attribute
    writers: Dict[str, Dict[str, TextIO]] = {}

    def get_writer(expl_impl: str, attribute: str) -> TextIO:
        expl_impl_key = _safe_filename(expl_impl)
        attribute_key = _safe_filename(attribute)

        if expl_impl_key not in writers:
            writers[expl_impl_key] = {}
        if attribute_key not in writers[expl_impl_key]:
            folder = out_dir / expl_impl_key
            folder.mkdir(parents=True, exist_ok=True)
            fp = folder / f"{attribute_key}.jsonl"
            writers[expl_impl_key][attribute_key] = fp.open("w", encoding="utf-8")
        return writers[expl_impl_key][attribute_key]

    skipped_missing_expl_impl = 0
    skipped_missing_attribute = 0
    written = 0

    # 4) Iterate rows and write projected JSONL
    for row in ds:
        expl_impl = row.get("expl_impl", None)
        attribute = row.get("attribute", None)

        if expl_impl is None:
            skipped_missing_expl_impl += 1
            continue
        if attribute is None:
            skipped_missing_attribute += 1
            continue

        out_obj = project_row(row)
        w = get_writer(expl_impl, attribute)
        w.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        written += 1

    # 5) Close files
    for by_attr in writers.values():
        for fh in by_attr.values():
            fh.close()

    print("Done.")
    print(f"Snapshot: {snapshot_path}")
    print(f"Output:   {out_dir.resolve()}")
    print(f"Written rows: {written}")
    if skipped_missing_expl_impl or skipped_missing_attribute:
        print(f"Skipped rows missing expl_impl: {skipped_missing_expl_impl}")
        print(f"Skipped rows missing attribute: {skipped_missing_attribute}")


if __name__ == "__main__":
    main()
