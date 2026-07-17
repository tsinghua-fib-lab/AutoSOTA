#!/bin/bash
# Download CORRECT-Error from Hugging Face and restore the per-trajectory
# JSON layout expected by src/inference_correct_error.py.
#
# After this script:
#   data/correct_error/{dataset}/individual_trajectories/{question_id}.json   (gpt-4o-mini split)
#   data/correct_error_gpt5nano/{dataset}/individual_trajectories/{question_id}.json   (gpt-5-nano split)
#
# Run from the CORRECT/ project root.
# Usage: bash scripts/download_data.sh [repo_id]

set -e

REPO=${1:-"yifanyu/CORRECT-Error"}
DEST_GPT4O="data/correct_error"
DEST_GPT5N="data/correct_error_gpt5nano"

echo "========================================"
echo "Downloading CORRECT-Error from Hugging Face"
echo "========================================"
echo "Repo:        $REPO"
echo "gpt-4o-mini -> $DEST_GPT4O"
echo "gpt-5-nano  -> $DEST_GPT5N"
echo "========================================"

python3 - "$REPO" "$DEST_GPT4O" "$DEST_GPT5N" <<'PY'
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset

repo_id, dest_gpt4o, dest_gpt5n = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

ds = load_dataset(repo_id, split="test")
print(f"Loaded {len(ds)} records from {repo_id}")

dest_for = {
    "gpt-4o-mini": dest_gpt4o,
    "gpt-5-nano":  dest_gpt5n,
}

# For gpt-5-nano + gaia, restore the original directory name "gaia_level1"
# so the inference scripts can locate the matching schemata cache.
def out_dataset_name(row):
    if row["generator_model"] == "gpt-5-nano" and row["dataset"] == "gaia":
        return "gaia_level1"
    return row["dataset"]

counts = {}
for row in ds:
    base = dest_for[row["generator_model"]]
    ds_name = out_dataset_name(row)
    out_dir = base / ds_name / "individual_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restore the original per-trajectory JSON schema (drop release-only fields).
    record = {
        "history":        row["history"],
        "question":       row["question"],
        "groundtruth":    row["groundtruth"],
        "is_corrected":   row["is_corrected"],
        "mistake_agent":  row["mistake_agent"],
        "mistake_reason": row["mistake_reason"],
        "mistake_step":   str(row["mistake_step"]),
        "question_ID":    row["question_id"],
        "level":          row["level"],
    }
    # File index inside individual_trajectories/ follows the question_id ordinal
    # used during inference; we reuse a running counter per dataset to keep names
    # short and contiguous (1.json, 2.json, ...).
    key = (row["generator_model"], ds_name)
    counts[key] = counts.get(key, 0) + 1
    out_path = out_dir / f"{counts[key]}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

print()
print("Per (generator_model, dataset):")
for (model, dataset), n in sorted(counts.items()):
    print(f"  {model:11s}  {dataset:14s}  {n:4d} trajectories")
PY

echo ""
echo "========================================"
echo "Download complete."
echo "========================================"
