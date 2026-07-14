#!/usr/bin/env python3

# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fix all prediction errors across all CSV files in data/.

Uses the proeval UnifiedCSVManager to scan each *_predictions.csv for
SKIPPED/ERROR/RATE_LIMITED/PARSE_ERROR rows and re-runs them with the
LLMPredictor.

Usage:
    # Fix all datasets, all models
    python utils/fix_all_errors.py

    # Fix a specific dataset
    python utils/fix_all_errors.py --dataset gsm8k

    # Fix a specific model across all datasets
    python utils/fix_all_errors.py --model gemini3_pro

    # Dry-run: just report errors without fixing
    python utils/fix_all_errors.py --dry-run

    # Sequential mode (no parallelism)
    python utils/fix_all_errors.py --sequential
"""

import argparse
import os
import sys

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from proeval.evaluator import (
    DATASET_CONFIGS,
    LLMPredictor,
    UnifiedCSVManager,
)
from proeval.evaluator.client import resolve_model_name


DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Datasets that need the original question list for fix_errors
# (question column name differs per dataset)
QUESTION_COLS = {
    "strategyqa": "question",
    "gsm8k": "question",
    "svamp": "question",
    "mmlu": None,  # dict-structured
    "mmlu_professionallaw": None,  # dict-structured
    "jigsaw": "question",
    "toxicchat": "question",
    "gqa": None,  # dict-structured (question + image_id)
    "dices": None,  # dict-structured (context + response)
    "dices_t2i": "question",
}


def get_csv_files(data_dir: str, dataset_filter: str = None):
    """Find all prediction CSV files."""
    files = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_predictions.csv"):
            ds_name = f.replace("_predictions.csv", "")
            if dataset_filter and ds_name != dataset_filter:
                continue
            if ds_name in DATASET_CONFIGS:
                files.append((ds_name, os.path.join(data_dir, f)))
    return files


def scan_errors(csv_path: str) -> dict:
    """Scan a CSV file for error counts per model."""
    df = pd.read_csv(csv_path)
    results = {}
    for col in df.columns:
        if col.startswith("label_"):
            model = col.replace("label_", "")
            pred_col = f"prediction_{model}"
            if pred_col not in df.columns:
                continue
            errors = df[pred_col].isin(
                ["SKIPPED", "ERROR", "RATE_LIMITED", "PARSE_ERROR"]
            ).sum()
            nan_labels = df[col].isna().sum()
            total = max(int(errors), int(nan_labels))
            if total > 0:
                results[model] = total
    return results


def fix_dataset(
    ds_name: str,
    csv_path: str,
    model_filter: str = None,
    parallel: bool = True,
    workers: int = 10,
    dry_run: bool = False,
    skip_error: bool = False,
    max_retry: int = 5,
):
    """Fix errors in a single dataset CSV."""
    df = pd.read_csv(csv_path)
    n_rows = len(df)

    # Build question and ground truth lists from the CSV
    questions = df["question"].tolist()
    ground_truths = df["ground_truth"].tolist()

    # Get dataset config
    config = DATASET_CONFIGS.get(ds_name)
    if config is None:
        print(f"  ⚠ No config for '{ds_name}', skipping")
        return

    # Create CSV manager
    csv_mgr = UnifiedCSVManager(ds_name, output_dir=os.path.dirname(csv_path))
    csv_mgr.load_or_create(questions, ground_truths)

    # Find models with errors
    error_summary = scan_errors(csv_path)
    if model_filter:
        error_summary = {k: v for k, v in error_summary.items() if k == model_filter}

    if not error_summary:
        print(f"  ✓ No errors to fix")
        return

    total_errors = sum(error_summary.values())
    print(f"  Models with errors ({total_errors} total):")
    for model, count in sorted(error_summary.items(), key=lambda x: -x[1]):
        print(f"    {model}: {count} errors")

    if dry_run:
        return

    # Fix each model
    for model_name, error_count in sorted(error_summary.items(), key=lambda x: -x[1]):
        print(f"\n  🔧 Fixing {model_name} ({error_count} errors)...")

        # Resolve the actual model identifier for API calls
        try:
            model_id = resolve_model_name(model_name)
        except Exception:
            model_id = model_name  # Use as-is if not in mapping

        predictor = LLMPredictor(model=model_id)

        try:
            csv_mgr.fix_errors(
                predictor,
                model_name,
                config,
                questions,
                ground_truths,
                parallel=parallel,
                workers=workers,
                max_parse_retries=max_retry,
                skip_error=skip_error,
            )
        except Exception as e:
            print(f"  ❌ Error fixing {model_name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Fix prediction errors across all CSV files")
    parser.add_argument("--dataset", type=str, default=None, help="Fix only this dataset")
    parser.add_argument("--model", type=str, default=None, help="Fix only this model")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument("--dry-run", action="store_true", help="Only report errors, don't fix")
    parser.add_argument("--sequential", action="store_true", help="Disable parallel processing")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers (default: 10)")
    parser.add_argument("--max-retry", type=int, default=5, help="Max retries per item")
    parser.add_argument("--skip-error", action="store_true", help="Mark unfixable as NaN not 1.0")
    args = parser.parse_args()

    csv_files = get_csv_files(args.data_dir, args.dataset)
    if not csv_files:
        print(f"No prediction CSVs found in {args.data_dir}")
        return

    print("=" * 60)
    print("ProEval — Fix All Prediction Errors")
    print("=" * 60)
    if args.dry_run:
        print("🔍 DRY RUN — reporting errors only\n")
    else:
        print(f"⚡ Mode: {'sequential' if args.sequential else f'parallel ({args.workers} workers)'}\n")

    grand_total = 0

    for ds_name, csv_path in csv_files:
        df = pd.read_csv(csv_path)
        errors = scan_errors(csv_path)
        model_errors = {k: v for k, v in errors.items() if args.model is None or k == args.model}
        ds_total = sum(model_errors.values())
        grand_total += ds_total

        print(f"\n{'─'*60}")
        print(f"📊 {ds_name} ({len(df)} rows, {ds_total} errors)")
        print(f"{'─'*60}")

        if ds_total == 0:
            print("  ✓ No errors to fix")
            continue

        fix_dataset(
            ds_name, csv_path,
            model_filter=args.model,
            parallel=not args.sequential,
            workers=args.workers,
            dry_run=args.dry_run,
            skip_error=args.skip_error,
            max_retry=args.max_retry,
        )

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"📋 Total errors found: {grand_total}")
    else:
        print(f"✅ Done! Processed {grand_total} errors across {len(csv_files)} datasets")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
