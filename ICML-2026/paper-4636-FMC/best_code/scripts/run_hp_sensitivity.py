#!/usr/bin/env python3
"""Hyperparameter Sensitivity Sweep for ICML Rebuttal.

Generates variant configs for LR × gradient clipping sweep,
runs training, and generates a sensitivity table.

Addresses Reviewer xKrR (Q2, Point 6).

Usage:
    # Step 1: Generate variant configs
    python scripts/run_hp_sensitivity.py generate

    # Step 2: Train foundation (only once)
    python run_benchmark.py command=train_foundation +experiment=rebuttal_hp_sensitivity

    # Step 3: Train all variants (can be parallelized on cluster)
    python scripts/run_hp_sensitivity.py train --experiment rebuttal_hp_sensitivity

    # Step 4: Evaluate all variants
    python run_benchmark.py command=evaluate +experiment=rebuttal_hp_sensitivity

    # Step 5: Generate sensitivity table
    python scripts/run_hp_sensitivity.py table --experiment rebuttal_hp_sensitivity
"""

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Hyperparameter grid
LR_VALUES = [1e-4, 5e-4, 1e-3, 5e-3]
GRAD_CLIP_VALUES = [0.5, 1.0, 5.0, None]  # None = no clipping (very large)


def variant_name(lr: float, gc: Optional[float]) -> str:
    """Generate variant name from HP values."""
    lr_str = f"lr{lr:.0e}".replace(".", "").replace("+", "").replace("-0", "")
    gc_str = f"gc{gc:.1f}".replace(".", "") if gc is not None else "gcNone"
    return f"hp_{lr_str}_{gc_str}"


def generate_configs():
    """Generate variant YAML configs for each HP combination."""
    variant_dir = PROJECT_ROOT / "configs" / "variant"

    for lr, gc in product(LR_VALUES, GRAD_CLIP_VALUES):
        name = variant_name(lr, gc)

        gc_yaml = (
            f"  grad_clip: {gc}"
            if gc is not None
            else "  grad_clip: 1000.0  # effectively no clipping"
        )

        config = f"""# Auto-generated variant for HP sensitivity sweep
# LR={lr}, grad_clip={gc}

variant_name: {name}
method: fm_post_transform

base_dist: fmpe

training_params:
  lr: {lr}
  epochs: 50
  batch_size: 256
  train_size: 0.8
{gc_yaml}

config:
  npe: ${{variant.base_dist}}

hpo:
  enabled: false
"""
        out_path = variant_dir / f"{name}.yaml"
        out_path.write_text(config)
        print(f"  Created: {out_path.name}")

    print(f"\nGenerated {len(LR_VALUES) * len(GRAD_CLIP_VALUES)} variant configs.")
    print("\nVariant names for training:")
    for lr, gc in product(LR_VALUES, GRAD_CLIP_VALUES):
        print(f"  {variant_name(lr, gc)}")


def train_all(experiment: str):
    """Print training commands for all variants."""
    print("Run these commands (or submit to cluster):\n")

    for lr, gc in product(LR_VALUES, GRAD_CLIP_VALUES):
        name = variant_name(lr, gc)
        cmd = (
            f"python run_benchmark.py command=train_variant "
            f"variant_name={name} +experiment={experiment}"
        )
        print(cmd)


def generate_table(experiment: str, results_dir: str = "results"):
    """Generate sensitivity table from evaluation results."""
    import numpy as np

    results_path = Path(results_dir) / experiment
    output_dir = results_path / "rebuttal_plots" / "hp_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect results
    rows = []
    for lr, gc in product(LR_VALUES, GRAD_CLIP_VALUES):
        name = variant_name(lr, gc)
        row = {"lr": lr, "grad_clip": gc, "variant": name}

        # Look for evaluation metrics
        eval_dir = results_path / "variants" / name / "evaluation"
        if not eval_dir.exists():
            print(f"  No evaluation results for {name}")
            rows.append(row)
            continue

        # Collect metrics across seeds and tasks
        metrics_by_seed = []
        for metrics_file in eval_dir.rglob("metrics.json"):
            try:
                with open(metrics_file) as f:
                    m = json.load(f)
                # Flatten nested "metrics" dict into top level
                if "metrics" in m and isinstance(m["metrics"], dict):
                    m = {**m, **m["metrics"]}
                metrics_by_seed.append(m)
            except Exception:
                continue

        if metrics_by_seed:
            # Average metrics across seeds
            all_keys = set()
            for m in metrics_by_seed:
                all_keys.update(m.keys())

            for key in all_keys:
                values = [
                    m[key] for m in metrics_by_seed if key in m and isinstance(m[key], (int, float))
                ]
                if values:
                    row[f"{key}_mean"] = float(np.mean(values))
                    row[f"{key}_std"] = float(np.std(values))

        rows.append(row)

    # Print table
    print("\n" + "=" * 100)
    print("HYPERPARAMETER SENSITIVITY TABLE")
    print("=" * 100)

    # Find available metric keys
    metric_keys = set()
    for row in rows:
        metric_keys.update(k for k in row if k.endswith("_mean"))
    metric_keys = sorted(metric_keys)

    # Header
    header = f"{'LR':>10s} | {'Grad Clip':>10s}"
    for mk in metric_keys:
        header += f" | {mk.replace('_mean', ''):>20s}"
    print(header)
    print("-" * len(header))

    for row in rows:
        gc_str = f"{row['grad_clip']}" if row["grad_clip"] is not None else "None"
        line = f"{row['lr']:>10.0e} | {gc_str:>10s}"
        for mk in metric_keys:
            mean = row.get(mk, float("nan"))
            std = row.get(mk.replace("_mean", "_std"), 0.0)
            if not np.isnan(mean):
                line += f" | {mean:>10.4f}±{std:.4f}"
            else:
                line += f" | {'N/A':>20s}"
        print(line)

    # Save as JSON
    json_path = output_dir / "sensitivity_table.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved to: {json_path}")

    # Save as CSV
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        csv_path = output_dir / "sensitivity_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved to: {csv_path}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="HP Sensitivity Sweep")
    parser.add_argument("action", choices=["generate", "train", "table"])
    parser.add_argument("--experiment", default="rebuttal_hp_sensitivity")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    if args.action == "generate":
        generate_configs()
    elif args.action == "train":
        train_all(args.experiment)
    elif args.action == "table":
        generate_table(args.experiment, args.results_dir)


if __name__ == "__main__":
    main()
