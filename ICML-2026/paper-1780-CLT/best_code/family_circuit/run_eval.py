#!/usr/bin/env python3
"""
Reproduction script for ProtoMech protein family classification (ESM2-8M).
Evaluates ProtoMech full-latent F1 score across protein families.

Usage:
    cd /repo/family_circuit
    python run_eval.py [--limit N] [--overwrite]
"""
import sys
import os
import json
import argparse
import statistics
from pathlib import Path
import subprocess

REPO_ROOT = "/repo"
FAMILY_DIR = os.path.join(REPO_ROOT, "family_circuit")


def run_step1(limit=None, overwrite=False):
    """Extract ESM2 embeddings for protein sequences."""
    print("=" * 60)
    print(" [Step 1] Extracting ESM embeddings")
    print("=" * 60)

    env = os.environ.copy()
    env["BATCH_SIZE"] = "16"
    env["MIN_POSITIVES"] = "50"
    env["CLT_CHECKPOINT"] = f"{REPO_ROOT}/models/CLT_L6_D3200/checkpoints/last.ckpt"
    env["ESM_WEIGHTS"] = f"{REPO_ROOT}/models/esm2_t6_8M_UR50D.pt"
    env["PARQUET_PATH"] = f"{REPO_ROOT}/data/swissprot_seqid30_75k_all_info_with_3di.parquet"
    env["OUTPUT_DIR"] = "families_8M"
    env["MASTER_NPZ_NAME"] = "all_acts_8M.npz"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/training:{REPO_ROOT}/training_block"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable, "01_extract_embeddings.py",
        "--layers", "6",
        "--hidden_size", "320",
        "--source", "mlp_output",
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if overwrite:
        cmd.append("--overwrite")

    result = subprocess.run(cmd, env=env, cwd=FAMILY_DIR)
    if result.returncode != 0:
        print("ERROR: Step 1 failed")
        sys.exit(1)


def run_step2(limit=None, overwrite=False):
    """Discover circuits and evaluate full-latent F1."""
    print("=" * 60)
    print(" [Step 2] Circuit Discovery (CLT Sequential)")
    print("=" * 60)

    env = os.environ.copy()
    env["BATCH_SIZE"] = "8"
    env["MIN_POSITIVES"] = "50"
    env["CLT_CHECKPOINT"] = f"{REPO_ROOT}/models/CLT_L6_D3200/checkpoints/last.ckpt"
    env["ESM_WEIGHTS"] = f"{REPO_ROOT}/models/esm2_t6_8M_UR50D.pt"
    env["PARQUET_PATH"] = f"{REPO_ROOT}/data/swissprot_seqid30_75k_all_info_with_3di.parquet"
    env["OUTPUT_DIR"] = "families_8M"
    env["MASTER_NPZ_NAME"] = "all_acts_8M.npz"
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/training:{REPO_ROOT}/training_block"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable, "02_discover_circuits_clt.py",
        "--recovery_ratio", "0.7",
        "--max_nodes", "1000",
        "--sequential",
        "--source", "mlp_output",
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if overwrite:
        cmd.append("--overwrite")

    result = subprocess.run(cmd, env=env, cwd=FAMILY_DIR)
    if result.returncode != 0:
        print("ERROR: Step 2 failed")
        sys.exit(1)


def compute_metrics():
    """Compute average F1 from saved results."""
    import glob as gb
    results_dir = os.path.join(FAMILY_DIR, "families_8M", "CLT_sequential")
    files = gb.glob(os.path.join(results_dir, "*.json"))

    if not files:
        print("ERROR: No result files found")
        sys.exit(1)

    max_f1s = []
    clean_f1s = []

    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        max_f1s.append(d["max_f1"])
        clean_f1s.append(d["clean_f1"])

    avg_max_f1 = sum(max_f1s) / len(max_f1s)
    avg_clean_f1 = sum(clean_f1s) / len(clean_f1s)
    std_max_f1 = statistics.stdev(max_f1s) if len(max_f1s) > 1 else 0.0

    print("=" * 60)
    print(" REPRODUCTION RESULTS")
    print("=" * 60)
    print(f"  Families evaluated:        {len(max_f1s)}")
    print(f"  ProtoMech F1 (full):       {avg_max_f1:.4f} +/- {std_max_f1:.4f}")
    print(f"  ESM2 probe F1 (clean):     {avg_clean_f1:.4f}")
    print(f"  Recovery ratio:            {avg_max_f1/avg_clean_f1*100:.1f}%")
    print("=" * 60)

    # Output machine-parseable metric line
    print(f"\nMETRIC:F1={avg_max_f1:.6f}")

    return avg_max_f1


def main():
    parser = argparse.ArgumentParser(description="ProtoMech Reproduction Eval")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of families")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--skip-step1", action="store_true", help="Skip embedding extraction")
    parser.add_argument("--skip-step2", action="store_true", help="Skip circuit discovery")
    parser.add_argument("--metric-only", action="store_true", help="Only compute metrics from existing results")
    args = parser.parse_args()

    if args.metric_only:
        compute_metrics()
        return

    if not args.skip_step1:
        run_step1(limit=args.limit, overwrite=args.overwrite)

    if not args.skip_step2:
        run_step2(limit=args.limit, overwrite=args.overwrite)

    compute_metrics()


if __name__ == "__main__":
    main()
