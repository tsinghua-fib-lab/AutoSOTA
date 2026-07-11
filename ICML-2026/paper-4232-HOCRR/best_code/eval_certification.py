#!/usr/bin/env python3
"""
Reproduction evaluation script for paper 4232:
"Higher-Order Certified Robustness for Regression"

Computes (E,C,G)+M certified metrics for MNIST rotation:
Absolute Accuracy, Conditional Accuracy, Mean Distance
at radius thresholds R in {0.05, 0.10, 0.15, 0.20, 0.25}.
"""
import json
import numpy as np
import subprocess
import sys
import argparse
from pathlib import Path

MODEL_PATH = "models/e2cnn_rotation_model.pth"
SIGMA = 0.75
EPS_Y_DEG = 10.0
N_TEST = 100
N_SAMPLES = 10000
N_TRIALS = 1
CONFIDENCE = 0.90
SEED = 42
R_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]
TOLERANCE = 10.0


def run_step(cmd, desc):
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Step failed with code {result.returncode}")
        sys.exit(1)


def compute_metrics(estimation_file, radii_file):
    with open(estimation_file) as f:
        est = json.load(f)
    with open(radii_file) as f:
        radii_data = json.load(f)

    radii = [r["radius"] for r in radii_data["results"]]
    clean_preds = []
    true_angles = []
    for r in radii_data["results"]:
        idx = r["test_dataset_idx"]
        for s in est["samples"]:
            if s["test_dataset_idx"] == idx:
                clean_preds.append(s.get("clean_pred_deg", 0))
                true_angles.append(s.get("true_angle_deg", 0))
                break

    errors = [abs(((p - t + 180) % 360) - 180) for p, t in zip(clean_preds, true_angles)]
    n = len(radii)

    print(f"\n{'='*70}")
    print(f"RESULTS: (E,C,G)+M, sigma={SIGMA}, eps_y={EPS_Y_DEG}deg, confidence={CONFIDENCE}")
    print(f"Samples: {n}, Mean radius: {np.mean(radii):.4f}, Mean error: {np.mean(errors):.2f} deg")
    print(f"{'='*70}")
    print(f"{'R':>8s}  {'AbsAcc%':>10s}  {'CondAcc%':>10s}  {'MeanDist':>10s}  {'Cert':>8s}")
    print("-" * 70)

    metrics = {}
    for R in R_VALUES:
        correct_cert = sum(1 for r, e in zip(radii, errors) if e <= TOLERANCE and r >= R)
        certified = sum(1 for r in radii if r >= R)
        abs_acc = 100.0 * correct_cert / n
        if certified > 0:
            cond_acc = 100.0 * correct_cert / certified
            mean_dist = np.mean([e for r, e in zip(radii, errors) if r >= R])
        else:
            cond_acc = float("nan")
            mean_dist = float("nan")
        print(f"{R:>8.2f}  {abs_acc:>10.1f}  {cond_acc:>10.1f}  {mean_dist:>10.2f}  {certified:>8d}")

        metrics[f"AbsAcc_R{R:.2f}"] = round(abs_acc, 1)
        metrics[f"CondAcc_R{R:.2f}"] = round(cond_acc, 1) if not np.isnan(cond_acc) else None
        metrics[f"MeanDist_R{R:.2f}"] = round(mean_dist, 2) if not np.isnan(mean_dist) else None

    print("=" * 70)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_estimation", action="store_true",
                        help="Skip estimation step (use existing files)")
    parser.add_argument("--skip_radii", action="store_true",
                        help="Skip radii computation (use existing file)")
    parser.add_argument("--estimation_file", default="outputs/mnist_sigma0p75_estimation.json")
    parser.add_argument("--radii_file", default="outputs/mnist_ecg_radii_sigma0p75.json")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    if not args.skip_estimation:
        cmd = (
            f"python3 experiments/mnist_rotation/mnist_rotation_full_certification.py "
            f"--model_path {MODEL_PATH} --use_rotation_dataset "
            f"--sigma {SIGMA} --n_test {N_TEST} --N_values {N_SAMPLES} "
            f"--n_trials {N_TRIALS} --confidence {CONFIDENCE} "
            f"--device cuda --stratified --seed {SEED} --skip_bootstrap "
            f"--output {args.estimation_file}"
        )
        run_step(cmd, "Certification estimation")

    if not args.skip_radii:
        cmd = (
            f"python3 experiments/mnist_rotation/compute_ecg_radii_from_estimates.py "
            f"--estimation_file {args.estimation_file} "
            f"--eps_y_deg {EPS_Y_DEG} --N {N_SAMPLES} --trial 0 "
            f"--ci_type analytical --confidence {CONFIDENCE} "
            f"--output {args.radii_file}"
        )
        run_step(cmd, "Compute (E,C,G)+M radii")

    metrics = compute_metrics(args.estimation_file, args.radii_file)

    # Save summary
    summary = {
        "paper_id": 4232,
        "parameters": {
            "sigma": SIGMA, "eps_y_deg": EPS_Y_DEG, "N": N_SAMPLES,
            "n_test": N_TEST, "n_trials": N_TRIALS,
            "confidence": CONFIDENCE, "seed": SEED,
            "certification_mode": "(E,C,G)+M"
        },
        "metrics": metrics
    }
    with open("outputs/reproduction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to outputs/reproduction_summary.json")


if __name__ == "__main__":
    main()
