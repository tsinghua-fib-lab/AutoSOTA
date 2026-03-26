#!/usr/bin/env python3
"""eval.py for paper-3868: Latent Score-Based Reweighting for Robust Classification

Runs classification training + testing on ACS Income AZ dataset.
Outputs mean_accuracy and worst_group_accuracy (averaged over SEX and race attributes).

Prerequisites (already set up in autosota/paper-3868:reproduced):
- /data/my_stored_dataset/ACS_income_AZ/  (preprocessed ACS Income AZ data)
- /repo/tabsyn/vae/ckpt/ACS_income_AZ/   (trained VAE encoder/decoder)
- /repo/tabsyn/ckpt/ACS_income_AZ/       (trained score models for class 0 & 1)
- /repo/weights_discrete_10_32/ACS_income_AZ/ (pre-computed score weights)
- /repo/classification_process/ckpt/ACS_income_AZ/encoder.pt
"""

import os
import sys
import subprocess
import json
import glob
import re
from pathlib import Path

REPO_DIR = "/repo"
SEED = 42
DATANAME = "ACS_income_AZ"
METHOD = "several_sigma_error_diff"
COMMENT = "several_sigma_error_diff"
ATTRS = ["SEX", "race"]
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
LOG_BASE_DIR = f"./classification_process/discrete_log_{SEED}"


def fix_num_workers():
    """Ensure num_workers=0 to avoid socket errors during DataLoader init."""
    main_py = os.path.join(REPO_DIR, "classification_process/main.py")
    with open(main_py, "r") as f:
        content = f.read()
    if "num_workers = 4" in content:
        content = content.replace("num_workers = 4", "num_workers = 0")
        with open(main_py, "w") as f:
            f.write(content)
        print("[setup] Fixed num_workers: 4 → 0")
    else:
        print("[setup] num_workers already set to 0, OK")


def run_training():
    """Train classification model with score-based reweighting."""
    cmd = [
        "python", "./classification_process/main.py",
        "--dataname", DATANAME,
        "--log_path", f"./classification_process/discrete_log",
        "--method", METHOD,
        "--mode", "train",
        "--comment", COMMENT,
        "--seed", str(SEED),
        "--use_weight",
        "--weight_criterion", "several_timestep_error_diff",
        "--timestep_weight_criterion", "EDM",
        "--temperature", "3",
        "--error_reflection", "softmax",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_DIR
    env["CUDA_VISIBLE_DEVICES"] = GPU

    print(f"[train] Running classification training (seed={SEED})...")
    result = subprocess.run(
        cmd, cwd=REPO_DIR, env=env,
        capture_output=True, text=True, timeout=1800
    )
    # Output goes to log files; returncode may be 0 even on silent finish
    if result.returncode != 0:
        print(f"[train] STDERR: {result.stderr[-2000:]}")
    print(f"[train] Done (returncode={result.returncode})")


def find_best_model():
    """Find the most recently trained best model checkpoint."""
    # Directory is named with COMMENT before EDM suffix is appended
    pattern = os.path.join(
        REPO_DIR, LOG_BASE_DIR, DATANAME, "train",
        f"{COMMENT}_train_*", "saved_models", f"{METHOD}_best.pt"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No best model found. Searched: {pattern}\n"
            "Training may have failed or early-stopped without saving."
        )
    best = matches[-1]
    print(f"[test] Best model: {best}")
    return best


def run_test(best_model, attr):
    """Run evaluation for a specific sensitive attribute."""
    cmd = [
        "python", "./classification_process/main.py",
        "--dataname", DATANAME,
        "--log_path", f"./classification_process/discrete_log",
        "--method", METHOD,
        "--mode", "test",
        "--comment", COMMENT,
        "--seed", str(SEED),
        "--use_weight",
        "--weight_criterion", "several_timestep_error_diff",
        "--timestep_weight_criterion", "EDM",
        "--temperature", "3",
        "--error_reflection", "softmax",
        "--evaluated_model_path", best_model,
        "--eval_attribute", attr,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_DIR
    env["CUDA_VISIBLE_DEVICES"] = GPU

    print(f"[test] Testing attribute: {attr}")
    result = subprocess.run(
        cmd, cwd=REPO_DIR, env=env,
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"[test] STDERR: {result.stderr[-1000:]}")
    print(f"[test] Done {attr} (returncode={result.returncode})")


def parse_test_results(attr):
    """Parse Average Acc and Worst Acc from the test log file."""
    # Dir named with original COMMENT; file named with COMMENT_EDM_3.0 (timestep + temp appended)
    FILE_COMMENT = f"{COMMENT}_EDM_3.0"
    pattern = os.path.join(
        REPO_DIR, LOG_BASE_DIR, DATANAME, "test", attr,
        f"{COMMENT}_test_*", f"{FILE_COMMENT}.txt"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No test result file found for attr={attr}. Pattern: {pattern}"
        )
    result_file = matches[-1]
    print(f"[parse] Reading: {result_file}")

    with open(result_file, "r") as f:
        content = f.read()

    avg_acc, worst_acc = None, None
    for line in content.split("\n"):
        if "Average Acc:" in line:
            m = re.search(r"Average Acc:\s*([\d.]+)", line)
            if m:
                avg_acc = float(m.group(1))
        if "Worst Acc:" in line:
            m = re.search(r"Worst Acc:\s*([\d.]+)", line)
            if m:
                worst_acc = float(m.group(1))

    if avg_acc is None or worst_acc is None:
        print(f"[parse] WARNING: Could not parse results from {result_file}")
        print(f"[parse] File contents:\n{content}")
        raise ValueError(f"Parsing failed for attr={attr}")

    print(f"[parse] {attr}: Average Acc={avg_acc:.4f}, Worst Acc={worst_acc:.4f}")
    return avg_acc, worst_acc


def main():
    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)

    # Step 0: Fix num_workers issue
    fix_num_workers()

    # Step 1: Train classification model
    print("\n=== Step 1: Training Classification Model ===")
    run_training()

    # Step 2: Find best model
    best_model = find_best_model()

    # Step 3: Test on each attribute
    print("\n=== Step 2: Testing on Sensitive Attributes ===")
    avg_accs = []
    worst_accs = []

    for attr in ATTRS:
        run_test(best_model, attr)
        avg_acc, worst_acc = parse_test_results(attr)
        avg_accs.append(avg_acc)
        worst_accs.append(worst_acc)

    # Step 4: Compute averages across attributes (as per paper Table 2)
    mean_accuracy = sum(avg_accs) / len(avg_accs)
    worst_group_accuracy = sum(worst_accs) / len(worst_accs)

    # Step 5: Output results
    results = {
        "mean_accuracy": round(mean_accuracy * 100, 2),
        "worst_group_accuracy": round(worst_group_accuracy * 100, 2),
        "sex_avg_acc": round(avg_accs[0] * 100, 2),
        "sex_worst_acc": round(worst_accs[0] * 100, 2),
        "race_avg_acc": round(avg_accs[1] * 100, 2),
        "race_worst_acc": round(worst_accs[1] * 100, 2),
    }

    print(f"\n=== Final Results ===")
    print(f"Mean Accuracy (avg across SEX+race):        {mean_accuracy*100:.2f}%")
    print(f"Worst-Group Accuracy (avg across SEX+race): {worst_group_accuracy*100:.2f}%")

    # Write to results/scores.jsonl
    results_dir = "/repo/results"
    os.makedirs(results_dir, exist_ok=True)
    scores_path = os.path.join(results_dir, "scores.jsonl")
    with open(scores_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    print(f"\nResults written to {scores_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
