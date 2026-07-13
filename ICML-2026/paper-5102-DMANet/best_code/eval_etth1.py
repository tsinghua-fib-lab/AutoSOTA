#!/usr/bin/env python3
"""DMANet ETTh1 evaluation script.
Trains and evaluates DMANet on ETTh1 for pred_len in {96, 192, 336, 720},
then reports the average MSE and MAE (the paper's primary metrics).

Paper settings from Appendix C Table 8:
  e_layers=1, down_sampling_layers=2, d_model=512, lr=2e-3,
  loss=frequency-domain MAE, batch_size=8, epochs=15,
  kernel_size=3, stride=2, channel_change_ratio=0.5, patience=3
"""

import os
import sys
import subprocess
import re

PRED_LENS = [96, 192, 336, 720]
BASE_ARGS = [
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--model", "DMANet",
    "--data", "ETTh1",
    "--root_path", "./all_datasets/ETT-small/",
    "--data_path", "ETTh1.csv",
    "--features", "M",
    "--seq_len", "96",
    "--label_len", "48",
    "--enc_in", "7",
    "--c_out", "7",
    "--d_model", "512",
    "--e_layers", "1",
    "--down_sampling_layers", "2",
    "--down_sampling_window", "2",
    "--down_sampling_c", "0.5",
    "--kernel_size", "5",
    "--d_ff", "2",
    "--learning_rate", "0.002",
    "--batch_size", "16",
    "--dropout", "0.05",
    "--train_epochs", "15",
    "--patience", "3",
    "--auxi_lambda", "1",
    "--auxi_loss", "MAE",
    "--lradj", "type1",
    "--num_workers", "0",
    "--itr", "1",
    "--seed", "2024",
    "--gpu", "0",
]

def run_pred_len(pred_len):
    """Run training + evaluation for a single pred_len. Returns (mse, mae)."""
    model_id = f"etth1_96_{pred_len}"

    args = [sys.executable, "-u", "run.py"] + BASE_ARGS + [
        "--model_id", model_id,
        "--pred_len", str(pred_len),
        "--des", "eval",
    ]

    # Clean previous state
    for d in ["checkpoints", "results", "test_results"]:
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)

    print(f"\n{'='*60}")
    print(f"Running pred_len={pred_len}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"

    result = subprocess.run(args, capture_output=True, text=True, env=env, cwd="/repo")
    output = result.stdout + "\n" + result.stderr

    # Extract metrics from the final test-line format: "96	| mse:0.374..., mae:0.396..."
    mse_match = re.search(r'mse:([0-9.]+).*mae:([0-9.]+)', output)

    if mse_match:
        mse = float(mse_match.group(1))
        mae = float(mse_match.group(2))
        print(f"pred_len={pred_len}: MSE={mse:.6f}, MAE={mae:.6f}")
        return mse, mae
    else:
        print(f"ERROR: Could not extract metrics for pred_len={pred_len}")
        print(f"Last 500 chars of output:\n{output[-500:]}")
        return None, None


def main():
    print("DMANet ETTh1 Evaluation")
    print("=" * 60)
    print(f"Prediction lengths: {PRED_LENS}")
    print(f"Paper target: MSE=0.428, MAE=0.429")
    print()

    mse_values = []
    mae_values = []

    for pred_len in PRED_LENS:
        mse, mae = run_pred_len(pred_len)
        if mse is not None:
            mse_values.append(mse)
            mae_values.append(mae)

    if len(mse_values) == 4:
        avg_mse = sum(mse_values) / 4
        avg_mae = sum(mae_values) / 4

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS (averaged over {PRED_LENS}):")
        print(f"  Average MSE: {avg_mse:.4f}")
        print(f"  Average MAE: {avg_mae:.4f}")
        print(f"  Paper MSE: 0.428")
        print(f"  Paper MAE: 0.429")
        print(f"{'='*60}")

        # Write results file for easy parsing
        with open("/repo/output/eval_results.json", "w") as f:
            import json
            json.dump({
                "mse": {str(p): v for p, v in zip(PRED_LENS, mse_values)},
                "mae": {str(p): v for p, v in zip(PRED_LENS, mae_values)},
                "avg_mse": round(avg_mse, 4),
                "avg_mae": round(avg_mae, 4),
                "paper_mse": 0.428,
                "paper_mae": 0.429,
            }, f, indent=2)
        print("\nResults saved to /repo/output/eval_results.json")
    else:
        print(f"\nERROR: Only {len(mse_values)}/4 pred_lens completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
