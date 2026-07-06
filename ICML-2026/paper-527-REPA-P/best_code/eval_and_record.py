#!/usr/bin/env python3
"""Evaluate a checkpoint and record the score."""
import sys, os, csv, json, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
parser.add_argument("--step", type=int, required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--iter", required=True)
parser.add_argument("--idea-id", required=True)
parser.add_argument("--title", required=True)
parser.add_argument("--is-best", default=None)
args = parser.parse_args()

os.chdir("/repo")

# Run reconstruction eval
print("=== Reconstruction eval ===")
subprocess.run([
    "python3", "sample.py",
    "--name", args.run_name,
    "--step", str(args.step),
    "--gpu", str(args.gpu),
    "--mode", "reconstruction",
    "--use-ema",
    "--num-batches", "10"
], check=True)

# Run generative eval
print("=== Generative eval ===")
subprocess.run([
    "python3", "sample.py",
    "--name", args.run_name,
    "--step", str(args.step),
    "--gpu", str(args.gpu),
    "--mode", "generative",
    "--use-ema",
    "--num-samples", "20"
], check=True)

# Parse reconstruction metrics
mse_values = []
recon_path = "trained_models/{}/evaluation/step_{}/reconstruction_metrics.csv".format(args.run_name, args.step)
with open(recon_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["sample"] != "mean":
            mse_values.append(float(row["mse"]))
data_loss = sum(mse_values) / len(mse_values)

# Parse generative statistics
gen_path = "trained_models/{}/evaluation/step_{}/generative_sample_statistics.csv".format(args.run_name, args.step)
rmae_mean = 0.0
with open(gen_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["sample"] == "mean":
            rmae_mean = float(row["mean_abs_residual"])

metrics = {"Data Loss (MSE)": round(data_loss, 6), "Physics Residual MAE (RMAE)": round(rmae_mean, 6)}
metrics_json = json.dumps(metrics)

print("Data Loss (MSE) = {:.6f}".format(data_loss))
print("RMAE = {:.6f}".format(rmae_mean))

# Record score
cmd = [
    "/tools/record_score.sh",
    "--scores", "/autosota_artifacts/paper-527/sota/scores.jsonl",
    "--iter", str(args.iter),
    "--idea-id", args.idea_id,
    "--title", args.title,
    "--status", "success",
    "--primary", str(data_loss),
    "--metrics", metrics_json,
    "--notes", "Evaluated at step {}. Config: {}.".format(args.step, args.run_name),
]
if args.is_best:
    cmd.extend(["--is-best", args.is_best])

subprocess.run(cmd, check=True)
print("Recorded iter={} primary={:.6f}".format(args.iter, data_loss))
