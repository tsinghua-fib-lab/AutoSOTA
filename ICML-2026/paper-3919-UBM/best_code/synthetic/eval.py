import pandas as pd
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default=None)
parser.add_argument("--bias_type", type=str, default="confounding_bias")
parser.add_argument("--u_type", type=str, default="disc")
parser.add_argument("--n_rct", type=int, default=50000)
parser.add_argument("--n_val", type=int, default=2000)
parser.add_argument("--d", type=int, default=6)
parser.add_argument("--pval", type=float, default=0.01)
args = parser.parse_args()

if args.results_dir:
    save_dir = Path(args.results_dir)
else:
    save_dir = Path(f"/repo/synthetic/results_U_{args.u_type}_ntrain-{args.n_rct}_nval-{args.n_val}/{args.bias_type}/d{args.d}")

df = pd.read_csv(save_dir / "results.csv")
n_trials = len(df)

print(f"Results from: {save_dir}")
print(f"Total trials: {n_trials}")
print()

results = {}
for key, label in [("SE_S", "rho_b1_S"), ("SE_A", "rho_b1_A"), ("SE_Y1", "rho_b1_Y")]:
    col_r = key + "_r"
    col_p = key + "_p"
    detected = len(df.query(f"`{col_r}` > 0 and `{col_p}` < {args.pval}"))
    detection_rate = detected / n_trials
    neg_detected = len(df.query(f"`{col_r}` < 0 and `{col_p}` < {args.pval}"))
    pos_sig = df.query(f"`{col_r}` > 0 and `{col_p}` < {args.pval}")
    avg_r = pos_sig[col_r].mean() if len(pos_sig) > 0 else float("nan")
    
    results[label] = {
        "detection_rate": detection_rate,
        "avg_r": avg_r,
        "n_detected": detected,
        "mean_r": df[col_r].mean(),
        "std_r": df[col_r].std(),
    }
    
    print(f"{label} Detection Rate: {detection_rate:.4f} ({detected}/{n_trials})")
    print(f"  Mean r: {df[col_r].mean():.4f} +/- {df[col_r].std():.4f}")

# For confounding bias under Figure 1b:
# - rho(b1,S) = 0 (no correlation expected) -> Specificity = 1 - false_positive_rate
# - rho(b1,A) > 0 (positive correlation expected) -> Detection Rate
# - rho(b1,Y) > 0 (positive correlation expected) -> Detection Rate

spec_s = 1.0 - results["rho_b1_S"]["detection_rate"]
det_a = results["rho_b1_A"]["detection_rate"]
det_y = results["rho_b1_Y"]["detection_rate"]

print()
print("=" * 60)
print("REPRODUCTION METRICS (Confounding Bias, nrct=50000, d=6)")
print("=" * 60)
print(f"rho(b1,S) Specificity:       {spec_s:.4f}  (paper: 0.99, CI: [0.9702, 1.0])")
print(f"rho(b1,A) Detection Rate:    {det_a:.4f}  (paper: 0.96, CI: [0.9408, 0.9792])")
print(f"rho(b1,Y) Detection Rate:    {det_y:.4f}  (paper: 0.95, CI: [0.931, 0.969])")
print("=" * 60)
