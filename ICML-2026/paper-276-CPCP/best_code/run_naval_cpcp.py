import os
import time
import warnings
import numpy as np
import pandas as pd

# Ensure working from /repo
os.chdir("/repo")

from utils import seed_everything, DEVICE
from data_utils import load_naval
from methods import run_rcp_density_improved, run_rcp
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

def rcp_protocol_split(X, Y, cal_size=0.2, seed=42):
    n_cal = int(len(X) * cal_size)
    X_rem, X_cal, Y_rem, Y_cal = train_test_split(X, Y, test_size=n_cal, random_state=seed)
    X_tr, X_te, Y_tr, Y_te = train_test_split(X_rem, Y_rem, test_size=0.25, random_state=seed)
    return X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te

def main():
    seed_everything(42)
    print(f"Using Device: {DEVICE}")
    alpha = 0.1
    n_seeds = 20

    # Load Naval data
    X, Y = load_naval("/repo/Datasets/")
    print(f"Naval Data Shape: X={X.shape}, Y={Y.shape}")

    methods = [
        ("RCP-Pinball", lambda *a: run_rcp(*a, "pinball")),
        ("CPCP-Clip+Mix-0.02", lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.03, mode="clip", clip_max=5.0, mix_ratio=0.5, **k)),
    ]

    results = {m[0]: [] for m in methods}
    total_start = time.time()

    for seed in range(n_seeds):
        seed_start = time.time()
        print(f"Seed {seed}...", end="", flush=True)
        X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te = rcp_protocol_split(X, Y, seed=42+seed)

        for name, func in methods:
            try:
                if "CPCP" in name:
                    res = func(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, dataset_name="naval", seed=seed)
                else:
                    res = func(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha)
                results[name].append(res)
            except Exception as e:
                print(f" Err({name}:{e})", end="")

        seed_dur = time.time() - seed_start
        total_elapsed = time.time() - total_start
        print(f" Done ({seed_dur/60:.2f}m seed | {total_elapsed/60:.2f}m total)")

    print("\n" + "="*80)
    print("NAVAL PROPULSION RESULTS")
    print("="*80)
    for name, mets in results.items():
        if not mets:
            print(f"{name}: No results")
            continue
        print(f"\n--- {name} ---")
        for k in ["Cov", "Size", "WSC", "MSCE_10", "MSCE_30", "L1-ERT", "L2-ERT"]:
            vals = [m[k] for m in mets]
            print(f"  {k}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")
        for k in ["Cov", "Size", "WSC", "MSCE_10", "MSCE_30", "L1-ERT", "L2-ERT"]:
            vals = [m[k] for m in mets]
            print(f"  {k}_raw: {vals}")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    summary_rows = []
    for name, mets in results.items():
        if not mets:
            continue
        row = {"Method": name}
        for k in ["Cov", "Size", "WSC", "MSCE_10", "MSCE_30", "L1-ERT", "L2-ERT"]:
            vals = [m[k] for m in mets]
            row[k] = f"{np.mean(vals):.6f} ± {np.std(vals):.6f}"
        summary_rows.append(row)
    df_res = pd.DataFrame(summary_rows)
    print("\nSummary Table:")
    print(df_res.to_string(index=False))
    df_res.to_csv("./results/naval_cpcp_results.csv", index=False)
    print("\nResults saved to ./results/naval_cpcp_results.csv")

if __name__ == "__main__":
    main()

# --- REPRODUCTION LOG ---
# Results from run at 2026-07-05
#
# Naval Propulsion (11934 samples, 16 features, 2 labels)
# Settings: alpha=0.1, n_seeds=20, data_split=6:2:2, calibrant_split=40:40:20
# CPCP settings: delta=0.02, mode=clip, clip_max=5.0, mix_ratio=0.5
# Nonconformity score: Linf norm of residuals
# WSC: M=1000, test_split=75:25
# ERT: logistic regression, 5-fold CV
# Volume: mean per-dimension log volume
#
# Results:
#   Metric        | CPCP-Clip+Mix            | Paper CPCP-Clip+Mix    | Within CI?
#   MSCE_10        | 0.001658 ± 0.001439      | 0.0019 ± 0.0009        | YES [0.0010, 0.0028]
#   WSC            | 0.829312 ± 0.053207      | 0.8320 ± 0.0304        | YES [0.8016, 0.8624]
#   L1-ERT         | 0.024541 ± 0.010984      | 0.0268 ± 0.0102        | YES [0.0166, 0.0370]
#   L2-ERT         | 0.000820 ± 0.000790      | 0.0012 ± 0.0010        | YES [0.0002, 0.0022]
#   Volume (Size)  | -0.881114 ± 0.270119     | -0.9101 ± 0.1810       | YES [-1.0911, -0.7291]
#
# RCP baseline:
#   MSCE_10: 0.003287 (paper 0.0029), WSC: 0.7793 (paper 0.8002)
#   L1-ERT: 0.0395 (paper 0.0353), L2-ERT: 0.0029 (paper 0.0023)
#   Volume: -0.9309 (paper -0.9301)
