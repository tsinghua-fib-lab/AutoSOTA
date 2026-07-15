"""
Fixed Generalization Protocol (Day 2 CPU)
==========================================
Addresses: Reviewer 6GDW ("results reported on best configuration")

Input: data/results/rebuttal/llama8b_full_extraction.json
Apply HFER at L24 (75th percentile) with threshold calibrated on 50 examples.

Usage:
    python scripts/rebuttal/fixed_protocol.py
"""

import os, sys, json
import numpy as np

INPUT_FILE = "data/results/rebuttal/llama8b_full_extraction.json"

def run_fixed_protocol():
    print("=" * 70)
    print("  FIXED GENERALIZATION PROTOCOL (Llama-3.1-8B)")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run day1_llama8b_batch.py first.")
        sys.exit(1)

    os.makedirs("output/rebuttal", exist_ok=True)
    os.makedirs("data/results/rebuttal", exist_ok=True)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Extract HFER at L24 and L30
    samples_L24, samples_L30 = [], []
    for entry in data:
        spec = entry.get("spectral", {})
        if isinstance(spec, dict) and "error" not in spec:
            label = 1 if entry["is_valid"] else 0
            h24 = spec.get("layer_24", {}).get("hfer")
            h30 = spec.get("layer_30", {}).get("hfer")
            if h24 is not None: samples_L24.append((h24, label))
            if h30 is not None: samples_L30.append((h30, label))

    np.random.seed(42)
    N_CALIB = 50
    N_REPEATS = 100

    for layer_name, samples in [("L24 (fixed)", samples_L24), ("L30 (best)", samples_L30)]:
        values = np.array([s[0] for s in samples])
        labels = np.array([s[1] for s in samples])
        n = len(values)

        accs = []
        for _ in range(N_REPEATS):
            idx = np.random.permutation(n)
            calib_idx, test_idx = idx[:N_CALIB], idx[N_CALIB:]

            # Calibrate threshold
            calib_v, calib_l = values[calib_idx], labels[calib_idx]
            best_acc, best_t, best_dir = 0, 0, "lt"
            for t in np.percentile(calib_v, np.linspace(0, 100, 50)):
                acc_lt = np.mean((calib_v < t) == calib_l)
                acc_gt = np.mean((calib_v > t) == calib_l)
                if acc_lt > best_acc: best_acc, best_t, best_dir = acc_lt, t, "lt"
                if acc_gt > best_acc: best_acc, best_t, best_dir = acc_gt, t, "gt"

            # Test
            test_v, test_l = values[test_idx], labels[test_idx]
            pred = (test_v < best_t) if best_dir == "lt" else (test_v > best_t)
            accs.append(np.mean(pred == test_l))

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"\n  {layer_name}: {mean_acc*100:.1f}% ± {std_acc*100:.1f}% "
              f"(n={n}, {N_REPEATS} repeats, {N_CALIB} calibration)")

    # Paper comparison
    print(f"\n  Paper (HFER@L30, full search): 94.1%")
    gap = 0.941 - np.mean(accs)  # accs is from L30 loop, but we want gap from L24
    # Redo for L24
    values_24 = np.array([s[0] for s in samples_L24])
    labels_24 = np.array([s[1] for s in samples_L24])
    accs_24 = []
    for _ in range(N_REPEATS):
        idx = np.random.permutation(len(values_24))
        calib_idx, test_idx = idx[:N_CALIB], idx[N_CALIB:]
        calib_v, calib_l = values_24[calib_idx], labels_24[calib_idx]
        best_acc, best_t, best_dir = 0, 0, "lt"
        for t in np.percentile(calib_v, np.linspace(0, 100, 50)):
            acc_lt = np.mean((calib_v < t) == calib_l)
            acc_gt = np.mean((calib_v > t) == calib_l)
            if acc_lt > best_acc: best_acc, best_t, best_dir = acc_lt, t, "lt"
            if acc_gt > best_acc: best_acc, best_t, best_dir = acc_gt, t, "gt"
        test_v, test_l = values_24[test_idx], labels_24[test_idx]
        pred = (test_v < best_t) if best_dir == "lt" else (test_v > best_t)
        accs_24.append(np.mean(pred == test_l))

    gap_24 = 0.941 - np.mean(accs_24)
    print(f"  Gap (L24 fixed vs L30 best): {gap_24*100:.1f}%")
    print(f"  Search cost: 150 evaluations → 1 evaluation + {N_CALIB}-sample calibration")

    # Save
    result = {
        "fixed_L24_acc_mean": round(float(np.mean(accs_24)), 4),
        "fixed_L24_acc_std": round(float(np.std(accs_24)), 4),
        "best_L30_acc": 0.941,
        "gap": round(float(gap_24), 4),
        "n_calib": N_CALIB, "n_repeats": N_REPEATS,
    }
    with open("data/results/rebuttal/fixed_protocol.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: data/results/rebuttal/fixed_protocol.json")

    # LaTeX
    latex = [
        r"\begin{table}[t]",
        r"\caption{Fixed protocol (HFER@L24, 50-sample calibration) vs.\ per-model optimized.}",
        r"\label{tab:fixed_protocol}",
        r"\centering",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Configuration & Accuracy & Search Cost \\",
        r"\midrule",
        f"  Fixed (HFER@L24) & {np.mean(accs_24)*100:.1f}\\% $\\pm$ {np.std(accs_24)*100:.1f}\\% & 1 eval \\\\",
        f"  Best (HFER@L30) & 94.1\\% & 150 evals \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open("output/rebuttal/fixed_protocol_table.tex", "w") as f:
        f.write("\n".join(latex))
    print(f"  Saved: output/rebuttal/fixed_protocol_table.tex")

if __name__ == "__main__":
    run_fixed_protocol()
