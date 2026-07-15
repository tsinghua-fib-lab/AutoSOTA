"""
Error Asymmetry / Cognitive Interpretation (Day 2 CPU)
=======================================================
Addresses: Reviewer mxpA Q2 ("asymmetry of error types")

Input: data/results/rebuttal/llama8b_full_extraction.json
Output: TP/TN/FP/FN quadrant analysis, scatter figure

Usage:
    python scripts/rebuttal/error_asymmetry.py
"""

import os, sys, json, re
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

INPUT_FILE = "data/results/rebuttal/llama8b_full_extraction.json"
TARGET_LAYER = "layer_24"


def parse_source(f):
    if f.startswith('aime'): return 'AIME'
    elif f.startswith('amc'): return 'AMC'
    elif f.startswith('mathd'): return 'MathD'
    return 'Other'


def run_error_analysis():
    print("=" * 70)
    print("  ERROR ASYMMETRY / COGNITIVE INTERPRETATION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found."); sys.exit(1)

    os.makedirs("output/rebuttal", exist_ok=True)
    os.makedirs("data/results/rebuttal", exist_ok=True)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Build samples
    samples = []
    for entry in data:
        spec = entry.get("spectral", {})
        if isinstance(spec, dict) and "error" not in spec:
            s = spec.get(TARGET_LAYER, {})
            hfer = s.get("hfer")
            fiedler = s.get("fiedler")
            if hfer is not None:
                samples.append({
                    "file": entry["file"],
                    "true_label": 1 if entry["is_valid"] else 0,
                    "hfer": hfer, "fiedler": fiedler,
                    "source": parse_source(entry["file"]),
                    "length": entry.get("proof_token_count", 0),
                })

    hfer_vals = np.array([s["hfer"] for s in samples])
    labels = np.array([s["true_label"] for s in samples])

    # Find optimal threshold
    best_acc, best_thresh = 0, 0
    for t in np.percentile(hfer_vals, np.linspace(0, 100, 200)):
        acc = max(np.mean((hfer_vals < t) == labels), np.mean((hfer_vals > t) == labels))
        if acc > best_acc: best_acc, best_thresh = acc, t

    # Determine direction (lower HFER = valid or higher?)
    acc_lt = np.mean((hfer_vals < best_thresh) == labels)
    acc_gt = np.mean((hfer_vals > best_thresh) == labels)
    predict_valid_if_lt = acc_lt >= acc_gt

    print(f"Threshold: HFER {'<' if predict_valid_if_lt else '>'} {best_thresh:.4f} → Valid (acc={best_acc*100:.1f}%)")

    # Classify
    quadrants = {"TP": [], "TN": [], "FP": [], "FN": []}
    for s in samples:
        pred = 1 if (s["hfer"] < best_thresh if predict_valid_if_lt else s["hfer"] > best_thresh) else 0
        true = s["true_label"]
        q = "TP" if pred == 1 and true == 1 else "TN" if pred == 0 and true == 0 else \
            "FP" if pred == 1 and true == 0 else "FN"
        quadrants[q].append(s)

    cognitive = {"TP": "Correct Confidence", "TN": "Correct Skepticism",
                 "FP": "Confident Wrongness", "FN": "Effortful Correctness"}

    print(f"\n{'Quadrant':<25} | {'Count':>5} | {'%':>6} | {'Mean HFER':>10} | {'Mean Len':>8}")
    print("-" * 65)
    for q in ["TP", "TN", "FP", "FN"]:
        items = quadrants[q]
        n = len(items)
        pct = n / len(samples) * 100
        mh = np.mean([i["hfer"] for i in items]) if items else 0
        ml = np.mean([i["length"] for i in items]) if items else 0
        print(f"{cognitive[q]:<25} | {n:>5} | {pct:>5.1f}% | {mh:>10.4f} | {ml:>8.1f}")

    # Statistical test: FP vs TP proof length
    if quadrants["FP"] and quadrants["TP"]:
        fp_len = [i["length"] for i in quadrants["FP"]]
        tp_len = [i["length"] for i in quadrants["TP"]]
        if len(fp_len) > 1 and len(tp_len) > 1:
            u, p = stats.mannwhitneyu(fp_len, tp_len)
            print(f"\n  FP vs TP length: U={u:.0f}, p={p:.4e}")

    # Figure
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"TP": "#1565C0", "TN": "#C62828", "FP": "#FF9800", "FN": "#7B1FA2"}
    for q in ["TP", "TN", "FP", "FN"]:
        items = quadrants[q]
        if items:
            h = [i["hfer"] for i in items]
            f = [i.get("fiedler", 0) for i in items]
            ax.scatter(h, f, c=colors[q], label=f"{cognitive[q]} (n={len(items)})",
                      alpha=0.5, s=25)
    ax.axvline(x=best_thresh, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("HFER (L24)"); ax.set_ylabel("Fiedler Value (L24)")
    ax.set_title("Error Asymmetry: Cognitive Quadrants")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.savefig("output/rebuttal/error_quadrants.pdf")
    plt.savefig("output/rebuttal/error_quadrants.png")
    print(f"\nSaved: output/rebuttal/error_quadrants.pdf")

    # Save
    result = {q: {"count": len(v), "cognitive_label": cognitive[q],
                   "mean_hfer": round(float(np.mean([i["hfer"] for i in v])), 4) if v else None}
               for q, v in quadrants.items()}
    with open("data/results/rebuttal/error_asymmetry.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: data/results/rebuttal/error_asymmetry.json")


if __name__ == "__main__":
    run_error_analysis()
