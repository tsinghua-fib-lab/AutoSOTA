"""
Full Baseline Analysis (Day 2 CPU)
====================================
Addresses: Reviewer 6GDW ("compare to other self-supervised signals"),
Reviewer mxpA Q1 ("Is HFER just a proxy for entropy?")

Input: data/results/rebuttal/llama8b_full_extraction.json
Output: comparison table, correlation analysis, ROC curves, figures

Usage:
    python scripts/rebuttal/analyze_baselines.py
"""

import os, sys, json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})


INPUT_FILE = "data/results/rebuttal/llama8b_full_extraction.json"
OUT_DIR = "output/rebuttal"
RESULT_DIR = "data/results/rebuttal"
TARGET_LAYER = "layer_24"  # 75th percentile
BEST_LAYER = "layer_30"    # best from paper


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 0: return 0
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)


def optimal_threshold_acc(values, labels):
    vals, labs = np.array(values), np.array(labels)
    thresholds = np.percentile(vals, np.linspace(0, 100, 200))
    best = 0
    for t in thresholds:
        best = max(best, np.mean((vals < t) == labs), np.mean((vals > t) == labs))
    return best


def run_analysis():
    print("=" * 70)
    print("  FULL BASELINE COMPARISON (Llama-3.1-8B)")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        print("Run day1_llama8b_batch.py first.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} proofs")

    # Build aligned feature vectors
    signals = {}
    labels = []
    filenames = []

    for entry in data:
        if isinstance(entry.get("spectral"), dict) and "error" not in entry["spectral"]:
            if isinstance(entry.get("token_baselines"), dict) and "error" not in entry["token_baselines"]:
                # Spectral signals at best layer
                spec_best = entry["spectral"].get(BEST_LAYER, {})
                spec_75 = entry["spectral"].get(TARGET_LAYER, {})
                tok = entry["token_baselines"]

                if spec_best.get("hfer") is None:
                    continue

                row = {
                    "hfer_L30": spec_best.get("hfer"),
                    "hfer_L24": spec_75.get("hfer"),
                    "fiedler_L30": spec_best.get("fiedler"),
                    "smoothness_L30": spec_best.get("smoothness"),
                    "spectral_entropy_L30": spec_best.get("entropy"),
                    "mean_logprob": tok.get("mean_logprob"),
                    "mean_token_entropy": tok.get("mean_entropy"),
                    "max_token_entropy": tok.get("max_entropy"),
                    "var_token_entropy": tok.get("var_entropy"),
                    "perplexity": tok.get("perplexity"),
                    "proof_length": entry.get("proof_token_count", 0),
                }

                # Skip if any key value is None
                if any(v is None for v in row.values()):
                    continue

                for k, v in row.items():
                    signals.setdefault(k, []).append(v)
                labels.append(1 if entry["is_valid"] else 0)
                filenames.append(entry["file"])

    labels = np.array(labels)
    n_valid = sum(labels)
    n_invalid = len(labels) - n_valid
    print(f"Aligned: {len(labels)} proofs ({n_valid} valid, {n_invalid} invalid)")

    # ── Per-signal analysis ──
    print(f"\n{'Signal':<25} | {'d':>6} | {'AUC':>5} | {'Acc':>6}")
    print("-" * 55)

    all_results = []
    for name in ["hfer_L30", "hfer_L24", "fiedler_L30", "smoothness_L30",
                  "spectral_entropy_L30", "mean_logprob", "mean_token_entropy",
                  "max_token_entropy", "var_token_entropy", "perplexity", "proof_length"]:
        vals = np.array(signals[name])
        valid_v = vals[labels == 1]
        invalid_v = vals[labels == 0]
        d = cohen_d(valid_v, invalid_v)

        # AUC (flip sign for signals where lower = valid)
        try:
            auc = roc_auc_score(labels, -vals if "hfer" in name else vals)
            if auc < 0.5: auc = 1 - auc
        except:
            auc = 0.5

        acc = optimal_threshold_acc(vals.tolist(), labels.tolist())
        print(f"{name:<25} | {d:>6.2f} | {auc:>5.3f} | {acc*100:>5.1f}%")
        all_results.append({
            "signal": name, "cohen_d": round(d, 3), "auc": round(auc, 3),
            "accuracy": round(acc, 4), "valid_mean": round(float(np.mean(valid_v)), 4),
            "invalid_mean": round(float(np.mean(invalid_v)), 4)
        })

    # ── Correlation matrix ──
    print("\n--- CORRELATIONS (HFER_L30 vs token baselines) ---")
    hfer = np.array(signals["hfer_L30"])
    corr_results = {}
    for name in ["mean_logprob", "mean_token_entropy", "perplexity", "proof_length"]:
        vals = np.array(signals[name])
        r_pearson, p_pearson = stats.pearsonr(hfer, vals)
        r_spearman, p_spearman = stats.spearmanr(hfer, vals)
        print(f"  HFER vs {name}: Pearson r={r_pearson:.3f} (p={p_pearson:.2e}), "
              f"Spearman ρ={r_spearman:.3f}")
        corr_results[name] = {
            "pearson_r": round(r_pearson, 4), "pearson_p": float(p_pearson),
            "spearman_r": round(r_spearman, 4), "spearman_p": float(p_spearman)
        }

    # ── Partial correlation ──
    print("\n--- PARTIAL CORRELATION (HFER → validity, controlling for token entropy) ---")
    from numpy.linalg import lstsq
    tok_ent = np.array(signals["mean_token_entropy"])
    # Residualize HFER on token entropy
    A = np.column_stack([tok_ent, np.ones(len(tok_ent))])
    beta, _, _, _ = lstsq(A, hfer, rcond=None)
    hfer_resid = hfer - A @ beta
    # Correlate residualized HFER with labels
    r_partial, p_partial = stats.pointbiserialr(labels, hfer_resid)
    print(f"  Partial r = {r_partial:.3f} (p = {p_partial:.2e})")
    print(f"  → HFER retains {'SIGNIFICANT' if p_partial < 0.05 else 'no significant'} "
          f"predictive power after controlling for token entropy")

    # ── Combined classifiers ──
    print("\n--- COMBINED CLASSIFIERS ---")
    X_hfer = np.array(signals["hfer_L30"]).reshape(-1, 1)
    X_tok = np.column_stack([signals[k] for k in ["mean_logprob", "mean_token_entropy", "perplexity"]])
    X_all = np.column_stack([X_hfer, X_tok])

    scaler = StandardScaler()
    for name, X in [("HFER alone", X_hfer), ("Token baselines only", X_tok),
                     ("HFER + token baselines", X_all)]:
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xs, labels)
        acc = clf.score(Xs, labels)
        print(f"  {name}: {acc*100:.1f}%")

    # ── Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) ROC curves
    for name, color, ls in [("hfer_L30", "#1565C0", "-"), ("mean_token_entropy", "#C62828", "--"),
                             ("perplexity", "#4CAF50", ":"), ("proof_length", "#FF9800", "-.")]:
        vals = np.array(signals[name])
        y_score = -vals if "hfer" in name else vals
        fpr, tpr, _ = roc_curve(labels, y_score)
        auc = roc_auc_score(labels, y_score)
        if auc < 0.5:
            fpr, tpr, _ = roc_curve(labels, -y_score)
            auc = 1 - auc
        axes[0].plot(fpr, tpr, color=color, linestyle=ls, label=f"{name} ({auc:.2f})", linewidth=2)
    axes[0].plot([0,1], [0,1], 'k--', alpha=0.3)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC Curves")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.2)

    # (b) Scatter: HFER vs token entropy
    c = ['#1565C0' if l == 1 else '#C62828' for l in labels]
    axes[1].scatter(signals["mean_token_entropy"], signals["hfer_L30"], c=c, alpha=0.4, s=12)
    axes[1].set_xlabel("Mean Token Entropy"); axes[1].set_ylabel("HFER (L30)")
    axes[1].set_title(f"HFER vs Token Entropy (r={corr_results['mean_token_entropy']['pearson_r']:.2f})")
    axes[1].grid(True, alpha=0.2)

    # (c) Bar chart: Cohen's d for each signal
    names = [r["signal"] for r in all_results]
    ds = [abs(r["cohen_d"]) for r in all_results]
    colors = ['#1565C0' if "hfer" in n or "fiedler" in n or "smooth" in n or "spectral" in n
              else '#FF9800' for n in names]
    axes[2].barh(range(len(names)), ds, color=colors)
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    axes[2].set_xlabel("|Cohen's d|"); axes[2].set_title("Effect Size Comparison")
    axes[2].grid(True, alpha=0.2, axis='x')
    axes[2].axvline(x=0.8, color='gray', linestyle='--', alpha=0.4, label="large effect")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "baseline_comparison.pdf")
    plt.savefig(fig_path); plt.savefig(fig_path.replace('.pdf', '.png'))
    print(f"\nSaved: {fig_path}")

    # ── Save results ──
    save_data = {
        "n_proofs": len(labels), "n_valid": int(n_valid), "n_invalid": int(n_invalid),
        "per_signal": all_results,
        "correlations_with_hfer": corr_results,
        "partial_correlation": {"r": round(r_partial, 4), "p": float(p_partial)},
    }
    json_path = os.path.join(RESULT_DIR, "baseline_comparison.json")
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {json_path}")

    # ── LaTeX table ──
    latex = [
        r"\begin{table}[t]",
        r"\caption{Spectral vs.\ token-level baselines (Llama-3.1-8B, corrected labels).}",
        r"\label{tab:baselines}",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Signal & $|d|$ & AUC & Acc \\",
        r"\midrule",
    ]
    for r in all_results:
        latex.append(f"  {r['signal'].replace('_', ' ')} & {abs(r['cohen_d']):.2f} & "
                    f"{r['auc']:.3f} & {r['accuracy']*100:.1f}\\% \\\\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path = os.path.join(OUT_DIR, "baseline_comparison.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    run_analysis()
