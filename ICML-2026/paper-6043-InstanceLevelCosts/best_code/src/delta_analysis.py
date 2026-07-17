"""
delta_analysis.py

Dataset-level Delta (annotator agreement) analysis and regression baselines.

Functions performed:
  1. Load processed data (comment_text, y_star, abs_delta=|Δ|).
  2. Reconstruct signed Δ (negative = majority non-toxic, positive = toxic).
  3. Summary stats (mean/max |Δ|).
  4. Signed-Δ regression (TF-IDF -> Ridge) with L2 loss.
  5. Metrics: MAE, RMSE, weighted MAE/RMSE (weights = |Δ|), sign accuracy, weighted sign accuracy.
  6. Signed-Δ histogram (prof requested). Also optional abs-Δ histogram.
  7. Sample comments from 5 signed-Δ quantile ranges (CSV).
  8. Optional discretized Δ-as-class fallback (--disc K): multinomial LR, expected Δ̂.

Usage:
  python -m src.delta_analysis
  python -m src.delta_analysis --sample 300000
  python -m src.delta_analysis --disc 20

Outputs (results/delta/):
  abs_delta_stats.csv
  signed_delta_stats.csv
  signed_delta_reg_preds.parquet
  abs_delta_hist.png
  signed_delta_hist.png
  signed_delta_samples.csv
  (optional) signed_delta_disc_stats_K<k>.csv
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils import load_processed


# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------
def weighted_mae(y_true, y_pred, w):
    return float(np.sum(w * np.abs(y_pred - y_true)) / np.sum(w))

def weighted_rmse(y_true, y_pred, w):
    return float(np.sqrt(np.sum(w * (y_pred - y_true) ** 2) / np.sum(w)))

def weighted_acc(y_true_bin, y_pred_bin, w):
    return float(np.sum(w * (y_true_bin == y_pred_bin)) / np.sum(w))


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------
def plot_hist(values, xlabel, title, out_path, bins=100, vline0=False):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    if vline0:
        plt.axvline(0, color="k", lw=1)
    plt.xlabel(xlabel)
    plt.ylabel("Number of comments")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------------------
# Sample table across signed-Δ quantiles
# ------------------------------------------------------------------
def sample_signed_delta(df, out_path, n_per_bin=3, qcuts=(0,.2,.4,.6,.8,1.0)):
    qs = np.quantile(df['delta_signed'], qcuts)
    # enforce strictly increasing edges (tie jitter)
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i] = qs[i-1] + 1e-8
    df = df.copy()
    df['Delta_bin'] = np.digitize(df['delta_signed'], qs, right=True)

    rows = []
    for b in range(1, len(qs)):
        block = df[df['Delta_bin'] == b].head(n_per_bin)
        if block.empty:
            rows.append({
                "Delta_bin": b,
                "Delta_min": qs[b-1],
                "Delta_max": qs[b],
                "delta_signed": "",
                "comment_text": "<empty>"
            })
        else:
            for _, r in block.iterrows():
                rows.append({
                    "Delta_bin": b,
                    "Delta_min": qs[b-1],
                    "Delta_max": qs[b],
                    "delta_signed": r['delta_signed'],
                    "comment_text": str(r['comment_text']).replace("\n", " "),
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return qs


# ------------------------------------------------------------------
# Discretized Δ-as-class fallback
# ------------------------------------------------------------------
def discretized_delta_baseline(X_tr_txt, X_te_txt, d_tr, d_te, K=20, max_features=60_000):
    """Train a multinomial classifier on Δ bins; decode Δ̂ as expectation."""
    dmin, dmax = d_tr.min(), d_tr.max()
    edges = np.linspace(dmin, dmax, K + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def to_bin(x):
        return np.digitize(x, edges, right=True) - 1

    y_tr = np.clip(to_bin(d_tr), 0, K-1)
    y_te = np.clip(to_bin(d_te), 0, K-1)

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_tr = vec.fit_transform(X_tr_txt)
    X_te = vec.transform(X_te_txt)

    clf = LogisticRegression(max_iter=1000, multi_class="auto", solver="saga", n_jobs=-1)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)      # [N,K]
    delta_hat = proba.dot(centers)       # expectation over bins
    return delta_hat


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(args):
    out_dir = pathlib.Path("results/delta")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load processed & optionally subsample
    # --------------------------------------------------------------
    df = load_processed("data/jigsaw_cost.csv")
    if args.sample is not None and args.sample < len(df):
        df = df.sample(int(args.sample), random_state=42)

    # (load_processed builds signed Δ if missing)
    # rename for local convenience
    df = df.rename(columns={'abs_delta': 'abs_delta'})  # noop but explicit

    # --------------------------------------------------------------
    # Basic stats (abs Δ)
    # --------------------------------------------------------------
    abs_mean = df['abs_delta'].mean()
    abs_max  = df['abs_delta'].max()
    abs_stats = pd.DataFrame([
        ("Mean|Delta|", abs_mean),
        ("Max|Delta|",  abs_max),
    ], columns=["Statistic", "Value"])
    abs_stats.to_csv(out_dir / "abs_delta_stats.csv", index=False)

    # --------------------------------------------------------------
    # Train/test split for regression
    # --------------------------------------------------------------
    X = df['comment_text'].values
    y_signed = df['delta_signed'].values
    w = df['abs_delta'].values
    y_label = df['y_star'].values  # 0/1

    X_tr_txt, X_te_txt, y_tr, y_te, w_tr, w_te, lbl_tr, lbl_te = train_test_split(
        X, y_signed, w, y_label,
        test_size=0.2, random_state=42, stratify=y_label
    )

    # --------------------------------------------------------------
    # TF-IDF → Ridge regression (signed Δ)
    # --------------------------------------------------------------
    vec = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X_tr = vec.fit_transform(X_tr_txt)
    X_te = vec.transform(X_te_txt)

    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_tr)
    d_hat = reg.predict(X_te)

    # errors
    mae  = mean_absolute_error(y_te, d_hat)
    rmse = np.sqrt(mean_squared_error(y_te, d_hat))

    w_mae  = weighted_mae(y_te, d_hat, w_te)
    w_rmse = weighted_rmse(y_te, d_hat, w_te)

    # sign accuracy
    lbl_pred = (d_hat >= 0).astype(int)
    sign_acc  = (lbl_pred == lbl_te).mean()
    w_sign_acc= weighted_acc(lbl_te, lbl_pred, w_te)

    # signed stats table
    signed_stats = pd.DataFrame([
        ("MAE", mae),
        ("RMSE", rmse),
        ("Weighted_MAE", w_mae),
        ("Weighted_RMSE", w_rmse),
        ("Sign_Accuracy", sign_acc),
        ("Weighted_Sign_Accuracy", w_sign_acc),
    ], columns=["Metric", "Value"])
    signed_stats.to_csv(out_dir / "signed_delta_stats.csv", index=False)

    # save per-example predictions for further analysis
    te_out = pd.DataFrame({
        "comment_text": X_te_txt,
        "y_star": lbl_te,
        "abs_delta": w_te,
        "delta_signed": y_te,
        "delta_hat": d_hat,
    })
    # write CSV to avoid optional parquet dependency
    te_out.to_csv(out_dir / "signed_delta_reg_preds.csv", index=False)

    # --------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------
    if not args.no_abs_hist:
        plot_hist(
            df['abs_delta'],
            xlabel="|Δ|  (annotator agreement log-ratio)",
            title="Distribution of |Δ|",
            out_path=out_dir / "abs_delta_hist.png",
            bins=50
        )

    plot_hist(
        df['delta_signed'],
        xlabel="Δ (signed: toxic > 0, non-toxic < 0)",
        title="Distribution of signed Δ",
        out_path=out_dir / "signed_delta_hist.png",
        bins=100,
        vline0=True
    )

    # --------------------------------------------------------------
    # Signed-Δ sample table
    # --------------------------------------------------------------
    sample_signed_delta(df, out_dir / "signed_delta_samples.csv", n_per_bin=3)

    # --------------------------------------------------------------
    # Optional discretized Δ baseline
    # --------------------------------------------------------------
    if args.disc is not None and args.disc > 1:
        d_hat_disc = discretized_delta_baseline(
            X_tr_txt, X_te_txt, y_tr, y_te, K=args.disc, max_features=args.max_features
        )
        mae_d  = mean_absolute_error(y_te, d_hat_disc)
        rmse_d = np.sqrt(mean_squared_error(y_te, d_hat_disc))
        w_mae_d  = weighted_mae(y_te, d_hat_disc, w_te)
        w_rmse_d = weighted_rmse(y_te, d_hat_disc, w_te)
        lbl_pred_d = (d_hat_disc >= 0).astype(int)
        sign_acc_d  = (lbl_pred_d == lbl_te).mean()
        w_sign_acc_d= weighted_acc(lbl_te, lbl_pred_d, w_te)

        pd.DataFrame([
            ("MAE", mae_d),
            ("RMSE", rmse_d),
            ("Weighted_MAE", w_mae_d),
            ("Weighted_RMSE", w_rmse_d),
            ("Sign_Accuracy", sign_acc_d),
            ("Weighted_Sign_Accuracy", w_sign_acc_d),
        ], columns=["Metric","Value"]).to_csv(
            out_dir / f"signed_delta_disc_stats_K{args.disc}.csv", index=False
        )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------
    print("\n=== |Δ| stats ===")
    print(abs_stats.to_string(index=False))
    print("\n=== signed Δ regression metrics ===")
    print(signed_stats.to_string(index=False))
    if args.disc:
        print(f"\n(Δ discretized K={args.disc} metrics written)")

    print(f"\nArtifacts written to: {out_dir.resolve()}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=None,
                    help="Rows to sample BEFORE split (for a quick run).")
    ap.add_argument("--max_features", type=int, default=60_000,
                    help="Max TF-IDF vocab size.")
    ap.add_argument("--disc", type=int, default=None,
                    help="If set (>1), run discretized Δ-as-class baseline with K bins.")
    ap.add_argument("--no_abs_hist", action="store_true",
                    help="Skip |Δ| histogram (saves time).")
    args = ap.parse_args()
    main(args)

