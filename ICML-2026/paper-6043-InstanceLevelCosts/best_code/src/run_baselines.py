"""
Train TF-IDF Logistic baselines (A–D) and save metrics table.
Usage: python -m src.run_baselines
"""
import numpy as np, pandas as pd, pathlib, argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils import load_processed

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

def weighted_acc(y_true, y_pred, w):
    return float((w * (y_true == y_pred)).sum() / w.sum())

def train_logreg_matrix(X_vec, y, sample_weight=None):
    clf = LogisticRegression(max_iter=1000, solver="saga", penalty="l2", n_jobs=-1)
    clf.fit(X_vec, y, sample_weight=sample_weight)
    return clf

def main(args):
    df = load_processed("data/jigsaw_cost.csv")

    # optional subsample
    if args.sample is not None and args.sample < len(df):
        df = df.sample(int(args.sample), random_state=42)

    X = df['comment_text'].values
    y = df['y_star'].values
    w = df['abs_delta'].values

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, w, test_size=0.2, stratify=y, random_state=42)

    # one shared TF-IDF vectorizer
    vec = TfidfVectorizer(max_features=args.max_features, ngram_range=(1,2))
    X_tr = vec.fit_transform(X_tr)
    X_te = vec.transform(X_te)

    # A baseline
    mdl_A = train_logreg_matrix(X_tr, y_tr, sample_weight=None)
    pred_A = mdl_A.predict(X_te)

    # B weighted
    mdl_B = train_logreg_matrix(X_tr, y_tr, sample_weight=w_tr)
    pred_B = mdl_B.predict(X_te)

    # D high-Δ
    mask_hi = w_tr >= np.median(w_tr)
    mdl_D = train_logreg_matrix(X_tr[mask_hi], y_tr[mask_hi], sample_weight=None)
    pred_D = mdl_D.predict(X_te)

    rows = [
        ("A_unw_unw", "none", "full",      "unweighted_acc", accuracy_score(y_te, pred_A)),
        ("B_w_unw",   "|Δ|",  "full",      "unweighted_acc", accuracy_score(y_te, pred_B)),
        ("B_w_w",     "|Δ|",  "full",      "weighted_acc",   weighted_acc(y_te, pred_B, w_te)),
        ("C_unw_w",   "none", "full",      "weighted_acc",   weighted_acc(y_te, pred_A, w_te)),
        ("D_hi_unw",  "none", "highΔ50%",  "unweighted_acc", accuracy_score(y_te, pred_D)),
        ("D_hi_w",    "none", "highΔ50%",  "weighted_acc",   weighted_acc(y_te, pred_D, w_te)),
    ]
    pd.DataFrame(rows, columns=["Model","Weighting","Train_subset","Metric","Score"])\
      .to_csv(RESULTS / "classification_metrics.csv", index=False)
    print("[+] wrote results/classification_metrics.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=None,
                    help="Rows to sample before split (quick smoke test).")
    ap.add_argument("--max_features", type=int, default=60_000)
    args = ap.parse_args()
    main(args)

