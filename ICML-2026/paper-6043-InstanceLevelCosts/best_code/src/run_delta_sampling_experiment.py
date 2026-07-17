# src/run_delta_sampling_experiment.py
import argparse, pathlib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

def weighted_acc(y_true, y_pred, w):
    w = np.asarray(w, dtype=float)
    return float((w * (y_true == y_pred)).sum() / w.sum())

def weighted_mae(y_true, y_pred, w):
    w = np.asarray(w, dtype=float)
    return float(np.abs(y_true - y_pred).dot(w) / w.sum())

def weighted_rmse(y_true, y_pred, w):
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(((y_true - y_pred)**2).dot(w) / w.sum()))

def _sanitize(df, name):
    """Drop bad rows; ensure required cols; coerce text to str; ensure weight."""
    need = ["comment_text", "y_star", "delta"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise AssertionError(f"{name} missing columns: {missing}")
    # Drop rows with NaN in required fields
    before = len(df)
    df = df.dropna(subset=need).copy()
    # Ensure text is string (avoid np.nan errors in TfidfVectorizer)
    df["comment_text"] = df["comment_text"].astype(str)
    # Ensure weight exists and is non-negative
    if "weight" not in df.columns:
        df["weight"] = df["delta"].abs()
    else:
        # fill NaN weights with |delta|
        df["weight"] = df["weight"].fillna(df["delta"].abs()).abs()
    dropped = before - len(df)
    if dropped:
        print(f"[sanitize] Dropped {dropped} rows from {name} due to NaNs")
    return df

def load_split(variant_dir, ext="csv"):
    read = pd.read_csv if ext == "csv" else pd.read_parquet
    tr = read(variant_dir / f"train.{ext}")
    va = read(variant_dir / f"val.{ext}")
    te = read(variant_dir / f"test.{ext}")
    tr = _sanitize(tr, f"{variant_dir.name}/train")
    va = _sanitize(va, f"{variant_dir.name}/val")
    te = _sanitize(te, f"{variant_dir.name}/test")
    return tr, va, te

def run_variant(variant_name, base_dir, out_dir, max_features=60000, reg_alpha=1.0, ext="csv"):
    variant_dir = base_dir / variant_name
    tr, va, te = load_split(variant_dir, ext=ext)

    Xtr, ytr, dtr, wtr = tr["comment_text"].values, tr["y_star"].values, tr["delta"].values, tr["weight"].values
    Xva, yva, dva, wva = va["comment_text"].values, va["y_star"].values, va["delta"].values, va["weight"].values
    Xte, yte, dte, wte = te["comment_text"].values, te["y_star"].values, te["delta"].values, te["weight"].values

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    Xtr_vec = vec.fit_transform(Xtr)
    Xva_vec = vec.transform(Xva)
    Xte_vec = vec.transform(Xte)

    results = []

    # Classification (unweighted train)
    clf_unw = LogisticRegression(max_iter=1000, solver="saga", penalty="l2", n_jobs=-1)
    clf_unw.fit(Xtr_vec, ytr)
    predA = clf_unw.predict(Xte_vec)
    results.append(("cls_unweighted_acc", float(accuracy_score(yte, predA))))
    results.append(("cls_weighted_acc",   float(weighted_acc(yte, predA, wte))))

    # Classification (weighted train) — control
    clf_w = LogisticRegression(max_iter=1000, solver="saga", penalty="l2", n_jobs=-1)
    clf_w.fit(Xtr_vec, ytr, sample_weight=wtr)
    predB = clf_w.predict(Xte_vec)
    results.append(("cls_unweighted_acc_weighted_train", float(accuracy_score(yte, predB))))
    results.append(("cls_weighted_acc_weighted_train",   float(weighted_acc(yte, predB, wte))))

    # |Δ| vs misclassification corr (on unweighted-train model)
    mis = (predA != yte).astype(float)
    abs_delta = np.abs(dte)
    if abs_delta.std() > 0 and mis.std() > 0:
        corr = float(np.corrcoef(abs_delta, mis)[0,1])
    else:
        corr = float("nan")
    results.append(("corr_absdelta_misclass", corr))

    # Regression: delta (signed)
    reg = Ridge(alpha=reg_alpha, random_state=0)
    reg.fit(Xtr_vec, dtr)
    d_hat = reg.predict(Xte_vec)

    mae  = mean_absolute_error(dte, d_hat)
    mse  = mean_squared_error(dte, d_hat)
    rmse = float(np.sqrt(mse))
    wmae = weighted_mae(dte, d_hat, wte)
    wrmse= weighted_rmse(dte, d_hat, wte)

    results.extend([
        ("reg_mae", mae), ("reg_rmse", rmse),
        ("reg_weighted_mae", wmae), ("reg_weighted_rmse", wrmse)
    ])

    sign_pred = (d_hat >= 0).astype(int)
    results.append(("reg_sign_acc",           float(accuracy_score(yte, sign_pred))))
    results.append(("reg_sign_weighted_acc",  float(weighted_acc(yte, sign_pred, wte))))

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"variant": variant_name, "metric": k, "value": v} for (k,v) in results]
    pd.DataFrame(rows).to_csv(out_dir / f"results_{variant_name}.csv", index=False)
    return {k: v for (k, v) in results}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="data/jigsaw")
    ap.add_argument("--variants", nargs="+", default=["U","P_up","Tdown50","Tdown30","Tdown70"])
    ap.add_argument("--ext", choices=["csv","parquet"], default="csv")
    ap.add_argument("--max_features", type=int, default=60000)
    ap.add_argument("--reg_alpha", type=float, default=1.0)
    ap.add_argument("--out_dir", default="results/sampling_runs")
    args = ap.parse_args()

    base_dir = pathlib.Path(args.base_dir)
    out_dir  = pathlib.Path(args.out_dir)

    all_rows = []
    for v in args.variants:
        print(f"[RUN] variant={v}")
        res = run_variant(v, base_dir, out_dir, max_features=args.max_features, reg_alpha=args.reg_alpha, ext=args.ext)
        for k, val in res.items():
            all_rows.append({"variant": v, "metric": k, "value": val})

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_dir / "sampling_summary.csv", index=False)
    pivot = summary.pivot(index="metric", columns="variant", values="value")
    pivot.to_csv(out_dir / "sampling_summary_pivot.csv")
    print("[OK] wrote:", out_dir / "sampling_summary.csv", "and", out_dir / "sampling_summary_pivot.csv")

if __name__ == "__main__":
    main()

