#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, auc
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROW_ORDER = ["normal", "advprompter", "autodan", "gcg"]

def _normalize_algo_col(df: pd.DataFrame) -> pd.Series:
    alg = df.get("algorithm", pd.Series(index=df.index, dtype=object)).astype(str).str.lower().str.strip()
    alg = alg.replace({"nan":"", "none":"", "": np.nan})
    is_adv = df["is_adversarial"].fillna(0).astype(int)
    out = np.where(is_adv==0, "normal", alg)
    return pd.Series(out, index=df.index).replace({"auto-dan":"autodan", "adv-prompter":"advprompter"})

def _subset(df: pd.DataFrame, eval_type: str) -> pd.DataFrame:
    if eval_type == "pooled": return df.copy()
    if eval_type == "pooled_no_gcg": return df[df["algorithm"]!="gcg"].copy()
    if eval_type.startswith("algo_"):
        algo = eval_type.split("algo_",1)[1].lower()
        return df[df["algorithm"].isin(["normal", algo])].copy()
    raise ValueError(f"Unknown eval_type {eval_type}")

def _balance_normals(sub: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    if mode=="none": return sub
    normals = sub[sub.is_adversarial==0]
    advs    = sub[sub.is_adversarial==1]
    if normals.empty or advs.empty: return sub
    if mode=="match_adv_total":
        k = min(len(normals), len(advs))
        return pd.concat([normals.sample(n=k, random_state=seed), advs], ignore_index=False).sort_index()
    return sub

def _metrics_from_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "threshold": float(thr),
        "n": int(len(y)),
        "acc": float((tp+tn)/len(y)),
        "precision_0": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_0": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "precision_1": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "fpr": float(fp / (fp+tn+1e-12)),
        "tpr": float(tp / (tp+fn+1e-12)),
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y))>1 else float("nan"),
        "pr_auc": float(auc(*precision_recall_curve(y, s)[1::-1])) if len(np.unique(y))>1 else float("nan"),
    }

def _pick_threshold_on(scores: np.ndarray, labels: np.ndarray, criterion: str, target_fpr: float) -> float:
    """Pick threshold on provided set (higher score => more adversarial)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    if len(np.unique(y)) < 2:
        return float("nan")

    if criterion == "youden":
        fpr, tpr, thr = roc_curve(y, s)
        return float(thr[int(np.argmax(tpr - fpr))])
    elif criterion == "fixed_fpr":
        fpr, tpr, thr = roc_curve(y, s)
        mask = fpr <= target_fpr
        best_idx = (np.where(mask)[0][np.argmax(tpr[mask])] if mask.any() else int(np.argmin(fpr)))
        return float(thr[best_idx])
    elif criterion == "f1":
        prec, rec, thr = precision_recall_curve(y, s)  # thr len = len(prec)-1
        f1 = 2*prec[:-1]*rec[:-1] / (prec[:-1]+rec[:-1]+1e-12)
        return float(thr[int(np.nanargmax(f1))])
    else:
        raise ValueError("criterion must be one of: youden, fixed_fpr, f1")

def _cv_recommend_threshold(
    s: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
    criterion: str,
    target_fpr: float,
    stratify: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Stratified CV: pick thr on train, eval on val.

    Returns: (recommended_thr, mean_val_metrics, std_val_metrics, fold_rows)
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    strat = np.asarray(stratify) if stratify is not None else y

    # Effective folds: cannot exceed min class count of the stratification labels
    strat_counts = pd.Series(strat).value_counts(dropna=False)
    min_strat_class = int(strat_counts.min()) if not strat_counts.empty else 0

    if min_strat_class < 2 or len(np.unique(y)) < 2:
        # No CV possible; fall back to single split on all data
        thr = _pick_threshold_on(s, y, criterion, target_fpr)
        metrics = _metrics_from_threshold(s, y, thr) if np.isfinite(thr) else {"threshold": float("nan")}
        return thr, metrics, {}, [{"fold":"na", "split":"all", **metrics}]

    n_splits = max(2, min(folds, min_strat_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows = []
    train_thrs = []
    for k, (tr, va) in enumerate(skf.split(s, strat), start=1):
        thr_k = _pick_threshold_on(s[tr], y[tr], criterion, target_fpr)
        train_thrs.append(thr_k)
        # eval on val
        val_metrics = _metrics_from_threshold(s[va], y[va], thr_k)
        val_metrics.update({"fold": k, "split": "val"})
        fold_rows.append(val_metrics)

    # Recommend median of train thresholds (robust)
    train_thrs_arr = np.array(train_thrs, dtype=float)
    rec_thr = float(np.nanmedian(train_thrs_arr))

    # Mean/std validation metrics across folds (for the chosen rec_thr we also compute on full set)
    val_keys = [k for k in fold_rows[0] if k not in ("fold", "split")]
    val_means = {k: float(np.nanmean([fr[k] for fr in fold_rows if k in fr])) for k in val_keys}
    val_stds = {k: float(np.nanstd([fr[k] for fr in fold_rows if k in fr])) for k in val_keys}

    overall_metrics = _metrics_from_threshold(s, y, rec_thr)
    overall_metrics.update({"fold": "recommended", "split": "all"})

    return rec_thr, val_means, val_stds, fold_rows + [overall_metrics]

def _cv_logistic(
    X: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
    stratify: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Stratified CV with LogisticRegression on multiple features.

    Returns mean metrics across folds.
    """
    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)
    strat = np.asarray(stratify) if stratify is not None else y

    # Remove rows with NaN features
    finite_mask = np.all(np.isfinite(X), axis=1)
    X, y = X[finite_mask], y[finite_mask]
    strat = strat[finite_mask]

    if len(np.unique(y)) < 2 or len(y) < 10:
        return {"f1_1": float("nan"), "roc_auc": float("nan"), "n": len(y),
                "cv_val_mean_f1_1": float("nan"), "cv_val_mean_roc_auc": float("nan"),
                "cv_val_std_f1_1": float("nan"), "cv_val_std_roc_auc": float("nan")}

    strat_counts = pd.Series(strat).value_counts(dropna=False)
    min_strat_class = int(strat_counts.min()) if not strat_counts.empty else 0
    n_splits = max(2, min(folds, min_strat_class, int((strat == 0).sum()), int((strat == 1).sum())))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_f1s, fold_aurocs = [], []
    all_y_true, all_y_pred, all_y_score = [], [], []

    for tr, va in skf.split(X, strat):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])

        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
        clf.fit(X_tr, y[tr])
        y_pred = clf.predict(X_va)
        y_score = clf.predict_proba(X_va)[:, 1]

        fold_f1s.append(f1_score(y[va], y_pred, pos_label=1, zero_division=0))
        fold_aurocs.append(roc_auc_score(y[va], y_score) if len(np.unique(y[va])) > 1 else float("nan"))

        all_y_true.extend(y[va].tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_score.extend(y_score.tolist())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)

    return {
        "f1_1": float(f1_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_y_true, all_y_score)) if len(np.unique(all_y_true)) > 1 else float("nan"),
        "tpr": float(recall_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)),
        "fpr": float(1.0 - recall_score(all_y_true, all_y_pred, pos_label=0, zero_division=0)),
        "n": len(all_y_true),
        "cv_val_mean_f1_1": float(np.nanmean(fold_f1s)),
        "cv_val_mean_roc_auc": float(np.nanmean(fold_aurocs)),
        "cv_val_std_f1_1": float(np.nanstd(fold_f1s)),
        "cv_val_std_roc_auc": float(np.nanstd(fold_aurocs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--features", nargs="+", default=None,
                    help="Feature columns to tune, e.g. PP EAP window_pp window_eap")
    ap.add_argument("--feature", default=None, help="(deprecated) Single feature to tune")
    ap.add_argument("--criterion", choices=["youden","fixed_fpr","f1"], default="youden")
    ap.add_argument("--target_fpr", type=float, default=0.01, help="Used when --criterion fixed_fpr")
    ap.add_argument("--eval_types", nargs="*", default=["pooled","pooled_no_gcg","algo_advprompter","algo_autodan","algo_gcg"])
    ap.add_argument("--balance_normals", choices=["none","match_adv_total"], default="none",
                    help="Optional downsampling for pooled_no_gcg.")
    ap.add_argument("--balance_seed", type=int, default=0)
    # NEW: CV controls
    ap.add_argument("--cv_folds", type=int, default=5, help="Number of Stratified folds (effective <= min class count).")
    ap.add_argument("--cv_seed", type=int, default=0, help="RNG seed for StratifiedKFold shuffle.")
    ap.add_argument("--stratify_by", choices=["label","algorithm"], default="label",
                    help="How to stratify CV folds: label=by is_adversarial, algorithm=by attack family (incl. normal).")
    ap.add_argument("--classifier", choices=["threshold","logistic"], default="threshold",
                    help="threshold=single-feature thresholding, logistic=multi-feature LR")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    for c in ["is_adversarial","algorithm"]:
        if c not in df.columns: raise ValueError(f"Missing column '{c}' in CSV.")

    feats: List[str] = []
    if args.features: feats += args.features
    if args.feature: feats += [args.feature]
    if not feats: raise ValueError("Provide --features (one or more) or --feature.")
    for f in feats:
        if f not in df.columns:
            raise ValueError(f"Column '{f}' not in CSV.")

    df["is_adversarial"] = df["is_adversarial"].fillna(0).astype(int)
    df["algorithm"] = _normalize_algo_col(df)
    df = df[df["algorithm"].isin(ROW_ORDER)].copy()

    rows_summary = []
    rows_folds   = []
    json_out: Dict[str, Dict[str, float]] = {}

    # Logistic regression mode: multi-feature classifier
    if args.classifier == "logistic":
        if len(feats) < 2:
            print("WARNING: logistic classifier needs >=2 features, falling back to single-feature")
        else:
            X = df[feats].values
            y = df["is_adversarial"].values
            strat = y if args.stratify_by == "label" else df["algorithm"].values
            result = _cv_logistic(X, y, args.cv_folds, args.cv_seed, stratify=strat)

            row = {
                "eval_type": "pooled", "feature": "logistic_regression",
                "criterion": "lr_cv",
                "threshold": float("nan"),
                "acc": float("nan"),
            }
            row.update(result)
            rows_summary.append(row)

            if args.out_csv:
                os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
                pd.DataFrame(rows_summary).to_csv(args.out_csv, index=False)

            print(f"pooled: logistic_regression: n={result.get('n','?')}, "
                  f"F1={result.get('f1_1',float('nan')):.4f}, "
                  f"AUROC={result.get('roc_auc',float('nan')):.4f}, "
                  f"cv_mean_F1={result.get('cv_val_mean_f1_1',float('nan')):.4f}, "
                  f"cv_mean_AUROC={result.get('cv_val_mean_roc_auc',float('nan')):.4f}")
            return

    for et in args.eval_types:
        sub = _subset(df, et)
        if et == "pooled_no_gcg":
            sub = _balance_normals(sub, args.balance_normals, args.balance_seed)

        if sub.empty or sub["is_adversarial"].nunique() < 2:
            for f in feats:
                rows_summary.append({"eval_type": et, "feature": f, "note":"empty_or_single_class"})
            continue

        json_out.setdefault(et, {})
        y = sub["is_adversarial"].values
        strat = y if args.stratify_by == "label" else sub["algorithm"].values
        for f in feats:
            s = sub[f].values
            rec_thr, val_means, val_stds, fold_rows = _cv_recommend_threshold(
                s, y, args.cv_folds, args.cv_seed, args.criterion, args.target_fpr, stratify=strat
            )

            # per-fold records
            for fr in fold_rows:
                rows_folds.append({"eval_type": et, "feature": f, **fr})

            # summary line (recommended threshold + overall metrics at that thr)
            overall = [r for r in fold_rows if r.get("fold")=="recommended"][0]
            rows_summary.append({
                "eval_type": et, "feature": f, "criterion": args.criterion,
                "threshold": rec_thr,
                "acc": overall.get("acc", np.nan),
                "f1_1": overall.get("f1_1", np.nan),
                "fpr": overall.get("fpr", np.nan),
                "tpr": overall.get("tpr", np.nan),
                "roc_auc": overall.get("roc_auc", np.nan),
                "pr_auc": overall.get("pr_auc", np.nan),
                "cv_val_mean_f1_1": val_means.get("f1_1", np.nan),
                "cv_val_mean_tpr": val_means.get("tpr", np.nan),
                "cv_val_mean_fpr": val_means.get("fpr", np.nan),
                "cv_val_mean_roc_auc": val_means.get("roc_auc", np.nan),
                "cv_val_mean_pr_auc": val_means.get("pr_auc", np.nan),
                "cv_val_std_f1_1": val_stds.get("f1_1", np.nan),
                "cv_val_std_tpr": val_stds.get("tpr", np.nan),
                "cv_val_std_fpr": val_stds.get("fpr", np.nan),
                "cv_val_std_roc_auc": val_stds.get("roc_auc", np.nan),
                "cv_val_std_pr_auc": val_stds.get("pr_auc", np.nan),
            })

            if np.isfinite(rec_thr):
                json_out[et][f] = float(rec_thr)

    # Write outputs
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        pd.DataFrame(rows_summary).to_csv(args.out_csv, index=False)
        # also save folds alongside (suffix _folds.csv)
        base, ext = os.path.splitext(args.out_csv)
        pd.DataFrame(rows_folds).to_csv(f"{base}_folds{ext or '.csv'}", index=False)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f: json.dump(json_out, f, indent=2)

    # Console summary + ready-to-paste JSON per eval_type
    # Logistic regression mode: multi-feature classifier
    if args.classifier == "logistic":
        if len(feats) < 2:
            print("WARNING: logistic classifier needs >=2 features, falling back to single-feature")
        else:
            X = df[feats].values
            y = df["is_adversarial"].values
            strat = y if args.stratify_by == "label" else df["algorithm"].values
            result = _cv_logistic(X, y, args.cv_folds, args.cv_seed, stratify=strat)

            row = {
                "eval_type": "pooled", "feature": "logistic_regression",
                "criterion": "lr_cv",
                "threshold": float("nan"),
                "acc": float("nan"),
            }
            row.update(result)
            rows_summary.append(row)

            if args.out_csv:
                os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
                pd.DataFrame(rows_summary).to_csv(args.out_csv, index=False)

            print(f"pooled: logistic_regression: n={result.get('n','?')}, "
                  f"F1={result.get('f1_1',float('nan')):.4f}, "
                  f"AUROC={result.get('roc_auc',float('nan')):.4f}, "
                  f"cv_mean_F1={result.get('cv_val_mean_f1_1',float('nan')):.4f}, "
                  f"cv_mean_AUROC={result.get('cv_val_mean_roc_auc',float('nan')):.4f}")
            return

    for et in args.eval_types:
        if et not in json_out or not json_out[et]:
            print(f"{et}: no thresholds (empty or single-class).")
            continue
        ordered_feats = [f for f in feats if f in json_out[et]]
        thr_line = ", ".join([f"{k}={json_out[et][k]:.3f}" for k in ordered_feats])
        print(f"{et}: {thr_line}")
        if et == "pooled_no_gcg":
            mapping = {k: round(float(json_out[et][k]), 3) for k in ordered_feats}
            print(f"fixed_thresholds JSON for {et}: {json.dumps(mapping)}")

if __name__ == "__main__":
    main()
