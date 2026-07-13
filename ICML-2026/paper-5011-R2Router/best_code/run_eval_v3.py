#!/usr/bin/env python3
"""
R2-Router evaluation v3: Calibrated KNN with variance scaling.

Key improvement over v2: KNN predictions are compressed toward the mean
(low variance). We calibrate by computing per-(model, budget) scale factors
on a held-out calibration set, then scaling predictions away from the mean.

Also supports L2 normalization of embeddings before KNN.
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-data", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./results_v3")
    p.add_argument("--lambda-points", type=int, default=200)
    p.add_argument("--k-neighbors", type=int, default=128)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--cal-frac", type=float, default=0.25,
                   help="Fraction of training data for calibration (0=no calibration)")
    p.add_argument("--l2-normalize", action="store_true",
                   help="L2-normalize embeddings before KNN")
    p.add_argument("--knn-metric", type=str, default="cosine",
                   help="KNN distance metric (cosine/euclidean)")
    p.add_argument("--qnc-target-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    use_cal = args.cal_frac > 0 and args.cal_frac < 1.0
    print("=" * 60)
    print("R2-Router Calibrated KNN Evaluation")
    print("  k=%d, metric=%s, L2-norm=%s, calibrate=%s" % (
        args.k_neighbors, args.knn_metric, args.l2_normalize,
        ("yes (%.0f%%)" % (args.cal_frac * 100)) if use_cal else "no"))
    print("=" * 60)

    with open(args.training_data, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    models_data = data["models"]
    n_queries = embeddings.shape[0]
    print("Loaded %d queries, %dd embeddings" % (n_queries, embeddings.shape[1]))

    rng = np.random.RandomState(args.seed)
    all_idx = np.arange(n_queries)
    n_train = int(n_queries * args.train_frac)
    train_idx = rng.choice(all_idx, n_train, replace=False)
    test_idx = np.array(sorted(set(all_idx) - set(train_idx.tolist())))

    # Further split training into fit + calibration
    if use_cal:
        n_cal = int(len(train_idx) * args.cal_frac)
        cal_perm = rng.permutation(len(train_idx))
        cal_sub_idx = train_idx[cal_perm[:n_cal]]
        fit_sub_idx = train_idx[cal_perm[n_cal:]]
        print("Train: %d (fit), %d (cal), Test: %d" % (len(fit_sub_idx), len(cal_sub_idx), len(test_idx)))
    else:
        fit_sub_idx = train_idx
        cal_sub_idx = np.array([], dtype=int)
        print("Train: %d, Test: %d" % (len(train_idx), len(test_idx)))

    # Standardize
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(embeddings[fit_sub_idx])
    X_cal = scaler.transform(embeddings[cal_sub_idx]) if use_cal else None
    X_test = scaler.transform(embeddings[test_idx])

    # L2 normalize
    if args.l2_normalize:
        X_fit = normalize(X_fit, norm="l2")
        if X_cal is not None:
            X_cal = normalize(X_cal, norm="l2")
        X_test = normalize(X_test, norm="l2")
        if args.knn_metric == "cosine":
            args.knn_metric = "euclidean"  # L2-norm + euclidean = cosine

    # Train KNN on fit set
    k = args.k_neighbors
    print("\nTraining KNN predictors...")
    preds_knn = {}
    for mn in sorted(models_data.keys()):
        budgets = sorted(models_data[mn].keys())
        if not budgets: continue
        preds_knn[mn] = {}
        for budget in budgets:
            bdata = models_data[mn][budget]
            y_all = bdata["accuracy"]
            y_fit = y_all[fit_sub_idx]
            valid_fit = ~np.isnan(y_fit)
            if valid_fit.sum() < k: continue
            knn = KNeighborsRegressor(
                n_neighbors=min(k, valid_fit.sum()),
                metric=args.knn_metric, weights="distance", n_jobs=-1,
            )
            knn.fit(X_fit[valid_fit], y_fit[valid_fit])
            preds_knn[mn][budget] = knn

    # Calibration: compute per-(model, budget) scale factors
    calib = {}
    if use_cal and len(cal_sub_idx) > 0:
        print("\nCalibrating prediction scales...")
        for mn in sorted(preds_knn.keys()):
            for budget in sorted(preds_knn[mn].keys()):
                y_cal = models_data[mn][budget]["accuracy"][cal_sub_idx]
                valid_cal = ~np.isnan(y_cal)
                if valid_cal.sum() < 10:
                    calib[(mn, budget)] = 1.0
                    continue
                yp = preds_knn[mn][budget].predict(X_cal[valid_cal])
                yt = y_cal[valid_cal]
                std_pred = np.std(yp)
                std_true = np.std(yt)
                if std_pred > 1e-10 and std_true > 1e-10:
                    scale = std_true / std_pred
                else:
                    scale = 1.0
                calib[(mn, budget)] = scale
        avg_scale = np.mean(list(calib.values()))
        print("  Avg scale factor: %.3f (1.0 = no scaling)" % avg_scale)
    else:
        for mn in sorted(preds_knn.keys()):
            for budget in sorted(preds_knn[mn].keys()):
                calib[(mn, budget)] = 1.0

    # Build options and predict
    options = [(mn, b) for mn in sorted(preds_knn.keys()) for b in sorted(preds_knn[mn].keys())]
    n_opts = len(options)
    n_test = len(test_idx)
    print("\nPredicting on test set (%d options x %d queries)..." % (n_opts, n_test))

    pred_q = np.zeros((n_test, n_opts))
    true_q = np.zeros((n_test, n_opts))
    costs = np.zeros((n_test, n_opts))

    for j, (mn, budget) in enumerate(options):
        y_test = models_data[mn][budget]["accuracy"][test_idx]
        valid_te = ~np.isnan(y_test)
        pred = np.full(n_test, np.nan)
        if valid_te.sum() > 0:
            pred[valid_te] = preds_knn[mn][budget].predict(X_test[valid_te])

        # Apply scale calibration
        scale = calib.get((mn, budget), 1.0)
        if scale != 1.0:
            valid_pred = ~np.isnan(pred)
            if valid_pred.sum() > 0:
                pred_mean = np.mean(pred[valid_pred])
                pred[valid_pred] = pred_mean + scale * (pred[valid_pred] - pred_mean)

        pred_q[:, j] = pred
        true_q[:, j] = y_test
        costs[:, j] = models_data[mn][budget]["output_tokens"][test_idx]

    # Cost normalization
    cost_norm = np.zeros_like(costs)
    for i in range(n_test):
        ci = costs[i]; valid = ~np.isnan(ci) & (ci > 0)
        if valid.sum() == 0: cost_norm[i] = 0
        else:
            cmin, cmax = ci[valid].min(), ci[valid].max()
            cost_norm[i] = np.clip((ci - cmin) / (cmax - cmin), 0, 1) if cmax > cmin else 0

    # Routing evaluation
    print("Evaluating routing...")
    lambdas = np.linspace(0, 1, args.lambda_points)
    results = []
    for lam in lambdas:
        risk = (1 - lam) * pred_q - lam * cost_norm
        best = np.nanargmax(risk, axis=1)
        sel_q = true_q[np.arange(n_test), best]
        sel_c = costs[np.arange(n_test), best]
        valid = ~np.isnan(sel_q) & ~np.isnan(sel_c) & (sel_c > 0)
        avg_q = sel_q[valid].mean() if valid.sum() > 10 else np.nanmean(sel_q)
        avg_c = sel_c[valid].mean() if valid.sum() > 10 else np.nanmean(sel_c)
        results.append({"lambda": lam, "cost": avg_c, "accuracy": avg_q})

    results_df = pd.DataFrame(results)
    oracle_acc = float(np.nanmean(np.nanmax(true_q, axis=1)))
    print("Oracle accuracy: %.4f" % oracle_acc)

    best_llm_acc = 0
    for mn in sorted(models_data.keys()):
        for budget in sorted(models_data[mn].keys()):
            acc = models_data[mn][budget]["accuracy"][test_idx]
            valid = ~np.isnan(acc)
            if valid.sum() > 100:
                m = float(acc[valid].mean())
                if m > best_llm_acc: best_llm_acc = m
    print("Best single LLM: %.4f" % best_llm_acc)

    sorted_df = results_df.sort_values("cost")
    cc = sorted_df["cost"].values; pc = sorted_df["accuracy"].values
    peak = float(pc.max())
    cmin, cmax = cc.min(), cc.max()
    nc = (cc - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(cc)
    audc = float(np.trapz(pc, nc))
    target = best_llm_acc * args.qnc_target_rate
    above = pc >= target
    qnc = float(nc[above][0]) if above.any() else 1.0

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print("  Peak Accuracy:    %.4f" % peak)
    print("  AUDC (norm cost): %.4f" % audc)
    print("  QNC:              %.4f" % qnc)
    print("  Oracle acc:       %.4f" % oracle_acc)
    print("  Best LLM acc:     %.4f" % best_llm_acc)

    metrics = {
        "peak_accuracy": peak, "AUDC": audc, "QNC": qnc,
        "oracle_accuracy": oracle_acc, "best_llm_accuracy": float(best_llm_acc),
        "qnc_target": float(target), "k_neighbors": args.k_neighbors,
        "calibrated": use_cal, "l2_normalize": args.l2_normalize,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    results_df.to_csv(os.path.join(args.output_dir, "routing_curves.csv"), index=False)
    print("\nResults saved to %s/" % args.output_dir)

if __name__ == "__main__":
    main()
