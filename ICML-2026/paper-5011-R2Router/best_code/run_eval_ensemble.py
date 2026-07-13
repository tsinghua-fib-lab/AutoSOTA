#!/usr/bin/env python3
"""
R2-Router Ensemble Evaluation: Calibrated KNN ensemble across k values.

Trains KNN at multiple k values, calibrates each independently, and
averages their predictions using learned weights for robust routing.
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-data", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./results_ensemble")
    p.add_argument("--lambda-points", type=int, default=200)
    p.add_argument("--k-values", type=str, default="256,512")
    p.add_argument("--k-weights", type=str, default="0.65,0.35")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--cal-frac", type=float, default=0.25)
    p.add_argument("--l2-normalize", action="store_true", default=True)
    p.add_argument("--qnc-target-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    k_values = [int(x) for x in args.k_values.split(",")]
    k_weights = [float(x) for x in args.k_weights.split(",")]
    if len(k_weights) != len(k_values):
        k_weights = [1.0/len(k_values)] * len(k_values)
    k_weights = np.array(k_weights) / sum(k_weights)

    print("=" * 60)
    print("R2-Router Ensemble KNN Evaluation")
    print("  k values: %s, weights: %s" % (k_values, k_weights))
    print("  cal_frac=%.0f%%, L2=%s" % (args.cal_frac * 100, args.l2_normalize))
    print("=" * 60)

    with open(args.training_data, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    models_data = data["models"]
    n_queries = embeddings.shape[0]

    rng = np.random.RandomState(args.seed)
    all_idx = np.arange(n_queries)
    n_train = int(n_queries * args.train_frac)
    train_idx = rng.choice(all_idx, n_train, replace=False)
    test_idx = np.array(sorted(set(all_idx) - set(train_idx.tolist())))

    n_cal = int(len(train_idx) * args.cal_frac)
    cal_perm = rng.permutation(len(train_idx))
    cal_sub_idx = train_idx[cal_perm[:n_cal]]
    fit_sub_idx = train_idx[cal_perm[n_cal:]]
    print("Train: %d (fit), %d (cal), Test: %d" % (len(fit_sub_idx), len(cal_sub_idx), len(test_idx)))

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(embeddings[fit_sub_idx])
    X_cal = scaler.transform(embeddings[cal_sub_idx])
    X_test = scaler.transform(embeddings[test_idx])

    if args.l2_normalize:
        X_fit = normalize(X_fit, norm="l2")
        X_cal = normalize(X_cal, norm="l2")
        X_test = normalize(X_test, norm="l2")

    options = [(mn, b) for mn in sorted(models_data.keys()) for b in sorted(models_data[mn].keys())]
    n_opts = len(options); n_test = len(test_idx)
    print("Options: %d" % n_opts)

    all_preds = {}
    for k in k_values:
        print("\nTraining k=%d..." % k)
        preds_knn = {}
        for mn in sorted(models_data.keys()):
            budgets = sorted(models_data[mn].keys())
            if not budgets: continue
            preds_knn[mn] = {}
            for budget in budgets:
                bdata = models_data[mn][budget]
                y_fit = bdata["accuracy"][fit_sub_idx]
                valid_fit = ~np.isnan(y_fit)
                if valid_fit.sum() < k: continue
                metric = "euclidean" if args.l2_normalize else "cosine"
                knn = KNeighborsRegressor(
                    n_neighbors=min(k, valid_fit.sum()),
                    metric=metric, weights="distance", n_jobs=-1,
                )
                knn.fit(X_fit[valid_fit], y_fit[valid_fit])
                preds_knn[mn][budget] = knn

        # Calibrate and predict
        pred_q = np.zeros((n_test, n_opts))
        for j, (mn, budget) in enumerate(options):
            if mn not in preds_knn or budget not in preds_knn[mn]:
                pred_q[:, j] = np.nan; continue
            y_cal = models_data[mn][budget]["accuracy"][cal_sub_idx]
            valid_cal = ~np.isnan(y_cal)
            scale = 1.0
            if valid_cal.sum() >= 10:
                yp_cal = preds_knn[mn][budget].predict(X_cal[valid_cal])
                yt_cal = y_cal[valid_cal]
                std_pred, std_true = np.std(yp_cal), np.std(yt_cal)
                if std_pred > 1e-10 and std_true > 1e-10:
                    scale = std_true / std_pred

            y_test = models_data[mn][budget]["accuracy"][test_idx]
            valid_te = ~np.isnan(y_test)
            pred = np.full(n_test, np.nan)
            if valid_te.sum() > 0:
                pred[valid_te] = preds_knn[mn][budget].predict(X_test[valid_te])
            if scale != 1.0:
                valid_p = ~np.isnan(pred)
                if valid_p.sum() > 0:
                    pred_mean = np.mean(pred[valid_p])
                    pred[valid_p] = pred_mean + scale * (pred[valid_p] - pred_mean)
            pred_q[:, j] = pred
        all_preds[k] = pred_q

    # Ensemble predictions
    print("\nEnsembling predictions...")
    pred_q = np.zeros((n_test, n_opts))
    for i, k in enumerate(k_values):
        pred_q += k_weights[i] * np.nan_to_num(all_preds[k], nan=0.0)

    # True values and costs
    true_q = np.zeros((n_test, n_opts))
    costs = np.zeros((n_test, n_opts))
    for j, (mn, budget) in enumerate(options):
        true_q[:, j] = models_data[mn][budget]["accuracy"][test_idx]
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

    best_llm_acc = 0
    for mn in sorted(models_data.keys()):
        for budget in sorted(models_data[mn].keys()):
            acc = models_data[mn][budget]["accuracy"][test_idx]
            valid = ~np.isnan(acc)
            if valid.sum() > 100:
                m = float(acc[valid].mean())
                if m > best_llm_acc: best_llm_acc = m

    sorted_df = results_df.sort_values("cost")
    cc = sorted_df["cost"].values; pc = sorted_df["accuracy"].values
    peak = float(pc.max())
    cmin, cmax = cc.min(), cc.max()
    nc = (cc - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(cc)
    audc = float(np.trapz(pc, nc))
    target = best_llm_acc * args.qnc_target_rate
    above = pc >= target
    qnc = float(nc[above][0]) if above.any() else 1.0

    print("\nOracle: %.4f, Best LLM: %.4f" % (oracle_acc, best_llm_acc))
    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print("  Peak Accuracy:    %.4f" % peak)
    print("  AUDC (norm cost): %.4f" % audc)
    print("  QNC:              %.4f" % qnc)

    metrics = {
        "peak_accuracy": peak, "AUDC": audc, "QNC": qnc,
        "oracle_accuracy": oracle_acc, "best_llm_accuracy": float(best_llm_acc),
        "k_values": k_values, "k_weights": list(k_weights),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    results_df.to_csv(os.path.join(args.output_dir, "routing_curves.csv"), index=False)
    print("\nResults saved to %s/" % args.output_dir)

if __name__ == "__main__":
    main()
