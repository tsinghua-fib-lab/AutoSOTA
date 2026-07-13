#!/usr/bin/env python3
"""R2-Router with XGBoost quality predictors."""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-data", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./results_xgb")
    p.add_argument("--lambda-points", type=int, default=200)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--early-stopping", type=int, default=20)
    p.add_argument("--qnc-target-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"R2-Router XGBoost Evaluation")
    print(f"  n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"  lr={args.learning_rate}, subsample={args.subsample}, reg_lambda={args.reg_lambda}")

    with open(args.training_data, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    models_data = data["models"]
    n_queries = embeddings.shape[0]
    print(f"  Queries: {n_queries}, Dim: {embeddings.shape[1]}")

    rng = np.random.RandomState(args.seed)
    all_idx = np.arange(n_queries)
    n_train = int(n_queries * args.train_frac)
    train_idx = rng.choice(all_idx, n_train, replace=False)
    test_idx = np.array(sorted(set(all_idx) - set(train_idx.tolist())))
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_idx])
    X_test = scaler.transform(embeddings[test_idx])

    print(f"\nTraining XGBoost predictors...")
    preds = {}
    n_trained = 0

    for mn in sorted(models_data.keys()):
        budgets = sorted(models_data[mn].keys())
        if not budgets: continue
        preds[mn] = {}
        mn_r2s = []
        for budget in budgets:
            bdata = models_data[mn][budget]
            y_all = bdata["accuracy"]
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            valid_train = ~np.isnan(y_train)
            valid_test = ~np.isnan(y_test)
            if valid_train.sum() < 32: continue

            Xtr = X_train[valid_train]; ytr = y_train[valid_train]
            Xte = X_test[valid_test]; yte = y_test[valid_test]

            xgb = XGBRegressor(
                n_estimators=args.n_estimators, max_depth=args.max_depth,
                learning_rate=args.learning_rate, subsample=args.subsample,
                reg_lambda=args.reg_lambda, random_state=args.seed,
                n_jobs=-1, verbosity=0, early_stopping_rounds=args.early_stopping,
            )
            xgb.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
            y_pred_valid = xgb.predict(Xte)
            ss_res = np.sum((yte - y_pred_valid) ** 2)
            ss_tot = np.sum((yte - yte.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            y_pred = np.full(len(test_idx), np.nan)
            y_pred[valid_test] = y_pred_valid
            preds[mn][budget] = {"pred_test": y_pred, "true_test": y_test, "test_score": r2}
            mn_r2s.append(r2)
            n_trained += 1
        if mn_r2s:
            print(f"  {mn}: {len(mn_r2s)}/{len(budgets)} budgets, avg R2={np.mean(mn_r2s):.4f}")
    print(f"  Total: {n_trained} predictors trained")

    # Routing evaluation (same as original)
    print(f"\nEvaluating routing...")
    n_test = len(test_idx)
    lambdas = np.linspace(0, 1, args.lambda_points)
    options = []
    for mn in sorted(preds.keys()):
        for budget in sorted(preds[mn].keys()):
            options.append((mn, budget))
    n_opts = len(options)
    pred_q = np.zeros((n_test, n_opts))
    true_q = np.zeros((n_test, n_opts))
    costs = np.zeros((n_test, n_opts))
    for j, (mn, budget) in enumerate(options):
        pdata = preds[mn][budget]
        pred_q[:, j] = pdata["pred_test"]
        true_q[:, j] = pdata["true_test"]
        costs[:, j] = models_data[mn][budget]["output_tokens"][test_idx]

    cost_norm = np.zeros_like(costs)
    for i in range(n_test):
        ci = costs[i]; valid = ~np.isnan(ci) & (ci > 0)
        if valid.sum() == 0: cost_norm[i] = 0
        else:
            cmin, cmax = ci[valid].min(), ci[valid].max()
            cost_norm[i] = np.clip((ci - cmin) / (cmax - cmin), 0, 1) if cmax > cmin else 0

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
    print(f"Oracle accuracy: {oracle_acc:.4f}")

    best_llm_acc = 0
    for mn in sorted(models_data.keys()):
        for budget in sorted(models_data[mn].keys()):
            acc = models_data[mn][budget]["accuracy"][test_idx]
            valid = ~np.isnan(acc)
            if valid.sum() > 100:
                m = float(acc[valid].mean())
                if m > best_llm_acc: best_llm_acc = m
    print(f"Best single LLM: {best_llm_acc:.4f}")

    sorted_df = results_df.sort_values("cost")
    cc = sorted_df["cost"].values; pc = sorted_df["accuracy"].values
    peak = float(pc.max())
    cmin, cmax = cc.min(), cc.max()
    nc = (cc - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(cc)
    audc = float(np.trapz(pc, nc))
    target = best_llm_acc * args.qnc_target_rate
    above = pc >= target
    qnc = float(nc[above][0]) if above.any() else 1.0

    print(f"\n" + "=" * 60)
    print(f"EVALUATION METRICS")
    print("=" * 60)
    print(f"  Peak Accuracy:    {peak:.4f}")
    print(f"  AUDC (norm cost): {audc:.4f}")
    print(f"  QNC:              {qnc:.4f}")
    print(f"  Oracle acc:       {oracle_acc:.4f}")
    print(f"  Best LLM acc:     {best_llm_acc:.4f}")

    metrics = {"peak_accuracy": peak, "AUDC": audc, "QNC": qnc,
               "oracle_accuracy": oracle_acc, "best_llm_accuracy": float(best_llm_acc),
               "predictor": "xgboost", "n_estimators": args.n_estimators}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    results_df.to_csv(os.path.join(args.output_dir, "routing_curves.csv"), index=False)
    print(f"\nResults saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
