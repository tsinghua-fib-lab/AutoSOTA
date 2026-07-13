#!/usr/bin/env python3
"""
R2-Router evaluation with 3-layer MLP predictor (matching paper Section 4.3.1).

Paper architecture: shared encoder + per-LLM 3-layer MLP heads
  - Hidden dims: [256, 128, 64]
  - ReLU activations
  - Sigmoid output
  - MSE loss, Adam optimizer, lr=1e-4, 100 epochs

This script trains MLP quality predictors and evaluates routing.
"""

import argparse, json, os, pickle, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--training-data', type=str, required=True)
    p.add_argument('--output-dir', type=str, default='./results_mlp')
    p.add_argument('--lambda-points', type=int, default=200)
    p.add_argument('--train-frac', type=float, default=0.8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden-dims', type=str, default='256,128,64')
    p.add_argument('--qnc-target-rate', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


class MLPHead(nn.Module):
    """3-layer MLP with ReLU, Sigmoid output (paper Section 4.3.1)"""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_test, y_test, input_dim, hidden_dims,
              epochs=100, lr=1e-4, batch_size=256, device='cuda:0', seed=42):
    """Train a single MLP head for one (model, budget) pair."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLPHead(input_dim, hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Prepare data
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_test_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        # Evaluate on test
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()
        # Compute R²
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return y_pred, r2, best_test_loss


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    print(f"R2-Router MLP Evaluation")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"  Device: {args.device}")

    # Load data
    with open(args.training_data, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    models_data = data['models']
    n_queries = embeddings.shape[0]
    input_dim = embeddings.shape[1]

    print(f"  Queries: {n_queries}, Dim: {input_dim}")

    # Train/test split
    rng = np.random.RandomState(args.seed)
    all_idx = np.arange(n_queries)
    n_train = int(n_queries * args.train_frac)
    train_idx = rng.choice(all_idx, n_train, replace=False)
    test_idx = np.array(sorted(set(all_idx) - set(train_idx.tolist())))
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_idx])
    X_test = scaler.transform(embeddings[test_idx])

    # Train MLP predictors for each (model, budget)
    print(f"\nTraining MLP predictors...")
    preds = {}
    n_trained = 0

    for mn in sorted(models_data.keys()):
        budgets = sorted(models_data[mn].keys())
        if not budgets:
            continue

        preds[mn] = {}
        mn_r2s = []

        for budget in budgets:
            bdata = models_data[mn][budget]
            y_all = bdata['accuracy']
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            valid_train = ~np.isnan(y_train)
            valid_test = ~np.isnan(y_test)

            if valid_train.sum() < 32:
                continue

            Xtr = X_train[valid_train]
            ytr = y_train[valid_train]
            Xte = X_test[valid_test]
            yte = y_test[valid_test]

            y_pred_valid, r2, test_loss = train_mlp(
                Xtr, ytr, Xte, yte,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
            )

            # Full test prediction (NaN for invalid)
            y_pred = np.full(len(test_idx), np.nan)
            y_pred[valid_test] = y_pred_valid

            preds[mn][budget] = {
                'pred_test': y_pred,
                'true_test': y_test,
                'test_score': r2,
            }
            mn_r2s.append(r2)
            n_trained += 1

        if mn_r2s:
            print(f"  {mn}: {len(mn_r2s)}/{len(budgets)} budgets, avg R²={np.mean(mn_r2s):.4f}")

    print(f"  Total: {n_trained} predictors trained")

    # Evaluate routing
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
        pred_q[:, j] = pdata['pred_test']
        true_q[:, j] = pdata['true_test']
        costs[:, j] = models_data[mn][budget]['output_tokens'][test_idx]

    # Normalize costs per query
    cost_norm = np.zeros_like(costs)
    for i in range(n_test):
        ci = costs[i]
        valid = ~np.isnan(ci) & (ci > 0)
        if valid.sum() == 0:
            cost_norm[i] = 0
        else:
            cmin, cmax = ci[valid].min(), ci[valid].max()
            if cmax > cmin:
                cost_norm[i] = np.clip((ci - cmin) / (cmax - cmin), 0, 1)
            else:
                cost_norm[i] = 0

    results = []
    for lam in lambdas:
        risk = (1 - lam) * pred_q - lam * cost_norm
        best = np.nanargmax(risk, axis=1)
        sel_q = true_q[np.arange(n_test), best]
        sel_c = costs[np.arange(n_test), best]
        valid = ~np.isnan(sel_q) & ~np.isnan(sel_c) & (sel_c > 0)
        avg_q = sel_q[valid].mean() if valid.sum() > 10 else np.nanmean(sel_q)
        avg_c = sel_c[valid].mean() if valid.sum() > 10 else np.nanmean(sel_c)
        results.append({'lambda': lam, 'cost': avg_c, 'accuracy': avg_q})

    results_df = pd.DataFrame(results)

    # Oracle
    oracle_acc = np.nanmax(true_q, axis=1)
    oracle_mean = float(np.nanmean(oracle_acc))
    print(f"Oracle accuracy: {oracle_mean:.4f}")

    # Best single LLM
    best_llm_acc = 0
    best_name = ""
    for mn in sorted(models_data.keys()):
        for budget in sorted(models_data[mn].keys()):
            acc = models_data[mn][budget]['accuracy'][test_idx]
            valid = ~np.isnan(acc)
            if valid.sum() > 100:
                m = float(acc[valid].mean())
                if m > best_llm_acc:
                    best_llm_acc = m
                    best_name = f"{mn}/{budget}"
    print(f"Best single LLM: {best_name} = {best_llm_acc:.4f}")

    # Metrics
    sorted_df = results_df.sort_values('cost')
    cc = sorted_df['cost'].values
    pc = sorted_df['accuracy'].values
    peak = float(pc.max())
    cmin, cmax = cc.min(), cc.max()
    nc = (cc - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(cc)
    try:
        audc = float(np.trapezoid(pc, nc))
    except AttributeError:
        audc = float(np.trapz(pc, nc))
    target = best_llm_acc * args.qnc_target_rate
    above = pc >= target
    qnc = float(nc[above][0]) if above.any() else 1.0

    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"  Peak Accuracy:    {peak:.4f}")
    print(f"  AUDC (norm cost): {audc:.4f}")
    print(f"  QNC:              {qnc:.4f}")
    print(f"  Oracle acc:       {oracle_mean:.4f}")
    print(f"  Best LLM acc:     {best_llm_acc:.4f}")

    metrics = {
        'peak_accuracy': peak, 'AUDC': audc, 'QNC': qnc,
        'oracle_accuracy': oracle_mean, 'best_llm_accuracy': float(best_llm_acc),
        'qnc_target': float(target), 'hidden_dims': hidden_dims,
        'epochs': args.epochs, 'lr': args.lr,
    }

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    results_df.to_csv(os.path.join(args.output_dir, 'routing_curves.csv'), index=False)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
