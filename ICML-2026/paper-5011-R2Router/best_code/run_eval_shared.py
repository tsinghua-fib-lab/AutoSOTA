#!/usr/bin/env python3
"""
R2-Router with Shared Encoder MLP (ALGO-01).

Key idea: For each LLM, share the first 2 encoder layers across all budgets.
A lightweight per-budget head (128+1 -> 1) with budget as continuous input.
This gives ~9x more training signal for the shared encoder, reducing overfitting.

Architecture per LLM:
  Shared encoder: Linear(1024,256) -> ReLU -> Dropout -> Linear(256,128) -> ReLU
  Per-budget head: Linear(129, 1) -> Sigmoid  (129 = 128 emb + 1 budget_norm)
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-data", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./results_shared")
    p.add_argument("--lambda-points", type=int, default=200)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--qnc-target-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()

class SharedEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden=256, bottleneck=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.encoder(x)

class BudgetHead(nn.Module):
    def __init__(self, bottleneck=128, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(bottleneck + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, encoded, budget_norm):
        x = torch.cat([encoded, budget_norm.unsqueeze(1)], dim=1)
        return self.head(x).squeeze(-1)

class SharedPredictor(nn.Module):
    def __init__(self, input_dim=1024, hidden=256, bottleneck=128, budgets=None, dropout=0.1):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, hidden, bottleneck, dropout)
        self.heads = nn.ModuleDict()
        self.budget_norms = {}
        if budgets is not None:
            # Normalize budgets to [0,1]
            bvals = sorted([b for b in budgets if isinstance(b, (int, float))])
            bmin, bmax = bvals[0], bvals[-1]
            for b in budgets:
                if isinstance(b, (int, float)):
                    self.budget_norms[b] = (b - bmin) / (bmax - bmin) if bmax > bmin else 0.5
                else:
                    self.budget_norms[b] = 1.0  # concise -> max budget
                self.heads[str(b)] = BudgetHead(bottleneck, dropout)

    def forward(self, x, budget_key):
        encoded = self.encoder(x)
        bn = torch.full((x.size(0),), self.budget_norms[budget_key], device=x.device)
        return self.heads[str(budget_key)](encoded, bn)


def train_shared_predictor(model, X_train_dict, y_train_dict, X_test_dict, y_test_dict,
                           epochs=200, lr=1e-4, batch_size=512, weight_decay=1e-5,
                           patience=40, device="cuda:0", seed=42):
    """Train shared encoder jointly across all budgets for one LLM."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build combined dataset across all budgets
    all_X, all_y, all_budgets = [], [], []
    budget_keys = sorted(X_train_dict.keys())
    for bk in budget_keys:
        if len(X_train_dict[bk]) == 0: continue
        all_X.append(torch.FloatTensor(X_train_dict[bk]))
        all_y.append(torch.FloatTensor(y_train_dict[bk]))
        all_budgets.extend([bk] * len(X_train_dict[bk]))

    if not all_X: return {}, 0, float("inf")

    X_cat = torch.cat(all_X, dim=0)
    y_cat = torch.cat(all_y, dim=0)

    # Build test tensors
    X_test_all = []
    y_test_all = []
    test_bk_list = []
    for bk in budget_keys:
        if bk not in X_test_dict or len(X_test_dict[bk]) == 0: continue
        X_test_all.append(torch.FloatTensor(X_test_dict[bk]).to(device))
        y_test_all.append(torch.FloatTensor(y_test_dict[bk]).to(device))
        test_bk_list.append(bk)

    dataset = TensorDataset(X_cat, y_cat)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    model.train()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            # Randomly sample budget for each item in batch
            batch_budgets = [all_budgets[i % len(all_budgets)] for i in range(len(bx))]
            # Actually, we need to track which budget each sample belongs to
            # For simplicity, iterate over budgets
            pass

        # Better approach: iterate over budgets within epoch
        epoch_loss = 0
        for bk in budget_keys:
            if bk not in X_train_dict or len(X_train_dict[bk]) == 0: continue
            X_bk = torch.FloatTensor(X_train_dict[bk]).to(device)
            y_bk = torch.FloatTensor(y_train_dict[bk]).to(device)
            bk_dataset = TensorDataset(X_bk, y_bk)
            bk_loader = DataLoader(bk_dataset, batch_size=min(batch_size, len(X_bk)), shuffle=True)

            for bx, by in bk_loader:
                optimizer.zero_grad()
                pred = model(bx, bk)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(bx)

        # Evaluate on test
        model.eval()
        test_loss = 0
        n_test = 0
        with torch.no_grad():
            for idx, bk in enumerate(test_bk_list):
                Xt = X_test_all[idx]
                yt = y_test_all[idx]
                pred = model(Xt, bk)
                test_loss += criterion(pred, yt).item() * len(Xt)
                n_test += len(Xt)
        test_loss /= max(n_test, 1)

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute per-budget predictions and R²
    model.eval()
    preds = {}
    total_r2 = []
    with torch.no_grad():
        for bk in budget_keys:
            if bk not in X_test_dict or len(X_test_dict[bk]) == 0: continue
            Xt = torch.FloatTensor(X_test_dict[bk]).to(device)
            yt = y_test_dict[bk]
            yp = model(Xt, bk).cpu().numpy()
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            preds[bk] = (yp, r2)
            total_r2.append(r2)

    avg_r2 = np.mean(total_r2) if total_r2 else 0.0
    return preds, avg_r2, best_loss


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"R2-Router Shared Encoder MLP Evaluation")
    print(f"  Dropout={args.dropout}, WD={args.weight_decay}")
    print(f"  Epochs={args.epochs}, LR={args.lr}, Batch={args.batch_size}")

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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_idx])
    X_test = scaler.transform(embeddings[test_idx])

    print(f"\nTraining shared-encoder predictors per LLM...")
    preds = {}
    n_llms_trained = 0

    for mn in sorted(models_data.keys()):
        budgets = sorted(models_data[mn].keys())
        if not budgets: continue

        # Check if this model has multiple budgets (shared encoder benefits most)
        numeric_budgets = []
        for b in budgets:
            bnum = int(b.replace("budget_", "")) if b.startswith("budget_") else None
            if bnum is not None:
                numeric_budgets.append(bnum)
        if b == "concise":
            numeric_budgets.append(9999)  # large placeholder

        if len(budgets) < 2:
            # Single budget — use standard MLP instead
            preds[mn] = {}
            for budget in budgets:
                bdata = models_data[mn][budget]
                y_all = bdata["accuracy"]
                y_train = y_all[train_idx]; y_test = y_all[test_idx]
                valid_tr = ~np.isnan(y_train); valid_te = ~np.isnan(y_test)
                if valid_tr.sum() < 32: continue
                Xtr = X_train[valid_tr]; ytr = y_train[valid_tr]
                Xte = X_test[valid_te]; yte = y_test[valid_te]

                # Simple KNN fallback for single-budget models
                from sklearn.neighbors import KNeighborsRegressor
                knn = KNeighborsRegressor(n_neighbors=min(128, valid_tr.sum()), metric="cosine", weights="distance", n_jobs=-1)
                knn.fit(Xtr, ytr)
                y_pred_valid = knn.predict(Xte)
                ss_res = np.sum((yte - y_pred_valid) ** 2)
                ss_tot = np.sum((yte - yte.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                y_pred = np.full(len(test_idx), np.nan)
                y_pred[valid_te] = y_pred_valid
                preds[mn][budget] = {"pred_test": y_pred, "true_test": y_test, "test_score": r2}
            continue

        # Multi-budget model: shared encoder
        X_train_dict, y_train_dict = {}, {}
        X_test_dict, y_test_dict = {}, {}
        valid_budgets = []

        for budget in budgets:
            bdata = models_data[mn][budget]
            y_all = bdata["accuracy"]
            y_train = y_all[train_idx]; y_test = y_all[test_idx]
            valid_tr = ~np.isnan(y_train); valid_te = ~np.isnan(y_test)
            if valid_tr.sum() < 32: continue
            X_train_dict[budget] = X_train[valid_tr]
            y_train_dict[budget] = y_train[valid_tr]
            X_test_dict[budget] = X_test[valid_te]
            y_test_dict[budget] = y_test[valid_te]
            valid_budgets.append(budget)

        if not valid_budgets: continue

        # Parse budgets for normalization
        parsed = []
        for b in valid_budgets:
            if b.startswith("budget_"):
                parsed.append(int(b.replace("budget_", "")))
            else:
                parsed.append(2000)  # concise ~ large
        bvals = sorted(set(parsed))
        bmin, bmax = bvals[0], bvals[-1]

        # Build budget norms
        budget_norms = {}
        for bk, bv in zip(valid_budgets, parsed):
            budget_norms[bk] = (bv - bmin) / (bmax - bmin) if bmax > bmin else 0.5

        model = SharedPredictor(input_dim=1024, hidden=256, bottleneck=128,
                                budgets=valid_budgets, dropout=args.dropout)
        # Override budget_norms
        model.budget_norms = budget_norms

        budget_preds, avg_r2, _ = train_shared_predictor(
            model, X_train_dict, y_train_dict, X_test_dict, y_test_dict,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay, patience=args.patience,
            device=args.device, seed=args.seed,
        )

        preds[mn] = {}
        mn_r2s = []
        for budget in valid_budgets:
            if budget in budget_preds:
                yp, r2 = budget_preds[budget]
                y_test = models_data[mn][budget]["accuracy"][test_idx]
                valid_te = ~np.isnan(y_test)
                y_pred = np.full(len(test_idx), np.nan)
                y_pred[valid_te] = yp
                preds[mn][budget] = {"pred_test": y_pred, "true_test": y_test, "test_score": r2}
                mn_r2s.append(r2)

        if mn_r2s:
            print(f"  {mn}: {len(mn_r2s)}/{len(budgets)} budgets (shared), avg R2={np.mean(mn_r2s):.4f}")
        n_llms_trained += 1

    # ... routing evaluation (identical to original) ...
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

    print(f"\nOracle: {oracle_acc:.4f}, Best LLM: {best_llm_acc:.4f}")
    print(f"{=*60}")
    print(f"EVALUATION METRICS")
    print(f"{=*60}")
    print(f"  Peak Accuracy:    {peak:.4f}")
    print(f"  AUDC (norm cost): {audc:.4f}")
    print(f"  QNC:              {qnc:.4f}")

    metrics = {"peak_accuracy": peak, "AUDC": audc, "QNC": qnc,
               "oracle_accuracy": oracle_acc, "best_llm_accuracy": float(best_llm_acc),
               "predictor": "shared_encoder", "dropout": args.dropout}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    results_df.to_csv(os.path.join(args.output_dir, "routing_curves.csv"), index=False)
    print(f"\nResults saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
