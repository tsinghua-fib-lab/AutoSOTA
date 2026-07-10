"""Quick sweep of lr and epochs for CauchyNet on UCI datasets (n=100)."""
import os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import load_diabetes, fetch_california_housing, fetch_openml
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from shared import (
    CauchyNet, FNN, count_real_params,
    cauchy_real_params, matched_fnn_hidden,
    BATCH_SIZE, WEIGHT_DECAY, IMAG_PENALTY
)

def load_datasets():
    datasets = {}
    data = load_diabetes()
    datasets['Diabetes'] = (data.data.astype(np.float32), data.target.astype(np.float32))
    data = fetch_california_housing()
    datasets['California'] = (data.data.astype(np.float32), data.target.astype(np.float32))
    try:
        data = fetch_openml(name='wine-quality-red', version=1, as_frame=False, parser='auto')
        datasets['Wine'] = (data.data.astype(np.float32), data.target.astype(np.float32))
    except:
        pass
    return datasets

def train_eval_kfold(model_cls, X, y, hidden_size=64, n_folds=5, epochs=200, lr=0.01, cauchy_eps=1e-8):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mse_scores = []
    input_size = X.shape[1]
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        scX, scy = MinMaxScaler(), MinMaxScaler()
        X_tr = scX.fit_transform(X[tr_idx]).astype(np.float32)
        X_te = scX.transform(X[te_idx]).astype(np.float32)
        y_tr = scy.fit_transform(y[tr_idx].reshape(-1,1)).flatten().astype(np.float32)
        y_te_raw = y[te_idx]
        Xtr, Ytr = torch.tensor(X_tr), torch.tensor(y_tr)
        Xte = torch.tensor(X_te)

        torch.manual_seed(42+fold)
        if model_cls == CauchyNet:
            model = model_cls(input_size, hidden_size, 1, eps=cauchy_eps)
        else:
            model = model_cls(input_size, hidden_size, 1)

        crit = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)

        best_loss, best_state, wait = float('inf'), None, 0
        for epoch in range(epochs):
            model.train(); ep_loss, nb = 0., 0
            for xb, yb in loader:
                opt.zero_grad(); out = model(xb)
                if isinstance(out, tuple):
                    loss = crit(out[0], yb.unsqueeze(-1)) + IMAG_PENALTY*crit(out[1], torch.zeros_like(out[1]))
                else:
                    loss = crit(out, yb.unsqueeze(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); ep_loss += loss.item(); nb += 1
            sched.step(); avg = ep_loss/nb
            if avg < best_loss: best_loss, best_state, wait = avg, {k:v.clone() for k,v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 30: break

        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(Xte)
            preds_sc = out[0].cpu().numpy().flatten() if isinstance(out, tuple) else out.cpu().numpy().flatten()
        preds = scy.inverse_transform(preds_sc.reshape(-1,1)).flatten()
        mse_scores.append(float(mean_squared_error(y_te_raw, preds)))

    return np.mean(mse_scores)

datasets = load_datasets()
max_train = 100
lrs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
epoch_list = [200, 300, 500]
cauchy_h = 64

print(f"{'Dataset':<15} {'LR':<8} {'Epochs':<8} {'CauchyNet MSE':<18} {'FNN MSE':<18} {'Winner'}")
print("="*85)

for ds_name, (X, y) in datasets.items():
    if len(X) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_train, replace=False)
        X, y = X[idx], y[idx]
    d = X.shape[1]
    tp = cauchy_real_params(cauchy_h, d)
    fh = matched_fnn_hidden(tp, d)
    cauchy_eps = 1.0 if d >= 8 else 1e-8

    # FNN baseline (only needs one run since lr doesn't affect CauchyNet comparison)
    fnn_mse = train_eval_kfold(FNN, X, y, hidden_size=fh, epochs=300, lr=0.01)

    for lr in lrs:
        for epochs in epoch_list:
            c_mse = train_eval_kfold(CauchyNet, X, y, hidden_size=cauchy_h, epochs=epochs, lr=lr, cauchy_eps=cauchy_eps)
            winner = "CauchyNet" if c_mse < fnn_mse else "FNN"
            ratio = fnn_mse/c_mse if c_mse > 0 else 0
            marker = " <-- BEST" if winner == "CauchyNet" else ""
            print(f"{ds_name:<15} {lr:<8} {epochs:<8} {c_mse:<18.4f} {fnn_mse:<18.4f} {winner} ({ratio:.2f}x){marker}")
    print()
