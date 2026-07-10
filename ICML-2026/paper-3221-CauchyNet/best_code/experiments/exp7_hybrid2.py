"""Sweep FNN->Cauchy hybrid sizes on UCI datasets."""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import load_diabetes, fetch_california_housing, fetch_openml
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from shared import (
    CauchyNet, FNN, count_real_params, ReciprocalActivation,
    BATCH_SIZE, WEIGHT_DECAY, IMAG_PENALTY
)


class HybridFNNCauchy(nn.Module):
    """FNN first layer -> CauchyNet output layer."""
    def __init__(self, input_size, fnn_h, cauchy_h, output_size, eps=1e-8):
        super().__init__()
        lambda_std = max(0.5, fnn_h / 5.0)
        self.fc1 = nn.Linear(input_size, fnn_h)
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=lambda_std, size=(cauchy_h, output_size), dtype=torch.cfloat)
        )
        angles = 2 * np.pi * torch.rand(cauchy_h, fnn_h)
        real_part = 2 * np.pi * torch.cos(angles)
        imaginary_part = torch.sin(angles)
        self.xi = nn.Parameter(torch.complex(real_part, imaginary_part))
        self.activation = ReciprocalActivation(eps=eps)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        xc = torch.complex(h, torch.zeros_like(h)).unsqueeze(1)
        reciprocals = self.activation(self.xi - xc)
        activated = reciprocals.prod(dim=-1)
        y = torch.matmul(activated, self.lambda_)
        return y.real.squeeze(-1).unsqueeze(-1), y.imag.squeeze(-1).unsqueeze(-1)


class HybridCauchyFNN(nn.Module):
    """CauchyNet first -> FNN output."""
    def __init__(self, input_size, cauchy_h, fnn_h, output_size, eps=1e-8):
        super().__init__()
        angles = 2 * np.pi * torch.rand(cauchy_h, input_size)
        real_part = 2 * np.pi * torch.cos(angles)
        imaginary_part = torch.sin(angles)
        self.xi = nn.Parameter(torch.complex(real_part, imaginary_part))
        self.activation = ReciprocalActivation(eps=eps)
        self.fc1 = nn.Linear(cauchy_h, fnn_h)
        self.fc2 = nn.Linear(fnn_h, output_size)

    def forward(self, x):
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(1)
        reciprocals = self.activation(self.xi - xc)
        activated = reciprocals.prod(dim=-1)
        features = activated.real
        h = torch.relu(self.fc1(features))
        return self.fc2(h)


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


def train_eval_kfold(model_factory, X, y, n_folds=5, epochs=200, lr=0.05):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mse_scores = []
    num_params = None
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        scX, scy = MinMaxScaler(), MinMaxScaler()
        X_tr = scX.fit_transform(X[tr_idx]).astype(np.float32)
        X_te = scX.transform(X[te_idx]).astype(np.float32)
        y_tr = scy.fit_transform(y[tr_idx].reshape(-1,1)).flatten().astype(np.float32)
        y_te_raw = y[te_idx]
        Xtr, Ytr = torch.tensor(X_tr), torch.tensor(y_tr)
        Xte = torch.tensor(X_te)

        torch.manual_seed(42+fold)
        model = model_factory()
        if num_params is None:
            num_params = count_real_params(model)

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

    return np.mean(mse_scores), np.std(mse_scores), num_params


datasets = load_datasets()
max_train = 100

# Configs: (fnn_h, cauchy_h) for FNN->Cauchy
fnn_cauchy_configs = [(8,8), (8,16), (16,8), (16,16), (16,32), (32,8), (32,16), (4,32), (4,16)]
# Configs: (cauchy_h, fnn_h) for Cauchy->FNN
cauchy_fnn_configs = [(8,32), (16,32), (32,32), (8,64), (16,16)]

print(f"{'Dataset':<12} {'Model':<30} {'Params':<7} {'MSE':<12} {'vs FNN'}")
print("="*75)

for ds_name, (X, y) in datasets.items():
    if len(X) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_train, replace=False)
        X, y = X[idx], y[idx]
    d = X.shape[1]

    # FNN baseline
    fnn_mse, fnn_std, fnn_p = train_eval_kfold(lambda d=d: FNN(d, 117, 1), X, y)
    print(f"{ds_name:<12} {'FNN (h=117)':<30} {fnn_p:<7} {fnn_mse:.4f}      ---")

    # CauchyNet standalone
    c_mse, c_std, c_p = train_eval_kfold(lambda d=d: CauchyNet(d, 64, 1, eps=1.0), X, y)
    tag = "WIN" if c_mse < fnn_mse else "LOSS"
    print(f"{'':12} {'CauchyNet (h=64)':<30} {c_p:<7} {c_mse:.4f}      {tag}")

    # FNN->Cauchy hybrids
    for fh, ch in fnn_cauchy_configs:
        m, s, p = train_eval_kfold(lambda d=d, fh=fh, ch=ch: HybridFNNCauchy(d, fh, ch, 1, eps=1.0), X, y)
        tag = "WIN" if m < fnn_mse else "LOSS"
        print(f"{'':12} {f'FNN({fh})->Cauchy({ch})':<30} {p:<7} {m:.4f}      {tag}")

    # Cauchy->FNN hybrids
    for ch, fh in cauchy_fnn_configs:
        m, s, p = train_eval_kfold(lambda d=d, ch=ch, fh=fh: HybridCauchyFNN(d, ch, fh, 1, eps=1.0), X, y)
        tag = "WIN" if m < fnn_mse else "LOSS"
        print(f"{'':12} {f'Cauchy({ch})->FNN({fh})':<30} {p:<7} {m:.4f}      {tag}")

    print()
