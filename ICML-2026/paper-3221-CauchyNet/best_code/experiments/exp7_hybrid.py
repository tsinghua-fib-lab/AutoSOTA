"""Test hybrid architectures: CauchyNet layer + FNN layers on UCI datasets."""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import load_diabetes, fetch_california_housing, fetch_openml
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from shared import (
    CauchyNet, FNN, count_real_params, ReciprocalActivation,
    BATCH_SIZE, WEIGHT_DECAY, IMAG_PENALTY
)


class HybridCauchyFNN(nn.Module):
    """CauchyNet first layer -> FNN second layer.
    CauchyNet maps input to h1 features (real part of pole outputs),
    then FNN maps h1 -> output."""
    def __init__(self, input_size, cauchy_h, fnn_h, output_size, eps=1e-8, lambda_std=None):
        super().__init__()
        if lambda_std is None:
            lambda_std = 0.1 if input_size <= 2 else max(0.5, input_size / 5.0)
        # CauchyNet poles
        angles = 2 * np.pi * torch.rand(cauchy_h, input_size)
        real_part = 2 * np.pi * torch.cos(angles)
        imaginary_part = torch.sin(angles)
        self.xi = nn.Parameter(torch.complex(real_part, imaginary_part))
        self.activation = ReciprocalActivation(eps=eps)
        # FNN on top of CauchyNet real outputs
        self.fc1 = nn.Linear(cauchy_h, fnn_h)
        self.fc2 = nn.Linear(fnn_h, output_size)

    def forward(self, x):
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(1)
        reciprocals = self.activation(self.xi - xc)
        activated = reciprocals.prod(dim=-1)  # (batch, cauchy_h) complex
        features = activated.real  # use real part as features
        h = torch.relu(self.fc1(features))
        return self.fc2(h)


class HybridFNNCauchy(nn.Module):
    """FNN first layer -> CauchyNet output layer.
    FNN reduces input, then CauchyNet applies pole structure."""
    def __init__(self, input_size, fnn_h, cauchy_h, output_size, eps=1e-8, lambda_std=None):
        super().__init__()
        if lambda_std is None:
            lambda_std = 0.1 if fnn_h <= 2 else max(0.5, fnn_h / 5.0)
        # FNN first layer
        self.fc1 = nn.Linear(input_size, fnn_h)
        # CauchyNet on reduced features
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
cauchy_h = 64

print(f"{'Dataset':<15} {'Model':<25} {'Params':<8} {'MSE':<20} {'Winner vs FNN'}")
print("="*80)

for ds_name, (X, y) in datasets.items():
    if len(X) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_train, replace=False)
        X, y = X[idx], y[idx]
    d = X.shape[1]
    cauchy_eps = 1.0

    models = {
        'CauchyNet': lambda d=d: CauchyNet(d, cauchy_h, 1, eps=cauchy_eps),
        'FNN': lambda d=d: FNN(d, 117, 1),
        'Cauchy->FNN (h=32,32)': lambda d=d: HybridCauchyFNN(d, 32, 32, 1, eps=cauchy_eps),
        'Cauchy->FNN (h=16,64)': lambda d=d: HybridCauchyFNN(d, 16, 64, 1, eps=cauchy_eps),
        'FNN->Cauchy (h=32,32)': lambda d=d: HybridFNNCauchy(d, 32, 32, 1, eps=cauchy_eps),
        'FNN->Cauchy (h=16,16)': lambda d=d: HybridFNNCauchy(d, 16, 16, 1, eps=cauchy_eps),
    }

    fnn_mse = None
    for name, factory in models.items():
        mse_mean, mse_std, nparams = train_eval_kfold(factory, X, y, lr=0.05)
        if name == 'FNN':
            fnn_mse = mse_mean
        winner = "WIN" if fnn_mse and mse_mean < fnn_mse else ("TIE" if fnn_mse and abs(mse_mean-fnn_mse)/fnn_mse < 0.05 else "LOSS")
        if name == 'FNN': winner = "---"
        print(f"{ds_name:<15} {name:<25} {nparams:<8} {mse_mean:.4f}+/-{mse_std:.4f}  {winner}")
    print()
