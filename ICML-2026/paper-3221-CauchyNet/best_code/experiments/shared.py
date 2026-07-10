"""
Shared utilities for all rebuttal experiments.
Hyperparameters match the paper (supp.tex Table 2).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# ── Paper hyperparameters (supp.tex Table 2) ──────────────────────
HIDDEN_SIZE = 64
BATCH_SIZE = 32
LR = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 200
IMAG_PENALTY = 0.1  # lambda for imaginary part


# ── Parameter counting ────────────────────────────────────────────

def count_real_params(model):
    """Count trainable parameters as real scalars (complex params count double)."""
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            n = p.numel()
            if p.is_complex():
                n *= 2
            total += n
    return total


# ── CauchyNet ────────────────────────────────────────────────────

class ReciprocalActivation(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return 1.0 / (x + self.eps)


class CauchyNet(nn.Module):
    """h_k = prod_{i=1}^N (xi_{k,i} - x_i + eps)^{-1}, y = sum_k lambda_k * h_k.
    Returns (y_real, y_imag) for imaginary penalty."""
    def __init__(self, input_size, hidden_size, output_size, eps=1e-8, lambda_std=None):
        super().__init__()
        # Lambda std: 0.1 for d<=2, max(0.5, d/5) for d>2
        if lambda_std is None:
            lambda_std = 0.1 if input_size <= 2 else max(0.5, input_size / 5.0)
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=lambda_std, size=(hidden_size, output_size), dtype=torch.cfloat)
        )
        # xi: (h, input_size) complex — one pole per hidden unit per input dim
        angles = 2 * np.pi * torch.rand(hidden_size, input_size)
        real_part = 2 * np.pi * torch.cos(angles)
        imaginary_part = torch.sin(angles)
        self.xi = nn.Parameter(torch.complex(real_part, imaginary_part))
        self.activation = ReciprocalActivation(eps=eps)
        self.input_size = input_size

    def forward(self, x):
        # x: (batch, input_size)
        xc = torch.complex(x, torch.zeros_like(x))  # (batch, input_size)
        xc = xc.unsqueeze(1)  # (batch, 1, input_size)
        # xi: (hidden_size, input_size) broadcasts to (batch, h, input_size)
        reciprocals = self.activation(self.xi - xc)  # (batch, h, input_size)
        # Product over input dimensions (paper eq: h_k = prod_i 1/(xi_{k,i} - x_i))
        activated = reciprocals.prod(dim=-1)  # (batch, h)
        y = torch.matmul(activated, self.lambda_)  # (batch, output_size) complex
        return y.real.squeeze(-1).unsqueeze(-1), y.imag.squeeze(-1).unsqueeze(-1)


class CauchyNet1D(CauchyNet):
    """1D input wrapper with scalar xi init."""
    def __init__(self, hidden_size, output_size=1, eps=1e-8):
        # Skip parent __init__, do our own
        nn.Module.__init__(self)
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=0.1, size=(hidden_size, output_size), dtype=torch.cfloat)
        )
        angles = 2 * np.pi * torch.rand(hidden_size)
        real_part = 2 * np.pi * torch.cos(angles)
        imaginary_part = torch.sin(angles)
        self.xi = nn.Parameter(torch.complex(real_part, imaginary_part))
        self.activation = ReciprocalActivation(eps=eps)
        self.input_size = 1

    def forward(self, x):
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(-1)  # (batch, 1, 1)
        xc = xc.view(x.size(0), 1, -1)  # (batch, 1, 1)
        activated = self.activation(self.xi - xc)  # (batch, h, 1)
        activated_flat = activated.view(x.size(0), -1)  # (batch, h)
        y = torch.matmul(activated_flat, self.lambda_)  # (batch, 1) complex
        return y.real.squeeze(-1).unsqueeze(-1), y.imag.squeeze(-1).unsqueeze(-1)


class HybridFNNCauchy(nn.Module):
    """FNN bottleneck -> CauchyNet output layer.
    Returns (y_real, y_imag) for imaginary penalty."""
    def __init__(self, input_size, fnn_h, cauchy_h, output_size=1, eps=1e-8):
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


# ── Parameter matching ────────────────────────────────────────────

def cauchy_real_params(hidden_size, input_size=1, output_size=1):
    """Real param count for CauchyNet with product structure:
    xi(h, d) + lambda(h, out), both complex → 2*(h*d + h*out) real."""
    return 2 * (hidden_size * input_size + hidden_size * output_size)

def matched_fnn_hidden(target_params, input_size=1, output_size=1):
    """Hidden size for single-layer FNN/SIREN to match target_params.
    FNN params = (input_size+1)*h + (h+1)*output_size = h*(input_size+1+output_size) + output_size
    """
    return (target_params - output_size) // (input_size + 1 + output_size)


# ── Baselines ─────────────────────────────────────────────────────

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class FNN_Deep(nn.Module):
    """2-hidden-layer FNN."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SIREN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, omega_0=30.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.omega_0 = omega_0
        with torch.no_grad():
            self.fc1.weight.uniform_(-1.0 / input_size, 1.0 / input_size)
            self.fc2.weight.uniform_(
                -np.sqrt(6.0 / hidden_size) / self.omega_0,
                np.sqrt(6.0 / hidden_size) / self.omega_0
            )

    def forward(self, x):
        return self.fc2(torch.sin(self.omega_0 * self.fc1(x)))


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads,
            dim_feedforward=hidden_size, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.fc(x)


class NBeatsBlock(nn.Module):
    """One N-BEATS block. Two FC+ReLU stages, two heads (backcast, forecast)."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.backcast_head = nn.Linear(hidden_size, input_size)
        self.forecast_head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.backcast_head(h), self.forecast_head(h)


class NBeats(nn.Module):
    """Standard N-BEATS: stacked blocks with residual backcast and summed forecast.

    Differs from a vanilla FNN by (i) backcast/forecast double-head per block,
    (ii) residual subtraction in the input stream, (iii) accumulated forecasts.
    With num_blocks=3 this is the configuration used in the Oreshkin et al.
    (ICLR 2020) ablations for generic blocks.
    """
    def __init__(self, input_size=1, hidden_size=128, output_size=1, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, output_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        residual = x
        forecast = 0
        for block in self.blocks:
            backcast, fc = block(residual)
            residual = residual - backcast
            forecast = forecast + fc
        return forecast


# ── Target function (paper Section 4.1) ──────────────────────────

def target_function(x):
    """f(x) from the paper's 1D experiment."""
    return (
        1.0 / ((x + 0.6) ** 2 + 0.005)
        - 40.0 * np.exp(-2.0 * (x + 0.4) ** 2)
        + 50.0 * np.sign(x) * np.abs(np.sin(3 * x) + 0.8) ** 1.5 * np.sin(10 * x)
    )


# ── Data preparation with MinMaxScaler ────────────────────────────

def prepare_1d_data(n_points=50, seed=0):
    """Generate and split 1D data with MinMaxScaler. 50/25/25 split."""
    x_np = np.linspace(-1, 1, n_points).astype(np.float32)
    y_np = target_function(x_np).astype(np.float32)

    # MinMaxScaler on targets
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_np.reshape(-1, 1)).flatten().astype(np.float32)

    # 50/25/25 split
    indices = np.arange(n_points)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    n_train = int(0.5 * n_points)
    n_val = int(0.25 * n_points)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train = torch.tensor(x_np[train_idx]).unsqueeze(-1)
    Y_train = torch.tensor(y_scaled[train_idx])
    X_val = torch.tensor(x_np[val_idx]).unsqueeze(-1)
    Y_val = torch.tensor(y_scaled[val_idx])
    X_test = torch.tensor(x_np[test_idx]).unsqueeze(-1)
    Y_test = torch.tensor(y_scaled[test_idx])
    # Keep unscaled test targets for reporting
    Y_test_raw = torch.tensor(y_np[test_idx])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_test_raw, scaler_y


# ── Training ─────────────────────────────────────────────────────

def _eval_loss(model, X, Y, criterion, imag_penalty, device):
    """Compute loss on a dataset (for validation-based early stopping)."""
    model.eval()
    with torch.no_grad():
        out = model(X.to(device))
        if isinstance(out, tuple):
            y_real, y_imag = out
            y_exp = Y.to(device).unsqueeze(-1)
            loss = criterion(y_real, y_exp) + imag_penalty * criterion(
                y_imag, torch.zeros_like(y_imag)
            )
        else:
            y_exp = Y.to(device).unsqueeze(-1)
            loss = criterion(out, y_exp)
    return loss.item()


def train_and_eval(model, X_train, Y_train, X_test, Y_test,
                   epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                   imag_penalty=IMAG_PENALTY, device='cpu',
                   return_grad_norms=False,
                   X_val=None, Y_val=None, patience=None):
    """Train with early stopping. Handles CauchyNet (real,imag) tuple output.
    If patience is set, stops after `patience` epochs with no improvement.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    # Paper: decay by 0.5 every 100 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    use_val = X_val is not None and Y_val is not None

    best_monitor_loss = float('inf')
    best_state = None
    best_epoch = 0
    wait = 0
    train_losses = []
    grad_norms = [] if return_grad_norms else None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_grad = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)

            if isinstance(out, tuple):
                y_real, y_imag = out
                y_exp = yb.unsqueeze(-1)
                loss = criterion(y_real, y_exp) + imag_penalty * criterion(
                    y_imag, torch.zeros_like(y_imag)
                )
            else:
                y_exp = yb.unsqueeze(-1)
                loss = criterion(out, y_exp)

            loss.backward()

            if return_grad_norms:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                epoch_grad += total_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        if return_grad_norms:
            grad_norms.append(epoch_grad / n_batches)

        # Monitor loss for early stopping: use val if available, else train
        if use_val:
            monitor_loss = _eval_loss(model, X_val, Y_val, criterion, imag_penalty, device)
        else:
            monitor_loss = avg_loss

        if monitor_loss < best_monitor_loss:
            best_monitor_loss = monitor_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if patience is not None and wait >= patience:
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(X_test.to(device))
        if isinstance(out, tuple):
            preds = out[0].cpu().numpy().flatten()
        else:
            preds = out.cpu().numpy().flatten()

    truths = Y_test.numpy().flatten()
    result = {
        'mse': float(mean_squared_error(truths, preds)),
        'mae': float(mean_absolute_error(truths, preds)),
        'train_losses': train_losses,
        'num_params': count_real_params(model),
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
    }
    if return_grad_norms:
        result['grad_norms'] = grad_norms
    return result
