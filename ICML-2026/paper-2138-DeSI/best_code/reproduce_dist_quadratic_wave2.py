#!/usr/bin/env python3
"""
Reproduction script for paper 2138: Deep Single-Index Fréchet Regression
Target: Dist. (Quad.), n=200, MPE via Wasserstein metric
Paper value: DeSI=0.2031 (0.0668)
"""
import os
import sys
import time

# Add the simulation_distribution directory to path
_REPO_DIR = '/repo'
_SUBMIT_DIR = os.path.join(_REPO_DIR, 'simulation_distribution')
if _SUBMIT_DIR not in sys.path:
    sys.path.insert(0, _SUBMIT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DeSI import DeSI_distribution
from generate_dist import generate_simulation_data_torch_true
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

# ── Paper settings ──────────────────────────────────────────────
N_SAMPLES = 200       # n in paper
P = 4                 # input_dim
QF_SIZE = 100         # quantile function grid
N_RUNS = 50          # Monte Carlo replications
LINK = "quadratic"    # link function
BATCH_SIZE = 64
N_EPOCHS = 10000
HIDDEN_DIM = 64
LR = 0.01
PATIENCE = 30
DELTA = 1e-6
LAMBDA_REG = 0.0005
LAMBDA_L1 = 1e-6           # L1 regularization (A-03)
SIGMA_AUG = 0.02          # noise augmentation scale (A-05)
BW_INIT = 0.1
SEED_START = 0

# ── Split ratios (matches simu.py) ──────────────────────────────
TRAIN_RATIO = 0.4
VAL_RATIO = 0.1
# test = remainder (100 for n=200)

def set_torch_threads():
    torch.set_num_threads(1)

class ThetaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)

    def forward(self, X):
        x = self.fc1(X)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        theta_raw = x
        theta_norm = torch.norm(theta_raw, dim=1, keepdim=True) + 1e-8
        theta = theta_raw / theta_norm
        sign = torch.where(theta[:, 0:1] < 0, -1.0, 1.0)
        theta = theta * sign
        return theta

class GlobalBandwidth(nn.Module):
    def __init__(self, bw_init=0.1):
        super().__init__()
        self.bw = nn.Parameter(torch.tensor([bw_init], dtype=torch.float32))

    @property
    def bandwidth(self):
        return torch.clamp(self.bw, min=0.01)


def run_single_reproduction(seed):
    """
    Run one Monte Carlo replication with paper settings.
    Returns: mpe (float) - mean prediction error (Wasserstein distance)
    """
    set_torch_threads()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data
    X, Y, theta_true, mu, sigma = generate_simulation_data_torch_true(
        n=N_SAMPLES, qf_size=QF_SIZE, p=P, link=LINK, seed=seed
    )
    qf_obs_torch = Y  # (n, qf_size)

    # Shuffle and split
    n = N_SAMPLES
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    n_test = n - n_train - n_val

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    qf_train, qf_val, qf_test = qf_obs_torch[idx_train], qf_obs_torch[idx_val], qf_obs_torch[idx_test]

    # Standardize
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Model
    model = ThetaMLP(P, HIDDEN_DIM, dropout_prob=0.3)
    global_bw = GlobalBandwidth(bw_init=BW_INIT)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(global_bw.parameters()),
        lr=LR, weight_decay=1e-3
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=50)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS - 50, eta_min=LR * 1e-4)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[50])

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_bw = None

    train_dataset = torch.utils.data.TensorDataset(X_train, qf_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for X_batch, qf_obs_batch in train_loader:
            # A-05: Gaussian noise augmentation on training quantile functions
            noise_std = SIGMA_AUG * qf_obs_batch.std()
            qf_obs_batch = qf_obs_batch + noise_std * torch.randn_like(qf_obs_batch)
            optimizer.zero_grad()
            theta_batch = model(X_batch)
            theta_batch = theta_batch / (torch.norm(theta_batch, dim=1, keepdim=True) + 1e-8)
            sign = torch.where(theta_batch[:, 0:1] < 0, -1.0, 1.0)
            theta_batch = theta_batch * sign
            Z_batch = torch.einsum('ij,ij->i', X_batch, theta_batch)

            y_batch = [qf_obs_batch[j] for j in range(qf_obs_batch.shape[0])]
            qf_pred = DeSI_distribution(
                y=y_batch, x=Z_batch, h=global_bw.bandwidth
            ).get('qf')

            l2_loss = torch.mean((qf_pred - qf_obs_batch) ** 2)
            y_batch_tensor = torch.stack(y_batch)
            mean_y = y_batch_tensor.mean(dim=0)
            frechet_var = torch.mean(torch.norm(y_batch_tensor - mean_y, dim=1) ** 2)
            denom = frechet_var + 1e-8
            norm_l2_loss = l2_loss / denom
            reg_term = LAMBDA_REG / (global_bw.bandwidth + 1e-8)
            # A-03: L1 sparsity penalty
            l1_penalty = LAMBDA_L1 * sum(p.abs().sum() for p in model.parameters())
            loss = norm_l2_loss + reg_term + l1_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(global_bw.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            theta_train = model(X_train)
            theta_train = theta_train / (torch.norm(theta_train, dim=1, keepdim=True) + 1e-8)
            sign = torch.where(theta_train[:, 0:1] < 0, -1.0, 1.0)
            theta_train = theta_train * sign
            Z_train = torch.einsum('ij,ij->i', X_train, theta_train)
            y_train = [qf_train[j] for j in range(qf_train.shape[0])]

            theta_val = model(X_val)
            theta_val = theta_val / (torch.norm(theta_val, dim=1, keepdim=True) + 1e-8)
            sign = torch.where(theta_val[:, 0:1] < 0, -1.0, 1.0)
            theta_val = theta_val * sign
            Z_val = torch.einsum('ij,ij->i', X_val, theta_val)

            result_val = DeSI_distribution(
                y=y_train, x=Z_train, xOut=Z_val, h=global_bw.bandwidth
            )
            qf_pred_val = result_val.get('qf')

            l2_loss_val = torch.mean((qf_pred_val - qf_val) ** 2)
            mean_y_val = qf_val.mean(dim=0)
            frechet_var_val = torch.mean(torch.norm(qf_val - mean_y_val, dim=1) ** 2)
            denom_val = frechet_var_val + 1e-8
            norm_l2_loss_val = l2_loss_val / denom_val
            reg_term_val = LAMBDA_REG / (global_bw.bandwidth + 1e-8)
            val_loss = (norm_l2_loss_val + reg_term_val).item()

            if val_loss < best_val_loss - DELTA:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_bw = global_bw.bandwidth.detach().cpu().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                if best_bw is not None:
                    global_bw.bw.data = best_bw.to(global_bw.bw.device)
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if best_bw is not None:
            global_bw.bw.data = best_bw.to(global_bw.bw.device)

    # ── Compute MPE on test set ──────────────────────────────────
    model.eval()
    with torch.no_grad():
        theta_test = model(X_test)
        theta_test = theta_test / (torch.norm(theta_test, dim=1, keepdim=True) + 1e-8)
        sign = torch.where(theta_test[:, 0:1] < 0, -1.0, 1.0)
        theta_test = theta_test * sign
        Z_test = torch.einsum('ij,ij->i', X_test, theta_test)

        # Use training data for local regression
        theta_train_final = model(X_train)
        theta_train_final = theta_train_final / (torch.norm(theta_train_final, dim=1, keepdim=True) + 1e-8)
        Z_train_final = torch.einsum('ij,ij->i', X_train, theta_train_final)
        y_train_list = [qf_train[j] for j in range(qf_train.shape[0])]

        # Predict quantile functions for test set
        result_test = DeSI_distribution(
            y=y_train_list, x=Z_train_final, xOut=Z_test, h=global_bw.bandwidth
        )
        qf_pred_test = result_test.get('qf')  # (n_test, qf_size)

        # Ground truth quantile functions
        qfSupp_np = np.linspace(0, 1, QF_SIZE + 2, dtype=np.float64)[1:-1]
        mu_test = mu[idx_test]
        sigma_test = sigma[idx_test]
        from scipy.stats import norm
        qf_true = np.zeros((n_test, QF_SIZE), dtype=np.float64)
        for i in range(n_test):
            sigma_i = max(float(sigma_test[i]), 1e-8)
            qf_true[i, :] = norm.ppf(qfSupp_np, loc=float(mu_test[i]), scale=sigma_i)
        qf_true_t = torch.tensor(qf_true, dtype=qf_pred_test.dtype, device=qf_pred_test.device)

        # Wasserstein-2 distance = sqrt(∫ (Q1 - Q2)² dp) ≈ ||diff||_2 / sqrt(M)
        wasserstein_distances = torch.norm(qf_pred_test - qf_true_t, dim=1) / np.sqrt(QF_SIZE)
        mpe = wasserstein_distances.mean().item()

    return mpe


def main():
    print(f"=== Reproduction: DeSI Dist. (Quad.) n={N_SAMPLES} ===")
    print(f"Settings: p={P}, qf_size={QF_SIZE}, link={LINK}, n_runs={N_RUNS}")
    print(f"Hyperparams: batch_size={BATCH_SIZE}, hidden_dim={HIDDEN_DIM}, lr={LR}")
    print(f"             n_epochs={N_EPOCHS}, patience={PATIENCE}, lambda_reg={LAMBDA_REG}")
    print(f"             bw_init={BW_INIT}")
    print(f"Split: train={int(TRAIN_RATIO*N_SAMPLES)}, val={int(VAL_RATIO*N_SAMPLES)}, test={N_SAMPLES - int(TRAIN_RATIO*N_SAMPLES) - int(VAL_RATIO*N_SAMPLES)}")
    print()

    mpe_list = []
    start_time = time.time()

    for run_idx in range(N_RUNS):
        seed = SEED_START + run_idx
        t0 = time.time()
        try:
            mpe = run_single_reproduction(seed)
            mpe_list.append(mpe)
            elapsed = time.time() - t0
            total_elapsed = time.time() - start_time
            avg_time = total_elapsed / (run_idx + 1)
            remaining = avg_time * (N_RUNS - run_idx - 1)

            print(f"[{run_idx+1:3d}/{N_RUNS}] seed={seed:3d}  MPE={mpe:.6f}  "
                  f"time={elapsed:.1f}s  elapsed={total_elapsed/60:.1f}m  "
                  f"eta={remaining/60:.1f}m")
        except Exception as e:
            print(f"[{run_idx+1:3d}/{N_RUNS}] seed={seed:3d}  FAILED: {e}")
            continue

    total_time = time.time() - start_time

    if len(mpe_list) == 0:
        print("ERROR: All runs failed!")
        return

    mpe_arr = np.array(mpe_list)
    mean_mpe = np.mean(mpe_arr)
    std_mpe = np.std(mpe_arr, ddof=1)

    print()
    print("=" * 60)
    print(f"RESULTS (over {len(mpe_list)} successful runs):")
    print(f"  MPE mean: {mean_mpe:.6f}")
    print(f"  MPE std:  {std_mpe:.6f}")
    print(f"  Paper:    DeSI=0.2031 (0.0668)")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print("=" * 60)

    # Check against rubric bounds
    lower = 0.1363
    upper = 0.2699
    paper_val = 0.2031
    print(f"\nRubric bounds: [{lower}, {upper}]")
    if lower <= mean_mpe <= upper:
        print(f"✓ Mean MPE {mean_mpe:.4f} within CI bounds")
        if mean_mpe <= upper:
            print(f"✓ Reproduced result better than or equal to upper bound ({upper})")
    else:
        print(f"✗ Mean MPE {mean_mpe:.4f} outside CI bounds [{lower}, {upper}]")

    # Save results
    results = {
        'n_runs': len(mpe_list),
        'mpe_mean': float(mean_mpe),
        'mpe_std': float(std_mpe),
        'mpe_list': [float(x) for x in mpe_list],
        'paper_value': paper_val,
        'paper_std': 0.0668,
        'rubric_ci_lower': lower,
        'rubric_ci_upper': upper,
        'total_time_minutes': total_time / 60.0,
    }
    import json
    with open('/repo/reproduction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /repo/reproduction_results.json")


if __name__ == '__main__':
    main()
