import os
import sys

# Use generate_dist.py and DeSI.py from this directory only (submitf)
_SUBMIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SUBMIT_DIR not in sys.path:
    sys.path.insert(0, _SUBMIT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DeSI import DeSI_distribution
from generate_dist import generate_simulation_data_torch_true
from torch.optim.lr_scheduler import StepLR
from scipy.stats import norm

# Parameters (copied from DeSI/simu_quad200.py)
p = 4
n = 500
qf_size = 100
batch_size = 64
n_epochs = 10000
hidden_dim = 64
lr = 0.01
patience = 10
delta = 1e-4  # Minimum improvement threshold for early stopping
lambda_reg = 0.0005
buffer = 5

def set_torch_threads():
    torch.set_num_threads(1)


def theta_estimate_l2_distance(theta_mean_np, theta_true, train_feat_std_torch=None):
    """
    ‖θ̂ - θ_eff‖ after normalization, identifying ±θ. If train_feat_std_torch gives
    per-column std σ of standardized inputs z=(x-mu)/σ, θ_eff ∝ θ ⊙ σ normalized.
    If None, compares θ with θ̂ without scale adjustment between spaces.
    """
    theta_hat = np.asarray(theta_mean_np, dtype=np.float64).reshape(-1)
    nh = np.linalg.norm(theta_hat)
    if nh < 1e-12:
        return float("nan")
    theta_hat = theta_hat / nh

    theta_t = theta_true.detach().cpu().numpy().reshape(-1).astype(np.float64)
    theta_t = theta_t / (np.linalg.norm(theta_t) + 1e-12)

    if train_feat_std_torch is not None:
        s = train_feat_std_torch.detach().cpu().numpy().reshape(-1)
        theta_t = theta_t * np.maximum(s, 1e-12)
        tn = np.linalg.norm(theta_t)
        if tn < 1e-12:
            return float("nan")
        theta_t = theta_t / tn

    if np.dot(theta_hat, theta_t) < 0:
        theta_t = -theta_t
    return float(np.linalg.norm(theta_hat - theta_t))


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

def run_single_sim(seed):
    set_torch_threads()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Available links: "linear", "quadratic", and "exp".
    # Use "linear" here for the linear single-index simulation setting.
    X, Y, theta, mu, sigma = generate_simulation_data_torch_true(n=n, qf_size=qf_size, p=p, link="linear", seed=seed)
    qf_obs = [Y[i,] for i in range(n)]
    qf_obs_torch = torch.stack(qf_obs)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.4 * n)
    n_val = int(0.1 * n)
    n_test = 200
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train+n_val]
    idx_test = idx[n_train+n_val:n_train+n_val+n_test]
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    qf_train, qf_val, qf_test = qf_obs_torch[idx_train], qf_obs_torch[idx_val], qf_obs_torch[idx_test]
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    model = ThetaMLP(p, hidden_dim, dropout_prob=0.3)
    global_bw = GlobalBandwidth(bw_init=0.1)
    optimizer = optim.Adam(list(model.parameters()) + list(global_bw.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_bw = None
    buffer_counter = 0
    train_dataset = torch.utils.data.TensorDataset(X_train, qf_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_losses = []
    train_losses = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for X_batch, qf_obs_batch in train_loader:
            optimizer.zero_grad()
            theta_batch = model(X_batch)
            theta_batch = theta_batch / (torch.norm(theta_batch, dim=1, keepdim=True) + 1e-8)
            sign = torch.where(theta_batch[:, 0:1] < 0, -1.0, 1.0)
            theta_batch = theta_batch * sign
            Z_batch = torch.einsum('ij,ij->i', X_batch, theta_batch)
            y_batch = [qf_obs_batch[j] for j in range(qf_obs_batch.shape[0])]
            qf_pred = DeSI_distribution(y=y_batch, x=Z_batch, h=global_bw.bandwidth).get('qf')
            l2_loss = torch.mean((qf_pred - qf_obs_batch) ** 2)
            y_batch_tensor = torch.stack(y_batch)
            mean_y = y_batch_tensor.mean(dim=0)
            frechet_var = torch.mean(torch.norm(y_batch_tensor - mean_y, dim=1) ** 2)
            denom = frechet_var + 1e-8
            norm_l2_loss = l2_loss / denom
            reg_term = lambda_reg / (global_bw.bandwidth + 1e-8)
            loss = norm_l2_loss + reg_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
        avg_train_loss = total_loss / total_samples
        train_losses.append(avg_train_loss)
        scheduler.step()
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[Seed {seed}] Epoch {epoch+1}/{n_epochs}, Loss: {avg_train_loss:.6f}")
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
            result_val = DeSI_distribution(y=y_train, x=Z_train, xOut=Z_val, h=global_bw.bandwidth)
            qf_pred_val = result_val.get('qf')
            l2_loss_val = torch.mean((qf_pred_val - qf_val) ** 2)
            mean_y_val = qf_val.mean(dim=0)
            frechet_var_val = torch.mean(torch.norm(qf_val - mean_y_val, dim=1) ** 2)
            denom_val = frechet_var_val + 1e-8
            norm_l2_loss_val = l2_loss_val / denom_val
            reg_term_val = lambda_reg / (global_bw.bandwidth + 1e-8)
            val_loss = (norm_l2_loss_val + reg_term_val).item()
            val_losses.append(val_loss)
            # Standard early stopping criterion:
            # Stop when val_loss(t) > min val_loss(s) - δ for 'patience' consecutive epochs
            current_val_loss = val_loss
            
            if current_val_loss < best_val_loss - delta:
                # Validation loss improved by at least delta
                best_val_loss = current_val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_bw = global_bw.bandwidth.detach().cpu().clone()
                patience_counter = 0
            else:
                # Validation loss did not improve by at least delta
                patience_counter += 1
            
            if patience_counter >= patience:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                if best_bw is not None:
                    global_bw.bw.data = best_bw.to(global_bw.bw.device)
                break
    # Compute Frechet mean theta on training set
    with torch.no_grad():
        theta_final = model(X_train)
        theta_final = theta_final / (torch.norm(theta_final, dim=1, keepdim=True) + 1e-8)
        theta_mean = theta_final.mean(dim=0)
        theta_mean_np = theta_mean.detach().cpu().numpy()
        theta_theta_l2 = theta_estimate_l2_distance(theta_mean_np, theta, X_std)
        # Also compute theta predictions for X_test
        theta_test = model(X_test)
        theta_test = theta_test / (torch.norm(theta_test, dim=1, keepdim=True) + 1e-8)
        sign = torch.where(theta_test[:, 0:1] < 0, -1.0, 1.0)
        theta_test = theta_test * sign
        theta_test_np = theta_test.detach().cpu().numpy()
        # Predict quantile functions for test set
        Z_test = torch.einsum('ij,ij->i', X_test, theta_test)
        y_train = [qf_train[j] for j in range(qf_train.shape[0])]
        Z_train = torch.einsum('ij,ij->i', X_train, theta_final)
        result_test = DeSI_distribution(y=y_train, x=Z_train, xOut=Z_test, h=global_bw.bandwidth)
        qf_pred_test = result_test.get('qf')
        # Calculate true quantile functions for test set using mu and sigma
        mu_test = mu[idx_test]
        sigma_test = sigma[idx_test]
        qf_true_test = np.zeros_like(qf_pred_test.cpu().numpy())
        # Interior grid: same as generate_dist.py / GFR_square.R
        qfSupp_np = np.linspace(0, 1, qf_size + 2, dtype=np.float64)[1:-1]
        for i in range(len(mu_test)):
            # Add safety checks for sigma
            sigma_i = max(sigma_test[i].item(), 1e-8)  # Ensure sigma is positive
            qf_true_test[i, :] = norm.ppf(qfSupp_np, loc=mu_test[i].item(), scale=sigma_i)
        qf_true_test = torch.tensor(qf_true_test, dtype=qf_pred_test.dtype, device=qf_pred_test.device)
        # Compute and print the average L2 distance between predicted and true quantile functions in the test set
        l2_distances = torch.norm(qf_pred_test - qf_true_test, dim=1)  # (n_test,)
        avg_l2_distance = l2_distances.mean().item()
        print(f"[Seed {seed}] L2 distance (theta estimate vs true, z-space match): {theta_theta_l2:.6f}")
        print(f"[Seed {seed}] Average L2 distance between predicted and true quantile functions (test set): {avg_l2_distance:.6f}")
    
    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if best_bw is not None:
            global_bw.bw.data = best_bw.to(global_bw.bw.device)
    
    return theta_mean_np, theta_test_np, avg_l2_distance, theta_theta_l2