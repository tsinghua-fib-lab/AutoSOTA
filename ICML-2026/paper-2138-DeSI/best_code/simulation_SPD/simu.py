import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_matrices import generate_spd_matrices
from DeSI import DeSI_SPD

# Parameters
p = 4
n = 500
q = 3
batch_size = 64
n_epochs = 1000
hidden_dim = 64
lr = 0.1
patience = 5
delta = 1e-4  # Minimum improvement threshold for early stopping
lambda_reg = 0.01  # Regularization hyperparameter for 1/h term

def set_torch_threads():
    torch.set_num_threads(1)

def add_diagonal_jitter(M_list, jitter=1e-8):
    """Add small diagonal jitter to a list of matrices to ensure SPD."""
    jittered_list = []
    for M in M_list:
        M_jittered = M.clone()
        M_jittered.diagonal().add_(jitter)
        jittered_list.append(M_jittered)
    return jittered_list

def add_jitter_if_needed(M_list, jitter=1e-3):
    """Add diagonal jitter only to matrices that fail Cholesky decomposition."""
    jittered_list = []
    for M in M_list:
        # Try Cholesky decomposition
        try:
            # Ensure symmetry
            M_sym = 0.5 * (M + M.transpose(-1, -2))
            torch.linalg.cholesky(M_sym)
            # Cholesky succeeded, use original matrix
            jittered_list.append(M)
        except RuntimeError:
            # Cholesky failed, add jitter
            M_jittered = M.clone()
            M_jittered.diagonal().add_(jitter)
            jittered_list.append(M_jittered)
    return jittered_list

def log_cholesky_distance(S1, S2):
    """
    Log-Cholesky distance using R-style upper Cholesky factor:
    d(S1,S2) = sqrt( ||sUT(U1) - sUT(U2)||_F^2 + ||log diag(U1) - log diag(U2)||_2^2 ),
    where U = chol(S) is upper-triangular (we take U = L^T with L = torch cholesky).
    """
    dtype0 = S1.dtype
    S1 = S1.to(torch.float64)
    S2 = S2.to(torch.float64)

    # Ensure symmetry
    S1 = 0.5 * (S1 + S1.transpose(-1, -2))
    S2 = 0.5 * (S2 + S2.transpose(-1, -2))
    
    # Cholesky decomposition
    L1 = torch.linalg.cholesky(S1)
    L2 = torch.linalg.cholesky(S2)
    
    # Convert to upper triangular (R-style)
    U1 = L1.transpose(-1, -2)
    U2 = L2.transpose(-1, -2)

    # Strictly upper parts (exclude diagonal)
    sUT1 = torch.triu(U1, diagonal=1)
    sUT2 = torch.triu(U2, diagonal=1)
    
    # Off-diagonal distance
    off_dist_sq = torch.norm(sUT1 - sUT2, p='fro') ** 2

    # Diagonal log distance
    d1 = torch.diagonal(U1, dim1=-2, dim2=-1)
    d2 = torch.diagonal(U2, dim1=-2, dim2=-1)
    
    logD_dist_sq = (torch.log(d1) - torch.log(d2)).pow(2).sum()

    # Total distance
    result = torch.sqrt(off_dist_sq + logD_dist_sq).to(dtype0)
    
    return result

def frechet_variance_log_cholesky(M_list):
    """Compute Fréchet variance using log-Cholesky distance"""
    n = len(M_list)
    if n <= 1:
        return torch.tensor(0.0, device=M_list[0].device)
    
    # Compute Fréchet mean using iterative algorithm in log-Cholesky space
    M_mean = compute_frechet_mean_log_cholesky(M_list)
    
    # Compute distances to mean
    dists = torch.stack([log_cholesky_distance(M, M_mean) for M in M_list])
    
    return dists.mean()

def compute_frechet_mean_log_cholesky(M_list):
    """Compute Fréchet mean using the same method as conditional means in generate_matrices.py"""
    n = len(M_list)
    if n == 1:
        return M_list[0]
    
    d = M_list[0].shape[0]
    
    # Calculate Fréchet mean using log-Cholesky metric
    # The minimizer is the arithmetic mean of sLT(L) and log(D(L))
    sLT_means = torch.zeros(d, d, device=M_list[0].device, dtype=M_list[0].dtype)
    logD_means = torch.zeros(d, d, device=M_list[0].device, dtype=M_list[0].dtype)
    
    for j in range(n):
        # Cholesky decomposition S = LL^T
        L = torch.linalg.cholesky(M_list[j])
        
        # Extract strictly lower triangular part sLT(L)
        sLT = torch.tril(L, diagonal=-1)
        sLT_means += sLT
        
        # Extract diagonal and take log
        D = torch.diag(torch.diag(L))
        logD_means += torch.diag_embed(torch.log(torch.diagonal(L)))
    
    sLT_mean = sLT_means / n
    logD_mean = logD_means / n
    
    # Reconstruct mean matrix
    expD = torch.diag(torch.exp(torch.diagonal(logD_mean)))
    L_mean = sLT_mean + expD
    M_mean = L_mean @ L_mean.T
    
    return M_mean

def lfr_loss_selffit(X, M_list, theta, h):
    """LFR loss with self-fitting (no separate validation)"""
    device = X.device
    n_samples = X.shape[0]
    
    # Compute Z = <X_i, theta_i> for each sample (theta is batch-aligned)
    Z = (X * theta).sum(dim=1, keepdim=True)
    
    # Add diagonal jitter only if Cholesky decomposition fails
    M_list_jittered = add_jitter_if_needed(M_list, jitter=1e-3)
    
    # M_list is already a list of n matrices (n, q, q)
    # Use LFR with log-Cholesky metric
    result = DeSI_SPD(
        x=Z, M=M_list_jittered, xout=Z, h=h, 
        metric="log_cholesky", dtype=torch.float64
    )
    
    M_hat_list = result['Mout']
    
    # Compute log-Cholesky distances
    distances = []
    for i in range(n_samples):
        dist = log_cholesky_distance(M_list[i], M_hat_list[i])
        distances.append(dist)
    
    return torch.stack(distances).mean()

class ThetaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super(ThetaMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        for layer in (self.fc1, self.fc2, self.fc3):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        
        # Normalize to unit vector and fix sign ambiguity
        norm = torch.norm(x, dim=1, keepdim=True)
        theta = x / (norm + 1e-8)
        sign = torch.where(theta[:, :1] < 0, -1.0, 1.0)
        theta = theta * sign
        
        return theta

class GlobalBandwidth(nn.Module):
    def __init__(self, bw_init=0.5):
        super().__init__()
        self.bw = nn.Parameter(torch.tensor([bw_init], dtype=torch.float32))
    
    @property
    def bandwidth(self):
        return torch.clamp(self.bw, min=0.1, max=1.0)

def run_single_sim(seed):
    set_torch_threads()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cpu')  # Use CPU for parallel processing
    
    # Generate data
    X, M_list, conditional_means = generate_spd_matrices(
        n=n, m=1000, p=q, random_seed=seed, device=device, dtype=torch.float32
    )
    X = X.to(device)
    M_list = [M.to(device) for M in M_list]
    conditional_means = [M.to(device) for M in conditional_means]
    
    # Split data
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.4 * n)
    n_val = int(0.4 * n)
    n_test = 200
    
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train+n_val]
    idx_test = idx[n_train+n_val:n_train+n_val+n_test]
    
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    M_train = [M_list[i] for i in idx_train]
    M_val = [M_list[i] for i in idx_val]
    M_test = [M_list[i] for i in idx_test]
    conditional_means_test = [conditional_means[i] for i in idx_test]
    
    # Normalize predictors using training statistics
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True).clamp_min(1e-8)
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    # Compute global Fréchet variance for normalization
    with torch.no_grad():
        frechet_var = frechet_variance_log_cholesky(M_train)
        frechet_var = frechet_var + 1e-8
    
    # Create model
    model = ThetaMLP(p, hidden_dim, dropout_rate=0.2).to(device)
    global_bw = GlobalBandwidth(bw_init=0.5)
    optimizer = optim.Adam(list(model.parameters()) + list(global_bw.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_h = None
    
    train_dataset = torch.utils.data.TensorDataset(X_train, torch.stack(M_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_losses = []
    train_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for X_batch, M_batch in train_loader:
            optimizer.zero_grad()
            
            theta_batch = model(X_batch)
            h_value = global_bw.bandwidth
            
            # Convert M_batch from tensor to list
            M_batch_list = [M_batch[j] for j in range(M_batch.shape[0])]
            
            lfr = lfr_loss_selffit(X_batch, M_batch_list, theta_batch, h_value)
            norm_loss = lfr / frechet_var
            reg_term = lambda_reg / (h_value + 1e-8)
            loss = norm_loss + reg_term
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            theta_val = model(X_val)
            h_eval = global_bw.bandwidth
            
            val_lfr = lfr_loss_selffit(X_val, M_val, theta_val, h_eval)
            val_loss = (val_lfr / frechet_var + lambda_reg / (h_eval + 1e-8)).item()
            val_losses.append(val_loss)
            
            # Standard early stopping criterion:
            # Stop when val_loss(t) > min val_loss(s) - δ for 'patience' consecutive epochs
            if val_loss < best_val_loss - delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_h = global_bw.bandwidth.detach().clone()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        global_bw.bw.data = best_h
                    break
    
    # Compute Fréchet mean theta on training set
    with torch.no_grad():
        model.eval()
        theta_final = model(X_train)
        theta_mean = theta_final.mean(dim=0)  # Column mean
        theta_mean_np = theta_mean.detach().cpu().numpy()
        
        # Also compute theta predictions for X_test
        theta_test = model(X_test)
        theta_test_np = theta_test.detach().cpu().numpy()
        
        # Predict matrices for test set
        h_final = global_bw.bandwidth
        M_pred_test = []
        
        for i in range(len(X_test)):
            Z_test = (X_test[i:i+1] * theta_test[i:i+1]).sum(dim=1, keepdim=True)
            Z_train = (X_train * theta_final).sum(dim=1, keepdim=True)
            
            # Add diagonal jitter only if Cholesky decomposition fails
            M_train_jittered = add_jitter_if_needed(M_train, jitter=1e-8)
            
            result_test = DeSI_SPD(
                x=Z_train.squeeze(),
                M=M_train_jittered,
                xout=Z_test,
                h=h_final,
                metric="log_cholesky",
                dtype=torch.float64
            )
            M_pred_test.append(result_test['Mout'][0])
        
        # Calculate average log-Cholesky distance between predicted and true conditional means
        log_chol_distances = []
        for i in range(len(M_test)):
            dist = log_cholesky_distance(M_pred_test[i], conditional_means_test[i])
            log_chol_distances.append(dist.item())
        
        avg_log_chol_distance = np.mean(log_chol_distances)
        print(f"[Seed {seed}] Average log-Cholesky distance between predicted and true conditional means (test set): {avg_log_chol_distance:.6f}")
    
    return theta_mean_np, theta_test_np, avg_log_chol_distance
