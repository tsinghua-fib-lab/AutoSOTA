"""Test per-item phi (CODE-04) for Beta-IRT."""
import numpy as np
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rasch_prob_np(theta, b):
    return 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))

def recovery(true_b, est_b):
    true_b = true_b - true_b.mean()
    est_b = est_b - est_b.mean()
    rmse = float(np.sqrt(np.mean((est_b - true_b) ** 2)))
    corr = float(np.corrcoef(true_b, est_b)[0, 1])
    return rmse, corr

def fit_beta_per_item_phi(Pobs, phi_global=250.0, maxiter=500, lr=1.0, lambda_reg=0.1):
    """Fit Beta-IRT with per-item precision parameters phi_j"""
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)
    
    # Logit init
    p_col_mean = np.clip(Pobs.mean(axis=0), 0.01, 0.99)
    b_init = -np.log(p_col_mean / (1 - p_col_mean))
    b_init = b_init - b_init.mean()
    p_row_mean = np.clip(Pobs.mean(axis=1), 0.01, 0.99)
    theta_init = np.log(p_row_mean / (1 - p_row_mean))
    theta_init = theta_init - theta_init.mean()
    
    # Per-item phi: initialize all to log(phi_global)
    log_phi_init = np.full(N, np.log(phi_global))
    
    theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(b_init, dtype=torch.float32, device=device, requires_grad=True)
    log_phi = torch.tensor(log_phi_init, dtype=torch.float32, device=device, requires_grad=True)
    
    optimizer = torch.optim.LBFGS([theta, b, log_phi], max_iter=maxiter, lr=lr,
                                   line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        bc = b - b.mean()
        tc = theta - b.mean()
        p = torch.sigmoid(tc[:, None] - bc[None, :])
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        
        # Per-item phi: shape (N,) -> broadcast to (M, N)
        phi_j = torch.exp(log_phi)  # (N,)
        alpha = phi_j[None, :] * p  # (M, N) * (M, N) broadcast
        beta_param = phi_j[None, :] * (1 - p)
        
        # Fisher weights
        fisher_weights = p * (1 - p)
        w = fisher_weights / (fisher_weights.mean() + 1e-8)
        w = torch.clamp(w, 0.1, 10.0)
        
        ll_per_ij = ((alpha - 1) * torch.log(Pobs_t) + (beta_param - 1) * torch.log(1 - Pobs_t) -
                     torch.lgamma(alpha) - torch.lgamma(beta_param) + torch.lgamma(alpha + beta_param))
        
        # L2 regularization on log_phi deviation from log(phi_global)
        reg = lambda_reg * torch.sum((log_phi - np.log(phi_global)) ** 2)
        
        loss = -torch.sum(w * ll_per_ij) + reg
        loss.backward()
        return loss
    
    optimizer.step(closure)
    with torch.no_grad():
        bc = b - b.mean()
        tc = theta - b.mean()
    return tc.cpu().numpy(), bc.cpu().numpy()

def evaluate(n_trials=50, seed=12345):
    N = 100
    rmses, corrs = [], []
    rng = np.random.default_rng(seed)
    for rep in range(n_trials):
        b_true = rng.normal(0, 1, size=N)
        theta = rng.normal(0, 1, size=2)
        p_true = rasch_prob_np(theta, b_true)
        P_obs = p_true + rng.normal(0, 0.01, size=p_true.shape)
        P_obs = np.clip(P_obs, 1e-6, 1 - 1e-6)
        _, b_hat = fit_beta_per_item_phi(P_obs)
        rmse, corr = recovery(b_true, b_hat)
        rmses.append(rmse)
        corrs.append(corr)
    return np.mean(rmses), np.std(rmses), np.mean(corrs)

t0 = time.time()
rmse_m, rmse_s, corr_m = evaluate(n_trials=50)
print("Per-item phi: RMSE=%.5f +/- %.5f, Corr=%.5f (%.1fs)" % (rmse_m, rmse_s, corr_m, time.time()-t0))
