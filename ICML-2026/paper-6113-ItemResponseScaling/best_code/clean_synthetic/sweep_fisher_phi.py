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

def fit_beta_fisher(Pobs, phi=200.0, maxiter=500, lr=1.0):
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)
    
    p_col_mean = np.clip(Pobs.mean(axis=0), 0.01, 0.99)
    b_init = -np.log(p_col_mean / (1 - p_col_mean))
    b_init = b_init - b_init.mean()
    p_row_mean = np.clip(Pobs.mean(axis=1), 0.01, 0.99)
    theta_init = np.log(p_row_mean / (1 - p_row_mean))
    theta_init = theta_init - theta_init.mean()
    
    theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(b_init, dtype=torch.float32, device=device, requires_grad=True)
    
    optimizer = torch.optim.LBFGS([theta, b], max_iter=maxiter, lr=lr,
                                   line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        bc = b - b.mean()
        tc = theta - b.mean()
        p = torch.sigmoid(tc[:, None] - bc[None, :])
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        alpha = phi * p
        bp = phi * (1 - p)
        
        # Fisher weights
        fisher_weights = p * (1 - p)
        w = fisher_weights / (fisher_weights.mean() + 1e-8)
        w = torch.clamp(w, 0.1, 10.0)
        
        ll_per_ij = ((alpha - 1) * torch.log(Pobs_t) + (bp - 1) * torch.log(1 - Pobs_t) -
                     torch.lgamma(alpha) - torch.lgamma(bp) + torch.lgamma(alpha + bp))
        loss = -torch.sum(w * ll_per_ij)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    with torch.no_grad():
        bc = b - b.mean()
        tc = theta - b.mean()
    return tc.cpu().numpy(), bc.cpu().numpy()

def evaluate(phi, n_trials=50, seed=12345):
    N = 100
    rmses, corrs = [], []
    rng = np.random.default_rng(seed)
    for rep in range(n_trials):
        b_true = rng.normal(0, 1, size=N)
        theta = rng.normal(0, 1, size=2)
        p_true = rasch_prob_np(theta, b_true)
        P_obs = p_true + rng.normal(0, 0.01, size=p_true.shape)
        P_obs = np.clip(P_obs, 1e-6, 1 - 1e-6)
        _, b_hat = fit_beta_fisher(P_obs, phi=phi)
        rmse, corr = recovery(b_true, b_hat)
        rmses.append(rmse)
        corrs.append(corr)
    return np.mean(rmses), np.std(rmses), np.mean(corrs)

phis = [50, 100, 150, 200, 250, 400, 800]
print("%-20s %10s %8s %8s" % ("Phi", "RMSE", "std", "Corr"))
print("-" * 50)
best_phi, best_rmse = None, float("inf")
for phi in phis:
    rmse_m, rmse_s, corr_m = evaluate(phi, n_trials=50)
    print("%-20s %10.5f %8.5f %8.5f" % (f"phi={phi}", rmse_m, rmse_s, corr_m))
    if rmse_m < best_rmse:
        best_rmse, best_phi = rmse_m, phi
print("\nBest: phi=%d (RMSE=%.5f)" % (best_phi, best_rmse))
