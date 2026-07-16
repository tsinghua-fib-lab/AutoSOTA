"""Quick M=2-only evaluation harness for rapid idea testing."""
import numpy as np
import torch
import sys, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rasch_prob_np(theta, b):
    return 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))

def fit_rasch_beta_torch(Pobs, phi=400.0, maxiter=500, lr=1.0):
    """Standard Beta-IRT fitting (matched to original)"""
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
    
    theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(b_init, dtype=torch.float32, device=device, requires_grad=True)
    
    optimizer = torch.optim.LBFGS([theta, b], max_iter=maxiter, lr=lr,
                                   line_search_fn="strong_wolfe")
    
    def closure():
        optimizer.zero_grad()
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()
        p = torch.sigmoid(theta_centered[:, None] - b_centered[None, :])
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        alpha = phi * p
        beta_param = phi * (1 - p)
        ll = torch.sum(
            (alpha - 1) * torch.log(Pobs_t) +
            (beta_param - 1) * torch.log(1 - Pobs_t) -
            torch.lgamma(alpha) - torch.lgamma(beta_param) + torch.lgamma(alpha + beta_param)
        )
        loss = -ll
        loss.backward()
        return loss
    
    optimizer.step(closure)
    with torch.no_grad():
        b_centered = b - b.mean()
        theta_centered = theta - b.mean()
    return theta_centered.cpu().numpy(), b_centered.cpu().numpy()

def recovery(true_b, est_b):
    true_b = true_b - true_b.mean()
    est_b = est_b - est_b.mean()
    rmse = float(np.sqrt(np.mean((est_b - true_b) ** 2)))
    corr = float(np.corrcoef(true_b, est_b)[0, 1])
    return rmse, corr

def evaluate_config(phi=200.0, maxiter=500, lr=1.0, n_trials=50, seed=42):
    """Evaluate a config over n_trials, return mean RMSE, mean corr, std RMSE"""
    N = 100
    rmses = []
    corrs = []
    rng = np.random.default_rng(seed)
    for rep in range(n_trials):
        b_true = rng.normal(0, 1, size=N)
        theta = rng.normal(0, 1, size=2)
        p_true = rasch_prob_np(theta, b_true)
        P_obs = p_true + rng.normal(0, 0.01, size=p_true.shape)
        P_obs = np.clip(P_obs, 1e-6, 1 - 1e-6)
        _, b_hat = fit_rasch_beta_torch(P_obs, phi=phi, maxiter=maxiter, lr=lr)
        rmse, corr = recovery(b_true, b_hat)
        rmses.append(rmse)
        corrs.append(corr)
    return np.mean(rmses), np.std(rmses), np.mean(corrs), np.std(corrs)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=float, default=200.0)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    rmse_m, rmse_s, corr_m, corr_s = evaluate_config(
        phi=args.phi, maxiter=args.maxiter, lr=args.lr,
        n_trials=args.trials, seed=args.seed
    )
    print(f"PHI={args.phi} MAXITER={args.maxiter} LR={args.lr}: RMSE={rmse_m:.5f}+/-{rmse_s:.5f} CORR={corr_m:.5f}+/-{corr_s:.5f}")
