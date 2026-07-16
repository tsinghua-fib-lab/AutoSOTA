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

def compute_beta_nll(Pobs_t, theta, b, phi):
    """Compute Beta NLL without backward"""
    bc = b - b.mean()
    tc = theta - b.mean()
    p = torch.sigmoid(tc[:, None] - bc[None, :])
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    alpha = phi * p
    bp = phi * (1 - p)
    ll = torch.sum((alpha - 1) * torch.log(Pobs_t) + (bp - 1) * torch.log(1 - Pobs_t) -
                  torch.lgamma(alpha) - torch.lgamma(bp) + torch.lgamma(alpha + bp))
    return (-ll).item()

def fit_beta(Pobs, phi=200.0, maxiter=500, lr=1.0, n_restarts=1):
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)
    
    p_col_mean = np.clip(Pobs.mean(axis=0), 0.01, 0.99)
    b_base = -np.log(p_col_mean / (1 - p_col_mean))
    b_base = b_base - b_base.mean()
    p_row_mean = np.clip(Pobs.mean(axis=1), 0.01, 0.99)
    theta_base = np.log(p_row_mean / (1 - p_row_mean))
    theta_base = theta_base - theta_base.mean()
    
    def run_one(theta_init, b_init):
        theta = torch.tensor(theta_init.copy(), dtype=torch.float32, device=device, requires_grad=True)
        b = torch.tensor(b_init.copy(), dtype=torch.float32, device=device, requires_grad=True)
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
            ll = torch.sum((alpha - 1) * torch.log(Pobs_t) + (bp - 1) * torch.log(1 - Pobs_t) -
                          torch.lgamma(alpha) - torch.lgamma(bp) + torch.lgamma(alpha + bp))
            loss = -ll
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            bc = b - b.mean()
            tc = theta - b.mean()
            final_loss = compute_beta_nll(Pobs_t, theta, b, phi)
        return tc.cpu().numpy().copy(), bc.cpu().numpy().copy(), final_loss
    
    best_theta, best_b, best_loss = run_one(theta_base, b_base)
    
    if n_restarts > 1:
        rng = np.random.default_rng(42)
        for _ in range(n_restarts - 1):
            tp = theta_base + rng.normal(0, 0.1, size=M)
            bp = b_base + rng.normal(0, 0.1, size=N)
            tr, br, lr_loss = run_one(tp, bp)
            if lr_loss < best_loss:
                best_theta, best_b, best_loss = tr, br, lr_loss
    
    return best_theta, best_b

def evaluate(phi=200, maxiter=500, lr=1.0, n_restarts=1, n_trials=50, seed=12345):
    N = 100
    rmses, corrs = [], []
    rng = np.random.default_rng(seed)
    for rep in range(n_trials):
        b_true = rng.normal(0, 1, size=N)
        theta = rng.normal(0, 1, size=2)
        p_true = rasch_prob_np(theta, b_true)
        P_obs = p_true + rng.normal(0, 0.01, size=p_true.shape)
        P_obs = np.clip(P_obs, 1e-6, 1 - 1e-6)
        _, b_hat = fit_beta(P_obs, phi=phi, maxiter=maxiter, lr=lr, n_restarts=n_restarts)
        rmse, corr = recovery(b_true, b_hat)
        rmses.append(rmse)
        corrs.append(corr)
    return np.mean(rmses), np.std(rmses), np.mean(corrs), np.std(corrs)

configs = [
    ("baseline_phi400", 400, 500, 1.0, 1),
    ("phi200", 200, 500, 1.0, 1),
    ("phi200_restart3", 200, 500, 1.0, 3),
    ("phi200_restart5", 200, 500, 1.0, 5),
    ("phi200_maxiter2000", 200, 2000, 1.0, 1),
    ("phi200_lr0.5", 200, 500, 0.5, 1),
    ("phi200_lr2.0", 200, 500, 2.0, 1),
    ("phi200_maxiter2000_lr0.5", 200, 2000, 0.5, 1),
    ("phi100", 100, 500, 1.0, 1),
    ("phi400_restart5", 400, 500, 1.0, 5),
    ("phi200_maxiter1000", 200, 1000, 1.0, 1),
]

print("%-30s %10s %8s %8s %6s" % ("Config", "RMSE", "std", "Corr", "Time"))
print("-" * 70)

results = []
for label, phi, maxiter, lr, n_restarts in configs:
    t0 = time.time()
    rmse_m, rmse_s, corr_m, corr_s = evaluate(
        phi=phi, maxiter=maxiter, lr=lr, n_restarts=n_restarts, n_trials=50)
    elapsed = time.time() - t0
    results.append((label, rmse_m, rmse_s, corr_m, corr_s, phi, maxiter, lr, n_restarts))
    print("%-30s %10.5f %8.5f %8.5f %5.1fs" % (label, rmse_m, rmse_s, corr_m, elapsed))

best = min(results, key=lambda r: r[1])
print("\nBest: %s (RMSE=%.5f, Corr=%.5f)" % (best[0], best[1], best[3]))
