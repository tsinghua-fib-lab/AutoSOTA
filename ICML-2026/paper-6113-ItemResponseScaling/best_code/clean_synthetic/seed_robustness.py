"""Seed robustness check for Fisher+phi=300."""
import numpy as np, torch, time
device = torch.device("cuda")

def recovery(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.sqrt(np.mean((b-a)**2))), float(np.corrcoef(a,b)[0,1])

def fit_fisher(Pobs, phi=300.0, maxiter=1000, clamp_min=0.1, clamp_max=10.0):
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)
    pcm = np.clip(Pobs.mean(axis=0), 0.01, 0.99)
    bi = -np.log(pcm / (1 - pcm)); bi -= bi.mean()
    prm = np.clip(Pobs.mean(axis=1), 0.01, 0.99)
    ti = np.log(prm / (1 - prm)); ti -= ti.mean()
    theta = torch.tensor(ti.astype(np.float32), device=device, requires_grad=True)
    b = torch.tensor(bi.astype(np.float32), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta, b], max_iter=maxiter, lr=1.0, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        bc = b - b.mean(); tc = theta - b.mean()
        p = torch.sigmoid(tc[:, None] - bc[None, :]); p = torch.clamp(p, 1e-6, 1 - 1e-6)
        alpha = phi * p; bp = phi * (1 - p)
        fw = p * (1 - p); w = fw / (fw.mean() + 1e-8)
        w = torch.clamp(w, clamp_min, clamp_max)
        ll = (alpha-1)*torch.log(Pobs_t)+(bp-1)*torch.log(1-Pobs_t)-torch.lgamma(alpha)-torch.lgamma(bp)+torch.lgamma(alpha+bp)
        loss = -torch.sum(w * ll)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        bc = b - b.mean(); tc = theta - b.mean()
    return tc.cpu().numpy(), bc.cpu().numpy()

# Test across 4 different seeds
seeds = [7, 42, 123, 1000]
N = 100; n_trials = 30

print("Seed robustness: Fisher+phi=300")
print("%-6s %10s %8s %8s" % ("Seed", "RMSE", "std", "Corr"))
print("-" * 40)

all_rmses = []
for seed in seeds:
    rmses = []; corrs = []
    rng = np.random.default_rng(seed)
    for rep in range(n_trials):
        bt = rng.normal(0, 1, size=N)
        th = rng.normal(0, 1, size=2)
        pt = 1.0/(1.0+np.exp(-(th[:,None]-bt[None,:])))
        Po = pt + rng.normal(0, 0.01, size=pt.shape)
        Po = np.clip(Po, 1e-6, 1-1e-6)
        _, bh = fit_fisher(Po)
        r, c = recovery(bt, bh)
        rmses.append(r); corrs.append(c)
    rmse_m = np.mean(rmses); rmse_s = np.std(rmses)
    corr_m = np.mean(corrs)
    print("%-6d %10.5f %8.5f %8.5f" % (seed, rmse_m, rmse_s, corr_m))
    all_rmses.extend(rmses)

print("\nOverall: RMSE=%.5f +/- %.5f" % (np.mean(all_rmses), np.std(all_rmses)))
