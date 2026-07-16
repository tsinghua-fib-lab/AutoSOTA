import numpy as np, torch, time
device = torch.device("cuda")

def recovery(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.sqrt(np.mean((b-a)**2))), float(np.corrcoef(a,b)[0,1])

def fit_beta_svd(Pobs, phi=250.0, maxiter=500, lr=1.0):
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)
    
    # ALGO-06: SVD-based initialization
    logit_P = np.log(Pobs / (1 - Pobs))
    logit_P_centered = logit_P - logit_P.mean()
    try:
        U, S, Vt = np.linalg.svd(logit_P_centered, full_matrices=False)
        # Rank-1 approximation
        theta_init = U[:, 0] * np.sqrt(S[0])
        b_init = -Vt[0, :] * np.sqrt(S[0])
        theta_init = theta_init - theta_init.mean()
        b_init = b_init - b_init.mean()
    except np.linalg.LinAlgError:
        # Fallback to logit init
        pcm = np.clip(Pobs.mean(axis=0), 0.01, 0.99)
        b_init = -np.log(pcm / (1 - pcm)); b_init -= b_init.mean()
        prm = np.clip(Pobs.mean(axis=1), 0.01, 0.99)
        theta_init = np.log(prm / (1 - prm)); theta_init -= theta_init.mean()
    
    theta = torch.tensor(theta_init.astype(np.float32), device=device, requires_grad=True)
    b = torch.tensor(b_init.astype(np.float32), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta, b], max_iter=maxiter, lr=lr, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        bc = b - b.mean(); tc = theta - b.mean()
        p = torch.sigmoid(tc[:, None] - bc[None, :]); p = torch.clamp(p, 1e-6, 1 - 1e-6)
        alpha = phi * p; bp = phi * (1 - p)
        fw = p * (1 - p); w = fw / (fw.mean() + 1e-8); w = torch.clamp(w, 0.1, 10.0)
        ll = (alpha-1)*torch.log(Pobs_t)+(bp-1)*torch.log(1-Pobs_t)-torch.lgamma(alpha)-torch.lgamma(bp)+torch.lgamma(alpha+bp)
        loss = -torch.sum(w * ll)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        bc = b - b.mean(); tc = theta - b.mean()
    return tc.cpu().numpy(), bc.cpu().numpy()

# Test
for label, use_svd in [("logit_init", False), ("svd_init", True)]:
    rmses = []; rng = np.random.default_rng(12345)
    for rep in range(50):
        bt = rng.normal(0, 1, size=100)
        th = rng.normal(0, 1, size=2)
        pt = 1.0/(1.0+np.exp(-(th[:,None]-bt[None,:])))
        Po = pt + rng.normal(0, 0.01, size=pt.shape)
        Po = np.clip(Po, 1e-6, 1-1e-6)
        # For logit_init test, temporarily modify the fallback
        if not use_svd:
            pcm = np.clip(Po.mean(axis=0), 0.01, 0.99)
            bi = -np.log(pcm / (1 - pcm)); bi -= bi.mean()
            prm = np.clip(Po.mean(axis=1), 0.01, 0.99)
            ti = np.log(prm / (1 - prm)); ti -= ti.mean()
            # user simpler fit
            from test_svd_init import fit_beta_svd
        _, bh = fit_beta_svd(Po, phi=250.0)
        r, _ = recovery(bt, bh); rmses.append(r)
    print("%s: RMSE=%.5f +/- %.5f" % (label, np.mean(rmses), np.std(rmses)))
