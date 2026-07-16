import numpy as np, torch, time
device = torch.device("cuda")

def fit_per_item(Pobs, phi_global=250.0, lambda_reg=1.0):
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)
    pcm = np.clip(Pobs.mean(axis=0), 0.01, 0.99)
    bi = -np.log(pcm / (1 - pcm)); bi -= bi.mean()
    prm = np.clip(Pobs.mean(axis=1), 0.01, 0.99)
    ti = np.log(prm / (1 - prm)); ti -= ti.mean()
    lpi = np.full(N, np.log(phi_global))
    theta = torch.tensor(ti, dtype=torch.float32, device=device, requires_grad=True)
    b = torch.tensor(bi, dtype=torch.float32, device=device, requires_grad=True)
    lp = torch.tensor(lpi, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta, b, lp], max_iter=500, lr=1.0, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        bc = b - b.mean(); tc = theta - b.mean()
        p = torch.sigmoid(tc[:, None] - bc[None, :]); p = torch.clamp(p, 1e-6, 1 - 1e-6)
        pj = torch.exp(lp)
        alpha = pj[None, :] * p; bp = pj[None, :] * (1 - p)
        fw = p * (1 - p); w = fw / (fw.mean() + 1e-8); w = torch.clamp(w, 0.1, 10.0)
        ll = (alpha-1)*torch.log(Pobs_t) + (bp-1)*torch.log(1-Pobs_t) - torch.lgamma(alpha) - torch.lgamma(bp) + torch.lgamma(alpha+bp)
        reg = lambda_reg * torch.sum((lp - np.log(phi_global))**2)
        loss = -torch.sum(w * ll) + reg
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        bc = b - b.mean(); tc = theta - b.mean()
    return tc.cpu().numpy(), bc.cpu().numpy()

def recovery(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.sqrt(np.mean((b-a)**2))), float(np.corrcoef(a,b)[0,1])

for lam in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    rmses = []
    rng = np.random.default_rng(12345)
    for rep in range(20):
        bt = rng.normal(0, 1, size=100)
        th = rng.normal(0, 1, size=2)
        pt = 1.0/(1.0+np.exp(-(th[:,None]-bt[None,:])))
        Po = pt + rng.normal(0, 0.01, size=pt.shape)
        Po = np.clip(Po, 1e-6, 1-1e-6)
        _, bh = fit_per_item(Po, lambda_reg=lam)
        r, _ = recovery(bt, bh); rmses.append(r)
    print("lambda_reg=%7.1f: RMSE=%.5f +/- %.5f" % (lam, np.mean(rmses), np.std(rmses)))
