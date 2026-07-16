import numpy as np, torch
device = torch.device("cuda")

def recovery(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.sqrt(np.mean((b-a)**2))), float(np.corrcoef(a,b)[0,1])

def fit_svd(Pobs, phi=250.0):
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)
    logit_P = np.log(Pobs / (1 - Pobs))
    logit_P_centered = logit_P - logit_P.mean()
    U, S, Vt = np.linalg.svd(logit_P_centered, full_matrices=False)
    ti = U[:, 0] * np.sqrt(S[0]); ti -= ti.mean()
    bi = -Vt[0, :] * np.sqrt(S[0]); bi -= bi.mean()
    theta = torch.tensor(ti.astype(np.float32), device=device, requires_grad=True)
    b = torch.tensor(bi.astype(np.float32), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta, b], max_iter=500, lr=1.0, line_search_fn="strong_wolfe")
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

def fit_logit(Pobs, phi=250.0):
    M, N = Pobs.shape
    Pobs = np.clip(Pobs, 1e-6, 1 - 1e-6)
    Pobs_t = torch.tensor(Pobs, dtype=torch.float32, device=device)
    pcm = np.clip(Pobs.mean(axis=0), 0.01, 0.99)
    bi = -np.log(pcm / (1 - pcm)); bi -= bi.mean()
    prm = np.clip(Pobs.mean(axis=1), 0.01, 0.99)
    ti = np.log(prm / (1 - prm)); ti -= ti.mean()
    theta = torch.tensor(ti.astype(np.float32), device=device, requires_grad=True)
    b = torch.tensor(bi.astype(np.float32), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta, b], max_iter=500, lr=1.0, line_search_fn="strong_wolfe")
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

rng = np.random.default_rng(12345)
svd_rmses = []; logit_rmses = []
for rep in range(50):
    bt = rng.normal(0, 1, size=100)
    th = rng.normal(0, 1, size=2)
    pt = 1.0/(1.0+np.exp(-(th[:,None]-bt[None,:])))
    Po = pt + rng.normal(0, 0.01, size=pt.shape)
    Po = np.clip(Po, 1e-6, 1-1e-6)
    _, bh_svd = fit_svd(Po)
    r_svd, _ = recovery(bt, bh_svd); svd_rmses.append(r_svd)
    _, bh_logit = fit_logit(Po)
    r_logit, _ = recovery(bt, bh_logit); logit_rmses.append(r_logit)

print("SVD init:   RMSE=%.5f +/- %.5f" % (np.mean(svd_rmses), np.std(svd_rmses)))
print("Logit init: RMSE=%.5f +/- %.5f" % (np.mean(logit_rmses), np.std(logit_rmses)))
