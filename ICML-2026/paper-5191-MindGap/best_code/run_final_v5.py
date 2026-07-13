import numpy as np
from scipy.stats import norm as scipy_norm
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# Load ecoli
data = fetch_openml(name="ecoli", version=1, parser="auto")
X_raw = data.data.values.astype(float)
y_raw = data.target.values.ravel()

# Preprocess
unique, counts = np.unique(y_raw, return_counts=True)
majority = unique[np.argmax(counts)]
y = np.where(y_raw == majority, 1, -1)
rmax = np.max(np.abs(X_raw), axis=1, keepdims=True)
rmax[rmax == 0] = 1.0
X = X_raw / rmax

N, d = X.shape
T, lam = 100, 1e-8
P = int(np.ceil(d/4))
noise_scale = 0.0385
L_j = 0.25 * np.mean(X**2, axis=0)
step_size = float(np.mean(1.0/L_j))

calib = {"sigma_AG": 21.105645147662777, "sigma_MG": 19.83505759494502,
         "sigma_QG": 21.035975001866223, "eps_per": 0.3242659560307008,
         "Delta_calib": 2.0, "K": 10}
eps_p = calib["eps_per"]
Delta_s = 2.0 * noise_scale
sig_ag = calib["sigma_AG"] * noise_scale
sig_mg = calib["sigma_MG"] * noise_scale
sig_qg = calib["sigma_QG"] * noise_scale

ks = np.arange(-10, 11)
mg_w = np.exp(-np.abs(ks)*eps_p); mg_w /= mg_w.sum()
mg_ctr = ks * Delta_s
qg_c = np.sqrt(2*np.pi)*sig_qg*(np.exp(eps_p) + 2*scipy_norm.cdf(Delta_s/sig_qg))
qg_w0 = np.sqrt(2*np.pi)*sig_qg*np.exp(eps_p)/qg_c

def stable_sigmoid(x):
    out = np.empty_like(x)
    pos = x > 0
    out[pos] = 1.0/(1.0+np.exp(x[pos]))
    out[~pos] = 1.0/(1.0+np.exp(x[~pos]))
    return out

def soft_threshold(z, lam):
    return np.sign(z)*np.maximum(np.abs(z)-lam, 0)

def dp_pcd(X_tr, y_tr, T, P, lam, noise_sampler=None, step_size=1.0):
    N_tr, d = X_tr.shape
    h = np.zeros(d)
    for _ in range(T):
        coords = np.random.choice(d, size=min(P,d), replace=False)
        for j in coords:
            margins = np.clip(y_tr*(X_tr@h), -100, 100)
            sig = stable_sigmoid(margins)
            g_j = -(X_tr[:,j]@(y_tr*sig))/N_tr
            h_j_new = soft_threshold(h[j]-step_size*g_j, step_size*lam)
            if noise_sampler is not None:
                h[j] = h_j_new + noise_sampler()
            else:
                h[j] = h_j_new
    return h

print(f"ns={noise_scale}, N={N}, d={d}, step={step_size:.2f}")
print(f"sigma_eff: AG={sig_ag:.4f}, MG={sig_mg:.4f}")

results = {"AG": {"in": [], "out": []}, "MG": {"in": [], "out": []}}
np.random.seed(42)

for s in range(500):
    idx = np.random.permutation(N)
    n_tr = int(N*0.8)
    X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
    y_tr, y_te = y[idx[:n_tr]], y[idx[n_tr:]]

    # AG
    h = dp_pcd(X_tr, y_tr, T, P, lam, noise_sampler=lambda: float(np.random.normal(0, sig_ag)), step_size=step_size)
    tp = np.sign(X_tr@h); tp[tp==0]=1
    ep = np.sign(X_te@h); ep[ep==0]=1
    results["AG"]["in"].append(100*(1-accuracy_score(y_tr, tp)))
    results["AG"]["out"].append(100*(1-accuracy_score(y_te, ep)))

    # MG
    h = dp_pcd(X_tr, y_tr, T, P, lam, noise_sampler=lambda: float(np.random.normal(mg_ctr[np.random.choice(len(ks), p=mg_w)], sig_mg)), step_size=step_size)
    tp = np.sign(X_tr@h); tp[tp==0]=1
    ep = np.sign(X_te@h); ep[ep==0]=1
    results["MG"]["in"].append(100*(1-accuracy_score(y_tr, tp)))
    results["MG"]["out"].append(100*(1-accuracy_score(y_te, ep)))

    if (s+1)%100==0:
        w = slice(-100,None)
        print(f"  [{s+1:3d}] AG_in={np.mean(results['AG']['in'][w]):.2f}% MG_in={np.mean(results['MG']['in'][w]):.2f}%")

mi_ag = np.mean(results["AG"]["in"]); mi_mg = np.mean(results["MG"]["in"])
mo_ag = np.mean(results["AG"]["out"]); mo_mg = np.mean(results["MG"]["out"])
mg_in_ok = 15.109 <= mi_mg <= 15.45
mg_out_ok = 13.387 <= mo_mg <= 14.19
print(f"\nAG in={mi_ag:.2f} out={mo_ag:.2f}")
print(f"MG in={mi_mg:.2f}{'*' if mg_in_ok else ''} out={mo_mg:.2f}{'*' if mg_out_ok else ''}")
print(f"MG in CI: {'PASS' if mg_in_ok else 'FAIL'}, MG out CI: {'PASS' if mg_out_ok else 'FAIL'}")
