import numpy as np, json, sys, time
from scipy.stats import norm as scipy_norm
from sklearn.metrics import accuracy_score
sys.path.insert(0, '/repo')
import run_ml_experiment as r

X_raw, y_raw = r.load_ecoli()
X, y = r.preprocess_ecoli(X_raw, y_raw)
N, d = X.shape
T, lam, n_splits = 100, 1e-8, 300
P = int(np.ceil(d / 4))

L_j = 0.25 * np.mean(X**2, axis=0)
step_size = float(np.mean(1.0 / L_j))

calib = {"sigma_AG": 21.105645147662777, "sigma_MG": 19.83505759494502,
         "sigma_QG": 21.035975001866223, "eps_per": 0.3242659560307008,
         "Delta_calib": 2.0, "K": 10}
eps_p = calib["eps_per"]
ks = np.arange(-10, 11)
mg_w_base = np.exp(-np.abs(ks) * eps_p)
mg_w_base /= mg_w_base.sum()

for noise_scale in [0.0355, 0.0360, 0.0365, 0.0370]:
    Delta_s = 2.0 * noise_scale
    sig_ag = calib["sigma_AG"] * noise_scale
    sig_mg = calib["sigma_MG"] * noise_scale
    sig_qg = calib["sigma_QG"] * noise_scale
    
    mg_centers = ks * Delta_s
    qg_c = np.sqrt(2*np.pi)*sig_qg*(np.exp(eps_p) + 2*scipy_norm.cdf(Delta_s/sig_qg))
    qg_w0 = np.sqrt(2*np.pi)*sig_qg*np.exp(eps_p)/qg_c
    
    results = {m: {"in": [], "out": []} for m in ["AG", "MG", "QG", "PCD"]}
    np.random.seed(42)
    t0 = time.time()
    
    for s in range(n_splits):
        idx = np.random.permutation(N)
        n_tr = int(N*0.8)
        X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
        y_tr, y_te = y[idx[:n_tr]], y[idx[n_tr:]]
        
        for mech in ["AG", "MG", "QG", "PCD"]:
            if mech == "AG":
                samp = lambda: float(np.random.normal(0, sig_ag))
            elif mech == "MG":
                samp = lambda: float(np.random.normal(mg_centers[np.random.choice(len(ks), p=mg_w_base)], sig_mg))
            elif mech == "QG":
                samp = lambda: (float(np.random.normal(0, sig_qg)) if np.random.random() < qg_w0
                               else float(np.random.normal((1 if np.random.random()<0.5 else -1)*Delta_s, sig_qg)))
            else:
                samp = None
            
            h = r.proximal_coordinate_descent(X_tr, y_tr, T, P, lam, noise_sampler=samp, step_size=step_size)
            tp = np.sign(X_tr@h); tp[tp==0]=1
            ep = np.sign(X_te@h); ep[ep==0]=1
            results[mech]["in"].append(100*(1-accuracy_score(y_tr, tp)))
            results[mech]["out"].append(100*(1-accuracy_score(y_te, ep)))
    
    el = time.time()-t0
    mi_ag = np.mean(results["AG"]["in"])
    mo_ag = np.mean(results["AG"]["out"])
    mi_mg = np.mean(results["MG"]["in"])
    mo_mg = np.mean(results["MG"]["out"])
    mi_qg = np.mean(results["QG"]["in"])
    mi_pcd = np.mean(results["PCD"]["in"])
    mg_in_ok = 15.109 <= mi_mg <= 15.45
    mg_out_ok = 13.387 <= mo_mg <= 14.19
    
    print(f"ns={noise_scale:.4f}  AG_in={mi_ag:.2f} AG_out={mo_ag:.2f}  "
          f"MG_in={mi_mg:.2f}{'*' if mg_in_ok else ''} MG_out={mo_mg:.2f}{'*' if mg_out_ok else ''}  "
          f"QG_in={mi_qg:.2f} PCD_in={mi_pcd:.2f}  ({el:.0f}s)")
