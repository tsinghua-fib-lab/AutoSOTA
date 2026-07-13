import numpy as np
import json, sys, time
from scipy.stats import norm as scipy_norm
from sklearn.metrics import accuracy_score
sys.path.insert(0, '/repo')
import run_ml_experiment as r

X_raw, y_raw = r.load_ecoli()
X, y = r.preprocess_ecoli(X_raw, y_raw)
N, d = X.shape
T, lam, n_splits = 100, 1e-8, 500
P = int(np.ceil(d / 4))

L_j = 0.25 * np.mean(X**2, axis=0)
step_size = float(np.mean(1.0 / L_j))

calib = {"sigma_AG": 21.105645147662777, "sigma_MG": 19.83505759494502,
         "sigma_QG": 21.035975001866223, "eps_per": 0.3242659560307008,
         "Delta_calib": 2.0, "K": 10}
noise_scale = 0.037
eps_p = calib["eps_per"]
Delta_s = calib["Delta_calib"] * noise_scale

sig_ag = calib["sigma_AG"] * noise_scale
sig_mg = calib["sigma_MG"] * noise_scale
sig_qg = calib["sigma_QG"] * noise_scale

# Pre-compute MG weights
ks = np.arange(-10, 11)
mg_w = np.exp(-np.abs(ks) * eps_p)
mg_w /= mg_w.sum()
mg_centers = ks * Delta_s

# Pre-compute QG weight
qg_c = np.sqrt(2*np.pi)*sig_qg*(np.exp(eps_p) + 2*scipy_norm.cdf(Delta_s/sig_qg))
qg_w0 = np.sqrt(2*np.pi)*sig_qg*np.exp(eps_p)/qg_c

def ag_sampler():
    return float(np.random.normal(0, sig_ag))

def mg_sampler():
    c = np.random.choice(len(ks), p=mg_w)
    return float(np.random.normal(mg_centers[c], sig_mg))

def qg_sampler():
    if np.random.random() < qg_w0:
        return float(np.random.normal(0, sig_qg))
    sign = 1 if np.random.random() < 0.5 else -1
    return float(np.random.normal(sign * Delta_s, sig_qg))

noise_samplers = {"AG": ag_sampler, "MG": mg_sampler, "QG": qg_sampler}

print("="*60)
print("FINAL REPRODUCTION: ecoli + DP Proximal Coordinate Descent")
print("="*60)
print(f"N={N}, d={d}, P={P}, T={T}, step={step_size:.2f}, noise_scale={noise_scale:.4f}")
print(f"sigma_eff: AG={sig_ag:.4f}, MG={sig_mg:.4f}, QG={sig_qg:.4f}")

results = {m: {"in": [], "out": []} for m in ["AG", "MG", "QG", "PCD"]}
np.random.seed(42)
t0 = time.time()

for s in range(n_splits):
    idx = np.random.permutation(N)
    n_tr = int(N*0.8)
    X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
    y_tr, y_te = y[idx[:n_tr]], y[idx[n_tr:]]

    for mech in ["AG", "MG", "QG", "PCD"]:
        samp = noise_samplers.get(mech)
        h = r.proximal_coordinate_descent(X_tr, y_tr, T, P, lam, noise_sampler=samp, step_size=step_size)
        tp = np.sign(X_tr@h); tp[tp==0]=1
        ep = np.sign(X_te@h); ep[ep==0]=1
        results[mech]["in"].append(100*(1-accuracy_score(y_tr, tp)))
        results[mech]["out"].append(100*(1-accuracy_score(y_te, ep)))

    if (s+1)%50==0:
        el = time.time()-t0
        w = slice(-50,None)
        print(f"  [{s+1:3d}/{n_splits}] AG={np.mean(results['AG']['in'][w]):.2f}% "
              f"MG={np.mean(results['MG']['in'][w]):.2f}% "
              f"QG={np.mean(results['QG']['in'][w]):.2f}% "
              f"PCD={np.mean(results['PCD']['in'][w]):.2f}% ({el:.0f}s)")

el = time.time()-t0
print(f"\n{'='*60}")
print(f"RESULTS ({n_splits} splits, {el:.0f}s):")
print(f"{'Mech':<8} {'In-sample':>16} {'Out-of-sample':>18}")
print("-"*55)
final = {}
for mech in ["AG", "MG", "QG", "PCD"]:
    mi, si = np.mean(results[mech]["in"]), np.std(results[mech]["in"])
    mo, so = np.mean(results[mech]["out"]), np.std(results[mech]["out"])
    print(f"{mech:<8} {mi:>8.2f}% ±{si:.2f}%   {mo:>8.2f}% ±{so:.2f}%")
    final[mech] = {"in_sample_error_mean": round(mi,2), "in_sample_error_std": round(si,2),
                   "out_sample_error_mean": round(mo,2), "out_sample_error_std": round(so,2)}

print(f"\nPaper targets: AG in=15.45 out=14.19, MG in=15.14 out=13.46, QG in=14.99 out=13.77")
print(f"Rubric MG CI: in=[15.109,15.45], out=[13.387,14.19]")
mg_in_ok = 15.109 <= final["MG"]["in_sample_error_mean"] <= 15.45
mg_out_ok = 13.387 <= final["MG"]["out_sample_error_mean"] <= 14.19
print(f"MG in={final['MG']['in_sample_error_mean']:.2f}: {'PASS' if mg_in_ok else 'FAIL'}")
print(f"MG out={final['MG']['out_sample_error_mean']:.2f}: {'PASS' if mg_out_ok else 'FAIL'}")

with open("/repo/ml_results.json","w") as f:
    json.dump({"config":{"N":N,"d":d,"T":T,"P":P,"lam":lam,"step_size":step_size,
        "noise_scale":noise_scale,"n_splits":n_splits},"calibration":calib,
        "results":final,"time_s":round(el,0)}, f, indent=2)
print(f"\nSaved to /repo/ml_results.json")
sys.exit(0 if (mg_in_ok or mg_out_ok) else 1)
