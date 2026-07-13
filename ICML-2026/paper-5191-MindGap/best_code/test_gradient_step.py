import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys; sys.path.insert(0, '/repo')
import run_ml_experiment as r

X_raw, y_raw = r.load_ecoli()
X, y = r.preprocess_ecoli(X_raw, y_raw)
N, d = X.shape

def compute_gradient_j(X_tr, y_tr, h, j):
    """Numerically stable gradient for coordinate j."""
    N_tr = X_tr.shape[0]
    margins = y_tr * (X_tr @ h)
    margins = np.clip(margins, -100, 100)
    sigmoid_neg = r.stable_sigmoid(margins)
    g_j = -(X_tr[:, j] @ (y_tr * sigmoid_neg)) / N_tr
    return float(g_j)

def pcd_gradient_step(X_tr, y_tr, T, P, lam, noise_sampler=None, step_size=1.0):
    N_tr, d = X_tr.shape
    h = np.zeros(d)
    for t in range(T):
        coords = np.random.choice(d, size=min(P, d), replace=False)
        for j in coords:
            g_j = compute_gradient_j(X_tr, y_tr, h, j)
            h_j_new = r.soft_threshold(h[j] - step_size * g_j, step_size * lam)
            if noise_sampler is not None:
                h[j] = h_j_new + noise_sampler()
            else:
                h[j] = h_j_new
    return h

n_trials = 30
np.random.seed(42)
seeds = np.random.randint(0, 2**31, n_trials)

print("=== Non-private PCD (gradient step) ===")
for step_size in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    errors_in = []
    errors_out = []
    for seed in seeds:
        np.random.seed(int(seed))
        n_tr = int(N * 0.8)
        idx = np.random.permutation(N)
        X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
        y_tr, y_te = y[idx[:n_tr]], y[idx[n_tr:]]
        
        np.random.seed(int(seed))
        h = pcd_gradient_step(X_tr, y_tr, 100, 2, 1e-8, step_size=step_size)
        p_tr = np.sign(X_tr @ h); p_tr[p_tr==0]=1
        p_te = np.sign(X_te @ h); p_te[p_te==0]=1
        errors_in.append(100*(1-accuracy_score(y_tr, p_tr)))
        errors_out.append(100*(1-accuracy_score(y_te, p_te)))
    print(f"step={step_size:.1f}: train={np.mean(errors_in):.2f}±{np.std(errors_in):.2f}%, test={np.mean(errors_out):.2f}±{np.std(errors_out):.2f}%")

print(f"\nPaper PCD reference: train=6.75%, test=4.41%")
print(f"Paper AG reference: train=15.45%, test=14.19%")

# Test DP with best step size
print(f"\n=== AG-DP (gradient step, step_size=10) ===")
for noise_scale in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]:
    errors_in = []
    errors_out = []
    for seed in seeds:
        np.random.seed(int(seed))
        n_tr = int(N * 0.8)
        idx = np.random.permutation(N)
        X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
        y_tr, y_te = y[idx[:n_tr]], y[idx[n_tr:]]
        
        np.random.seed(int(seed))
        sampler = r.make_ag_sampler(21.105645147662777, noise_scale)
        h = pcd_gradient_step(X_tr, y_tr, 100, 2, 1e-8, noise_sampler=sampler, step_size=10.0)
        p_tr = np.sign(X_tr @ h); p_tr[p_tr==0]=1
        p_te = np.sign(X_te @ h); p_te[p_te==0]=1
        errors_in.append(100*(1-accuracy_score(y_tr, p_tr)))
        errors_out.append(100*(1-accuracy_score(y_te, p_te)))
    print(f"scale={noise_scale:.3f} (σ_eff={(21.1*noise_scale):.3f}): train={np.mean(errors_in):.2f}±{np.std(errors_in):.2f}%, test={np.mean(errors_out):.2f}±{np.std(errors_out):.2f}%")
