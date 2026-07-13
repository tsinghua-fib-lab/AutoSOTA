import numpy as np
from sklearn.metrics import accuracy_score
import sys; sys.path.insert(0, '/repo')
import run_ml_experiment as r

X_raw, y_raw = r.load_ecoli()
X, y = r.preprocess_ecoli(X_raw, y_raw)
N, d = X.shape

# Use fixed step_size for consistency
L_j = 0.25 * np.mean(X**2, axis=0)
step_size = float(np.mean(1.0 / L_j))

# Test with 100 splits for better statistics
n_trials = 100
np.random.seed(42)
seeds = np.random.randint(0, 2**31, n_trials)

print(f"step_size={step_size:.2f}")
print(f"{'noise_scale':>12s} {'sigma_AG_eff':>12s} {'AG_train':>10s} {'AG_test':>10s} {'MG_train':>10s} {'MG_test':>10s} {'PCD_train':>10s} {'PCD_test':>10s}")
print("-" * 90)

for noise_scale in [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060]:
    ag_in, ag_out, mg_in, mg_out, pcd_in, pcd_out = [], [], [], [], [], []
    
    for seed in seeds:
        np.random.seed(int(seed))
        idx = np.random.permutation(N)
        n_tr = int(N * 0.8)
        X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
        y_tr, y_te = y[idx[:n_tr]], y[idx[n_tr:]]
        
        # PCD
        np.random.seed(int(seed))
        h = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8, step_size=step_size)
        pr = np.sign(X_tr @ h); pr[pr==0]=1
        pe = np.sign(X_te @ h); pe[pe==0]=1
        pcd_in.append(100*(1-accuracy_score(y_tr, pr)))
        pcd_out.append(100*(1-accuracy_score(y_te, pe)))
        
        # AG
        np.random.seed(int(seed))
        sig_ag = 21.105645147662777 * noise_scale
        s_ag = r.make_ag_sampler(sig_ag)
        h = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8, noise_sampler=s_ag, step_size=step_size)
        pr = np.sign(X_tr @ h); pr[pr==0]=1
        pe = np.sign(X_te @ h); pe[pe==0]=1
        ag_in.append(100*(1-accuracy_score(y_tr, pr)))
        ag_out.append(100*(1-accuracy_score(y_te, pe)))
        
        # MG
        np.random.seed(int(seed))
        sig_mg = 19.83505759494502 * noise_scale
        s_mg = r.make_mg_sampler(sig_mg, 0.3242659560307008, 2.0*noise_scale, 10)
        h = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8, noise_sampler=s_mg, step_size=step_size)
        pr = np.sign(X_tr @ h); pr[pr==0]=1
        pe = np.sign(X_te @ h); pe[pe==0]=1
        mg_in.append(100*(1-accuracy_score(y_tr, pr)))
        mg_out.append(100*(1-accuracy_score(y_te, pe)))
    
    print(f"{noise_scale:12.6f} {21.1*noise_scale:12.4f} {np.mean(ag_in):10.2f} {np.mean(ag_out):10.2f} {np.mean(mg_in):10.2f} {np.mean(mg_out):10.2f} {np.mean(pcd_in):10.2f} {np.mean(pcd_out):10.2f}")

print(f"\nPaper targets: AG in=15.45, AG out=14.19, MG in=15.14, MG out=13.46, PCD in=6.75, PCD out=4.41")
