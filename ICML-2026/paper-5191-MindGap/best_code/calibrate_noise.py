import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys; sys.path.insert(0, '/repo')
import run_ml_experiment as r

X_raw, y_raw = r.load_ecoli()
X, y = r.preprocess_ecoli(X_raw, y_raw)
N, d = X.shape

# Run 20 random splits and compute mean error for each noise scale
n_trials = 20
np.random.seed(42)
all_seeds = np.random.randint(0, 2**31, n_trials)

# Noise scales to test
noise_scales = [1/N, 2/N, 5/N, 10/N, 20/N, 50/N, 1.0, 2.0, 1.0/20, 1.0/10, 1.0/5, 1.0/2]

print(f"{'noise_scale':>12s} {'sigma_eff':>10s} {'AG_train':>10s} {'AG_test':>10s} {'PCD_train':>10s} {'PCD_test':>10s}")
print("-" * 70)

for noise_scale in noise_scales:
    ag_train_errs = []
    ag_test_errs = []
    pcd_train_errs = []
    pcd_test_errs = []
    
    for seed in all_seeds:
        np.random.seed(int(seed))
        n_train = int(N * 0.8)
        indices = np.random.permutation(N)
        X_tr, X_te = X[indices[:n_train]], X[indices[n_train:]]
        y_tr, y_te = y[indices[:n_train]], y[indices[n_train:]]
        
        # Non-private
        np.random.seed(int(seed))
        h_pcd = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8)
        p_tr = np.sign(X_tr @ h_pcd); p_tr[p_tr==0]=1
        p_te = np.sign(X_te @ h_pcd); p_te[p_te==0]=1
        pcd_train_errs.append(100*(1-accuracy_score(y_tr, p_tr)))
        pcd_test_errs.append(100*(1-accuracy_score(y_te, p_te)))
        
        # AG-DP
        np.random.seed(int(seed))
        sampler = r.make_ag_sampler(21.105645147662777, noise_scale)
        h_ag = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8, noise_sampler=sampler)
        p_tr = np.sign(X_tr @ h_ag); p_tr[p_tr==0]=1
        p_te = np.sign(X_te @ h_ag); p_te[p_te==0]=1
        ag_train_errs.append(100*(1-accuracy_score(y_tr, p_tr)))
        ag_test_errs.append(100*(1-accuracy_score(y_te, p_te)))
    
    print(f"{noise_scale:12.6f} {21.1*noise_scale:10.4f} {np.mean(ag_train_errs):10.2f} {np.mean(ag_test_errs):10.2f} {np.mean(pcd_train_errs):10.2f} {np.mean(pcd_test_errs):10.2f}")

print(f"\nPaper target: AG train=15.45, AG test=14.19, PCD train=6.75, PCD test=4.41")
