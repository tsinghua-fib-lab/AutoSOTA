import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
sys.path.insert(0, '/repo')
import run_ml_experiment as r

# Load and preprocess
X_raw, y_raw = r.load_ecoli()
X, y = r.preprocess_ecoli(X_raw, y_raw)
N, d = X.shape
print(f"N={N}, d={d}")

# Test different step sizes
np.random.seed(42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8, random_state=42)

# Compute per-coordinate Lipschitz constants
L_j = np.zeros(d)
for j in range(d):
    L_j[j] = 0.25 * np.mean(X_tr[:, j]**2)
print(f"L_j: min={L_j.min():.4f}, max={L_j.max():.4f}, mean={L_j.mean():.4f}")

# Optimal step size for each coordinate
eta_j = 1.0 / L_j
print(f"Step sizes: min={eta_j.min():.2f}, max={eta_j.max():.2f}, mean={eta_j.mean():.2f}")

# Test PCD with different approaches
for step_size in [1.0, 4.0, 10.0, 20.0, 50.0]:
    np.random.seed(42)
    h = np.zeros(d)
    for t in range(100):
        coords = np.random.choice(d, size=min(2, d), replace=False)
        for j in coords:
            g_j = r.compute_gradient_j(X_tr, y_tr, h, j)
            h[j] = r.soft_threshold(h[j] - step_size * g_j, step_size * 1e-8)
    
    train_pred = np.sign(X_tr @ h); train_pred[train_pred==0]=1
    test_pred = np.sign(X_te @ h); test_pred[test_pred==0]=1
    print(f"step={step_size:.0f}: train_err={100*(1-accuracy_score(y_tr,train_pred)):.2f}%, test_err={100*(1-accuracy_score(y_te,test_pred)):.2f}%")

# Test with more iterations
print("\nWith more iterations:")
for T in [100, 200, 500, 1000]:
    np.random.seed(42)
    h = np.zeros(d)
    step = 4.0
    for t in range(T):
        coords = np.random.choice(d, size=min(2, d), replace=False)
        for j in coords:
            g_j = r.compute_gradient_j(X_tr, y_tr, h, j)
            h[j] = r.soft_threshold(h[j] - step * g_j, step * 1e-8)
    
    train_pred = np.sign(X_tr @ h); train_pred[train_pred==0]=1
    test_pred = np.sign(X_te @ h); test_pred[test_pred==0]=1
    print(f"T={T}: train_err={100*(1-accuracy_score(y_tr,train_pred)):.2f}%, test_err={100*(1-accuracy_score(y_te,test_pred)):.2f}%")

# Test using sklearn LogisticRegression (non-DP) as reference
print("\nSklearn reference:")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=1/1e-8, solver='saga', max_iter=1000, tol=1e-8)
lr.fit(X_tr, y_tr)
train_pred = lr.predict(X_tr)
test_pred = lr.predict(X_te)
print(f"sklearn: train_err={100*(1-accuracy_score(y_tr,train_pred)):.2f}%, test_err={100*(1-accuracy_score(y_te,test_pred)):.2f}%")
