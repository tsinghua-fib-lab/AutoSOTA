import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys; sys.path.insert(0, '/repo')
import run_ml_experiment as r

X_raw, y_raw = r.load_ecoli()
X, y = r.preprocess_ecoli(X_raw, y_raw)
N, d = X.shape
print(f"N={N}, d={d}")

np.random.seed(42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8, random_state=42)

# Non-private PCD
h = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8)
p_tr = np.sign(X_tr @ h); p_tr[p_tr==0]=1
p_te = np.sign(X_te @ h); p_te[p_te==0]=1
print(f"PCD (T=100, Newton): train={100*(1-accuracy_score(y_tr,p_tr)):.2f}%, test={100*(1-accuracy_score(y_te,p_te)):.2f}%")

# With more iterations
h = r.proximal_coordinate_descent(X_tr, y_tr, 500, 2, 1e-8)
p_tr = np.sign(X_tr @ h); p_tr[p_tr==0]=1
p_te = np.sign(X_te @ h); p_te[p_te==0]=1
print(f"PCD (T=500, Newton): train={100*(1-accuracy_score(y_tr,p_tr)):.2f}%, test={100*(1-accuracy_score(y_te,p_te)):.2f}%")

# DP
noise_scale = 1.0 / N
sampler = r.make_ag_sampler(21.105645147662777, noise_scale)
np.random.seed(42)
h = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8, noise_sampler=sampler)
p_tr = np.sign(X_tr @ h); p_tr[p_tr==0]=1
p_te = np.sign(X_te @ h); p_te[p_te==0]=1
print(f"AG-DP (T=100, Newton): train={100*(1-accuracy_score(y_tr,p_tr)):.2f}%, test={100*(1-accuracy_score(y_te,p_te)):.2f}%")
print(f"h: mean={np.mean(h):.3f}, std={np.std(h):.3f}")
