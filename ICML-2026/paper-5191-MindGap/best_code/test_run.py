import run_ml_experiment as r
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess
X_raw, y_raw = r.load_ecoli()
X, y = r.preprocess_ecoli(X_raw, y_raw)
print(f"N={X.shape[0]}, d={X.shape[1]}")
print(f"Label dist: +1={np.sum(y==1)}, -1={np.sum(y==-1)}")
print(f"X range: [{X.min():.3f}, {X.max():.3f}]")

# Test non-private PCD
np.random.seed(42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8, random_state=42)
h = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8, noise_sampler=None)
train_pred = np.sign(X_tr @ h); train_pred[train_pred==0] = 1
test_pred = np.sign(X_te @ h); test_pred[test_pred==0] = 1
print(f"Non-private PCD: train_err={100*(1-accuracy_score(y_tr, train_pred)):.2f}%, test_err={100*(1-accuracy_score(y_te, test_pred)):.2f}%")

# Test with AG noise
N = X.shape[0]
noise_scale = 1.0 / N
sampler = r.make_ag_sampler(21.105645147662777, noise_scale)
np.random.seed(42)
h = r.proximal_coordinate_descent(X_tr, y_tr, 100, 2, 1e-8, noise_sampler=sampler)
train_pred = np.sign(X_tr @ h); train_pred[train_pred==0] = 1
test_pred = np.sign(X_te @ h); test_pred[test_pred==0] = 1
print(f"AG DP: train_err={100*(1-accuracy_score(y_tr, train_pred)):.2f}%, test_err={100*(1-accuracy_score(y_te, test_pred)):.2f}%")
print(f"h stats: mean={np.mean(h):.4f}, std={np.std(h):.4f}, min={np.min(h):.4f}, max={np.max(h):.4f}")
