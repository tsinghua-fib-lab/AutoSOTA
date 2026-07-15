import json
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

print("=== 1. L8 COHEN'S D BOOTSTRAP (1000 iterations) ===")
with open('data/results/rebuttal/llama8b_full_extraction.json', 'r') as f:
    data = json.load(f)

# Extract L8 metrics
feats = []
for item in data:
    if 'spectral' not in item or 'layer_8' not in item['spectral']:
        continue
    label_str = item.get('label_corrected', item.get('label_original', ''))
    lbl = 1 if label_str.lower() == 'valid' else 0
    ent = item['spectral']['layer_8'].get('entropy', 0.0)
    hfer = item['spectral']['layer_8'].get('hfer', 0.0)
    feats.append((lbl, ent, hfer))
feats = np.array(feats)

def compute_d(v, iv):
    if len(v)==0 or len(iv)==0: return 0
    var1, var2 = np.var(v, ddof=1), np.var(iv, ddof=1)
    if var1 == 0 and var2 == 0: return 0
    pooled_sd = np.sqrt(((len(v)-1)*var1 + (len(iv)-1)*var2) / (len(v)+len(iv)-2))
    return abs(np.mean(v) - np.mean(iv)) / pooled_sd if pooled_sd > 0 else 0

d_ents, d_hfers = [], []
np.random.seed(42)
for _ in range(1000):
    idx = np.random.choice(len(feats), size=len(feats), replace=True)
    samp = feats[idx]
    v = samp[samp[:,0] == 1]
    iv = samp[samp[:,0] == 0]
    d_ents.append(compute_d(v[:,1], iv[:,1]))
    d_hfers.append(compute_d(v[:,2], iv[:,2]))

ent_lb, ent_ub = np.percentile(d_ents, [2.5, 97.5])
hfer_lb, hfer_ub = np.percentile(d_hfers, [2.5, 97.5])
print(f"L8 Entropy d: {np.mean(d_ents):.3f} (95% CI: [{ent_lb:.3f}, {ent_ub:.3f}])")
print(f"L8 HFER d: {np.mean(d_hfers):.3f} (95% CI: [{hfer_lb:.3f}, {hfer_ub:.3f}])")

print("\n=== 2. HFER CALIBRATION BOOTSTRAP (200 iterations per N) ===")
# Reuse L30 HFER baseline items
hfers_l30 = []
for item in data:
    if 'spectral' not in item or 'layer_30' not in item['spectral']: continue
    label_str = item.get('label_corrected', item.get('label_original', ''))
    lbl = 1 if label_str.lower() == 'valid' else 0
    hfer = item['spectral']['layer_30'].get('hfer', 0.0)
    hfers_l30.append((lbl, hfer))
hfers_l30 = np.array(hfers_l30)
X_hf = hfers_l30[:, 1]
y_hf = hfers_l30[:, 0]

for n in [10, 25, 50, 100]:
    accs = []
    for _ in range(200):
        # Sample N indices, ensuring at least one of each class if possible, but uniform random is what the real protocol does
        calib_idx = np.random.choice(len(X_hf), n, replace=False)
        test_idx = np.setdiff1d(np.arange(len(X_hf)), calib_idx)
        
        cX, cy = X_hf[calib_idx], y_hf[calib_idx]
        tX, ty = X_hf[test_idx], y_hf[test_idx]
        
        if len(np.unique(cy)) < 2:
            preds = np.zeros_like(ty) # Fail safely
        else:
            best_acc, best_t = 0, 0
            # Sweep thresholds
            for t in np.linspace(cX.min(), cX.max(), 100):
                preds = (cX < t).astype(int)
                acc = accuracy_score(cy, preds)
                if acc > best_acc:
                    best_acc, best_t = acc, t
            preds = (tX < best_t).astype(int)
        accs.append(accuracy_score(ty, preds))
    print(f"N={n}: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")

print("\n=== 3. SUPERVISED CROSS-SEED VARIANCE (10 Seeds) ===")
npz = np.load('data/results/rebuttal/hidden_states_llama8b.npz')
X = npz['hidden_L30']
y = npz['labels']

seeds = [0, 1, 2, 3, 4, 42, 123, 456, 789, 1000]
lp_results = []
mlp_results = []

for s in seeds:
    # LP
    lp = LogisticRegression(max_iter=1000)
    param_grid = {'C': [0.1, 1, 10]}
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=s)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    lp_clf = GridSearchCV(lp, param_grid, cv=inner_cv)
    
    lp_accs_inner = []
    for train_ix, test_ix in outer_cv.split(X, y):
        lp_clf.fit(X[train_ix], y[train_ix])
        lp_accs_inner.append(accuracy_score(y[test_ix], lp_clf.predict(X[test_ix])))
    lp_results.append(np.mean(lp_accs_inner))
    
    # MLP (Reducing max_iter slightly for speed, standard 200 is sufficient for robust convergence)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, early_stopping=True, random_state=s)
    mlp_accs_inner = []
    for train_ix, test_ix in outer_cv.split(X, y):
        mlp.fit(X[train_ix], y[train_ix])
        mlp_accs_inner.append(accuracy_score(y[test_ix], mlp.predict(X[test_ix])))
    mlp_results.append(np.mean(mlp_accs_inner))
    
    print(f"Seed {s:4d}: LP = {lp_results[-1]*100:.1f}%, MLP = {mlp_results[-1]*100:.1f}%")

print(f"\nFinal LP:  {np.mean(lp_results)*100:.1f}% ± {np.std(lp_results)*100:.1f}%")
print(f"Final MLP: {np.mean(mlp_results)*100:.1f}% ± {np.std(mlp_results)*100:.1f}%")
