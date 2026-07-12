import sys
sys.path.insert(0, "/repo/src")
from utils import GenSynthDataset
from semi_KO import Semi_KO
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np
import time

seed = 0
n, p = 300, 50
print(f"Starting seed {seed}...", flush=True)

X_comp, y_comp, true_imp = GenSynthDataset(n=n+1000, d=p, setting="adjacent", seed=seed)
print(f"Data generated: X={X_comp.shape}", flush=True)

X = X_comp[:n]
y = y_comp[:n]

model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
print("Fitting model...", flush=True)
t0 = time.time()
model.fit(X, y)
print(f"Model fit done in {time.time()-t0:.1f}s", flush=True)

imputer = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
cpi_ko = Semi_KO(
    estimator=model,
    imputation_model=clone(imputer),
    imputation_model_y=clone(imputer),
    random_state=seed,
    n_jobs=1
)
print("Fitting SKO...", flush=True)
t0 = time.time()
cpi_ko.fit(X, y)
print(f"SKO fit done in {time.time()-t0:.1f}s", flush=True)

for n_perm in [1, 5]:
    t0 = time.time()
    score_dict = cpi_ko.score(X, y, n_perm=n_perm, p_val="wilcox")
    elapsed = time.time() - t0
    print(f"Score(n_perm={n_perm}) done in {elapsed:.1f}s", flush=True)
    
    pvals = score_dict["pval"].reshape(p)
    pvals = np.nan_to_num(pvals, nan=1.0)
    selected = pvals <= 0.05
    power = sum(selected[true_imp == 1]) / max(sum(true_imp == 1), 1)
    type_I = sum(selected[true_imp == 0]) / max(sum(true_imp == 0), 1)
    auc = roc_auc_score(true_imp, 1 - pvals)
    print(f"n_perm={n_perm}: Power={power:.4f}, TypeI={type_I:.4f}, AUC={auc:.4f}", flush=True)

print("Done!", flush=True)
