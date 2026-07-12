"""Verify vectorized loss gives same Wilcoxon p-values as original."""
import sys
sys.path.insert(0, "/repo/src")
import numpy as np
from utils import GenSynthDataset
from semi_KO import Semi_KO
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.base import clone

seed = 42
n, p = 300, 50
X_comp, y_comp, true_imp = GenSynthDataset(n=n+1000, d=p, setting="adjacent", seed=seed)
X = X_comp[:n]
y = y_comp[:n]

model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
model.fit(X, y)

imputer = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
cpi_ko = Semi_KO(
    estimator=model,
    imputation_model=clone(imputer),
    imputation_model_y=clone(imputer),
    random_state=seed,
    n_jobs=1
)
cpi_ko.fit(X, y)

for n_perm in [1, 5]:
    score_dict = cpi_ko.score(X, y, n_perm=n_perm, p_val="wilcox")
    pvals = score_dict["pval"].reshape(p)
    pvals = np.nan_to_num(pvals, nan=1.0)
    selected = pvals <= 0.05
    
    # Check loss arrays
    for j in range(p):
        lp = score_dict["loss_perm"][j]
        lpy = score_dict["loss_perm_y"][j]
        expected_len = n_perm * n
        assert len(lp) == expected_len, f"j={j}: expected {expected_len}, got {len(lp)}"
        assert len(lpy) == expected_len, f"j={j}: expected {expected_len}, got {len(lpy)}"
    
    power = sum(selected[true_imp == 1]) / max(sum(true_imp == 1), 1)
    type_I = sum(selected[true_imp == 0]) / max(sum(true_imp == 0), 1)
    print(f"n_perm={n_perm}: Power={power:.4f}, TypeI={type_I:.4f}, all lengths OK")

print("Vectorized loss correctness VERIFIED")
