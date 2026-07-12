"""
Evaluation script with BH procedure option for feature selection.
"""
import numpy as np
import pandas as pd
import sys
import os
import time
sys.path.insert(0, "/repo/src")
from utils import GenSynthDataset, _bhq_threshold
from semi_KO import Semi_KO
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.base import clone

def run_single_seed(seed, n_perm_list=None, use_bh=False, fdr=0.05):
    if n_perm_list is None:
        n_perm_list = [1]
    n, p, n_r2 = 300, 50, 1000

    X_comp, y_comp, true_imp = GenSynthDataset(n=n+n_r2, d=p, setting="adjacent", seed=seed)
    X = X_comp[:n]
    y = y_comp[:n]
    X_r2 = X_comp[n:]
    y_r2 = y_comp[n:]

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
    model.fit(X, y)
    r2_val = r2_score(y_r2, model.predict(X_r2))

    imputer = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
    
    results = []
    start = time.time()
    cpi_ko = Semi_KO(
        estimator=model,
        imputation_model=clone(imputer),
        imputation_model_y=clone(imputer),
        random_state=seed,
        n_jobs=1
    )
    cpi_ko.fit(X, y)
    fit_time = time.time() - start

    for n_perm in n_perm_list:
        start = time.time()
        score_dict = cpi_ko.score(X, y, n_perm=n_perm, p_val="wilcox")
        pvals = score_dict["pval"].reshape(p)
        elapsed = time.time() - start + fit_time
        
        pvals = np.nan_to_num(pvals, nan=1.0)
        
        if use_bh:
            threshold = _bhq_threshold(pvals, fdr=fdr)
            selected = pvals <= threshold
            method_suffix = "_BH"
        else:
            selected = pvals <= 0.05
            method_suffix = ""
        
        power = sum(selected[true_imp == 1]) / max(sum(true_imp == 1), 1)
        type_I = sum(selected[true_imp == 0]) / max(sum(true_imp == 0), 1)
        auc = roc_auc_score(true_imp, 1 - pvals)
        
        key = "SKO_Wcx" if n_perm == 1 else "SKO_Wcx_p%d" % n_perm
        key = key + method_suffix
        
        results.append({
            "method": key, "power": power, "type_I": type_I, "auc": auc,
            "r2": r2_val, "time": elapsed, "seed": seed
        })
    
    return results

def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
    use_bh = "--bh" in sys.argv
    fdr = 0.05
    for i, a in enumerate(sys.argv):
        if a == "--fdr" and i+1 < len(sys.argv):
            fdr = float(sys.argv[i+1])
    
    env_nperm = os.environ.get("SKO_NPERM", "1")
    n_perm_list = [int(x.strip()) for x in env_nperm.split(",")]
    
    label = f"BH(fdr={fdr})" if use_bh else "fixed_alpha=0.05"
    print(f"Running {n_seeds} seeds with n_perm={n_perm_list}, selection={label}", flush=True)
    
    all_results = []
    for seed in range(n_seeds):
        try:
            res = run_single_seed(seed, n_perm_list=n_perm_list, use_bh=use_bh, fdr=fdr)
            all_results.extend(res)
            for r in res:
                print("Seed %d %s: Power=%.4f, TypeI=%.4f, AUC=%.4f, R2=%.4f" % (seed, r["method"], r["power"], r["type_I"], r["auc"], r["r2"]), flush=True)
        except Exception as e:
            import traceback
            print("Seed %d FAILED: %s" % (seed, str(e)), flush=True)
            traceback.print_exc()
    
    df = pd.DataFrame(all_results)
    print("\n=== AGGREGATE RESULTS ===", flush=True)
    for method in sorted(df["method"].unique()):
        subset = df[df["method"] == method]
        print("%s: N=%d, Power=%.4f+/-%.4f, TypeI=%.4f+/-%.4f, AUC=%.4f+/-%.4f, R2=%.4f+/-%.4f" % (
            method, len(subset),
            subset["power"].mean(), subset["power"].std(),
            subset["type_I"].mean(), subset["type_I"].std(),
            subset["auc"].mean(), subset["auc"].std(),
            subset["r2"].mean(), subset["r2"].std()), flush=True)
    
    for method in sorted(df["method"].unique()):
        subset = df[df["method"] == method]
        print("%s: N=%d, Power=%.4f, TypeI=%.4f, AUC=%.4f, R2=%.4f" % (
            method, len(subset),
            subset["power"].mean(), subset["type_I"].mean(),
            subset["auc"].mean(), subset["r2"].mean()), flush=True)
    
    df.to_csv("/repo/results/reproduction_metrics.csv", index=False)
    print("\nSaved %d rows to /repo/results/reproduction_metrics.csv" % len(df), flush=True)

if __name__ == "__main__":
    main()
