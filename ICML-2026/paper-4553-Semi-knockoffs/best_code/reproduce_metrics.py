"""
Focused reproduction script for Semi-knockoffs paper.
Runs only the Semi_KO methods for the Adjacent Support setting
with GradientBoosting: n=300, p=50, n_repetitions=50,
data=Gaussian_AR1_rho0.6, y=linear_with_noise,
important_features=first_0.25p_beta_in_[1,2], significance_level=0.05

Usage:
  python3 reproduce_metrics.py <n_seeds> [--nperm N] [--imputer IMPUTER] [--model MODEL]
  
  Environment: SKO_NPERM, SKO_IMPUTER, SKO_MODEL
"""
import numpy as np
import pandas as pd
import sys
import os
import time
sys.path.insert(0, "/repo/src")
from utils import GenSynthDataset
from semi_KO import Semi_KO
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone

def get_imputer(name, seed):
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
    elif name == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=10, random_state=seed, n_jobs=1)
    elif name == "lasso":
        return LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed)
    elif name == "elasticnet":
        return ElasticNetCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed)
    else:
        raise ValueError(f"Unknown imputer: {name}")

def get_model(name, seed):
    if name == "gb":
        return GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
    elif name == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=None, random_state=seed, n_jobs=1)
    elif name == "nn":
        return MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {name}")

def run_single_seed(seed, setting="adjacent", n=300, p=50, n_r2=1000, n_perm_list=None, imputer_name="ridge", model_name="gb"):
    if n_perm_list is None:
        n_perm_list = [1]
    
    X_comp, y_comp, true_imp = GenSynthDataset(n=n+n_r2, d=p, setting=setting, seed=seed)
    X = X_comp[:n]
    y = y_comp[:n]
    X_r2 = X_comp[n:]
    y_r2 = y_comp[n:]
    
    model = get_model(model_name, seed)
    model.fit(X, y)
    r2_val = r2_score(y_r2, model.predict(X_r2))
    
    imputer = get_imputer(imputer_name, seed)
    
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
        selected = pvals <= 0.05
        power = sum(selected[true_imp == 1]) / max(sum(true_imp == 1), 1)
        type_I = sum(selected[true_imp == 0]) / max(sum(true_imp == 0), 1)
        auc = roc_auc_score(true_imp, 1 - pvals)
        
        if n_perm == 1:
            key = "SKO_Wcx"
        else:
            key = "SKO_Wcx_p%d" % n_perm
        
        imputer_tag = "" if imputer_name == "ridge" else "_%s" % imputer_name
        model_tag = "" if model_name == "gb" else "_%s" % model_name
        key = key + imputer_tag + model_tag
        
        results.append({
            "method": key, "power": power, "type_I": type_I, "auc": auc, 
            "r2": r2_val, "time": elapsed, "seed": seed
        })
    
    return results

def main():
    args = sys.argv[1:]
    n_seeds = 50
    n_perm_list = None
    
    if args and args[0].isdigit():
        n_seeds = int(args[0])
        args = args[1:]
    
    imputer_name = os.environ.get("SKO_IMPUTER", "ridge")
    model_name = os.environ.get("SKO_MODEL", "gb")
    
    for i, a in enumerate(args):
        if a == "--nperm" and i+1 < len(args):
            n_perm_list = [int(x.strip()) for x in args[i+1].split(",")]
        elif a == "--imputer" and i+1 < len(args):
            imputer_name = args[i+1]
        elif a == "--model" and i+1 < len(args):
            model_name = args[i+1]
    
    if n_perm_list is None:
        env_nperm = os.environ.get("SKO_NPERM", "1")
        n_perm_list = [int(x.strip()) for x in env_nperm.split(",")]
    
    print(f"Running {n_seeds} seeds with n_perm={n_perm_list}, imputer={imputer_name}, model={model_name}", flush=True)
    
    all_results = []
    for seed in range(n_seeds):
        try:
            res = run_single_seed(seed, n_perm_list=n_perm_list, imputer_name=imputer_name, model_name=model_name)
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
    
    # Old-format for backward compat
    for method in sorted(df["method"].unique()):
        subset = df[df["method"] == method]
        print("%s: N=%d, Power=%.4f, TypeI=%.4f, AUC=%.4f, R2=%.4f" % (
            method, len(subset),
            subset["power"].mean(),
            subset["type_I"].mean(),
            subset["auc"].mean(),
            subset["r2"].mean()), flush=True)
    
    df.to_csv("/repo/results/reproduction_metrics.csv", index=False)
    print("\nSaved %d rows to /repo/results/reproduction_metrics.csv" % len(df), flush=True)

if __name__ == "__main__":
    main()
