import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../"))
import numpy as np
import pickle
from plotting_utils import longest_true_sequence

with open("./results/AXP.pkl", "rb") as f:
    all_results = pickle.load(f)

print("Models:", list(all_results.keys()))

for model_name, results in all_results.items():
    if not isinstance(results, dict):
        continue
    print(f"\n=== {model_name} ===")
    T_burnin = results.get("T_burnin", 100)
    alpha = results.get("alpha", 0.05)
    print(f"T_burnin={T_burnin}, alpha={alpha}")
    
    for method_key in results:
        if method_key in ["scores", "alpha", "T_burnin", "quantiles_given", 
                          "multiple_series", "real_data", "score_function_name",
                          "asymmetric", "forecasts", "data"]:
            continue
        
        method_results = results[method_key]
        if not isinstance(method_results, dict):
            continue
            
        for lr, lr_results in method_results.items():
            if not isinstance(lr_results, dict) or "coverages" not in lr_results:
                continue
                
            c = np.array(lr_results["coverages"])
            q = np.array([np.array(x) for x in lr_results["q"]])
            sets = lr_results["sets"]
            
            # Post burn-in metrics
            c_burnin = c[T_burnin:]
            q_burnin = q[T_burnin:]
            
            # Compute set sizes
            if q.ndim == 1:
                set_sizes = 2 * q_burnin
            else:
                set_sizes = q_burnin[:, 1] - q_burnin[:, 0]
            
            coverage = np.mean(c_burnin)
            avg_size = np.mean(set_sizes)
            median_size = np.median(set_sizes)
            p75 = np.percentile(set_sizes, 75)
            p90 = np.percentile(set_sizes, 90)
            p95 = np.percentile(set_sizes, 95)
            
            errors = 1 - c_burnin
            longest_err = longest_true_sequence(errors.astype(bool))
            
            print(f"\n  {method_key} (lr={lr}):")
            print(f"    Marginal Coverage: {coverage:.4f}")
            print(f"    Longest Err. Seq.: {longest_err}")
            print(f"    Avg Set Size: {avg_size:.2f}")
            print(f"    Median Set Size: {median_size:.2f}")
            print(f"    75% Quantile Size: {p75:.2f}")
            print(f"    90% Quantile Size: {p90:.2f}")
            print(f"    95% Quantile Size: {p95:.2f}")
