"""Sweep SOCOP lambda_param to find optimal value for COVID dataset."""
import sys, time, numpy as np
sys.path.insert(0, "/repo")
from pathlib import Path
from evaluation import evaluate_seed, get_dataset_config, mean_and_se

def sweep_socop_lambda(num_seeds=3, start_seed=23):
    config = get_dataset_config("covid")
    prob_dir = Path("/repo/results/Covid_data/probabilities")
    seed_files = sorted(prob_dir.glob("seed_*.npz"))
    seed_files = [(int(p.stem.split("_")[1]), p) for p in seed_files
                  if int(p.stem.split("_")[1]) >= start_seed]
    seed_files.sort(key=lambda x: x[0])
    seed_files = seed_files[:num_seeds]
    
    loss_matrix = config["loss_matrix"]
    alpha_list = config["alpha_list"]
    critical_labels = config["critical_labels"]
    
    lambda_values = [0.01, 0.03125, 0.0625, 0.125, 0.25, 0.5]
    
    # Load all seed data first
    all_data = []
    for seed, path in seed_files:
        with np.load(path) as data:
            all_data.append({
                "seed": seed,
                "cal_probs": data["cal_probs"],
                "cal_labels": data["cal_labels"],
                "test_probs": data["test_probs"],
                "test_labels": data["test_labels"],
            })
    
    results_by_lambda = {}
    
    for lam in lambda_values:
        print(f"\n=== SOCOP lambda={lam:.5f} ===", flush=True)
        t0 = time.time()
        
        # Temporarily modify SOCOP default in baselines
        import baselines
        orig_init = baselines.SOCOPConformal.__init__
        
        def new_init(self, actions, lambda_param=lam):
            orig_init(self, actions, lambda_param=lambda_param)
        baselines.SOCOPConformal.__init__ = new_init
        
        seed_results = []
        for d in all_data:
            result = evaluate_seed(
                d["cal_probs"], d["cal_labels"],
                d["test_probs"], d["test_labels"],
                loss_matrix, alpha_list, critical_labels,
            )
            seed_results.append(result)
        
        # Restore original
        baselines.SOCOPConformal.__init__ = orig_init
        
        alpha_idx = 5  # alpha=0.05
        wcr_vals = []
        for r in seed_results:
            # SOCOP a_ROCP worst-case risk at alpha=0.05
            wcr_vals.append(r["worst_case_risk_scores"]["SOCOP"]["a_ROCP"][alpha_idx])
        
        wcr_mean, wcr_se = mean_and_se(wcr_vals)
        
        # Also get ROCP baseline for comparison
        rocp_vals = [r["worst_case_risk"]["ROCP"][alpha_idx] for r in seed_results]
        rocp_mean, _ = mean_and_se(rocp_vals)
        
        elapsed = time.time() - t0
        print(f"  SOCOP a_ROCP WCR: {wcr_mean:.4f} +/- {wcr_se:.4f}", flush=True)
        print(f"  ROCP WCR:          {rocp_mean:.4f}", flush=True)
        print(f"  Time: {elapsed:.0f}s", flush=True)
        
        results_by_lambda[lam] = {
            "socop_wcr_mean": float(wcr_mean),
            "socop_wcr_se": float(wcr_se),
            "rocp_wcr_mean": float(rocp_mean),
            "time_s": elapsed,
        }
    
    # Find best lambda
    print("\n=== SWEEP SUMMARY ===")
    best_lam = None
    best_wcr = float("inf")
    for lam, res in results_by_lambda.items():
        marker = ""
        if res["socop_wcr_mean"] < best_wcr:
            best_wcr = res["socop_wcr_mean"]
            best_lam = lam
            marker = " <-- BEST"
        print(f"  lambda={lam:.5f}: SOCOP WCR={res['socop_wcr_mean']:.4f} +/- {res['socop_wcr_se']:.4f}{marker}")
    
    print(f"\nBest lambda: {best_lam} (SOCOP WCR = {best_wcr:.4f})")
    return best_lam, results_by_lambda

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--start", type=int, default=23)
    args = p.parse_args()
    
    print(f"SOCOP lambda sweep: {args.seeds} seeds, starting at {args.start}")
    best, results = sweep_socop_lambda(num_seeds=args.seeds, start_seed=args.start)
    print(f"\nRECOMMENDATION: Use lambda_param={best}")
