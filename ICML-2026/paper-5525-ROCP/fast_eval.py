"""Fast evaluation harness: runs on a subset of seeds for quick validation."""
import sys, time, json, numpy as np
sys.path.insert(0, "/repo")
from pathlib import Path
from evaluation import evaluate_seed, get_dataset_config, mean_and_se

def fast_evaluate(dataset="covid", num_seeds=5, start_seed=23):
    config = get_dataset_config(dataset)
    prob_dir = Path(f"/repo/results/Covid_data/probabilities")
    seed_files = sorted(prob_dir.glob("seed_*.npz"))
    seed_files = [(int(p.stem.split("_")[1]), p) for p in seed_files 
                  if int(p.stem.split("_")[1]) >= start_seed]
    seed_files.sort(key=lambda x: x[0])
    seed_files = seed_files[:num_seeds]
    
    if not seed_files:
        print("No seed files found!", file=sys.stderr)
        return None
    
    alpha_list = config["alpha_list"]
    critical_labels = config["critical_labels"]
    loss_matrix = config["loss_matrix"]
    label_names = config["label_names"]
    
    print(f"Fast eval: {len(seed_files)} seeds, starting at seed {seed_files[0][0]}")
    t0 = time.time()
    
    seed_results = []
    for seed, path in seed_files:
        t_seed = time.time()
        with np.load(path) as data:
            result = evaluate_seed(
                data["cal_probs"], data["cal_labels"],
                data["test_probs"], data["test_labels"],
                loss_matrix, alpha_list, critical_labels,
            )
        seed_results.append(result)
        elapsed = time.time() - t_seed
        print(f"  Seed {seed}: {elapsed:.0f}s", flush=True)
    
    total_time = time.time() - t0
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # Extract metrics at alpha=0.05 (index 5)
    alpha_idx = 5  # alpha_list[5] = 0.05
    
    wcr_rocp_vals = [r["worst_case_risk"]["ROCP"][alpha_idx] for r in seed_results]
    wcr_rac_vals = [r["worst_case_risk"]["RAC"][alpha_idx] for r in seed_results]
    rl_rocp_vals = [r["realized_loss"]["ROCP"][alpha_idx] for r in seed_results]
    rl_rac_vals = [r["realized_loss"]["RAC"][alpha_idx] for r in seed_results]
    mis_rocp_vals = [r["miscoverage"]["ROCP"][alpha_idx] for r in seed_results]
    
    cm_rocp = {}
    cm_rac = {}
    cm_best = {}
    for lbl in critical_labels:
        cm_rocp[lbl] = [r["critical_mistake"]["rocp"][lbl] for r in seed_results]
        cm_rac[lbl] = [r["critical_mistake"]["rac"][lbl] for r in seed_results]
        cm_best[lbl] = [r["critical_mistake"]["best"][lbl] for r in seed_results]
    
    wcr_mean, wcr_se = mean_and_se(wcr_rocp_vals)
    rl_mean, rl_se = mean_and_se(rl_rocp_vals)
    mis_mean, mis_se = mean_and_se(mis_rocp_vals)
    
    # Average critical mistake rate across critical labels
    cm_rocp_avg = []
    for i in range(len(seed_results)):
        avg = np.mean([cm_rocp[lbl][i] for lbl in critical_labels])
        cm_rocp_avg.append(avg)
    cm_mean_val, cm_se_val = mean_and_se(cm_rocp_avg)
    
    result = {
        "num_seeds": len(seed_files),
        "total_time_s": total_time,
        "alpha": 0.05,
        "worst_case_risk_rocp": {"mean": float(wcr_mean), "se": float(wcr_se)},
        "realized_loss_rocp": {"mean": float(rl_mean), "se": float(rl_se)},
        "miscoverage_rocp": {"mean": float(mis_mean), "se": float(mis_se)},
        "critical_mistake_rate_rocp_pct": {
            "mean": float(cm_mean_val * 100),
            "se": float(cm_se_val * 100),
        },
        "worst_case_risk_rac": {
            "mean": float(np.mean(wcr_rac_vals)),
            "se": float(np.std(wcr_rac_vals, ddof=1) / np.sqrt(len(wcr_rac_vals))) if len(wcr_rac_vals) > 1 else 0,
        },
        "per_seed_wcr_rocp": [float(v) for v in wcr_rocp_vals],
        "per_seed_cm_rocp_pct": {str(lbl): [float(v * 100) for v in cm_rocp[lbl]] for lbl in critical_labels},
    }
    
    print(f"\n=== RESULTS (alpha=0.05, {len(seed_files)} seeds) ===")
    print(f"Worst-Case Risk ROCP: {wcr_mean:.4f} +/- {wcr_se:.4f}")
    print(f"Realized Loss ROCP:    {rl_mean:.4f} +/- {rl_se:.4f}")
    print(f"Miscoverage ROCP:      {mis_mean:.4f} +/- {mis_se:.4f}")
    print(f"Critical Mistake ROCP: {cm_mean_val*100:.2f}% +/- {cm_se_val*100:.2f}%")
    for lbl in critical_labels:
        cm_lbl_vals = cm_rocp[lbl]
        cm_lbl_mean = np.mean(cm_lbl_vals)
        print(f"  {label_names[lbl]}: {cm_lbl_mean*100:.2f}%")
    
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--start", type=int, default=23)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    result = fast_evaluate(num_seeds=args.seeds, start_seed=args.start)
    if args.json and result:
        print(json.dumps(result, indent=2))
