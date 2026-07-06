"""Test a specific SOCOP lambda with 1 seed."""
import sys, time, numpy as np
sys.path.insert(0, "/repo")
from pathlib import Path
from evaluation import evaluate_seed, get_dataset_config

def test_lambda(lam, seed=23):
    config = get_dataset_config("covid")
    path = Path(f"/repo/results/Covid_data/probabilities/seed_{seed}.npz")
    with np.load(path) as data:
        cal_probs = data["cal_probs"]
        cal_labels = data["cal_labels"]
        test_probs = data["test_probs"]
        test_labels = data["test_labels"]
    
    # Monkey-patch SOCOP default lambda
    import baselines
    orig_init = baselines.SOCOPConformal.__init__
    def new_init(self, actions, lambda_param=lam):
        orig_init(self, actions, lambda_param=lambda_param)
    baselines.SOCOPConformal.__init__ = new_init
    
    t0 = time.time()
    result = evaluate_seed(cal_probs, cal_labels, test_probs, test_labels,
                          config["loss_matrix"], config["alpha_list"],
                          config["critical_labels"])
    
    baselines.SOCOPConformal.__init__ = orig_init
    
    alpha_idx = 5
    socop_wcr = result["worst_case_risk_scores"]["SOCOP"]["a_ROCP"][alpha_idx]
    rocp_wcr = result["worst_case_risk"]["ROCP"][alpha_idx]
    socop_rl = result["realized_loss_scores"]["SOCOP"]["a_ROCP"][alpha_idx]
    rocp_rl = result["realized_loss"]["ROCP"][alpha_idx]
    elapsed = time.time() - t0
    
    cm = result["critical_mistake"]["rocp"]
    cm_avg = np.mean([cm[l] for l in config["critical_labels"]]) * 100
    
    return {
        "lambda": lam,
        "socop_wcr": float(socop_wcr),
        "rocp_wcr": float(rocp_wcr),
        "socop_rl": float(socop_rl),
        "rocp_rl": float(rocp_rl),
        "cm_avg_pct": float(cm_avg),
        "time_s": elapsed,
    }

if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0625
    print(f"Testing SOCOP lambda={lam}...", flush=True)
    res = test_lambda(lam)
    print(f"  SOCOP a_ROCP WCR: {res['socop_wcr']:.4f}", flush=True)
    print(f"  ROCP WCR:          {res['rocp_wcr']:.4f}", flush=True)
    print(f"  SOCOP RL:          {res['socop_rl']:.4f}", flush=True)
    print(f"  ROCP RL:           {res['rocp_rl']:.4f}", flush=True)
    print(f"  CM avg:            {res['cm_avg_pct']:.2f}%", flush=True)
    print(f"  Time:              {res['time_s']:.0f}s", flush=True)
