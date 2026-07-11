import json
import numpy as np
from pathlib import Path

results = []
for d in sorted(Path(".logs").iterdir()):
    if not d.is_dir():
        continue
    try:
        config = json.load(open(d / "config.json"))
        s = config["seed"]
        sp = config["dataset_config"]["spread"]
    except Exception:
        continue
    
    metrics = [json.loads(l) for l in open(d / "metrics.jsonl")]
    
    oracle = [m["value"] for m in metrics if m.get("type") == "oracle"][0]
    grid_utils = [m["value"] for m in metrics if m.get("type") == "grid"]
    best_grid = max(grid_utils)
    
    pre_ew = [m for m in metrics if m.get("type") == "ew" and m.get("t") == 0][0]["value"]
    ew_metrics = [m for m in metrics if m.get("type") == "ew"]
    final_ew = ew_metrics[-1]["value"]
    
    mse_init = [m for m in metrics if m.get("name") == "mse" and m.get("t") == 0][0]["value"]
    mse_metrics = [m for m in metrics if m.get("name") == "mse"]
    mse_final = mse_metrics[-1]["value"]
    
    results.append({
        "seed": s, "spread": sp,
        "utility_gap": best_grid - final_ew,
        "utility_improvement": final_ew - pre_ew,
        "mse_reduction": mse_init - mse_final,
    })

print("=== Per-experiment results ===")
for r in results:
    print("seed={:2d} spread={} | Gap={:.5f} | Impr={:.5f} | MSE_red={:.5f}".format(
        r["seed"], r["spread"], r["utility_gap"], r["utility_improvement"], r["mse_reduction"]))

gaps = [r["utility_gap"] for r in results]
improvements = [r["utility_improvement"] for r in results]
mse_reductions = [r["mse_reduction"] for r in results]

print("\n=== Aggregate (mean +/- std across {} runs) ===".format(len(results)))
print("Utility Gap:        {:.5f} +/- {:.5f}".format(np.mean(gaps), np.std(gaps)))
print("Utility Improvement: {:.5f} +/- {:.5f}".format(np.mean(improvements), np.std(improvements)))
print("MSE Reduction:       {:.5f} +/- {:.5f}".format(np.mean(mse_reductions), np.std(mse_reductions)))

print("\n=== Paper values (estimated from figures) ===")
print("Utility Gap:        0.00500")
print("Utility Improvement: 0.08500")
print("MSE Reduction:       0.01500")
