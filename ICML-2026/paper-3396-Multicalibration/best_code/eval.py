"""Parse experiment metrics and report reproduction results.

Run after experiments to extract utility gap, utility improvement, and MSE reduction.
"""
import json
import numpy as np
from pathlib import Path

def parse_results(logs_dir=".logs"):
    results = []
    for d in sorted(Path(logs_dir).iterdir()):
        if not d.is_dir():
            continue
        try:
            config = json.load(open(d / "config.json"))
            s = config["seed"]
            sp = config["dataset_config"]["spread"]
        except Exception:
            continue

        metrics = [json.loads(l) for l in open(d / "metrics.jsonl")]

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

    if not results:
        print("ERROR: No experiment results found in", logs_dir)
        return

    gaps = [r["utility_gap"] for r in results]
    improvements = [r["utility_improvement"] for r in results]
    mse_reductions = [r["mse_reduction"] for r in results]

    print("=" * 60)
    print("Reproduction Results: Multicalibration Yields Better Matchings")
    print("Experiment: Best Action, m=4, eps=0.25, 10-d features")
    print("=" * 60)
    print()
    print("Per-experiment results:")
    for r in results:
        print("  seed={:2d}  spread={}  |  Gap={:+.5f}  Impr={:+.5f}  MSE_red={:+.5f}".format(
            r["seed"], r["spread"], r["utility_gap"], r["utility_improvement"], r["mse_reduction"]))

    print()
    print("Aggregate (mean +/- std, n={}):".format(len(results)))
    print("  Utility Gap:        {:.5f} +/- {:.5f}".format(np.mean(gaps), np.std(gaps)))
    print("  Utility Improvement: {:.5f} +/- {:.5f}".format(np.mean(improvements), np.std(improvements)))
    print("  MSE Reduction:       {:.5f} +/- {:.5f}".format(np.mean(mse_reductions), np.std(mse_reductions)))

    print()
    print("Paper reference values (from Figure 2a/3a):")
    print("  Utility Gap:        0.00500")
    print("  Utility Improvement: 0.08500")
    print("  MSE Reduction:       0.01500")

    print()
    print("CI bounds check:")
    gap_ok = 0.0 <= np.mean(gaps) <= 0.0055
    impr_ok = 0.0 <= np.mean(improvements) <= 0.0935
    mse_ok = 0.0 <= np.mean(mse_reductions) <= 0.0165
    print("  Utility Gap ({:.5f}): {}".format(np.mean(gaps), "WITHIN [0.0, 0.0055]" if gap_ok else "OUTSIDE CI"))
    print("  Utility Impr ({:.5f}): {}".format(np.mean(improvements), "WITHIN [0.0, 0.0935]" if impr_ok else "OUTSIDE CI"))
    print("  MSE Red ({:.5f}):     {}".format(np.mean(mse_reductions), "WITHIN [0.0, 0.0165]" if mse_ok else "OUTSIDE CI"))

    # Print metrics in machine-parseable format
    print()
    print("METRICS_JSON:", json.dumps({
        "Utility_Gap": round(np.mean(gaps), 5),
        "Utility_Improvement": round(np.mean(improvements), 5),
        "MSE_Reduction": round(np.mean(mse_reductions), 5),
    }))

if __name__ == "__main__":
    parse_results()
