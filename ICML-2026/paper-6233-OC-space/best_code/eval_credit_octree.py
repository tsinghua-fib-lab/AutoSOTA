#!/usr/bin/env python3
"""Reproduce OC-Tree adversarial robustness verification time on Credit dataset.

Reproduces: geometric mean time (ms) across Credit Pareto front models
using OC-Tree index for closest adversarial example search (L-infinity norm).

Paper value: 0.027 ms (Table 6, ICML 2026)
"""

import json, os, math, time, sys, gc
import numpy as np
import pandas as pd
import veritas

os.environ["PRADA_DATA_DIR"] = os.environ.get("PRADA_DATA_DIR", "/datasets/prada")
os.makedirs(os.environ["PRADA_DATA_DIR"], exist_ok=True)

sys.path.insert(0, "/repo/src")
import util
from verification import run_adversarial_robustness_tasks
from depth_first_ocenum import enumerate_ocs_to_disk

COMPRESSED_FILE = "/repo/data/raw/OC-space_paper_compression.txt"
SAVE_DIR = os.environ.get("OC_SPACE_DIR", "/autosota_cache/oc_space")
SEED = 5823
os.makedirs(SAVE_DIR, exist_ok=True)


def load_credit_pareto_models():
    """Load all Credit dataset models on the Pareto front."""
    results = {}
    with open(COMPRESSED_FILE, "r") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            line_dict = json.loads(line.strip())
            if line_dict["dname"] != "Credit":
                continue
            p = line_dict["params"]
            key = (int(p["n_estimators"]), int(p["max_depth"]),
                   float(p["learning_rate"]), int(line_dict["fold"]))
            results[key] = line_dict

    # Find LOP refinement index
    for v in results.values():
        refinements = v["refinements"]
        idx_list = [i for i, r in enumerate(refinements)
                    if r["penalty"] in ["lop", "ours"]]
        if idx_list:
            refinements_idx = idx_list[0]
            break

    # Build Pareto front
    df_data = []
    for depth in [4, 6, 8]:
        for n_est in [10, 25, 50, 100]:
            for lr in [0.1, 0.25, 0.5, 1.0]:
                avg_mtest = 0.0
                avg_nleafs = 0.0
                for fold in range(5):
                    key = (n_est, depth, lr, fold)
                    if key in results:
                        r = results[key]["refinements"][refinements_idx]
                        avg_mtest += r["mtest"] / 5
                        avg_nleafs += r["nleafs"] / 5
                df_data.append([depth, n_est, lr, avg_mtest, avg_nleafs])

    df = pd.DataFrame(df_data, columns=["max_depth", "n_estimators",
                      "learning_rate", "mtest", "nleafs"])
    on_front, _ = util.pareto_front_xy(
        df["nleafs"].to_numpy(), df["mtest"].to_numpy())
    df["on_front"] = on_front

    pareto_configs = []
    for row in df[df["on_front"]].itertuples():
        for fold in range(5):
            pareto_configs.append({
                "fold": fold,
                "learning_rate": float(row.learning_rate),
                "max_depth": int(row.max_depth),
                "n_estimators": int(row.n_estimators),
            })
    return pareto_configs, results, refinements_idx


def main():
    print("=== OC-space Paper 6233: Credit OC-Tree Verification ===")
    print(f"Compressed models: {COMPRESSED_FILE}")
    print(f"Save directory: {SAVE_DIR}")

    pareto_configs, all_results, idx = load_credit_pareto_models()
    print(f"Pareto front models: {len(pareto_configs)}")

    verify_results = []
    total_start = time.time()

    for i, cfg in enumerate(pareto_configs):
        fold = cfg["fold"]
        lr = cfg["learning_rate"]
        md = cfg["max_depth"]
        ne = cfg["n_estimators"]
        key = (ne, md, lr, fold)

        if key not in all_results:
            continue

        line_dict = all_results[key]
        model_json = line_dict["refinements"][idx]["model_json"]
        model = veritas.AddTree.from_json(model_json)

        # Enumerate OC-space
        run_id = time.strftime("%Y%m%d_%H%M%S")
        oc_file = (f"{SAVE_DIR}/OC_enum_Credit_{ne}_{md}_{lr}"
                   f"_fold{fold}_{run_id}.zarr")

        enum_result = enumerate_ocs_to_disk(
            model, buffer_size=1000 * 8194, filename=oc_file, timeout=3600)
        if enum_result.get("failed", False):
            print(f"[{i+1}/{len(pareto_configs)}] ENUM_FAILED")
            continue

        oc_space = enum_result["oc_space"]

        # Load dataset
        d, dtrain, dvalid, dtest = util.get_dataset("Credit", SEED, fold, True)

        # Run OC-Tree verification
        verify = run_adversarial_robustness_tasks(
            model, dtest.X, dtest.y,
            oc_file=oc_file, storage_options=None,
            timeout=3600, n=500)

        octree = verify.get("octree_index", {})
        n_inst = octree.get("emp_rob_n", 0)
        total_t = octree.get("emp_rob_time", 0)
        avg_ms = total_t / n_inst * 1000 if n_inst > 0 else 0.0

        print(f"[{i+1}/{len(pareto_configs)}] "
              f"|O|={oc_space:7d} avg={avg_ms:.4f}ms")

        verify_results.append({
            "fold": fold, "lr": lr, "depth": md, "trees": ne,
            "oc_space": oc_space, "avg_time_ms": avg_ms,
        })
        gc.collect()

    total_time = time.time() - total_start

    # Compute geometric mean
    times = [r["avg_time_ms"] for r in verify_results if r["avg_time_ms"] > 0]
    log_sum = sum(math.log(t) for t in times)
    geo_mean = math.exp(log_sum / len(times))

    print(f"\n=== REPRODUCTION RESULT ===")
    print(f"Models verified: {len(verify_results)}")
    print(f"Geometric mean time: {geo_mean:.6f} ms")
    print(f"Paper value:         0.027 ms")
    print(f"Total wall time:     {total_time:.1f} s")

    # Save results
    output = {
        "paper_id": 6233,
        "dataset": "Credit",
        "metric": "OC-Tree verification time (geometric mean ms)",
        "paper_value_ms": 0.027,
        "reproduced_value_ms": geo_mean,
        "n_models": len(verify_results),
        "n_instances_per_model": 500,
        "norm": "L_infinity",
        "wall_time_s": total_time,
        "per_model": verify_results,
    }
    out_path = f"{SAVE_DIR}/credit_octree_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")

    return geo_mean


if __name__ == "__main__":
    main()
