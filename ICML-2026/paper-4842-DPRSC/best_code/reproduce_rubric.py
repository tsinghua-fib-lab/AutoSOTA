#!/usr/bin/env python3
"""Targeted reproduction for DPRSC paper (ID 4842) — unbuffered output.

Rubric settings:
  Dataset: Wiki-Squirrel (n=5201, d=1)
  Pattern: triangle, epsilon=2.0, delta=1e-5
  |Q| = ceil(n^1.5) = 375244
  n_runs = 20
"""

import math, time, logging, json, sys, os, random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import preprocessing, ourAlg, baseline

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stderr)
logger = logging.getLogger(__name__)

DATASET, PATTERN = "musae-squirrel", "triangle"
N, D = 5201, 1
EPSILON, DELTA = 2.0, 1e-5
N_RUNS = 20
Q_NUM = math.ceil(N ** 1.5)
OUT = "/repo/reproduction_metrics.json"

def log(msg):
    """Print with flush for reliable progress output."""
    print(msg, flush=True)

def main():
    log("=" * 70)
    log("DPRSC Reproduction — Rubric Metrics (unbuffered)")
    log(f"  Dataset: {DATASET} (n={N}, d={D})  Pattern: {PATTERN}")
    log(f"  epsilon={EPSILON}, delta={DELTA:.0e}  |Q|={Q_NUM}  n_runs={N_RUNS}")
    log("=" * 70)

    # ── Load data ──
    log("\n[1/6] Loading graph data...")
    t0 = time.perf_counter()
    d_max, test_input_nodes, edges, h, m = preprocessing.graph_data_load(
        DATASET, PATTERN, N, logger, D)
    log(f"  Done in {time.perf_counter()-t0:.1f}s: d_max={d_max}, m={m}, "
        f"n_edges={len(edges)}, n_occurrences={len(test_input_nodes)}")

    # ── Generate queries ──
    log(f"\n[2/6] Generating {Q_NUM} queries...")
    t0 = time.perf_counter()
    Q = preprocessing.generate_queries(Q_NUM, m, D)
    log(f"  Done in {time.perf_counter()-t0:.1f}s: {len(Q)} queries")

    # ── Ground truth ──
    log("\n[3/6] Computing ground truth...")
    t0 = time.perf_counter()
    true = ourAlg.query_true(N, m, D, Q, test_input_nodes, logger)
    log(f"  Done in {time.perf_counter()-t0:.1f}s")

    # ── 20 error runs ──
    log(f"\n[4/6] Running ADP_RSC error measurement ({N_RUNS} runs × {Q_NUM} queries)...")
    total_sum_err = 0.0
    total_sum_err2 = 0.0
    run_err_means = []
    phase_t0 = time.perf_counter()
    for run in range(N_RUNS):
        t0 = time.perf_counter()
        sum_err, sum_err2 = ourAlg.approx_DP(
            N, m, D, Q, d_max, EPSILON, DELTA,
            test_input_nodes, PATTERN, logger)
        dt = time.perf_counter() - t0
        total_sum_err += sum_err
        total_sum_err2 += sum_err2
        run_mean = sum_err / Q_NUM
        run_err_means.append(run_mean)
        running_mean = total_sum_err / ((run + 1) * Q_NUM)
        log(f"  Run {run+1:2d}/{N_RUNS}: mean_err={run_mean:.6f}  running_avg={running_mean:.6f}  [{dt:.1f}s]")

    phase_dt = time.perf_counter() - phase_t0
    count = Q_NUM * N_RUNS
    rel_err_mean = total_sum_err / count
    rel_err_std = math.sqrt((total_sum_err2 - total_sum_err * rel_err_mean) / (count - 1))
    run_arr = np.array(run_err_means)
    run_mean = run_arr.mean()
    run_std = run_arr.std(ddof=1)

    log(f"\n  Results after {phase_dt:.1f}s:")
    log(f"    Relative Error    = {rel_err_mean:.8f} ± {rel_err_std:.8f}")
    log(f"    Per-run mean      = {run_mean:.8f} ± {run_std:.8f}")

    # ── Query time ──
    log(f"\n[5/6] Measuring query time (ADP_RSC, {len(Q)} queries)...")
    t0 = time.perf_counter()
    qt_mu, qt_se = ourAlg.approx_DP_qtime(
        N, m, D, Q, d_max, EPSILON, DELTA, test_input_nodes, PATTERN)
    log(f"  Done in {time.perf_counter()-t0:.1f}s")
    qt_sec = qt_mu / 1e6
    log(f"    Query Time = {qt_mu:.3f} µs/query = {qt_sec:.8f} s/query  (SE={qt_se:.3f} µs)")

    # ── Preprocessing time ──
    log(f"\n[6/6] Measuring preprocessing time (ADP_RSC, {N_RUNS} repeats)...")
    t0 = time.perf_counter()
    pr_mu, pr_se = ourAlg.approx_DP_prtime(
        N, m, D, EPSILON, DELTA, edges, h, N_RUNS, PATTERN)
    log(f"  Done in {time.perf_counter()-t0:.1f}s")
    pr_sec = pr_mu * 60.0
    log(f"    Preprocessing Time = {pr_mu:.3f} min = {pr_sec:.1f}s  (SE={pr_se:.6f} min)")

    # ── Total time ──
    total_sec = pr_sec + Q_NUM * qt_sec
    total_min = total_sec / 60.0
    log(f"\n    Total Time = {pr_sec:.1f}s + {Q_NUM}×{qt_sec:.8f}s = {total_sec:.1f}s = {total_min:.3f} min")

    # ── Baselines ──
    log("\n── Baselines ──")
    t0 = time.perf_counter()
    base_qt, _ = baseline.base_comp_qtime(N, D, Q[:20], EPSILON, edges, h, PATTERN)
    log(f"  PDP_Comp Query Time: {base_qt:.3f} µs/query  [{time.perf_counter()-t0:.1f}s]")

    t0 = time.perf_counter()
    log("  Computing PDP_Comp relative error...")
    base_err_mu, base_err_std = baseline.base_comp(
        N, Q_NUM, N_RUNS, EPSILON, true, PATTERN, logger)
    log(f"  PDP_Comp Relative Error: {base_err_mu:.6f} ± {base_err_std:.6f}  [{time.perf_counter()-t0:.1f}s]")

    base_total_sec = Q_NUM * (base_qt / 1e6)
    base_total_min = base_total_sec / 60.0
    log(f"  PDP_Comp Total Time: {base_total_sec:.1f}s = {base_total_min:.3f} min")

    # ── Output ──
    metrics = {
        "relative_error_adp_rsc": {
            "value": round(rel_err_mean, 8),
            "std": round(rel_err_std, 8),
            "per_run_mean": round(run_mean, 8),
            "per_run_std": round(run_std, 8),
            "unit": "ratio",
            "n_runs": N_RUNS,
            "n_queries": Q_NUM,
        },
        "query_time_adp_rsc": {
            "value_us": round(qt_mu, 3),
            "value_sec": round(qt_sec, 8),
            "se_us": round(qt_se, 3),
            "unit": "microseconds_per_query",
        },
        "preprocessing_time_adp_rsc": {
            "value_min": round(pr_mu, 3),
            "value_sec": round(pr_sec, 2),
            "se_min": round(pr_se, 6),
            "unit": "minutes",
            "n_repeats": N_RUNS,
        },
        "total_time_adp_rsc": {
            "value_sec": round(total_sec, 2),
            "value_min": round(total_min, 3),
        },
        "baseline_pdp_comp": {
            "relative_error": round(base_err_mu, 6),
            "relative_error_std": round(base_err_std, 6),
            "query_time_us": round(base_qt, 3),
            "total_time_min": round(base_total_min, 3),
        },
        "parameters": {
            "dataset": DATASET, "n": N, "d": D, "pattern": PATTERN,
            "epsilon": EPSILON, "delta": DELTA,
            "Q_num": Q_NUM, "n_runs": N_RUNS, "seed": SEED,
        },
    }

    log("\n" + "=" * 70)
    log(json.dumps(metrics, indent=2))
    with open(OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"\nMetrics written to {OUT}")

if __name__ == "__main__":
    main()
