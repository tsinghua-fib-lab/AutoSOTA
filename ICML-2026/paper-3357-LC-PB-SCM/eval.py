#!/usr/bin/env python3
"""
Evaluation script for paper 3357 (Table 1, Case 2).
Reproduces: On the Identifiability of Poisson Branching Structural Causal Model
Under Latent Confounding (ICML 2026).

Settings: sample_size=10000, n_runs=50, alpha in [0.1, 0.9], mu in [0.02, 0.08]
Metric: F1 score (mark-level compatibility, PPADMG comparison)
"""
import numpy as np
import time
import sys
import json
from util import pbscm, mag2graph
from pgf_confounder_partial import pgf_confounder_partial

# Configuration
SAMPLE_SIZE = 10000
N_RUNS = 50
ALPHA_MIN, ALPHA_MAX = 0.1, 0.9
MU_MIN, MU_MAX = 0.02, 0.08
BOOTSTRAP_ROUND = 200
P_VALUE = 0.05
N_JOBS = 4
SEED = 42

# Case 2 graph structure: latent confounder L between X0 and X1, collider at X2
# Indices: 0=X0, 1=X1, 2=X2 (observed, n=3), 3=L (latent)
CASE2_BASE_GRAPH = np.array([
    [0, 0, 1, 0],    # X0 -> X2
    [0, 0, 1, 0],    # X1 -> X2
    [0, 0, 0, 0],    # X2 (sink)
    [1, 1, 0, 0],    # L -> X0, L -> X1
], dtype=np.float64)

# Ground-truth PPADMG for 3 observed variables:
# X0<->X1 (bidirected due to latent L), X0->X2, X1->X2
# mag encoding: 1=arrowhead, -1=tail, 0=no edge
GROUND_TRUTH_MAG = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [-1, -1, 0],
], dtype=np.int32)


def compute_f1(learned_mag, truth_mag):
    """
    F1 at the mark level with circle compatibility.
    A circle mark (value 2) is compatible with both arrow (1) and tail (-1).
    This matches the PPADMG comparison described in the paper.
    """
    n = truth_mag.shape[0]
    tp = fp = fn = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lm = learned_mag[i][j]   # learned mark
            tm = truth_mag[i][j]      # true mark

            if tm != 0:
                # Truth has a mark
                if lm == tm:
                    tp += 1           # exact match
                elif lm == 2:
                    tp += 1           # circle is compatible with any mark
                elif lm == 0:
                    fn += 1           # missing mark
                else:
                    fp += 1
                    fn += 1           # wrong mark
            else:
                # Truth has no mark
                if lm not in (0, 2):
                    fp += 1           # spurious mark (circles tolerated)

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def main():
    rng = np.random.RandomState(SEED)
    f1_scores = []
    run_times = []

    print("Paper 3357 Evaluation: Case 2 (Table 1)")
    print("Ground truth PPADMG:")
    print(mag2graph(GROUND_TRUTH_MAG))
    print()
    print("Settings: sample_size=%d, n_runs=%d, alpha=[%.1f,%.1f], mu=[%.2f,%.2f]" %
          (SAMPLE_SIZE, N_RUNS, ALPHA_MIN, ALPHA_MAX, MU_MIN, MU_MAX))
    print("bootstrap_round=%d, p_value=%.2f, seed=%d" % (BOOTSTRAP_ROUND, P_VALUE, SEED))
    sys.stdout.flush()

    for run_idx in range(N_RUNS):
        run_seed = SEED + run_idx * 1000
        t0 = time.time()

        # Randomize graph parameters
        graph = CASE2_BASE_GRAPH.copy().astype(np.float64)
        mask = graph > 0
        n_edges = int(mask.sum())
        graph[mask] = rng.uniform(ALPHA_MIN, ALPHA_MAX, size=n_edges)
        mu = rng.uniform(MU_MIN, MU_MAX, size=4).tolist()

        # Generate data from PB-SCM
        data = pbscm(graph=graph, mu=mu, sample=SAMPLE_SIZE, seed=run_seed)
        data_obs = data[:, :3]  # Only observed variables

        # Learn causal structure
        terms, mag = pgf_confounder_partial(
            data_obs,
            bootstrap_round=BOOTSTRAP_ROUND,
            p_value=P_VALUE,
            verbose=False,
            n_jobs=N_JOBS,
            seed=run_seed,
        )

        # Compute F1 score
        f1 = compute_f1(mag, GROUND_TRUTH_MAG)
        f1_scores.append(f1)
        elapsed = time.time() - t0
        run_times.append(elapsed)

        if (run_idx + 1) % 10 == 0 or run_idx == 0:
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0
            print("  [%2d/%d] F1=%.4f  running_mean=%.4f+/-%.4f  avg_t=%.1fs  ETA=%.1fmin" %
                  (run_idx + 1, N_RUNS, f1, mean_f1, std_f1,
                   np.mean(run_times), np.mean(run_times) * (N_RUNS - run_idx - 1) / 60))
            sys.stdout.flush()

    # Final statistics
    f1_scores = np.array(f1_scores)
    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores, ddof=1))

    print()
    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("  F1 Score:  %.4f +/- %.4f" % (mean_f1, std_f1))
    print("  Paper:     0.72 +/- 0.19")
    print("  Total time: %.1f min" % (np.sum(run_times) / 60))
    print("=" * 72)

    # Output machine-parseable result
    result = {
        "metric": "F1",
        "value": mean_f1,
        "std": std_f1,
        "n_runs": N_RUNS,
    }
    print()
    print("RESULT_JSON: " + json.dumps(result))

    # Save detailed results
    with open("eval_results.json", "w") as f:
        json.dump({
            "paper_id": 3357,
            "case": "Case 2",
            "f1_mean": mean_f1,
            "f1_std": std_f1,
            "f1_scores": [float(x) for x in f1_scores],
            "total_time_seconds": float(np.sum(run_times)),
            "n_runs": N_RUNS,
            "sample_size": SAMPLE_SIZE,
        }, f, indent=2)

    return mean_f1


if __name__ == "__main__":
    main()
