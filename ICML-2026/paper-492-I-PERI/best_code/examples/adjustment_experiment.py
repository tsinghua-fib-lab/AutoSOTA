"""
Experiment: compare adjustment sets for estimating effect of X on Y
- Builds a fixed DAG with vertices X,Y,Z,W,U,V and edges:
    X -> Y
    Z -> X, Z -> Y
    W -> X, W -> Y
    U -> X
    V -> Y
- For 100 random linear SCMs (different weights), simulate observational data
  with create_random_linear_scm_from_dag and scm.generate_data.
- Compute the true total effect of X on Y from the SCM (linear case,
  total effect = ((I - B)^{-1})[Y,X] where B is the directed-edges weight matrix
  among observed variables).
- Estimate effect using GComputation, SLearner, TLearner, XLearner with several
  adjustment sets.
- For TLearner and XLearner, binarize X by thresholding at the empirical median
  (they require binary exposure). The true effect for a unit change in X is
  comparable and equals the matrix-derived total effect (per unit).
- Report mean error and variance and plot results.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyciphod.utils import DirectedAcyclicGraph
from pyciphod.utils import create_random_linear_scm_from_dag, LinearSCM
from pyciphod.causal_estimation.outcome_regression import GComputation
# meta-learners imports not needed for the current run (only GComputation used)
from pyciphod.causal_estimation.meta_learners import SLearner, TLearner, XLearner


def build_target_dag():
    dag = DirectedAcyclicGraph()
    vertices = ["X", "Y", "Z", "W", "U", "V"]
    for v in vertices:
        dag.add_vertex(v)
    # Edges as described
    dag.add_directed_edge("X", "Y")
    dag.add_directed_edge("Z", "X")
    dag.add_directed_edge("Z", "Y")
    dag.add_directed_edge("W", "X")
    dag.add_directed_edge("W", "Y")
    dag.add_directed_edge("U1", "X")
    dag.add_directed_edge("U2", "X")
    dag.add_directed_edge("U3", "X")
    dag.add_directed_edge("V1", "Y")
    dag.add_directed_edge("V2", "Y")
    return dag


def compute_true_total_effect(dag, coefficients, source: str, target: str):
    """Compute total causal effect of `source` on `target` for linear SCM
    using the directed-observed adjacency matrix B and formula (I - B)^{-1}.

    coefficients: dict mapping (src, tgt) -> weight (may include latent->obs keys)
    dag: DirectedAcyclicGraph instance (observed vertices)
    """
    vlist = sorted(dag.get_vertices())
    n = len(vlist)
    idx = {v: i for i, v in enumerate(vlist)}
    B = np.zeros((n, n), dtype=float)
    for (s, t), w in coefficients.items():
        # Only include observed->observed directed edges
        if s in idx and t in idx:
            i = idx[t]
            j = idx[s]
            B[i, j] = float(w)
    try:
        M = np.linalg.inv(np.eye(n) - B)
    except np.linalg.LinAlgError:
        # numerical issue: fallback to pseudo-inverse
        M = np.linalg.pinv(np.eye(n) - B)
    # total effect per unit change in source
    return float(M[idx[target], idx[source]])


def run_experiment(n_scms: int = 100, n_samples: int = 1000, seed_base: int = 0):
    dag = build_target_dag()

    adjustment_sets = [
        ("Z", "W"),
        ("Z", "W", "V1", "V2"),
        ("Z", "W", "U1", "U2"),
        ("Z", "W", "U1", "U2", "V1", "V2")
    ]

    # estimators = ["GComp", "SLearner", "TLearner", "XLearner"]
    estimators = ["GComp"]
    # estimators = ["SLearner"]

    results = {est: {adj: [] for adj in adjustment_sets} for est in estimators}

    for i in range(n_scms):
        print(f"Running SCM {i+1}/{n_scms} (seed {seed_base + i})")
        seed = seed_base + i
        try:
            scm, coeffs, intercepts = create_random_linear_scm_from_dag(dag, seed=seed)
        except Exception as e:
            print(f"Failed to create SCM for seed {seed}: {e}")
            continue
        # ensure X has a direct effect on Y for identifiability: update coefficients
        coeffs[("X", "Y")] = 1.0
        # Rebuild the SCM so data generation matches the (possibly modified) coefficients
        try:
            scm = LinearSCM(v=list(scm.v), u=list(scm.u), coefficients=coeffs, intercepts=intercepts, u_dist=None)
        except Exception:
            # If rebuild fails, fall back to original scm and proceed (we keep coeffs for truth computation)
            pass

        # true effect per unit (X on Y)
        try:
            true_effect = compute_true_total_effect(dag, coeffs, "X", "Y")
        except Exception as e:
            print(f"Failed to compute true effect for seed {seed}: {e}")
            continue

        # simulate observational data
        try:
            df = scm.generate_data(n_samples=n_samples, include_latent=False, seed=seed)
        except Exception as e:
            print(f"Data generation failed seed {seed}: {e}")
            continue

        # ensure required columns
        for col in ["X", "Y", "Z", "W", "U", "V"]:
            if col not in df.columns:
                print(f"Column {col} missing in generated data for seed {seed}")
                continue

        # run estimators for each adjustment set
        for adj in adjustment_sets:
            adj_list = list(adj)

            # GComputation (continuous exposure supported)
            try:
                gcomp = GComputation("X", "Y", z=adj_list, w=None, model=None, seed=seed)
                res = gcomp.run(df)
                est = float(res.get("cate"))
                err = est - true_effect
                results["GComp"][adj].append(err)
            except Exception as e:
                # append NaN to keep alignment
                results["GComp"][adj].append(np.nan)

            # # SLearner (works for continuous exposures)
            # try:
            #     slearner = SLearner("X", "Y", z=adj_list, w=None, model=None, seed=seed)
            #     res = slearner.run(df)
            #     est = float(res.get("cate"))
            #     err = est - true_effect
            #     results["SLearner"][adj].append(err)
            # except Exception as e:
            #     results["SLearner"][adj].append(np.nan)

    return results


def summarize_and_plot(results, title: str = "Adjustment experiment"):
    # compute RMSE and variance for each estimator x adjustment set
    adjustment_sets = list(next(iter(results.values())).keys())
    estimators = list(results.keys())

    # We'll compute:
    # - var_err: sample variance of errors across SCMs
    # - rmse: sqrt(mean(err**2))
    # - var_sq: sample variance of squared errors (used to estimate stderr of RMSE)
    var_err = np.zeros((len(estimators), len(adjustment_sets)))
    rmse = np.zeros_like(var_err)
    var_sq = np.zeros_like(var_err)
    counts = np.zeros_like(var_err)

    for i, est in enumerate(estimators):
        for j, adj in enumerate(adjustment_sets):
            arr = np.asarray(results[est][adj], dtype=float)
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                var_err[i, j] = np.nan
                rmse[i, j] = np.nan
                var_sq[i, j] = np.nan
                counts[i, j] = 0
            else:
                # variance of raw errors
                var_err[i, j] = float(np.nanvar(valid, ddof=1)) if valid.size > 1 else 0.0
                sq = valid ** 2
                mean_sq = float(np.nanmean(sq))
                rmse[i, j] = float(np.sqrt(mean_sq))
                var_sq[i, j] = float(np.nanvar(sq, ddof=1)) if valid.size > 1 else 0.0
                counts[i, j] = valid.size

    # Estimate standard error for RMSE using the delta method:
    # RMSE = sqrt(m), m = mean(sq); Var(RMSE) ≈ Var(m) / (4 * m) where Var(m) = var_sq / N
    with np.errstate(divide='ignore', invalid='ignore'):
        stderr_rmse = np.full_like(rmse, np.nan)
        mask = (counts > 0) & (rmse > 0)
        stderr_rmse[mask] = np.sqrt(var_sq[mask] / counts[mask]) / (2.0 * rmse[mask])

    # Pretty print table with RMSE, Var(error), StdErr(RMSE)
    header = "Estimator\tAdjustment\tN\tRMSE\tVar(Error)\tStdErr(RMSE)"
    print(header)
    for i, est in enumerate(estimators):
        for j, adj in enumerate(adjustment_sets):
            r = rmse[i, j]
            ve = var_err[i, j]
            se_r = stderr_rmse[i, j]
            n = int(counts[i, j])
            print(f"{est}\t{adj}\t{n}\t{r:.6f}\t{ve:.6f}\t{(se_r if not np.isnan(se_r) else float('nan')):.6f}")

    # Plot: two stacked plots (RMSE on top, Variance of error below)
    x = np.arange(len(adjustment_sets))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: RMSE with stderr bars
    for i, est in enumerate(estimators):
        offsets = (i - (len(estimators)-1)/2.0) * width
        y = rmse[i]
        yerr = stderr_rmse[i]
        ax1.bar(x + offsets, y, width=width, label=est, yerr=yerr, capsize=5)
    ax1.set_ylabel("RMSE (root mean squared error)")
    ax1.set_title(title)
    ax1.legend()

    # Bottom: Variance of error (sample variance across SCMs)
    for i, est in enumerate(estimators):
        offsets = (i - (len(estimators)-1)/2.0) * width
        y = var_err[i]
        ax2.bar(x + offsets, y, width=width, label=est)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["+".join(adj) for adj in adjustment_sets])
    ax2.set_ylabel("Variance of error")
    ax2.set_xlabel("Adjustment set")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Run experiment
    res = run_experiment(n_scms=100, n_samples=500, seed_base=0)
    summarize_and_plot(res, title="Adjustment sets: mean error and variance (100 SCMs)")
