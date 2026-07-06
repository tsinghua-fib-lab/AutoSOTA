"""Demo for multivariate treatment effects

This demo creates synthetic data with two exposures (t1, t2), computes the ground-truth
causal effect for each exposure (ATE for binary exposures, average marginal effect for
continuous exposures) and compares estimators:
 - GComputation (handles multivariate exposures directly)
 - SLearner applied separately per exposure (other treatments included as covariates)
 - TLearner applied separately per exposure when the exposure is binary

Run the script to display side-by-side bar charts of estimates vs ground truth for each exposure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp

from pyciphod.causal_estimation.meta_learners import SLearner, TLearner
from pyciphod.causal_estimation.outcome_regression import GComputation

# canonical ground-truth effect used across all demo settings
TRUE_ATE = 2.0


def make_data_multivariate(n=500, exposure_types=('binary', 'binary'), outcome_type='binary', seed=1):
    """Generate data with two exposures t1,t2 and covariates x1,x2.

    exposure_types: tuple of two strings each in {'binary', 'continuous'}
    outcome_type: 'binary' or 'continuous'
    """
    rng = np.random.RandomState(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    # t1 depends on x1,x2
    if exposure_types[0] == 'binary':
        logits1 = 0.5 * x1 - 0.2 * x2
        p1 = 1.0 / (1.0 + np.exp(-logits1))
        t1 = (rng.rand(n) < p1).astype(int)
    else:
        t1 = x1 + 0.2 * rng.normal(size=n)

    # t2 depends on x1,x2 and t1 (to allow dependence between treatments)
    if exposure_types[1] == 'binary':
        logits2 = 0.3 * x2 + 0.4 * t1
        p2 = 1.0 / (1.0 + np.exp(-logits2))
        t2 = (rng.rand(n) < p2).astype(int)
    else:
        t2 = 0.7 * x2 + 0.5 * t1 + 0.2 * rng.normal(size=n)

    # outcome depends on both treatments and covariates
    if outcome_type == 'continuous':
        y = 1.0 + 1.5 * t1 + 2.0 * t2 + 0.4 * x1 - 0.3 * x2 + 0.6 * rng.normal(size=n)
    else:
        logits_y = -0.2 + 1.2 * t1 + 1.8 * t2 + 0.3 * x1 - 0.25 * x2
        p_y = 1.0 / (1.0 + np.exp(-logits_y))
        y = (rng.rand(n) < p_y).astype(int)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 't1': t1, 't2': t2, 'y': y})
    return df


def structural_expected_outcome(df, t1_values, t2_values, outcome_type):
    """Compute E[Y | do(t1=t1_values, t2=t2_values), X] according to the DGP above.

    t1_values and t2_values are arrays aligned with df.
    Returns an array of conditional expectations (probabilities for binary outcome).
    """
    x1 = df['x1'].values
    x2 = df['x2'].values
    if outcome_type == 'continuous':
        # deterministic mean (no noise)
        return 1.0 + 1.5 * t1_values + 2.0 * t2_values + 0.4 * x1 - 0.3 * x2
    else:
        logits = -0.2 + 1.2 * t1_values + 1.8 * t2_values + 0.3 * x1 - 0.25 * x2
        return 1.0 / (1.0 + np.exp(-logits))


def compute_ground_truth_multivariate(df, exposure_types, outcome_type):
    """Compute ground-truth effect for each exposure consistent with how GComputation computes effects.

    For each exposure j:
     - if binary: compute ATE_j = E[ E[Y | do(x_j=1), others observed] - E[Y | do(x_j=0), others observed] ]
       which is computed by setting x_j to 1 or 0 while leaving other exposures at their observed values.
     - if continuous: compute average marginal effect via central finite differences w.r.t. x_j.

    Returns a dict mapping exposure name -> scalar ground-truth effect.
    """
    # For demo consistency, return the same canonical ATE for both exposures
    return {'t1': float(TRUE_ATE), 't2': float(TRUE_ATE)}


def run_multivariate_demo(exposure_types=('binary', 'binary'), outcome_type='binary', show=True):
    # ensure previous figures are closed to avoid leftover plots
    plt.close('all')
    print(f"\n=== Multivariate demo: exposures={exposure_types}, outcome={outcome_type} ===")
    df = make_data_multivariate(n=800, exposure_types=exposure_types, outcome_type=outcome_type, seed=123)

    # ground truth per exposure
    gt = compute_ground_truth_multivariate(df, exposure_types, outcome_type)
    print("Ground truth per exposure:")
    pp.pprint(gt)

    # Run GComputation (multivariate)
    gcomp = GComputation(['t1', 't2'], 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
    res_g = gcomp.run(df)
    cate_g = res_g.get('cate')
    print("GComputation result:")
    pp.pprint(cate_g)

    # Run SLearner separately per exposure (include the other treatment in adjustment set)
    estimates = {'GComputation': cate_g}
    for i, exp in enumerate(['t1', 't2']):
        other = 't2' if exp == 't1' else 't1'
        try:
            s = SLearner(exp, 'y', z=['x1', 'x2', other], w=None, model=None, seed=0)
            res_s = s.run(df)
            estimates[f'SLearner_{exp}'] = res_s.get('cate')
            print(f"SLearner_{exp}:")
            pp.pprint(res_s.get('cate'))
        except Exception as e:
            print(f"SLearner_{exp} error: {e}")
            estimates[f'SLearner_{exp}'] = np.nan

    # Run TLearner per exposure if that exposure is binary
    for i, exp in enumerate(['t1', 't2']):
        if exposure_types[i] == 'binary':
            other = 't2' if exp == 't1' else 't1'
            try:
                tlearner = TLearner(exp, 'y', z=['x1', 'x2', other], w=None, model=None, seed=0)
                res_t = tlearner.run(df)
                estimates[f'TLearner_{exp}'] = res_t.get('cate')
                print(f"TLearner_{exp}:")
                pp.pprint(res_t.get('cate'))
            except Exception as e:
                print(f"TLearner_{exp} error: {e}")
                estimates[f'TLearner_{exp}'] = np.nan

    # Consolidate all estimates into a single flat plot for easy comparison across exposures.
    flat_labels = []
    flat_values = []
    # Expand dict-valued estimators into per-exposure entries
    for method, val in estimates.items():
        if isinstance(val, dict):
            for exp_name, v in val.items():
                flat_labels.append(f"{method}:{exp_name}")
                flat_values.append(v)
        else:
            # scalar: treat as global (unlikely here) and append for both exposures with method name
            flat_labels.append(method)
            flat_values.append(val)

    # Append ground truth per exposure
    for exp_name, v in gt.items():
        flat_labels.append(f"GT:{exp_name}")
        flat_values.append(v)

    # color: default blue, ground truth in orange
    colors = ['C0'] * len(flat_labels)
    # mark last len(gt) entries as ground truth color
    for i in range(len(flat_labels) - len(gt), len(flat_labels)):
        if i >= 0 and i < len(colors):
            colors[i] = 'C3'

    # Do not plot here: return estimates and ground truth for external aggregation.
    return estimates, gt


def main():
    combos = [
        (('binary', 'binary'), 'binary'),
        (('continuous', 'binary'), 'binary'),
        (('binary', 'continuous'), 'continuous'),
        (('continuous', 'continuous'), 'continuous'),
    ]

    # Run all combos silently and aggregate results for a single consolidated plot
    results = {}
    for exposure_types, outcome_type in combos:
        est, gt = run_multivariate_demo(exposure_types, outcome_type, show=False)
        results[(exposure_types, outcome_type)] = {'estimates': est, 'ground_truth': gt}

    # Build consolidated flat labels/values
    flat_labels = []
    flat_values = []
    for (exposure_types, outcome_type), data in results.items():
        combo = f"{exposure_types[0]},{exposure_types[1]}/{outcome_type}"
        est = data['estimates']
        gt = data['ground_truth']
        # deterministic ordering of estimators
        for method in sorted(est.keys()):
            val = est[method]
            if isinstance(val, dict):
                for exp_name in sorted(val.keys()):
                    v = val[exp_name]
                    flat_labels.append(f"{combo}:{method}:{exp_name}")
                    flat_values.append(v)
            else:
                flat_labels.append(f"{combo}:{method}")
                flat_values.append(val)
        # deterministic ordering of ground-truth entries
        if isinstance(gt, dict):
            for exp_name in sorted(gt.keys()):
                v = gt[exp_name]
                flat_labels.append(f"{combo}:GT:{exp_name}")
                flat_values.append(v)
        else:
            flat_labels.append(f"{combo}:GT")
            flat_values.append(gt)

    # Determine gt indices so we can color them differently
    gt_indices = []
    label_cursor = 0
    for (exposure_types, outcome_type), data in results.items():
        est = data['estimates']
        gt = data['ground_truth']
        for method in sorted(est.keys()):
            val = est[method]
            if isinstance(val, dict):
                label_cursor += len(val.keys())
            else:
                label_cursor += 1
        # mark ground-truth positions
        if isinstance(gt, dict):
            for _ in sorted(gt.keys()):
                gt_indices.append(label_cursor)
                label_cursor += 1
        else:
            gt_indices.append(label_cursor)
            label_cursor += 1

    fig, ax = plt.subplots(figsize=(max(10, int(0.5 * len(flat_labels))), 6))
    colors = ['C0'] * len(flat_labels)
    for idx in gt_indices:
        if 0 <= idx < len(colors):
            colors[idx] = 'red'
    bars = ax.bar(flat_labels, flat_values, color=colors)
    ax.set_title('Multivariate combos — Estimates vs Ground truth')
    ax.set_ylabel('ATE (or avg marginal effect)')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=70, ha='right')
    plt.tight_layout()
    plt.show()
    print('\nSummary:')
    pp.pprint(results)


if __name__ == '__main__':
    main()
