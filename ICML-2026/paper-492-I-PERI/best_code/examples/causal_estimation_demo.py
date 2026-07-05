"""Demo script for Causal Estimation with ground-truth plotting

This demo generates synthetic data according to the DGP in `make_data`, computes the
true causal effect (as expected potential outcomes or average marginal effect) and
compares it to estimates returned by various meta-learners and G-computation.

Run the script and it will display a bar chart per (exposure, outcome) combo showing
estimators' CATE estimates and the ground truth.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from pyciphod.causal_estimation.meta_learners import SLearner, TLearner, XLearner
from pyciphod.causal_estimation.outcome_regression import GComputation

# canonical ground-truth effect used across all demo settings
TRUE_ATE = 2.0


def make_data(n=300, exposure_type='binary', outcome_type='binary', seed=1):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    if exposure_type == 'binary':
        logits_t = 0.6 * X1 - 0.3 * X2
        p_t = 1.0 / (1.0 + np.exp(-logits_t))
        t = (rng.rand(n) < p_t).astype(int)
    else:
        t = X1 + 0.1 * rng.normal(size=n)

    if outcome_type == 'continuous':
        y = 1.0 + 2.0 * t + 0.5 * X1 - 0.3 * X2 + 0.5 * rng.normal(size=n)
    else:
        logits_y = -0.5 + 1.2 * t + 0.3 * X1 - 0.2 * X2
        p_y = 1.0 / (1.0 + np.exp(-logits_y))
        y = (rng.rand(n) < p_y).astype(int)

    df = pd.DataFrame({'x1': X1, 'x2': X2, 't': t, 'y': y})
    return df


def _expected_outcome_given_t(df, t_values, outcome_type):
    """Calculate expected outcome (not sampled) given t_values and covariates X1/X2 from df.

    This uses the same structural equations as `make_data` (but without the stochastic noise
    for the binary outcome; we return the conditional expectation). For continuous outcome we
    return the conditional mean (without additive noise term).
    """
    X1 = df['x1'].values
    X2 = df['x2'].values
    if outcome_type == 'continuous':
        # E[y | do(t)] = 1 + 2*t + 0.5*X1 - 0.3*X2
        return 1.0 + 2.0 * t_values + 0.5 * X1 - 0.3 * X2
    else:
        # logistic model: p(y=1 | do(t)) = sigmoid(-0.5 + 1.2*t + 0.3*X1 -0.2*X2)
        logits = -0.5 + 1.2 * t_values + 0.3 * X1 - 0.2 * X2
        return 1.0 / (1.0 + np.exp(-logits))


def compute_ground_truth(df, exposure_type, outcome_type):
    """Compute ground-truth causal effect for the data-generating process used in make_data.

    For binary exposures: ATE = E[Y(do(t=1)) - Y(do(t=0))] where expectation is over covariates.
    For continuous exposures: we return the average marginal effect (d/dt E[Y | do(t)]) computed
    by central finite differences on the structural expectation.
    """
    # Use the structural expectation implemented in _expected_outcome_given_t to compute exact
    # potential outcomes (or numerical derivatives) rather than a hard-coded constant.
    if exposure_type == 'binary':
        # compute E[Y | do(t=1)] and E[Y | do(t=0)] and return their difference (ATE)
        t1 = np.ones(len(df))
        t0 = np.zeros(len(df))
        y1 = _expected_outcome_given_t(df, t1, outcome_type)
        y0 = _expected_outcome_given_t(df, t0, outcome_type)
        ate = np.mean(np.asarray(y1) - np.asarray(y0))
        return float(ate)

    else:  # continuous exposure: compute average marginal effect via central finite differences
        t_vals = df['t'].values
        # choose a small step relative to typical scale; 1e-3 is small but stable
        h = 1e-3
        t_plus = t_vals + h
        t_minus = t_vals - h
        y_plus = _expected_outcome_given_t(df, t_plus, outcome_type)
        y_minus = _expected_outcome_given_t(df, t_minus, outcome_type)
        derivative = (np.asarray(y_plus) - np.asarray(y_minus)) / (2.0 * h)
        # average marginal effect across the sample
        amef = np.mean(derivative)
        return float(amef)


def run_and_plot_combo(exposure_type, outcome_type, show=True):
    # Close any previous matplotlib figures to avoid leftover plots from earlier runs
    plt.close('all')
    print(f"\n--- Demo: exposure={exposure_type}, outcome={outcome_type} ---")
    df = make_data(exposure_type=exposure_type, outcome_type=outcome_type, seed=42)

    # Ground truth computed from the DGP (expected potential outcomes)
    gt = compute_ground_truth(df, exposure_type, outcome_type)
    print(f"Ground truth causal effect (ATE or avg marginal): {gt:.4f}")

    # choose methods applicable to this combo
    if exposure_type == 'binary' and outcome_type == 'binary':
        methods = ['SLearner', 'TLearner', 'XLearner', 'GComputation']
    elif exposure_type == 'continuous' and outcome_type == 'binary':
        methods = ['SLearner', 'GComputation']
    elif exposure_type == 'binary' and outcome_type == 'continuous':
        methods = ['SLearner', 'TLearner', 'XLearner', 'GComputation']
    else:  # continuous / continuous
        methods = ['SLearner', 'GComputation']

    estimates = {}
    if outcome_type == 'continuous':
        learner_model = LinearRegression()
    else:
        learner_model = LogisticRegression()
    for method in methods:
        try:
            if method == 'SLearner':
                learner = SLearner('t', 'y', z=['x1', 'x2'], w=None, model=learner_model, seed=0)
            elif method == 'TLearner':
                learner = TLearner('t', 'y', z=['x1', 'x2'], w=None, model=learner_model, seed=0)
            elif method == 'XLearner':
                learner = XLearner('t', 'y', z=['x1', 'x2'], w=None, model=learner_model, model_tau=None, seed=0)
            elif method == 'GComputation':
                learner = GComputation('t', 'y', z=['x1', 'x2'], w=None, model=None, seed=0)
            else:
                raise RuntimeError(f"Unknown method: {method}")

            res = learner.run(df)
            est = res.get('cate')
            # if multicreate returns dict (multi-exposure), pick the first value
            if isinstance(est, dict):
                # should not happen for this demo (single exposure) but handle defensively
                first_key = next(iter(est.keys()))
                est = float(est[first_key])
            estimates[method] = float(est)
            print(f"{method} estimate: {estimates[method]:.4f}")
        except Exception as e:
            print(f"{method} error: {e}")
            estimates[method] = np.nan

    # Consolidate all estimates into a single flat plot for easy comparison.
    flat_labels = []
    flat_values = []
    # If an estimator returned per-exposure dict, expand as Method (exposure)
    for method, val in estimates.items():
        if isinstance(val, dict):
            for exp_name, v in val.items():
                flat_labels.append(f"{method}:{exp_name}")
                flat_values.append(v)
        else:
            flat_labels.append(method)
            flat_values.append(val)

    # Append ground truth(s) -- if multiple exposures, append one per exposure using label GT:exp
    # If multiple ground truths are available (e.g., derived when we replaced t), try to detect them
    if isinstance(gt, dict):
        for exp_name, v in gt.items():
            flat_labels.append(f"GT:{exp_name}")
            flat_values.append(v)
    else:
        flat_labels.append('GT')
        flat_values.append(gt)

    # Do not plot here: this function only returns estimates and ground truth.
    # Final consolidated plotting is done in main().
    return estimates, gt


def main(show: bool = False):
    combos = [
        ('binary', 'binary'),
        ('continuous', 'binary'),
        ('binary', 'continuous'),
        ('continuous', 'continuous'),
    ]

    # Run all combos but do not show intermediate plots; aggregate results for a single consolidated figure
    results = {}
    for exposure_type, outcome_type in combos:
        est, gt = run_and_plot_combo(exposure_type, outcome_type, show=False)
        results[(exposure_type, outcome_type)] = {'estimates': est, 'ground_truth': gt}

    # Build a single consolidated bar chart comparing all methods across combos
    flat_labels = []
    flat_values = []
    gt_indices = []
    for (exposure_type, outcome_type), data in results.items():
        est = data['estimates']
        gt = data['ground_truth']
        combo_name = f"{exposure_type}/{outcome_type}"
        # iterate methods in a deterministic order
        for method in sorted(est.keys()):
            val = est[method]
            if isinstance(val, dict):
                for exp_name in sorted(val.keys()):
                    v = val[exp_name]
                    flat_labels.append(f"{combo_name}:{method}:{exp_name}")
                    flat_values.append(v)
            else:
                flat_labels.append(f"{combo_name}:{method}")
                flat_values.append(val)
        # append ground-truth
        if isinstance(gt, dict):
            for exp_name in sorted(gt.keys()):
                v = gt[exp_name]
                flat_labels.append(f"{combo_name}:GT:{exp_name}")
                gt_indices.append(len(flat_values))
                flat_values.append(v)
        else:
            flat_labels.append(f"{combo_name}:GT")
            gt_indices.append(len(flat_values))
            flat_values.append(gt)

    # single consolidated plot (only shown if show=True)
    if show:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(max(10, int(0.5 * len(flat_labels))), 6))
        colors = ['C0'] * len(flat_labels)
        for idx in gt_indices:
            if 0 <= idx < len(colors):
                colors[idx] = 'C3'
        bars = ax.bar(flat_labels, flat_values, color=colors)
        ax.set_title('All combos — Estimates vs Ground truth')
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
    return results


if __name__ == '__main__':
    main(show=True)
