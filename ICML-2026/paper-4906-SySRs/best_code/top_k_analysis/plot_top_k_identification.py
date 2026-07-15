"""
Plot Top-k Identification: error-rate curve (left) + CB quantile curve (right).

Left panel  (a): Mean error rate (100 - metric) vs budget %.
Right panel (b): Minimum budget % to reach each target rate (CB quantile).

Metrics (both computed from tracked empirical means, using tie correction):
  top_m_identification_rate  – binary: predicted top-m SET matches true top-m
  top_m_ranking_rate         – binary: predicted top-m ORDER matches true order
  per_rank_accuracy          – continuous: fraction of k positions correctly ranked
  per_rank_accuracy_elim     – same, but using elimination ranks as predictor

Usage
-----
  # Mean figure across all datasets (default metric: identification rate)
  python plot_top_k_identification.py

  # Ranking rate metric
  python plot_top_k_identification.py --metric top_m_ranking_rate

  # Per-rank accuracy metric
  python plot_top_k_identification.py --metric per_rank_accuracy

  # Single dataset, top-3
  python plot_top_k_identification.py --datasets commonsense --k 3

  # Custom axes
  python plot_top_k_identification.py --max-pct 35 --min-target 60 --max-target 100
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS = [
    'arc_challenge', 'bbh', 'commonsense', 'gpqa', 'gsm', 'ifeval',
    'legalbench', 'math', 'med_qa', 'mmlu', 'mmlu_pro',
    'musr', 'narrative_qa', 'natural_qa', 'wmt_14',
]

DATASET_DISPLAY = {
    'arc_challenge': 'ARC Challenge',
    'bbh':           'BIG-Bench Hard',
    'commonsense':   'Commonsense',
    'gpqa':          'GPQA',
    'gsm':           'GSM',
    'ifeval':        'IFEval',
    'legalbench':    'LegalBench',
    'math':          'MATH',
    'med_qa':        'MedQA',
    'mmlu':          'MMLU',
    'mmlu_pro':      'MMLU-Pro',
    'musr':          'MuSR',
    'narrative_qa':  'NarrativeQA',
    'natural_qa':    'NaturalQA',
    'wmt_14':        'WMT-14',
}

# Raw algorithm name (from pkl) → display name
NAME_MAPPING = {
    'Naive-Baseline':      'SyUS',
    'Uniform-Pulls':       'US',
    'SR':                  'SR',
    'Smart-SR':            'SySRs (Ours)',
    'UCB-E (a=0.1)':       'UCB-E (a=0.1)',
    'UCB-E (a=1.0)':       'UCB-E (a=1.0)',
    'UCB-E (a=10.0)':      'UCB-E (a=10.0)',
    'UCB-E (a=100.0)':     'UCB-E (a=100.0)',
    'SyUCB-E (a=0.1)':     'SyUCB-E (a=0.1) (Ours)',
    'SyUCB-E (a=1.0)':     'SyUCB-E (a=1.0) (Ours)',
    'SyUCB-E (a=10.0)':    'SyUCB-E (a=10.0) (Ours)',
    'SyUCB-E (a=100.0)':   'SyUCB-E (a=100.0) (Ours)',
}

DEFAULT_ALGOS = [
    'SySRs (Ours)', 'SR',
    'SyUCB-E (a=0.1) (Ours)', 'SyUCB-E (a=1.0) (Ours)', 'SyUCB-E (a=10.0) (Ours)',
    'UCB-E (a=0.1)', 'UCB-E (a=1.0)', 'UCB-E (a=10.0)',
    'SyUS', 'US',
]

LEGEND_ORDER = [
    'SySRs (Ours)', 'SR',
    'SyUCB-E (a=0.1) (Ours)', 'SyUCB-E (a=1.0) (Ours)', 'SyUCB-E (a=10.0) (Ours)',
    'SyUCB-E (a=100.0) (Ours)',
    'UCB-E (a=0.1)', 'UCB-E (a=1.0)', 'UCB-E (a=10.0)', 'UCB-E (a=100.0)',
    'SyUS', 'US',
]

ALGORITHM_STYLES = {
    'SySRs (Ours)': {
        'color': '#DC143C', 'marker': '*', 'markersize': 8,
        'linestyle': '-', 'linewidth': 3.5, 'alpha': 1.0, 'zorder': 10,
    },
    'SR': {
        'color': '#FF8C00', 'marker': 's', 'markersize': 6,
        'linestyle': '--', 'linewidth': 2.0, 'alpha': 0.7, 'zorder': 9,
    },
    'SyUCB-E (a=0.1) (Ours)': {
        'color': '#90EE90', 'marker': '^', 'markersize': 6,
        'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 8,
    },
    'SyUCB-E (a=1.0) (Ours)': {
        'color': '#32CD32', 'marker': 'v', 'markersize': 6,
        'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 8,
    },
    'SyUCB-E (a=10.0) (Ours)': {
        'color': '#228B22', 'marker': '<', 'markersize': 6,
        'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 8,
    },
    'SyUCB-E (a=100.0) (Ours)': {
        'color': '#006400', 'marker': '>', 'markersize': 6,
        'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 8,
    },
    'UCB-E (a=0.1)': {
        'color': '#87CEEB', 'marker': 'D', 'markersize': 6,
        'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 7,
    },
    'UCB-E (a=1.0)': {
        'color': '#1E90FF', 'marker': 'p', 'markersize': 6,
        'linestyle': '-.', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 7,
    },
    'UCB-E (a=10.0)': {
        'color': '#4169E1', 'marker': 'h', 'markersize': 6,
        'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 7,
    },
    'UCB-E (a=100.0)': {
        'color': '#00008B', 'marker': 'H', 'markersize': 6,
        'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 7,
    },
    'SyUS': {
        'color': '#696969', 'marker': 'x', 'markersize': 6,
        'linestyle': ':', 'linewidth': 2.0, 'alpha': 0.4, 'zorder': 1,
    },
    'US': {
        'color': '#A9A9A9', 'marker': '+', 'markersize': 6,
        'linestyle': ':', 'linewidth': 2.0, 'alpha': 0.4, 'zorder': 1,
    },
}

# Marker positions matching original paper style
MARKER_BUDGETS = list(range(5, 101, 5))


# ---------------------------------------------------------------------------
# Metric computation from empirical means (matches run script logic)
# ---------------------------------------------------------------------------

def _metric_value(predictors: np.ndarray, true_ranking, true_accuracies, k_top: int,
                  k_runs: int, metric: str) -> float:
    """Compute one metric value (%) for a single (algo, budget, k_top).

    Parameters
    ----------
    predictors     : (n_models, k_runs) — higher value → better predicted arm.
                     Pass empirical means for standard metrics, or elimination
                     ranks (higher = later eliminated = better) for *_elim.
    true_ranking   : (n_models,) sorted model indices, best first
    true_accuracies: (n_models,) mean accuracy per model
    k_top          : m for top-m
    k_runs         : number of runs in predictors
    metric         : 'top_m_identification_rate' | 'top_m_ranking_rate' | 'per_rank_accuracy'
                     or the *_elim variants of the first two (suffix stripped for comparison)

    Returns
    -------
    float in [0, 100] — average over runs (as percentage)
    """
    base_metric    = metric.replace('_elim', '')
    true_kth_acc   = true_accuracies[true_ranking[k_top - 1]]
    n_must_true    = int(np.sum(true_accuracies > true_kth_acc + 1e-9))
    n_tie_slots    = k_top - n_must_true
    true_topk_accs = true_accuracies[true_ranking[:k_top]]   # (k_top,)

    rates = []
    for run in range(k_runs):
        predicted_topk = np.argsort(-predictors[:, run])[:k_top]
        pred_accs     = true_accuracies[predicted_topk]

        if base_metric == 'top_m_identification_rate':
            n_pred_must = int(np.sum(pred_accs > true_kth_acc + 1e-9))
            n_pred_tie  = int(np.sum(np.abs(pred_accs - true_kth_acc) < 1e-9))
            n_correct   = n_pred_must + min(n_pred_tie, n_tie_slots)
            rates.append(100.0 if n_correct == k_top else 0.0)
        elif base_metric == 'per_rank_accuracy':
            exact_count = int(np.sum(np.abs(pred_accs - true_topk_accs) < 1e-9))
            rates.append(exact_count / k_top * 100.0)
        else:   # top_m_ranking_rate
            exact_count = int(np.sum(np.abs(pred_accs - true_topk_accs) < 1e-9))
            rates.append(100.0 if exact_count == k_top else 0.0)

    return float(np.mean(rates))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(data_dir: Path, k_values=None, metric='top_m_identification_rate'):
    """Load per-dataset pkl files and return:

        data[dataset][algo][k_top] -> sorted list of (budget_pct, value%)

    Metrics are recomputed from raw empirical_means (or elim_ranks for *_elim
    metrics) stored in pkl files.
    """
    if k_values is None:
        k_values = [3, 5, 10]
    k_set = set(k_values)

    is_elim = metric.endswith('_elim')

    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for fp in sorted(data_dir.glob("*_top_k_identification_results.pkl")):
        try:
            with fp.open('rb') as fh:
                obj = pickle.load(fh)
        except Exception as exc:
            print(f"Warning: could not load {fp.name}: {exc}", file=sys.stderr)
            continue

        dataset       = obj.get('dataset')
        budget_fracs  = obj.get('budget_fracs', [])
        k_runs        = int(obj.get('k_runs', 100))
        true_ranking  = np.asarray(obj['true_ranking'])
        true_accuracies = np.asarray(obj['true_accuracies'])
        em_by_frac    = obj.get('empirical_means', {})
        elim_by_frac  = obj.get('elim_ranks', {})

        if dataset is None or not budget_fracs:
            print(f"Warning: {fp.name} missing dataset/budget_fracs; skipping.",
                  file=sys.stderr)
            continue

        n_models = true_ranking.shape[0]

        for frac in budget_fracs:
            pct = int(round(frac * 100))
            em_by_algo = em_by_frac.get(frac, {})
            if not em_by_algo:
                # fallback: budget_fracs stored as strings (unlikely but safe)
                em_by_algo = em_by_frac.get(str(frac), {})
            elim_by_algo = elim_by_frac.get(frac, {}) or elim_by_frac.get(str(frac), {})

            for algo_raw, em_raw in em_by_algo.items():
                algo = NAME_MAPPING.get(algo_raw, algo_raw)
                em   = np.asarray(em_raw)
                if em.ndim != 2 or em.shape[0] != n_models:
                    continue

                if is_elim:
                    elim_raw = elim_by_algo.get(algo_raw)
                    if elim_raw is not None:
                        predictors = np.asarray(elim_raw, dtype=float)
                    else:
                        predictors = em  # fallback: use empirical means for non-SR
                else:
                    predictors = em

                for k_top in k_values:
                    if k_top > n_models:
                        continue
                    val = _metric_value(predictors, true_ranking, true_accuracies,
                                        k_top, k_runs, metric)
                    data[dataset][algo][k_top].append((pct, val))

    # Sort each curve by budget percentage
    for ds in data:
        for algo in data[ds]:
            for k in data[ds][algo]:
                data[ds][algo][k].sort(key=lambda x: x[0])

    return data


# ---------------------------------------------------------------------------
# CB helpers (right panel)
# ---------------------------------------------------------------------------

def find_cb(curve, target):
    """Minimum budget_pct where metric >= target AND stays there. Cap at 100."""
    for i, (bpct, val) in enumerate(curve):
        if val >= target:
            if all(v >= target for _, v in curve[i:]):
                return float(bpct)
    return 100.0


def compute_cb_curve(curve, target_range):
    return [(t, find_cb(curve, t)) for t in target_range]


# ---------------------------------------------------------------------------
# Build plot-ready curves
# ---------------------------------------------------------------------------

def build_error_curves(data, dataset, k, algos, max_pct):
    """Return dict algo -> (bpcts, errors) for left panel."""
    curves = {}
    for algo in algos:
        pts = data.get(dataset, {}).get(algo, {}).get(k, [])
        if not pts:
            continue
        bpcts   = np.array([p[0] for p in pts if p[0] <= max_pct])
        vals    = np.array([p[1] for p in pts if p[0] <= max_pct])
        if len(bpcts) == 0:
            continue
        curves[algo] = (bpcts, np.clip(100.0 - vals, 0.0, 100.0))
    return curves


def build_cb_curves(data, dataset, k, algos, target_range):
    """Return dict algo -> (targets, cbs) for right panel, single dataset."""
    curves = {}
    for algo in algos:
        pts = data.get(dataset, {}).get(algo, {}).get(k, [])
        if not pts:
            continue
        cb_vals = compute_cb_curve(pts, target_range)
        curves[algo] = (
            np.array([t for t, _ in cb_vals]),
            np.array([c for _, c in cb_vals]),
        )
    return curves


def build_mean_error_curves(data, k, algos, max_pct, datasets=None):
    """Mean error rate across datasets, interpolated to MARKER_BUDGETS grid."""
    if datasets is None:
        datasets = list(data.keys())
    grid = np.array([b for b in MARKER_BUDGETS if b <= max_pct])
    algo_all: dict = defaultdict(list)
    for ds in datasets:
        for algo in algos:
            pts = data.get(ds, {}).get(algo, {}).get(k, [])
            if not pts:
                continue
            bpcts   = np.array([p[0] for p in pts])
            errors  = np.clip(100.0 - np.array([p[1] for p in pts]), 0.0, 100.0)
            interp  = np.interp(grid, bpcts, errors, left=errors[0], right=errors[-1])
            algo_all[algo].append(interp)
    return {algo: (grid, np.mean(arrs, axis=0)) for algo, arrs in algo_all.items()}


def build_worst_case_cb_curves(data, k, algos, target_range, datasets=None):
    """Worst-case (max over datasets) CB for each target rate."""
    if datasets is None:
        datasets = list(data.keys())
    algo_cbs: dict = defaultdict(lambda: defaultdict(list))
    for ds in datasets:
        for algo in algos:
            pts = data.get(ds, {}).get(algo, {}).get(k, [])
            if not pts:
                continue
            for t in target_range:
                algo_cbs[algo][t].append(find_cb(pts, t))
    worst = {}
    for algo, t_dict in algo_cbs.items():
        targets = np.array(sorted(t_dict.keys()))
        wc = np.array([max(t_dict[t]) if t_dict[t] else 100.0 for t in targets])
        worst[algo] = (targets, wc)
    return worst


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _get_style(algo):
    return ALGORITHM_STYLES.get(algo, {
        'color': 'gray', 'marker': 'o', 'markersize': 5,
        'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7, 'zorder': 5,
    })


def _sort_algos(algos):
    return sorted(algos, key=lambda a: LEGEND_ORDER.index(a) if a in LEGEND_ORDER else 99)


def plot_combined_figure(
    title_left, title_right,
    error_curves,   # algo -> (bpcts, errors)
    cb_curves,      # algo -> (targets, cbs)
    out_path: Path,
    max_pct=100,
    max_y_pct=100,
    min_target=90,
    max_target=100,
    min_budget_y=0,
    max_budget_y=105,
    k_value=3,
    left_ylabel=None,
    right_xlabel=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), dpi=150)

    visible   = set(ALGORITHM_STYLES) | set(DEFAULT_ALGOS)
    left_algs = _sort_algos([a for a in error_curves if a in visible])
    rght_algs = _sort_algos([a for a in cb_curves    if a in visible])
    handles   = {}
    ticks     = [b for b in MARKER_BUDGETS if b <= max_pct]

    # ---- Left panel ----
    for algo in left_algs:
        if algo not in error_curves:
            continue
        bpcts, errors = error_curves[algo]
        st = _get_style(algo)
        mk_idx = [int(np.argmin(np.abs(bpcts - t))) for t in ticks
                  if bpcts.min() <= t <= bpcts.max()]
        line, = ax1.plot(bpcts, errors, label=algo,
                         color=st['color'], linestyle=st['linestyle'],
                         marker=st['marker'], markersize=st['markersize'],
                         markevery=mk_idx or None,
                         linewidth=st['linewidth'], alpha=st['alpha'],
                         zorder=st['zorder'])
        handles[algo] = line

    ax1.set_xlabel('Percentage of Full Evaluation Budget', fontsize=18, fontweight='bold')
    ax1.set_ylabel(
        left_ylabel or f'Top-${k_value}$ Identification Error Rate (%)',
        fontsize=16, fontweight='bold')
    ax1.set_title(title_left, fontsize=18, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(0, max_pct)
    ax1.set_ylim(0, max_y_pct)
    ax1.tick_params(axis='both', labelsize=14)

    # ---- Right panel ----
    for algo in rght_algs:
        if algo not in cb_curves:
            continue
        targets, cbs = cb_curves[algo]
        mask = (targets >= min_target) & (targets <= max_target)
        if not mask.any():
            continue
        st = _get_style(algo)
        line, = ax2.plot(targets[mask], cbs[mask], label=algo,
                         color=st['color'], linestyle=st['linestyle'],
                         marker=st['marker'], markersize=st['markersize'],
                         markevery=1,
                         linewidth=st['linewidth'], alpha=st['alpha'],
                         zorder=st['zorder'])
        handles[algo] = line

    ax2.set_xlabel(
        right_xlabel or f'Target Top-${k_value}$ Identification Rate (%)',
        fontsize=18, fontweight='bold')
    ax2.set_ylabel('Budget Percentage Required (%)', fontsize=18, fontweight='bold')
    ax2.set_title(title_right, fontsize=18, fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim(min_target, max_target)
    ax2.set_ylim(min_budget_y, max_budget_y)
    ax2.tick_params(axis='both', labelsize=14)

    # ---- Shared legend ----
    ord_h, ord_l = [], []
    for algo in LEGEND_ORDER:
        if algo in handles:
            ord_h.append(handles[algo])
            ord_l.append(algo)
    for algo, h in handles.items():
        if algo not in LEGEND_ORDER:
            ord_h.append(h)
            ord_l.append(algo)

    fig.legend(ord_h, ord_l,
               loc='lower center', ncol=min(6, len(ord_h)),
               fontsize=12, bbox_to_anchor=(0.5, -0.12),
               frameon=True, fancybox=True, shadow=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all_k_combined_figure(
    k_values,
    mean_error_dict,    # k -> algo -> (bpcts, errors)
    worst_cb_dict,      # k -> algo -> (targets, cbs)
    out_path: Path,
    max_pct=100, max_y_pct=100,
    min_target=90, max_target=100,
    min_budget_y=0, max_budget_y=105,
    metric='top_m_identification_rate',
    left_title_tmpl=None,
    right_title_tmpl=None,
):
    n   = len(k_values)
    fig, axes = plt.subplots(n, 2, figsize=(20, 7 * n), dpi=150)
    if n == 1:
        axes = [axes]

    handles  = {}
    ticks    = [b for b in MARKER_BUDGETS if b <= max_pct]
    _base_metric = metric.replace('_elim', '')
    _elim_label  = ''

    for row, k in enumerate(k_values):
        ax1, ax2   = axes[row]
        err_curves = mean_error_dict.get(k, {})
        cb_curves  = worst_cb_dict.get(k, {})
        visible    = set(ALGORITHM_STYLES) | set(DEFAULT_ALGOS)
        left_algs  = _sort_algos([a for a in err_curves if a in visible])
        rght_algs  = _sort_algos([a for a in cb_curves  if a in visible])

        for algo in left_algs:
            if algo not in err_curves:
                continue
            bpcts, errors = err_curves[algo]
            st = _get_style(algo)
            mk_idx = [int(np.argmin(np.abs(bpcts - t))) for t in ticks
                      if bpcts.min() <= t <= bpcts.max()]
            line, = ax1.plot(bpcts, errors, label=algo,
                             color=st['color'], linestyle=st['linestyle'],
                             marker=st['marker'], markersize=st['markersize'],
                             markevery=mk_idx or None,
                             linewidth=st['linewidth'], alpha=st['alpha'],
                             zorder=st['zorder'])
            handles[algo] = line

        if _base_metric == 'top_m_ranking_rate':
            left_ylabel  = f'Top-${k}$ Ranking Rate{_elim_label} Error (%)'
        elif _base_metric == 'per_rank_accuracy':
            left_ylabel  = f'Top-${k}$ Per-Rank Accuracy Error (%)'
        else:
            left_ylabel  = f'Top-${k}$ Identification Rate{_elim_label} Error (%)'

        _lt = (left_title_tmpl or 'Mean Performance — Top-{k}').format(k=k)
        ax1.set_xlabel('Percentage of Full Evaluation Budget', fontsize=14, fontweight='bold')
        ax1.set_ylabel(left_ylabel, fontsize=13, fontweight='bold')
        ax1.set_title(_lt, fontsize=14, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.set_xlim(0, max_pct)
        ax1.set_ylim(0, max_y_pct)
        ax1.tick_params(axis='both', labelsize=12)

        for algo in rght_algs:
            if algo not in cb_curves:
                continue
            targets, cbs = cb_curves[algo]
            mask = (targets >= min_target) & (targets <= max_target)
            if not mask.any():
                continue
            st = _get_style(algo)
            line, = ax2.plot(targets[mask], cbs[mask], label=algo,
                             color=st['color'], linestyle=st['linestyle'],
                             marker=st['marker'], markersize=st['markersize'],
                             markevery=1,
                             linewidth=st['linewidth'], alpha=st['alpha'],
                             zorder=st['zorder'])
            handles[algo] = line

        if _base_metric == 'top_m_ranking_rate':
            right_xlabel = f'Target Top-${k}$ Ranking Rate{_elim_label} (%)'
        elif _base_metric == 'per_rank_accuracy':
            right_xlabel = f'Target Top-${k}$ Per-Rank Accuracy (%)'
        else:
            right_xlabel = f'Target Top-${k}$ Identification Rate{_elim_label} (%)'

        _rt = (right_title_tmpl or 'Budget Requirements — Top-{k}').format(k=k)
        ax2.set_xlabel(right_xlabel, fontsize=14, fontweight='bold')
        ax2.set_ylabel('Budget Percentage Required (%)', fontsize=14, fontweight='bold')
        ax2.set_title(_rt, fontsize=14, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.set_xlim(min_target, max_target)
        ax2.set_ylim(min_budget_y, max_budget_y)
        ax2.tick_params(axis='both', labelsize=12)

    # Shared legend
    ord_h, ord_l = [], []
    for algo in LEGEND_ORDER:
        if algo in handles:
            ord_h.append(handles[algo])
            ord_l.append(algo)
    for algo, h in handles.items():
        if algo not in LEGEND_ORDER:
            ord_h.append(h)
            ord_l.append(algo)

    fig.legend(ord_h, ord_l,
               loc='lower center', ncol=min(6, len(ord_h)),
               fontsize=12, bbox_to_anchor=(0.5, -0.02),
               frameon=True, fancybox=True, shadow=False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Plot top-k identification combined figure (codebase version).')
    p.add_argument(
        '--data-dir', type=Path,
        default=Path(__file__).parent / 'results' / 'top_k_identification',
        help='Directory containing *_top_k_identification_results.pkl files.',
    )
    p.add_argument(
        '--out-dir', type=Path, default=None,
        help='Output directory (default: --data-dir/plots/top_k_{metric}).',
    )
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Datasets to process (default: all 15).')
    p.add_argument('--k', type=int, nargs='+', default=[3, 5, 10],
                   help='Top-k values to plot (default: 3 5 10).')
    p.add_argument('--max-pct',    type=float, default=100,
                   help='Max budget %% for left panel x-axis (default: 100).')
    p.add_argument('--max-y-pct',  type=float, default=100,
                   help='Max error rate %% for left panel y-axis (default: 100).')
    p.add_argument('--min-target', type=float, default=90,
                   help='Min target rate %% for right panel x-axis (default: 90).')
    p.add_argument('--max-target', type=float, default=100,
                   help='Max target rate %% for right panel x-axis (default: 100).')
    p.add_argument('--min-budget-y', type=float, default=0,
                   help='Min y for right panel (default: 0).')
    p.add_argument('--max-budget-y', type=float, default=105,
                   help='Max y for right panel (default: 105).')
    p.add_argument(
        '--metric',
        choices=[
            'top_m_identification_rate',
            'top_m_ranking_rate',
            'per_rank_accuracy',
            'top_m_identification_rate_elim',
            'top_m_ranking_rate_elim',
            'per_rank_accuracy_elim',
        ],
        default='top_m_identification_rate',
        help='Metric to plot (default: top_m_identification_rate).',
    )
    p.add_argument('--no-mean',       action='store_true',
                   help='Skip mean-across-datasets figure.')
    p.add_argument('--no-individual', action='store_true',
                   help='Skip per-dataset figures.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    datasets = args.datasets or DATASETS
    metric   = args.metric

    # Metric-dependent labels / subdirectories
    is_elim     = metric.endswith('_elim')
    base_metric = metric.replace('_elim', '')
    elim_sfx    = '_elim' if is_elim else ''
    elim_label  = ''

    if base_metric == 'top_m_ranking_rate':
        out_subdir         = f'top_k_combined_ranking_rate{elim_sfx}'
        left_ylabel_tmpl   = f'Top-${{k}}$ Ranking Rate{elim_label} Error (%)'
        right_xlabel_tmpl  = f'Target Top-${{k}}$ Ranking Rate{elim_label} (%)'
        mean_left_tmpl     = f'(a) Mean Ranking Rate{elim_label} Error — Top-{{k}}'
        mean_right_tmpl    = f'(b) Worst-case Budget Requirements (Ranking Rate{elim_label}) — Top-{{k}}'
        ds_left_tmpl       = f'(a) Ranking Rate{elim_label} Error on {{dname}} — Top-{{k}}'
        ds_right_tmpl      = f'(b) Budget Requirements (Ranking Rate{elim_label}) on {{dname}} — Top-{{k}}'
        fname_prefix       = f'top{{k}}_ranking_rate{elim_sfx}_combined'
        all_k_fname        = f'all_k_ranking_rate{elim_sfx}_combined.png'
    elif base_metric == 'per_rank_accuracy':
        out_subdir         = f'top_k_combined_per_rank_accuracy{elim_sfx}'
        left_ylabel_tmpl   = f'Top-${{k}}$ Per-Rank Accuracy{elim_label} Error (%)'
        right_xlabel_tmpl  = f'Target Top-${{k}}$ Per-Rank Accuracy{elim_label} (%)'
        mean_left_tmpl     = f'(a) Mean Per-Rank Accuracy{elim_label} Error — Top-{{k}}'
        mean_right_tmpl    = f'(b) Worst-case Budget Requirements (Per-Rank Accuracy{elim_label}) — Top-{{k}}'
        ds_left_tmpl       = f'(a) Per-Rank Accuracy{elim_label} Error on {{dname}} — Top-{{k}}'
        ds_right_tmpl      = f'(b) Budget Requirements (Per-Rank Accuracy{elim_label}) on {{dname}} — Top-{{k}}'
        fname_prefix       = f'top{{k}}_per_rank_accuracy{elim_sfx}_combined'
        all_k_fname        = f'all_k_per_rank_accuracy{elim_sfx}_combined.png'
    else:   # top_m_identification_rate
        out_subdir         = f'top_k_combined_identification_rate{elim_sfx}'
        left_ylabel_tmpl   = f'Top-${{k}}$ Identification Rate{elim_label} Error (%)'
        right_xlabel_tmpl  = f'Target Top-${{k}}$ Identification Rate{elim_label} (%)'
        mean_left_tmpl     = f'(a) Mean Identification Rate{elim_label} Error — Top-{{k}}'
        mean_right_tmpl    = f'(b) Worst-case Budget Requirements (Identification Rate{elim_label}) — Top-{{k}}'
        ds_left_tmpl       = f'(a) Identification Rate{elim_label} Error on {{dname}} — Top-{{k}}'
        ds_right_tmpl      = f'(b) Budget Requirements (Identification Rate{elim_label}) on {{dname}} — Top-{{k}}'
        fname_prefix       = f'top{{k}}_identification_rate{elim_sfx}_combined'
        all_k_fname        = f'all_k_identification_rate{elim_sfx}_combined.png'

    out_dir = args.out_dir or (args.data_dir / 'plots' / out_subdir)

    print(f"Loading results from {args.data_dir} (metric={metric}) ...")
    data = load_all_results(args.data_dir, k_values=args.k, metric=metric)
    print(f"Found data for {len(data)} dataset(s).")

    target_range = np.arange(args.min_target, args.max_target + 0.5, 1.0)

    mean_error_dict: dict = {}
    worst_cb_dict:   dict = {}

    for k in args.k:
        print(f"\n=== k = {k} ===")
        ly = left_ylabel_tmpl.replace('${k}', str(k))
        rx = right_xlabel_tmpl.replace('${k}', str(k))

        if not args.no_mean:
            print("  Building mean figure ...")
            me = build_mean_error_curves(data, k, DEFAULT_ALGOS, args.max_pct, datasets)
            wc = build_worst_case_cb_curves(data, k, DEFAULT_ALGOS, target_range, datasets)
            mean_error_dict[k] = me
            worst_cb_dict[k]   = wc

            plot_combined_figure(
                title_left=mean_left_tmpl.format(k=k),
                title_right=mean_right_tmpl.format(k=k),
                error_curves=me,
                cb_curves=wc,
                out_path=out_dir / f'{fname_prefix.format(k=k)}_mean.png',
                max_pct=args.max_pct,       max_y_pct=args.max_y_pct,
                min_target=args.min_target, max_target=args.max_target,
                min_budget_y=args.min_budget_y, max_budget_y=args.max_budget_y,
                k_value=k, left_ylabel=ly, right_xlabel=rx,
            )

        if not args.no_individual:
            for ds in datasets:
                if ds not in data:
                    print(f"  Skipping {ds}: no data.")
                    continue
                dname = DATASET_DISPLAY.get(ds, ds.replace('_', ' ').title())
                print(f"  {dname} ...")
                ec = build_error_curves(data, ds, k, DEFAULT_ALGOS, args.max_pct)
                cc = build_cb_curves(data, ds, k, DEFAULT_ALGOS, target_range)
                plot_combined_figure(
                    title_left=ds_left_tmpl.format(dname=dname, k=k),
                    title_right=ds_right_tmpl.format(dname=dname, k=k),
                    error_curves=ec,
                    cb_curves=cc,
                    out_path=out_dir / f'{fname_prefix.format(k=k)}_{ds}.png',
                    max_pct=args.max_pct,       max_y_pct=args.max_y_pct,
                    min_target=args.min_target, max_target=args.max_target,
                    min_budget_y=args.min_budget_y, max_budget_y=args.max_budget_y,
                    k_value=k, left_ylabel=ly, right_xlabel=rx,
                )

    # All-k combined figure (one row per k)
    if not args.no_mean and len(args.k) > 1 and mean_error_dict:
        print("\nBuilding all-k combined figure ...")
        plot_all_k_combined_figure(
            k_values=args.k,
            mean_error_dict=mean_error_dict,
            worst_cb_dict=worst_cb_dict,
            out_path=out_dir / all_k_fname,
            max_pct=args.max_pct,       max_y_pct=args.max_y_pct,
            min_target=args.min_target, max_target=args.max_target,
            min_budget_y=args.min_budget_y, max_budget_y=args.max_budget_y,
            metric=metric,
            left_title_tmpl=mean_left_tmpl,
            right_title_tmpl=mean_right_tmpl,
        )


if __name__ == '__main__':
    main()
