"""
Plot BAI Combined Figure: error rate curve (left) + CB quantile curve (right).

For each dataset (and for the mean across all datasets), produces a two-panel
figure:
  Left  panel (a): Mean BAI error rate (100 – accuracy) vs budget %.
  Right panel (b): Minimum budget % required to reach each target accuracy
                   ("CB quantile function").

Usage
-----
# Mean figure + all individual dataset figures (default)
python plot_bai_combined_figure.py

# Only the mean figure
python plot_bai_combined_figure.py --no-individual

# Single dataset
python plot_bai_combined_figure.py --datasets commonsense

# Customise axes
python plot_bai_combined_figure.py --max-pct 50 --min-target 70 --max-target 100

# Concise mode (5 main algorithms, combined variants — matches paper figure)
python plot_bai_combined_figure.py --concise
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from collections import defaultdict

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

NAME_MAPPING = {
    'Naive-Baseline':       'SyUS',
    'Uniform-Pulls':        'US',
    'SR':                   'SR',
    'Smart-SR':             'SySRs (Ours)',
    'UCB-E (a=0.1)':       'UCB-E (a=0.1)',
    'UCB-E (a=1.0)':       'UCB-E (a=1.0)',
    'UCB-E (a=10)':        'UCB-E (a=10.0)',
    'UCB-E (a=100)':       'UCB-E (a=100.0)',
    'Smart-UCB-E (a=0.1)': 'SyUCB-E (a=0.1) (Ours)',
    'Smart-UCB-E (a=1.0)': 'SyUCB-E (a=1.0) (Ours)',
    'Smart-UCB-E (a=10)':  'SyUCB-E (a=10.0) (Ours)',
    'Smart-UCB-E (a=100)': 'SyUCB-E (a=100.0) (Ours)',
}

LEGEND_ORDER = [
    'UCB-E-LRF', 'UCB-E-LRF (No Warm-up)',
    'SySRs (Ours)', 'SR',
    'SyUCB-E (a=0.1) (Ours)', 'SyUCB-E (a=1.0) (Ours)',
    'SyUCB-E (a=10.0) (Ours)', 'SyUCB-E (a=100.0) (Ours)',
    'UCB-E (a=0.1)', 'UCB-E (a=1.0)', 'UCB-E (a=10.0)', 'UCB-E (a=100.0)',
    'SyUS', 'US',
]

DEFAULT_ALGOS = list(LEGEND_ORDER)  # includes LRF, updated when LRF files present

ALGORITHM_STYLES = {
    # UCB-E-LRF variants — Purple shades (most prominent: highest zorder)
    'UCB-E-LRF': {
        'color': '#9370DB', 'marker': '*', 'markersize': 9,
        'linestyle': '-', 'linewidth': 3.0, 'alpha': 0.95, 'zorder': 12,
    },
    'UCB-E-LRF (No Warm-up)': {
        'color': '#8B008B', 'marker': 'P', 'markersize': 7,
        'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 12,
    },
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

# Budget positions where markers are placed on the left panel curves
MARKER_BUDGETS = list(range(5, 101, 5))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(data_dir: Path):
    """Load per-budget pkl files (bai and lrf) and return accuracy curves.

    Returns
    -------
    data : nested dict  data[dataset][algo] -> sorted list of (budget_pct, accuracy)
        accuracy is in [0, 100] (percentage of runs that selected arm 0).
    """
    data = defaultdict(lambda: defaultdict(list))

    all_files = sorted(set(
        list(data_dir.glob('*_bai_results.pkl')) +
        list(data_dir.glob('*_lrf_results.pkl'))
    ))

    for fp in all_files:
        if 'pct' not in fp.name:
            continue
        try:
            with fp.open('rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"Warning: could not load {fp.name}: {e}", file=sys.stderr)
            continue

        dataset         = obj.get('dataset')
        budget_fraction = obj.get('budget_fraction')
        selected_arms   = obj.get('selected_arms', {})

        if dataset is None or budget_fraction is None or not selected_arms:
            continue

        pct = int(round(budget_fraction * 100))

        for algo_raw, arms in selected_arms.items():
            if not hasattr(arms, '__len__'):
                continue
            algo    = NAME_MAPPING.get(algo_raw, algo_raw)
            arms_np = np.asarray(arms)
            accuracy = float((arms_np == 0).mean()) * 100.0
            data[dataset][algo].append((pct, accuracy))

    # Sort by budget percentage
    for dataset in data:
        for algo in data[dataset]:
            data[dataset][algo].sort(key=lambda x: x[0])

    return data


# ---------------------------------------------------------------------------
# CB computation
# ---------------------------------------------------------------------------

def find_cb(curve, target):
    """Minimum budget_pct where accuracy >= target AND stays >= target.

    Returns 100.0 (capped) if never reached within observed budgets.
    """
    for i, (bpct, acc) in enumerate(curve):
        if acc >= target:
            if all(a >= target for _, a in curve[i:]):
                return float(bpct)
    return 100.0


def compute_cb_curve(curve, target_range):
    """Return [(target, cb_pct)] for each target in target_range."""
    return [(t, find_cb(curve, t)) for t in target_range]


# ---------------------------------------------------------------------------
# Build curves for plotting
# ---------------------------------------------------------------------------

def build_error_curves(data, dataset, algos, max_pct):
    """error = 100 – accuracy, filtered to budget_pct <= max_pct."""
    curves = {}
    for algo in algos:
        pts = data.get(dataset, {}).get(algo, [])
        if not pts:
            continue
        bpcts   = np.array([p[0] for p in pts if p[0] <= max_pct])
        accs    = np.array([p[1] for p in pts if p[0] <= max_pct])
        if len(bpcts) == 0:
            continue
        curves[algo] = (bpcts, np.clip(100.0 - accs, 0.0, 100.0))
    return curves


def build_cb_curves(data, dataset, algos, target_range):
    """CB quantile curves for one dataset."""
    curves = {}
    for algo in algos:
        pts = data.get(dataset, {}).get(algo, [])
        if not pts:
            continue
        cb_vals = compute_cb_curve(pts, target_range)
        curves[algo] = (
            np.array([t for t, _ in cb_vals]),
            np.array([c for _, c in cb_vals]),
        )
    return curves


def build_mean_error_curves(data, algos, max_pct, datasets=None):
    """Mean error rate across datasets, averaged directly at each budget pct."""
    if datasets is None:
        datasets = list(data.keys())
    # Accumulate errors per (algo, pct) across datasets
    algo_pct_errors = defaultdict(lambda: defaultdict(list))
    for dataset in datasets:
        for algo in algos:
            pts = data.get(dataset, {}).get(algo, [])
            if not pts:
                continue
            for pct, acc in pts:
                if pct <= max_pct:
                    algo_pct_errors[algo][pct].append(np.clip(100.0 - acc, 0.0, 100.0))
    result = {}
    for algo, pct_dict in algo_pct_errors.items():
        sorted_pcts = np.array(sorted(pct_dict.keys()))
        means = np.array([np.mean(pct_dict[p]) for p in sorted_pcts])
        result[algo] = (sorted_pcts, means)
    return result


def build_worst_case_cb_curves(data, algos, target_range, datasets=None):
    """Worst-case (max across datasets) CB for each target accuracy."""
    if datasets is None:
        datasets = list(data.keys())
    algo_cbs = defaultdict(lambda: defaultdict(list))
    for dataset in datasets:
        for algo in algos:
            pts = data.get(dataset, {}).get(algo, [])
            if not pts:
                continue
            for t in target_range:
                algo_cbs[algo][t].append(find_cb(pts, t))
    worst_curves = {}
    for algo, t_dict in algo_cbs.items():
        targets = np.array(sorted(t_dict.keys()))
        wc = np.array([max(t_dict[t]) if t_dict[t] else 100.0 for t in targets])
        worst_curves[algo] = (targets, wc)
    return worst_curves


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _default_style(algo):
    return {'color': 'gray', 'marker': 'o', 'markersize': 5,
            'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7, 'zorder': 5}


def _get_style(algo):
    return ALGORITHM_STYLES.get(algo, _default_style(algo))


def _sort_algos(algos):
    return sorted(algos, key=lambda a: LEGEND_ORDER.index(a) if a in LEGEND_ORDER else 99)


def _build_legend(handles_dict):
    ordered_handles, ordered_labels = [], []
    for algo in LEGEND_ORDER:
        if algo in handles_dict:
            ordered_handles.append(handles_dict[algo])
            ordered_labels.append(algo)
    for algo, h in handles_dict.items():
        if algo not in LEGEND_ORDER:
            ordered_handles.append(h)
            ordered_labels.append(algo)
    return ordered_handles, ordered_labels


# ---------------------------------------------------------------------------
# Two-panel figure
# ---------------------------------------------------------------------------

def plot_combined_figure(
    title_left, title_right,
    error_curves,   # algo -> (bpcts, errors)
    cb_curves,      # algo -> (targets, cbs)
    out_path: Path,
    max_pct=100,
    max_y_pct=100,
    min_target=70,
    max_target=100,
    min_budget_y=0,
    max_budget_y=105,
    concise=False,
):
    """Save a two-panel BAI figure."""
    _CONCISE_RENAME = {'SyUS': 'Uniform Sampling', 'UCB-E (a=1.0)': 'UCB-E'}
    _CONCISE_ORDER  = ['SySRs (Ours)', 'SR', 'UCB-E', 'UCB-E-LRF', 'Uniform Sampling']

    # Concise mode: combine algorithm variants and restrict to 5 main algorithms
    if concise:
        error_curves = dict(error_curves)
        cb_curves    = dict(cb_curves)
        common_grid  = np.arange(1, int(max_pct) + 1, 1, dtype=float)

        # Combine UCB-E-LRF variants — take minimum error (best performance)
        lrf_keys_e = [k for k in ('UCB-E-LRF', 'UCB-E-LRF (No Warm-up)') if k in error_curves]
        if len(lrf_keys_e) >= 2:
            arrs = [np.interp(common_grid, error_curves[k][0], error_curves[k][1],
                              left=error_curves[k][1][0], right=error_curves[k][1][-1])
                    for k in lrf_keys_e]
            error_curves['UCB-E-LRF'] = (common_grid, np.minimum.reduce(arrs))
        lrf_keys_cb = [k for k in ('UCB-E-LRF', 'UCB-E-LRF (No Warm-up)') if k in cb_curves]
        if len(lrf_keys_cb) >= 2:
            t = cb_curves['UCB-E-LRF'][0]
            arrs = [cb_curves[k][1] for k in lrf_keys_cb]
            cb_curves['UCB-E-LRF'] = (t, np.minimum.reduce(arrs))

        # Combine all UCB-E hyperparameter variants — take minimum error / minimum budget
        ucb_e_keys = [f'UCB-E (a={v})' for v in ('0.1', '1.0', '10.0', '100.0')
                      if f'UCB-E (a={v})' in error_curves]
        if len(ucb_e_keys) >= 2:
            arrs = [np.interp(common_grid, error_curves[k][0], error_curves[k][1],
                              left=error_curves[k][1][0], right=error_curves[k][1][-1])
                    for k in ucb_e_keys]
            error_curves['UCB-E (a=1.0)'] = (common_grid, np.minimum.reduce(arrs))
        ucb_e_keys_cb = [f'UCB-E (a={v})' for v in ('0.1', '1.0', '10.0', '100.0')
                         if f'UCB-E (a={v})' in cb_curves]
        if len(ucb_e_keys_cb) >= 2:
            ref_t = cb_curves[ucb_e_keys_cb[0]][0]
            arrs  = [cb_curves[k][1] for k in ucb_e_keys_cb]
            cb_curves['UCB-E (a=1.0)'] = (ref_t, np.minimum.reduce(arrs))

        concise_algos = ['SySRs (Ours)', 'SR', 'UCB-E-LRF', 'UCB-E (a=1.0)', 'SyUS']
        error_curves  = {k: v for k, v in error_curves.items() if k in concise_algos}
        cb_curves     = {k: v for k, v in cb_curves.items()    if k in concise_algos}

        # Use _concise suffix in output filename
        out_path = out_path.parent / (out_path.stem + '_concise' + out_path.suffix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), dpi=150)

    algos_left  = _sort_algos([a for a in error_curves if a in DEFAULT_ALGOS or a in ALGORITHM_STYLES])
    algos_right = _sort_algos([a for a in cb_curves    if a in DEFAULT_ALGOS or a in ALGORITHM_STYLES])
    marker_ticks = [b for b in MARKER_BUDGETS if b <= max_pct]
    handles_dict = {}

    # ---- Left panel: error rate vs budget ----
    for algo in algos_left:
        bpcts, errors = error_curves[algo]
        style = _get_style(algo)
        marker_indices = [
            int(np.argmin(np.abs(bpcts - t)))
            for t in marker_ticks if bpcts.min() <= t <= bpcts.max()
        ]
        legend_name = (_CONCISE_RENAME.get(algo, algo) if concise else algo)
        line, = ax1.plot(
            bpcts, errors,
            label=legend_name,
            color=style['color'], linestyle=style['linestyle'],
            marker=style['marker'], markersize=style['markersize'],
            markevery=marker_indices if marker_indices else None,
            linewidth=style['linewidth'], alpha=style['alpha'], zorder=style['zorder'],
        )
        handles_dict[legend_name] = line

    ax1.set_xlabel('Percentage of Full Evaluation Budget', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Best-Arm Identification Error Rate (%)', fontsize=16, fontweight='bold')
    ax1.set_title(title_left, fontsize=18, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(0, max_pct)
    ax1.set_ylim(0, max_y_pct)
    ax1.set_xticks(np.arange(0, int(max_pct) + 1, 5))
    ax1.tick_params(axis='both', labelsize=14)

    # ---- Right panel: CB quantile ----
    for algo in algos_right:
        targets, cbs = cb_curves[algo]
        mask = (targets >= min_target) & (targets <= max_target)
        if mask.sum() == 0:
            continue
        style = _get_style(algo)
        legend_name = (_CONCISE_RENAME.get(algo, algo) if concise else algo)
        line, = ax2.plot(
            targets[mask], cbs[mask],
            label=legend_name,
            color=style['color'], linestyle=style['linestyle'],
            marker=style['marker'], markersize=style['markersize'],
            markevery=1,
            linewidth=style['linewidth'], alpha=style['alpha'], zorder=style['zorder'],
        )
        handles_dict[legend_name] = line

    ax2.set_xlabel('Target Best-Arm Identification Accuracy (%)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Budget Percentage Required (%)', fontsize=18, fontweight='bold')
    ax2.set_title(title_right, fontsize=18, fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim(min_target, max_target)
    ax2.set_ylim(min_budget_y, max_budget_y)
    ax2.set_xticks(np.arange(int(min_target), int(max_target) + 1, 1))
    ax2.tick_params(axis='both', labelsize=14)

    if concise:
        ordered_handles = [handles_dict[l] for l in _CONCISE_ORDER if l in handles_dict]
        ordered_labels  = [l for l in _CONCISE_ORDER if l in handles_dict]
        fig.legend(
            ordered_handles, ordered_labels,
            loc='lower center',
            ncol=len(ordered_labels),
            bbox_to_anchor=(0.5, -0.05),
            framealpha=0.9,
            edgecolor='black',
            prop={'weight': 'bold', 'size': 18},
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        ordered_handles, ordered_labels = _build_legend(handles_dict)
        fig.legend(
            ordered_handles, ordered_labels,
            loc='lower center',
            ncol=min(6, len(ordered_handles)),
            fontsize=12,
            bbox_to_anchor=(0.5, -0.12),
            frameon=True, fancybox=True, shadow=False,
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Plot BAI combined figures.')
    p.add_argument('--data-dir', type=Path,
                   default=Path(__file__).parent / 'results',
                   help='Directory containing *_bai_results.pkl files.')
    p.add_argument('--out-dir', type=Path,
                   default=Path(__file__).parent / 'results' / 'plots',
                   help='Output directory for figures.')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Datasets to plot (default: all available).')
    p.add_argument('--max-pct', type=float, default=35,
                   help='Max budget %% for left panel x-axis (default: 35).')
    p.add_argument('--max-y-pct', type=float, default=90,
                   help='Max error rate %% for left panel y-axis (default: 90).')
    p.add_argument('--min-target', type=float, default=90,
                   help='Min target accuracy %% for right panel x-axis (default: 90).')
    p.add_argument('--max-target', type=float, default=100,
                   help='Max target accuracy %% for right panel x-axis (default: 100).')
    p.add_argument('--min-budget-y', type=float, default=0,
                   help='Min y for right panel (default: 0).')
    p.add_argument('--max-budget-y', type=float, default=105,
                   help='Max y for right panel (default: 105).')
    p.add_argument('--no-mean', action='store_true',
                   help='Skip the mean-across-datasets figure.')
    p.add_argument('--no-individual', action='store_true',
                   help='Skip individual dataset figures.')
    p.add_argument('--concise', action='store_true',
                   help='Show only 5 main algorithms (SySRs, SR, UCB-E-LRF, UCB-E, SyUS) '
                        'with UCB-E-LRF variants combined and UCB-E hyperparameter '
                        'variants combined into their best envelope.')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading results from {args.data_dir} ...")
    data = load_all_results(args.data_dir)
    datasets_available = sorted(data.keys())
    if not datasets_available:
        print("No data found. Run run_bai_experiments.py first.")
        return
    print(f"Loaded data for {len(datasets_available)} datasets.")

    datasets = args.datasets or datasets_available
    algos    = DEFAULT_ALGOS
    target_range = np.arange(args.min_target, args.max_target + 0.5, 1.0)

    # ---- Mean figure ----
    if not args.no_mean:
        print("\nBuilding mean figure ...")
        mean_err = build_mean_error_curves(data, algos, args.max_pct, datasets=datasets)
        worst_cb = build_worst_case_cb_curves(data, algos, target_range, datasets=datasets)
        out_path = args.out_dir / 'bai_combined_figure_mean.png'
        plot_combined_figure(
            title_left='(a) Mean BAI Error Rate',
            title_right='(b) Worst-case Budget Requirements',
            error_curves=mean_err,
            cb_curves=worst_cb,
            out_path=out_path,
            max_pct=args.max_pct,
            max_y_pct=args.max_y_pct,
            min_target=args.min_target,
            max_target=args.max_target,
            min_budget_y=args.min_budget_y,
            max_budget_y=args.max_budget_y,
            concise=args.concise,
        )

    # ---- Per-dataset figures ----
    if not args.no_individual and not args.concise:
        for dataset in datasets:
            if dataset not in data:
                print(f"  Skipping {dataset}: no data.")
                continue
            dname = DATASET_DISPLAY.get(dataset, dataset.replace('_', ' ').title())
            print(f"\nBuilding figure for {dname} ...")
            err_curves = build_error_curves(data, dataset, algos, args.max_pct)
            cb_curves_ds = build_cb_curves(data, dataset, algos, target_range)
            out_path = args.out_dir / f'bai_combined_figure_{dataset}.png'
            plot_combined_figure(
                title_left=f'(a) BAI Error Rate — {dname}',
                title_right=f'(b) Budget Requirements — {dname}',
                error_curves=err_curves,
                cb_curves=cb_curves_ds,
                out_path=out_path,
                max_pct=args.max_pct,
                max_y_pct=args.max_y_pct,
                min_target=args.min_target,
                max_target=args.max_target,
                min_budget_y=args.min_budget_y,
                max_budget_y=args.max_budget_y,
                concise=args.concise,
            )

    print(f"\nFigures saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
