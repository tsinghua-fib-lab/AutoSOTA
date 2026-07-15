"""
Compute BAI Confidence Budget (CB) statistics.

The Confidence Budget (CB) for a given algorithm at a given target accuracy T
is the minimum budget percentage b* such that the algorithm's mean accuracy
is ≥ T for every budget ≥ b*.

Outputs
-------
bai_confidence_budget_per_dataset.csv
    One row per (dataset × algorithm × threshold), showing the CB value.

bai_confidence_budget_summary.csv
    One row per (algorithm × threshold), showing mean CB across datasets.

LaTeX tables are printed to stdout.

Usage
-----
# Compute CB at default thresholds (70, 80, 90, 95, 100)
python compute_bai_confidence_budget.py

# Custom thresholds
python compute_bai_confidence_budget.py --thresholds 80 90 95 100

# Restrict to specific datasets
python compute_bai_confidence_budget.py --datasets commonsense mmlu gsm

# Cap CB at a different max (e.g. only evaluate up to 50%)
python compute_bai_confidence_budget.py --max-pct 50
"""

import argparse
import sys
import csv
import numpy as np
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

# Display order for tables and LaTeX output
TABLE_ROW_ORDER = [
    'UCB-E-LRF',
    'UCB-E-LRF (No Warm-up)',
    'US',
    'SyUS',
    'SR',
    'SySRs (Ours)',
    'UCB-E (a=0.1)',
    'UCB-E (a=1.0)',
    'UCB-E (a=10.0)',
    'UCB-E (a=100.0)',
    'SyUCB-E (a=0.1) (Ours)',
    'SyUCB-E (a=1.0) (Ours)',
    'SyUCB-E (a=10.0) (Ours)',
    'SyUCB-E (a=100.0) (Ours)',
]

LATEX_GROUPS = [
    ('UCB-E-LRF',          ['UCB-E-LRF', 'UCB-E-LRF (No Warm-up)']),
    ('Baselines',          ['US', 'SyUS']),
    ('Successive Rejects', ['SR', 'SySRs (Ours)']),
    ('UCB-E',             ['UCB-E (a=0.1)', 'UCB-E (a=1.0)', 'UCB-E (a=10.0)', 'UCB-E (a=100.0)']),
    ('SyUCB-E',           ['SyUCB-E (a=0.1) (Ours)', 'SyUCB-E (a=1.0) (Ours)',
                           'SyUCB-E (a=10.0) (Ours)', 'SyUCB-E (a=100.0) (Ours)']),
]

DEFAULT_THRESHOLDS = [70, 80, 90, 95, 100]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(data_dir: Path, max_pct=100):
    """Load per-budget pkl files (bai and lrf) and return accuracy curves.

    Returns
    -------
    data : nested dict  data[dataset][algo] -> sorted list of (budget_pct, accuracy)
        accuracy is in [0, 100] (% runs that selected arm 0).
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
        if pct > max_pct:
            continue

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

def find_cb(curve, threshold):
    """Minimum budget_pct where accuracy >= threshold AND stays >= threshold.

    If the threshold is never sustainably reached, returns None.

    Parameters
    ----------
    curve     : list of (budget_pct, accuracy_pct) sorted by budget_pct
    threshold : float, accuracy target in [0, 100]

    Returns
    -------
    float or None
    """
    for i, (bpct, acc) in enumerate(curve):
        if acc >= threshold:
            if all(a >= threshold for _, a in curve[i:]):
                return float(bpct)
    return None


def compute_cb_for_all(data, thresholds, cap_pct=100.0):
    """Compute CB for every (dataset, algo, threshold) combination.

    Returns
    -------
    cb_data : dict  cb_data[(dataset, algo, threshold)] = cb_pct or cap_pct
    """
    cb_data = {}
    for dataset, algo_dict in data.items():
        for algo, curve in algo_dict.items():
            for thr in thresholds:
                cb = find_cb(curve, thr)
                cb_data[(dataset, algo, thr)] = cb if cb is not None else cap_pct
    return cb_data


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def summarize_cb(cb_data, datasets, algos, thresholds, cap_pct=100.0):
    """Return mean and max CB per (algo, threshold) across provided datasets.

    Returns
    -------
    summary : dict  summary[(algo, threshold)] = {'mean': float, 'max': float}
    """
    summary = {}
    for algo in algos:
        for thr in thresholds:
            vals = [
                cb_data.get((ds, algo, thr), cap_pct)
                for ds in datasets
                if (ds, algo, thr) in cb_data
            ]
            if vals:
                summary[(algo, thr)] = {
                    'mean': float(np.mean(vals)),
                    'max':  float(np.max(vals)),
                }
            else:
                summary[(algo, thr)] = {'mean': cap_pct, 'max': cap_pct}
    return summary


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def write_per_dataset_csv(cb_data, datasets, algos, thresholds, out_path: Path):
    """Write bai_confidence_budget_per_dataset.csv."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'algorithm', 'threshold', 'confidence_budget'])
        for dataset in datasets:
            for algo in algos:
                for thr in thresholds:
                    key = (dataset, algo, thr)
                    if key in cb_data:
                        writer.writerow([dataset, algo, thr, f"{cb_data[key]:.1f}"])
    print(f"Saved per-dataset CSV: {out_path}")


def write_summary_csv(summary, algos, thresholds, out_path: Path):
    """Write bai_confidence_budget_summary.csv."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        header = ['algorithm']
        for t in thresholds:
            header += [f'mean_CB@{t}%', f'max_CB@{t}%']
        writer.writerow(header)
        for algo in algos:
            row = [algo]
            for thr in thresholds:
                entry = summary.get((algo, thr), {'mean': 100.0, 'max': 100.0})
                row += [f"{entry['mean']:.1f}", f"{entry['max']:.1f}"]
            writer.writerow(row)
    print(f"Saved summary CSV:     {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def format_latex_cell(entry, cap_pct, is_best=False):
    """Format a single CB cell for LaTeX showing 'mean (max)'."""
    def _fmt(v):
        return '100' if v >= cap_pct else f'{v:.1f}'
    s = f"{_fmt(entry['mean'])} ({_fmt(entry['max'])})"
    if is_best:
        return r'\textbf{' + s + r'}'
    return s


def print_latex_table(summary, thresholds, algos, cap_pct=100.0):
    """Print a grouped LaTeX table of CB values (mean across datasets).

    Groups or individual rows whose algos are not in *algos* are silently
    skipped, so the table is clean when e.g. LRF results are absent.
    """
    algos_set = set(algos)
    active_groups = [
        (gname, [a for a in galgos if a in algos_set])
        for gname, galgos in LATEX_GROUPS
        if any(a in algos_set for a in galgos)
    ]

    cols = ' & '.join([f'CB@{t}\\%' for t in thresholds])
    n_cols = 1 + len(thresholds)
    col_spec = 'l' + 'r' * len(thresholds)

    # Best per threshold (by mean) only over present algos
    best_cb = {
        thr: min(
            (summary.get((a, thr), {'mean': cap_pct})['mean'] for a in algos),
            default=cap_pct,
        )
        for thr in thresholds
    }

    print('\n% ===== BAI Confidence Budget LaTeX Table =====')
    print(r'\begin{table}[t]')
    print(r'  \centering')
    print(r'  \small')
    print(r'  \begin{tabular}{' + col_spec + r'}')
    print(r'    \toprule')
    print(f'    Algorithm & {cols} \\\\')
    print(r'    \midrule')

    for g_idx, (group_name, group_algos) in enumerate(active_groups):
        if g_idx > 0:
            print(r'    \midrule')
        print(f'    \\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{group_name}}}}} \\\\')
        for algo in group_algos:
            cells = []
            for thr in thresholds:
                entry = summary.get((algo, thr), {'mean': cap_pct, 'max': cap_pct})
                cells.append(format_latex_cell(entry, cap_pct,
                                               is_best=(entry['mean'] == best_cb[thr])))
            algo_tex = algo.replace('%', r'\%').replace('_', r'\_')
            print(f'    {algo_tex} & {" & ".join(cells)} \\\\')

    print(r'    \bottomrule')
    print(r'  \end{tabular}')
    print(r'  \caption{BAI Confidence Budget: minimum budget \% to sustain target accuracy.}')
    print(r'  \label{tab:bai_confidence_budget}')
    print(r'\end{table}')
    print()


# ---------------------------------------------------------------------------
# Human-readable table (console)
# ---------------------------------------------------------------------------

def print_console_table(summary, algos, thresholds, cap_pct=100.0):
    """Print a formatted table to stdout."""
    col_w = 30
    thr_w = 16

    header = f"{'Algorithm':<{col_w}}" + "".join(
        f"  CB@{t}% (mean/max)".rjust(thr_w + 2) for t in thresholds
    )
    sep_w = col_w + len(thresholds) * (thr_w + 2)
    print('\n' + '=' * sep_w)
    print('BAI Confidence Budget Summary (mean / max across datasets)')
    print('=' * sep_w)
    print(header)
    print('-' * sep_w)

    for algo in algos:
        row = f"{algo:<{col_w}}"
        for thr in thresholds:
            entry = summary.get((algo, thr), {'mean': cap_pct, 'max': cap_pct})
            def _fmt(v):
                return '100' if v >= cap_pct else f'{v:.1f}'
            cell = f"{_fmt(entry['mean'])} / {_fmt(entry['max'])}"
            row += cell.rjust(thr_w + 2)
        print(row)
    print('=' * sep_w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Compute BAI Confidence Budget.')
    p.add_argument('--data-dir', type=Path,
                   default=Path(__file__).parent / 'results',
                   help='Directory containing *_bai_results.pkl files.')
    p.add_argument('--out-dir', type=Path,
                   default=Path(__file__).parent / 'results',
                   help='Directory for output CSV files.')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Restrict to these datasets (default: all available).')
    p.add_argument('--thresholds', nargs='+', type=float,
                   default=DEFAULT_THRESHOLDS,
                   help='Target accuracy thresholds (default: 70 80 90 95 100).')
    p.add_argument('--max-pct', type=float, default=100.0,
                   help='Maximum budget %% to consider (default: 100).')
    p.add_argument('--no-latex', action='store_true',
                   help='Skip LaTeX table output.')
    return p.parse_args()


def main():
    args = parse_args()
    thresholds = sorted(args.thresholds)
    cap_pct    = args.max_pct

    print(f"Loading results from {args.data_dir} ...")
    data = load_all_results(args.data_dir, max_pct=int(args.max_pct))
    datasets_available = sorted(data.keys())
    if not datasets_available:
        print("No data found. Run run_bai_experiments.py first.")
        return
    print(f"Loaded data for {len(datasets_available)} datasets.")

    datasets = sorted(args.datasets) if args.datasets else datasets_available

    # Collect all algorithms seen in the data
    seen_algos = set()
    for ds in datasets:
        for algo in data.get(ds, {}):
            seen_algos.add(algo)
    algos = [a for a in TABLE_ROW_ORDER if a in seen_algos]
    missing = seen_algos - set(TABLE_ROW_ORDER)
    if missing:
        algos += sorted(missing)

    # Compute CB
    cb_data = compute_cb_for_all(data, thresholds, cap_pct=cap_pct)
    summary  = summarize_cb(cb_data, datasets, algos, thresholds, cap_pct=cap_pct)

    # Console output
    print_console_table(summary, algos, thresholds, cap_pct=cap_pct)

    # LaTeX output
    if not args.no_latex:
        print_latex_table(summary, thresholds, algos, cap_pct=cap_pct)

    # CSV output
    write_per_dataset_csv(
        cb_data, datasets, algos, thresholds,
        args.out_dir / 'bai_confidence_budget_per_dataset.csv',
    )
    write_summary_csv(
        summary, algos, thresholds,
        args.out_dir / 'bai_confidence_budget_summary.csv',
    )

    print(f"\nDone. Output written to: {args.out_dir}")


if __name__ == '__main__':
    main()
