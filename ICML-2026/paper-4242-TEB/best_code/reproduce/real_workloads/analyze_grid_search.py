#!/usr/bin/env python3
"""
Analyze TB × BS grid search results on real datasets.

Applies to results produced by run_grid_search_real.sh.
Directory structure: tb{TB}/bs{BS}/bench_*.json

Usage:
    python analyze_grid_search_real.py <experiment_dir>

Outputs:
    - grid_summary.json: full data summary
    - heatmap.png: heatmap
    - optimal_comparison.png: optimal configuration comparison plot
    - analysis_report.txt: text analysis report
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping plots")


def load_bench_result_fast(filepath: Path) -> Optional[Dict]:
    """Quickly load a benchmark result file (only reads metadata at head and tail).

    For large files (100MB+), only the first 100KB and last 100KB are read to
    extract key metrics, avoiding loading the whole file into memory.

    Note: vLLM benchmark JSON file structure:
    - Head contains: throughput metrics, completed, failed, etc.
    - Tail contains: TTFT, TPOT, ITL and other latency metrics
    """
    if not filepath.exists():
        return None

    try:
        file_size = filepath.stat().st_size

        # Small files: load directly
        if file_size < 10 * 1024 * 1024:  # < 10MB
            with open(filepath) as f:
                return json.load(f)

        # Large files: read head and tail
        with open(filepath, 'r') as f:
            header = f.read(100 * 1024)  # read first 100KB
            # Read last 100KB (latency metrics are at the end of the file)
            f.seek(max(0, file_size - 100 * 1024))
            footer = f.read()

        # Combine head and tail content for searching
        combined = header + footer

        result = {}

        # Fields to extract
        # Throughput is at the head, latency metrics are at the tail
        fields = [
            'request_throughput', 'output_throughput', 'total_token_throughput',
            'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
            'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms',
            'mean_itl_ms', 'median_itl_ms', 'p99_itl_ms',
            'completed', 'failed', 'num_prompts'
        ]

        for field in fields:
            # Match "field": value pattern
            pattern = rf'"{field}":\s*([^,\}}\]]+)'
            match = re.search(pattern, combined)
            if match:
                value_str = match.group(1).strip()
                try:
                    result[field] = json.loads(value_str)
                except json.JSONDecodeError:
                    pass

        return result if result else None

    except Exception:
        return None


def extract_metrics(bench_result: Dict) -> Dict[str, float]:
    """Extract key metrics from benchmark result."""
    return {
        "throughput": bench_result.get("request_throughput", 0),
        "output_throughput": bench_result.get("output_throughput", 0),
        "mean_ttft_ms": bench_result.get("mean_ttft_ms", 0),
        "median_ttft_ms": bench_result.get("median_ttft_ms", 0),
        "p99_ttft_ms": bench_result.get("p99_ttft_ms", 0),
        "mean_tpot_ms": bench_result.get("mean_tpot_ms", 0),
        "median_tpot_ms": bench_result.get("median_tpot_ms", 0),
        "p99_tpot_ms": bench_result.get("p99_tpot_ms", 0),
        "mean_itl_ms": bench_result.get("mean_itl_ms", 0),
        "median_itl_ms": bench_result.get("median_itl_ms", 0),
        "p99_itl_ms": bench_result.get("p99_itl_ms", 0),
        "completed": bench_result.get("completed", 0),
        "failed": bench_result.get("failed", 0),
    }


def collect_grid_results(exp_dir: Path) -> Dict:
    """Collect grid search results.

    Directory structure: tb{TB}/bs{BS}/bench_{scheduler}.json

    Returns:
        {
            "tb_values": [...],
            "bs_values": [...],
            "results": {
                (tb, bs): {"v1": metrics, "eb_kratio": metrics, ...}
            }
        }
    """
    tb_values = set()
    bs_values = set()
    results = {}

    for tb_dir in sorted(exp_dir.iterdir()):
        if not tb_dir.is_dir() or not tb_dir.name.startswith("tb"):
            continue

        tb = int(tb_dir.name[2:])
        tb_values.add(tb)

        for bs_dir in sorted(tb_dir.iterdir()):
            if not bs_dir.is_dir() or not bs_dir.name.startswith("bs"):
                continue

            bs = int(bs_dir.name[2:])
            bs_values.add(bs)

            key = (tb, bs)
            results[key] = {}

            # Dynamically detect all bench_*.json files
            for bench_file in bs_dir.glob("bench_*.json"):
                # Extract scheduler name from filename: bench_eb_1.json -> pd_ifr_1
                scheduler = bench_file.stem[6:]  # strip "bench_" prefix
                bench_result = load_bench_result_fast(bench_file)
                if bench_result:
                    results[key][scheduler] = extract_metrics(bench_result)

    return {
        "tb_values": sorted(tb_values),
        "bs_values": sorted(bs_values),
        "results": results
    }


def find_optimal_configs(data: Dict) -> Dict:
    """Find the optimal configuration for each scheduler."""
    optimal = {}
    # Dynamically detect all schedulers
    all_schedulers = set()
    for sched_results in data["results"].values():
        all_schedulers.update(sched_results.keys())

    for scheduler in sorted(all_schedulers):
        best_key = None
        best_throughput = 0

        for (tb, bs), sched_results in data["results"].items():
            if scheduler in sched_results:
                tp = sched_results[scheduler].get("throughput", 0)
                if tp > best_throughput:
                    best_throughput = tp
                    best_key = (tb, bs)

        if best_key:
            optimal[scheduler] = {
                "tb": best_key[0],
                "bs": best_key[1],
                "metrics": data["results"][best_key][scheduler]
            }

    return optimal


def get_best_eb_variant(sched_results: Dict) -> Optional[str]:
    """Return the best available EB variant (selected by throughput)."""
    # Dynamically detect all schedulers starting with eb
    best_variant = None
    best_tp = 0

    for variant, metrics in sched_results.items():
        if variant.startswith("eb"):
            tp = metrics.get("throughput", 0)
            if tp > best_tp:
                best_tp = tp
                best_variant = variant

    return best_variant


def compute_improvement_grid(data: Dict, metric: str, higher_better: bool = True) -> Tuple[np.ndarray, List, List]:
    """Compute the EB-vs-v1 improvement grid."""
    tb_values = data["tb_values"]
    bs_values = data["bs_values"]
    results = data["results"]

    matrix = np.full((len(tb_values), len(bs_values)), np.nan)

    for i, tb in enumerate(tb_values):
        for j, bs in enumerate(bs_values):
            key = (tb, bs)
            if key in results:
                v1_val = results[key].get("v1", {}).get(metric, 0)
                eb_variant = get_best_eb_variant(results[key])
                eb_val = results[key].get(eb_variant, {}).get(metric, 0) if eb_variant else 0

                if v1_val > 0 and eb_val > 0:
                    if higher_better:
                        improvement = (eb_val - v1_val) / v1_val * 100
                    else:
                        improvement = (v1_val - eb_val) / v1_val * 100
                    matrix[i, j] = improvement

    return matrix, tb_values, bs_values


def plot_heatmaps(data: Dict, output_dir: Path):
    """Plot heatmaps."""
    if not HAS_MATPLOTLIB:
        return

    metrics_to_plot = [
        ("throughput", "Throughput Improvement (%)", True),
        ("mean_itl_ms", "Mean ITL Improvement (%)", False),
        ("mean_ttft_ms", "Mean TTFT Improvement (%)", False),
        ("output_throughput", "Output Throughput Improvement (%)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (metric, title, higher_better) in enumerate(metrics_to_plot):
        ax = axes[idx]
        matrix, tb_vals, bs_vals = compute_improvement_grid(data, metric, higher_better)

        if np.all(np.isnan(matrix)):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 10)
        vmin = -vmax

        cmap = plt.cm.RdYlGn
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_xticks(range(len(bs_vals)))
        ax.set_xticklabels([str(b) for b in bs_vals], rotation=45)
        ax.set_yticks(range(len(tb_vals)))
        ax.set_yticklabels([str(t) for t in tb_vals])

        ax.set_xlabel('max_num_seqs (BS)')
        ax.set_ylabel('max_num_batched_tokens (TB)')
        ax.set_title(title)

        for i in range(len(tb_vals)):
            for j in range(len(bs_vals)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            color=color, fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('PD vs Baseline Improvement', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_optimal_comparison(optimal: Dict, output_dir: Path):
    """Plot optimal configuration comparison."""
    if not HAS_MATPLOTLIB:
        return

    metrics_to_plot = [
        ("throughput", "Request Throughput (req/s)", True),
        ("output_throughput", "Output Throughput (tok/s)", True),
        ("mean_itl_ms", "Mean ITL (ms)", False),
        ("p99_itl_ms", "P99 ITL (ms)", False),
        ("mean_ttft_ms", "Mean TTFT (ms)", False),
        ("p99_ttft_ms", "P99 TTFT (ms)", False),
    ]

    schedulers = [
        ("v1",        '#2ecc71', 'v1'),
        ("eb_kratio", '#3498db', 'EB (fixed θ*)'),
        ("eb",        '#9b59b6', 'EB(k̂*)'),
        ("ebplus",    '#1abc9c', 'EB+'),
    ]

    available_scheds = [(key, color, label) for key, color, label in schedulers if key in optimal]

    if len(available_scheds) < 2:
        print("Not enough schedulers for comparison")
        return

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for j, (metric, ylabel, higher_better) in enumerate(metrics_to_plot):
        ax = axes[j]

        values = []
        colors = []
        labels = []
        for key, color, label in available_scheds:
            opt = optimal[key]
            val = opt["metrics"].get(metric, 0)
            values.append(val)
            colors.append(color)
            labels.append(f'{label}\nTB={opt["tb"]}\nBS={opt["bs"]}')

        x = np.arange(len(values))
        bars = ax.bar(x, values, color=colors)

        # Compute EB-vs-v1 improvement
        v1_val_opt = optimal.get("v1", {}).get("metrics", {}).get(metric, 0)
        # Find best PD (dynamically detect all schedulers starting with pd_)
        best_eb_val = 0
        for key, opt_data in optimal.items():
            if key.startswith("pd_"):
                val = opt_data["metrics"].get(metric, 0)
                if val > best_eb_val:
                    best_eb_val = val

        if v1_val > 0 and best_eb_val > 0:
            if higher_better:
                improvement = (best_eb_val - v1_val) / v1_val * 100
            else:
                improvement = (v1_val - best_eb_val) / v1_val * 100
            imp_str = f"Best PD vs Base: {improvement:+.1f}%"
            ax.annotate(imp_str, xy=(0.5, 0.95), xycoords='axes fraction',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       color='green' if improvement > 0 else 'red')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(ylabel, fontsize=10)

        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=7)

        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle('Optimal Configuration Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "optimal_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(data: Dict, optimal: Dict) -> str:
    """Generate text analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("TB × BS grid search analysis report (real dataset)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"TB values: {data['tb_values']}")
    lines.append(f"BS values: {data['bs_values']}")
    lines.append(f"Completed experiments: {len([k for k, v in data['results'].items() if v])}")
    lines.append("")

    # Optimal configurations
    if optimal:
        lines.append("=" * 80)
        lines.append("Optimal configurations (by throughput)")
        lines.append("=" * 80)
        lines.append("")

        # Dynamically list optimal configs for all schedulers
        for scheduler in sorted(optimal.keys()):
            opt = optimal[scheduler]
            lines.append(f"  {scheduler:>15}: TB={opt['tb']}, BS={opt['bs']}, "
                       f"throughput={opt['metrics']['throughput']:.2f} req/s")

        lines.append("")

        # Comparison table: v1 and all eb_* schedulers
        available = [(s, optimal[s]) for s in sorted(optimal.keys())]
        if len(available) >= 2:
            lines.append("-" * 80)
            header = f"{'Metric':<25}"
            for sched, _ in available:
                header += f" {sched:<15}"
            header += " Improvement"
            lines.append(header)
            lines.append("-" * 80)

            metrics_compare = [
                ("throughput", "Throughput (req/s)", True),
                ("output_throughput", "Output (tok/s)", True),
                ("mean_itl_ms", "Mean ITL (ms)", False),
                ("mean_ttft_ms", "Mean TTFT (ms)", False),
                ("p99_itl_ms", "P99 ITL (ms)", False),
            ]

            for metric, metric_name, higher_better in metrics_compare:
                row = f"{metric_name:<25}"
                vals = []
                for sched, opt in available:
                    val = opt["metrics"].get(metric, 0)
                    vals.append(val)
                    row += f" {val:<15.2f}"

                # Compute improvement (v1 vs best EB)
                v1_val_opt = optimal.get("v1", {}).get("metrics", {}).get(metric, 0)
                best_eb_val = 0
                for s, opt_data in optimal.items():
                    if s.startswith("pd_"):
                        v = opt_data["metrics"].get(metric, 0)
                        if v > best_eb_val:
                            best_eb_val = v

                if v1_val > 0 and best_eb_val > 0:
                    if higher_better:
                        imp = (best_eb_val - v1_val) / v1_val * 100
                    else:
                        imp = (v1_val - best_eb_val) / v1_val * 100
                    row += f" {imp:+.2f}%"

                lines.append(row)

        lines.append("")

    # Conclusion
    lines.append("=" * 80)
    lines.append("Conclusion")
    lines.append("=" * 80)

    if optimal:
        throughputs = {s: optimal[s]["metrics"]["throughput"] for s in optimal}
        winner = max(throughputs, key=throughputs.get)
        lines.append(f"Highest throughput: {winner} ({throughputs[winner]:.2f} req/s)")

        if "v1" in throughputs:
            for pd in sorted(throughputs.keys()):
                if pd.startswith("pd_"):
                    imp = (throughputs[pd] - throughputs["v1"]) / throughputs["v1"] * 100
                    lines.append(f"  {eb_val} vs v1: {imp:+.2f}%")

    lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_grid_search_real.py <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f"Directory not found: {exp_dir}")
        sys.exit(1)

    print(f"Analyzing experiment directory: {exp_dir}")
    print("")

    # Collect data
    data = collect_grid_results(exp_dir)

    if not data["results"]:
        print("No experiment results found")
        sys.exit(1)

    # Count completed experiments
    completed = sum(1 for v in data["results"].values() if v)
    total = len(data["tb_values"]) * len(data["bs_values"])

    print(f"Found {len(data['tb_values'])} TB values: {data['tb_values']}")
    print(f"Found {len(data['bs_values'])} BS values: {data['bs_values']}")
    print(f"Completed experiments: {completed}/{total}")
    print("")

    # Find optimal configurations
    optimal = find_optimal_configs(data)

    # Generate report
    report = generate_report(data, optimal)
    print(report)

    # Save report
    report_path = exp_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # Plots
    plot_heatmaps(data, exp_dir)
    plot_optimal_comparison(optimal, exp_dir)

    # Save summary JSON
    summary = {
        "tb_values": data["tb_values"],
        "bs_values": data["bs_values"],
        "optimal": optimal,
        "all_results": {
            f"tb{tb}_bs{bs}": sched_results
            for (tb, bs), sched_results in data["results"].items()
        }
    }

    summary_path = exp_dir / "grid_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
