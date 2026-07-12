#!/usr/bin/env python3
"""
Analyze TB × BS grid search results from multi-turn benchmarks.

Usage:
    python pd_exp/multiturn/analyze_results.py <experiment_dir>

Outputs:
    - grid_summary.json: full data summary
    - heatmap_{metric}.png: heatmap
    - optimal_comparison.png: optimal configuration comparison plot
    - analysis_report.txt: text analysis report
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping plots")


def load_bench_result(filepath: Path) -> Optional[Dict]:
    """Load a benchmark result file.

    For large files (10MB+), only the first 100KB and last 100KB are read to
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
        import re
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
            'mean_e2e_latency_ms', 'median_e2e_latency_ms', 'p99_e2e_latency_ms',
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
        "mean_e2e_latency_ms": bench_result.get("mean_e2e_latency_ms", 0),
        "median_e2e_latency_ms": bench_result.get("median_e2e_latency_ms", 0),
        "p99_e2e_latency_ms": bench_result.get("p99_e2e_latency_ms", 0),
    }


def collect_grid_results(exp_dir: Path) -> Dict:
    """Collect grid search results.

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

    # Walk directory structure: tb{TB}/bs{BS}/
    for tb_dir in sorted(exp_dir.iterdir()):
        if not tb_dir.is_dir() or not tb_dir.name.startswith("tb"):
            continue

        try:
            tb = int(tb_dir.name[2:])
        except ValueError:
            continue
        tb_values.add(tb)

        for bs_dir in sorted(tb_dir.iterdir()):
            if not bs_dir.is_dir() or not bs_dir.name.startswith("bs"):
                continue

            try:
                bs = int(bs_dir.name[2:])
            except ValueError:
                continue
            bs_values.add(bs)

            key = (tb, bs)
            if key not in results:
                results[key] = {}

            # Dynamically detect all bench_*.json files
            for bench_file in bs_dir.glob("bench_*.json"):
                # Extract scheduler name from filename: bench_eb_1.json -> pd_ifr_1
                scheduler = bench_file.stem[6:]  # strip "bench_" prefix
                bench_result = load_bench_result(bench_file)
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


def compute_improvement_grid(data: Dict, metric: str, eb_scheduler: str = "eb",
                             higher_better: bool = True) -> Tuple[np.ndarray, list, list]:
    """Compute the EB-vs-v1 improvement grid.

    Returns:
        (improvement_matrix, tb_values, bs_values)
    """
    tb_values = data["tb_values"]
    bs_values = data["bs_values"]
    results = data["results"]

    matrix = np.full((len(tb_values), len(bs_values)), np.nan)

    for i, tb in enumerate(tb_values):
        for j, bs in enumerate(bs_values):
            key = (tb, bs)
            if key in results:
                v1_val = results[key].get("v1", {}).get(metric, 0)
                eb_val = results[key].get(eb_scheduler, {}).get(metric, 0)

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

    # Detect available PD schedulers
    pd_schedulers = []
    for key in ["eb", "eb_kratio", "pd"]:
        for sched_results in data["results"].values():
            if key in sched_results:
                pd_schedulers.append(key)
                break

    if not pd_schedulers:
        print("No PD scheduler results found, skipping heatmap")
        return

    for pd_sched in pd_schedulers:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (metric, title, higher_better) in enumerate(metrics_to_plot):
            ax = axes[idx]
            matrix, tb_vals, bs_vals = compute_improvement_grid(
                data, metric, pd_sched, higher_better)

            # Set color range (green=improvement, red=regression)
            vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 10)
            vmin = -vmax

            cmap = plt.cm.RdYlGn
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

            # Set ticks
            ax.set_xticks(range(len(bs_vals)))
            ax.set_xticklabels([str(b) for b in bs_vals], rotation=45)
            ax.set_yticks(range(len(tb_vals)))
            ax.set_yticklabels([str(t) for t in tb_vals])

            ax.set_xlabel('max_num_seqs (BS)')
            ax.set_ylabel('max_num_batched_tokens (TB)')
            ax.set_title(title)

            # Add value annotations
            for i in range(len(tb_vals)):
                for j in range(len(bs_vals)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > vmax * 0.5 else 'black'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                color=color, fontsize=8)

            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.suptitle(f'Multi-Turn: {pd_sched} vs Baseline Improvement', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"heatmap_{pd_sched}.png"
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

    # Scheduler configuration: (key, color, label)
    schedulers = [
        ("v1",        '#2ecc71', 'v1'),
        ("eb_kratio", '#3498db', 'EB (fixed θ*)'),
        ("eb",        '#1abc9c', 'EB(k̂*)'),
        ("ebplus",    '#9b59b6', 'EB+'),
    ]

    # Filter out schedulers with no data
    available_scheds = [(key, color, label) for key, color, label in schedulers if key in optimal]

    if len(available_scheds) < 2:
        print("Fewer than 2 schedulers available, skipping comparison plot")
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
        width = 0.6 if len(values) <= 3 else 0.4
        bars = ax.bar(x, values, width=width, color=colors)

        # Compute EB-vs-v1 improvement
        v1_val_opt = optimal.get("v1", {}).get("metrics", {}).get(metric, 0)
        eb_val_opt = optimal.get("eb", {}).get("metrics", {}).get(metric, 0)
        if v1_val > 0 and eb_val > 0:
            if higher_better:
                improvement = (eb_val - v1_val) / v1_val * 100
            else:
                improvement = (v1_val - eb_val) / v1_val * 100
            imp_str = f"PD(direct) vs Base: {improvement:+.1f}%"
        else:
            imp_str = ""
            improvement = 0

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ylabel, fontsize=10)

        # Annotate values above bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=7)

        # Annotate improvement percentage
        if imp_str:
            ax.annotate(imp_str, xy=(0.5, 0.95), xycoords='axes fraction',
                       ha='center', va='top', fontsize=8, fontweight='bold',
                       color='green' if improvement > 0 else 'red' if improvement < 0 else 'gray')

        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle('Multi-Turn: Optimal Configuration Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "optimal_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(data: Dict, optimal: Dict) -> str:
    """Generate text analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("Multi-Turn TB × BS grid search analysis report")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"TB values: {data['tb_values']}")
    lines.append(f"BS values: {data['bs_values']}")
    lines.append("")

    # Dynamically detect and collect available schedulers
    def get_display_name(key: str) -> str:
        """Convert a scheduler key to a display name."""
        name_map = {
            "v1": "Baseline",
            "eb_kratio": "PD (ratio)",
            "eb": "PD (IFR)",
            "pd": "PD (legacy)",
        }
        if key in name_map:
            return name_map[key]
        # For schedulers with suffixes (e.g. pd_ifr_1), generate a friendly name
        if key.startswith("pd_"):
            return f"PD ({key[3:]})"
        return key

    available = []
    for key in sorted(optimal.keys()):
        available.append((key, get_display_name(key), optimal[key]))

    if len(available) >= 2:
        lines.append("=" * 80)
        lines.append("Optimal configurations")
        lines.append("=" * 80)
        lines.append("")
        for key, display_name, opt in available:
            lines.append(f"  {display_name:>15}: TB={opt['tb']}, BS={opt['bs']}")
        lines.append("")

        # Get optimal config for specific schedulers
        baseline_opt = optimal.get("v1", {})
        pd_ifr_opt = optimal.get("eb", {})
        pd_ratio_opt = optimal.get("eb_kratio", {})

        # Compare key metrics
        header = f"{'Metric':<25}"
        for key, display_name, opt in available:
            header += f" {display_name:<15}"
        if baseline_opt and pd_ifr_opt:
            header += f" {'PD(dir) vs Base':<15}"
        if baseline_opt and pd_ratio_opt:
            header += f" {'PD(rat) vs Base':<15}"
        lines.append(header)

        extra_cols = (1 if baseline_opt and pd_ifr_opt else 0) + (1 if baseline_opt and pd_ratio_opt else 0)
        lines.append("-" * (25 + 15 * len(available) + 15 * extra_cols))

        metrics_compare = [
            ("throughput", "Throughput (req/s)", True),
            ("output_throughput", "Output (tok/s)", True),
            ("mean_itl_ms", "Mean ITL (ms)", False),
            ("mean_ttft_ms", "Mean TTFT (ms)", False),
            ("p99_itl_ms", "P99 ITL (ms)", False),
            ("p99_ttft_ms", "P99 TTFT (ms)", False),
            ("mean_e2e_latency_ms", "Mean E2E (ms)", False),
        ]

        for metric, metric_name, higher_better in metrics_compare:
            row = f"{metric_name:<25}"
            for key, display_name, opt in available:
                val = opt["metrics"].get(metric, 0)
                row += f" {val:<15.2f}"

            # Compute PD(IFR) vs Baseline improvement
            if baseline_opt and pd_ifr_opt:
                b_val = baseline_opt["metrics"].get(metric, 0)
                p_val = pd_ifr_opt["metrics"].get(metric, 0)
                if b_val > 0:
                    if higher_better:
                        improvement = (p_val - b_val) / b_val * 100
                    else:
                        improvement = (b_val - p_val) / b_val * 100
                    imp_str = f"{improvement:+.2f}%"
                else:
                    imp_str = "N/A"
                row += f" {imp_str:<15}"

            # Compute PD(ratio) vs Baseline improvement
            if baseline_opt and pd_ratio_opt:
                b_val = baseline_opt["metrics"].get(metric, 0)
                p_val = pd_ratio_opt["metrics"].get(metric, 0)
                if b_val > 0:
                    if higher_better:
                        improvement = (p_val - b_val) / b_val * 100
                    else:
                        improvement = (b_val - p_val) / b_val * 100
                    imp_str = f"{improvement:+.2f}%"
                else:
                    imp_str = "N/A"
                row += f" {imp_str:<15}"

            lines.append(row)

        # Determine winner (by throughput)
        throughputs = {display_name: opt["metrics"].get("throughput", 0)
                      for key, display_name, opt in available}
        winner = max(throughputs, key=throughputs.get)
        best_tp = throughputs[winner]

        lines.append("")
        lines.append(f"Conclusion: {winner} wins on throughput ({best_tp:.2f} req/s)")

    # Summary
    lines.append("")
    lines.append("=" * 80)
    lines.append("Overall conclusion")
    lines.append("=" * 80)

    # Tally winner (dynamically detect all schedulers starting with pd_)
    baseline_tp = optimal.get("v1", {}).get("metrics", {}).get("throughput", 0)
    best_pd_tp = 0
    best_pd_name = None
    for v, opt_data in optimal.items():
        if v.startswith("pd_"):
            tp = opt_data.get("metrics", {}).get("throughput", 0)
            if tp > best_pd_tp:
                best_pd_tp = tp
                best_pd_name = v

    lines.append("")
    if baseline_tp > 0 and best_pd_tp > 0:
        if best_pd_tp > baseline_tp:
            improvement = (best_pd_tp - baseline_tp) / baseline_tp * 100
            lines.append(f"Overall: {best_pd_name} wins (throughput improvement {improvement:+.2f}%)")
        elif baseline_tp > best_pd_tp:
            improvement = (baseline_tp - best_pd_tp) / baseline_tp * 100
            lines.append(f"Overall: Baseline wins (PD throughput dropped {improvement:.2f}%)")
        else:
            lines.append("Overall: Baseline ties with PD")
    else:
        lines.append("Overall: insufficient data for comparison")

    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <experiment_dir>")
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

    print(f"Found {len(data['tb_values'])} TB values: {data['tb_values']}")
    print(f"Found {len(data['bs_values'])} BS values: {data['bs_values']}")
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
