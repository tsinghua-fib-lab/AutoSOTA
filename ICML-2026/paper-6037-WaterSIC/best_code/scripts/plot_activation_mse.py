"""Plot activation MSE (||X - X̂||) per layer from Qronos statistics.

The relative MSE is computed from Qronos matrices:
  MSE = tr(Sig_X) - 2*tr(Sig_X_hX) + tr(Sig_hX)
  relMSE = sqrt(MSE) / sqrt(tr(Sig_X))

This measures how much the activations have drifted due to quantization
of previous layers.

Usage:
    python scripts/plot_activation_mse.py --run_dir /path/to/run
    python scripts/plot_activation_mse.py --run_dir /path/to/run --output plot.png
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_qronos_stats(qronos_dir: Path) -> Dict[str, Dict]:
    """Load all Qronos stats from a directory."""
    stats = {}
    for pkl_file in sorted(qronos_dir.glob("*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        # Extract layer name from filename (e.g., "layers.0.attention.wq.pkl" -> "layers.0.attention.wq")
        name = pkl_file.stem
        stats[name] = data
    return stats


def compute_activation_mse(stats: Dict) -> Tuple[float, float, float]:
    """Compute activation MSE from Qronos stats.

    Returns:
        (mse, rel_mse, correlation): Raw MSE, relative MSE, and correlation
    """
    Sig_X = stats["Sig_X"].double()
    Sig_hX = stats["Sig_hX"].double()
    Sig_X_hX = stats["Sig_X_hX"].double()

    tr_X = Sig_X.trace().item()
    tr_hX = Sig_hX.trace().item()
    tr_X_hX = Sig_X_hX.trace().item()

    # MSE = E[||X - X̂||²] = tr(Sig_X) - 2*tr(Sig_X_hX) + tr(Sig_hX)
    mse = tr_X - 2 * tr_X_hX + tr_hX

    # Handle numerical issues (MSE should be >= 0)
    mse = max(0, mse)

    # Relative MSE = sqrt(MSE) / sqrt(tr(Sig_X))
    if tr_X > 0:
        rel_mse = np.sqrt(mse) / np.sqrt(tr_X)
    else:
        rel_mse = 0.0

    # Correlation = tr(Sig_X_hX) / sqrt(tr(Sig_X) * tr(Sig_hX))
    if tr_X > 0 and tr_hX > 0:
        correlation = tr_X_hX / np.sqrt(tr_X * tr_hX)
    else:
        correlation = 0.0

    return mse, rel_mse, correlation


def parse_layer_name(name: str) -> Tuple[int, str, str]:
    """Parse layer name like 'layers.0.attention.wq' into (layer_id, block_type, weight_type)."""
    parts = name.split(".")
    # Expected format: layers.{id}.{block_type}.{weight_type}
    if len(parts) >= 4 and parts[0] == "layers":
        layer_id = int(parts[1])
        block_type = parts[2]  # "attention" or "feed_forward"
        weight_type = parts[3]  # "wq", "wk", "wv", "wo", "w1", "w3", "w2"
        return layer_id, block_type, weight_type
    return -1, "", name


def get_sort_key(name: str) -> Tuple[int, int, int]:
    """Get sort key for layer ordering (by layer_id, then block, then weight)."""
    layer_id, block_type, weight_type = parse_layer_name(name)

    # Block order: attention before feed_forward
    block_order = 0 if block_type == "attention" else 1

    # Weight order within block
    weight_order_map = {
        "wq": 0, "wk": 1, "wv": 2, "wo": 3,  # attention
        "w1": 0, "w3": 1, "w2": 2,  # feed_forward (w2 last, depends on w1 & w3)
    }
    weight_order = weight_order_map.get(weight_type, 99)

    return (layer_id, block_order, weight_order)


def short_name(name: str, merge_inputs: bool = False) -> str:
    """Convert 'layers.0.attention.wq' to 'L0_wq'.

    If merge_inputs=True, merge weights with identical inputs:
      - wq, wk, wv -> qkv
      - w1, w3 -> w1w3
    """
    layer_id, block_type, weight_type = parse_layer_name(name)
    if layer_id >= 0:
        if merge_inputs:
            if weight_type in ("wq", "wk", "wv"):
                return f"L{layer_id}_qkv"
            elif weight_type in ("w1", "w3"):
                return f"L{layer_id}_w1w3"
        return f"L{layer_id}_{weight_type}"
    return name


def get_merged_weight_type(weight_type: str) -> str:
    """Get merged weight type for grouping."""
    if weight_type in ("wq", "wk", "wv"):
        return "qkv"
    elif weight_type in ("w1", "w3"):
        return "w1w3"
    return weight_type


def get_weight_type_color(weight_type: str) -> str:
    """Get color for weight type using tab palette."""
    color_map = {
        # Attention weights (original)
        "wq": "tab:blue",
        "wk": "tab:red",
        "wv": "#E377C2",  # pink/magenta
        "wo": "tab:cyan",
        # FFN weights (original)
        "w1": "tab:orange",
        "w2": "tab:green",
        "w3": "tab:purple",
        # Merged types
        "qkv": "tab:blue",
        "w1w3": "tab:orange",
    }
    return color_map.get(weight_type, "tab:gray")


def _load_cached_metrics(run_dir: Path) -> Dict[str, float]:
    """Try to load activation_metrics_cache_*.pt from compute_activation_mse.py."""
    cache_files = sorted(run_dir.glob("activation_metrics_cache_*.pt"))
    if not cache_files:
        return None
    cache_path = cache_files[0]  # take first available
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    print(f"Loaded cached metrics from {cache_path.name}")
    return data["mse"]  # dict: module_name -> relative MSE float


def _prepare_plot_data(
    run_dir: Path,
    merge_inputs: bool = True,
):
    """Load Qronos stats or cached metrics and compute relMSE for all layers.

    Returns (results, plot_results) where plot_results may be merged.
    """
    qronos_dir = run_dir / "qronos_stats"
    cached_mse = None

    if qronos_dir.exists():
        all_stats = load_qronos_stats(qronos_dir)
        if all_stats:
            print(f"Loaded {len(all_stats)} layer stats from {qronos_dir}")
            results = []
            for name, stats in all_stats.items():
                mse, rel_mse, corr = compute_activation_mse(stats)
                layer_id, block_type, weight_type = parse_layer_name(name)
                results.append({
                    "name": name,
                    "short_name": short_name(name, merge_inputs=merge_inputs),
                    "mse": mse,
                    "rel_mse": rel_mse,
                    "correlation": corr,
                    "sort_key": get_sort_key(name),
                    "weight_type": weight_type,
                    "merged_type": get_merged_weight_type(weight_type) if merge_inputs else weight_type,
                    "layer_id": layer_id,
                })
        else:
            cached_mse = _load_cached_metrics(run_dir)
    else:
        cached_mse = _load_cached_metrics(run_dir)

    if cached_mse is not None:
        # Build results from cached relative MSE dict
        results = []
        for name, rel_mse_val in cached_mse.items():
            layer_id, block_type, weight_type = parse_layer_name(name)
            # Skip norm layers for consistency with qronos-based plotting
            if weight_type in ("attention_norm", "ffn_norm"):
                continue
            results.append({
                "name": name,
                "short_name": short_name(name, merge_inputs=merge_inputs),
                "mse": 0.0,
                "rel_mse": rel_mse_val,
                "correlation": 0.0,
                "sort_key": get_sort_key(name),
                "weight_type": weight_type,
                "merged_type": get_merged_weight_type(weight_type) if merge_inputs else weight_type,
                "layer_id": layer_id,
            })
        print(f"Using {len(results)} layers from cached metrics")
    elif not qronos_dir.exists() or not results:
        print(f"Error: No qronos_stats/ or activation_metrics_cache_*.pt in {run_dir}")
        return None, None

    results.sort(key=lambda x: x["sort_key"])

    if merge_inputs:
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in results:
            key = (r["layer_id"], r["merged_type"])
            grouped[key].append(r)

        merged_results = []
        for (layer_id, merged_type), items in grouped.items():
            avg_rel_mse = np.mean([r["rel_mse"] for r in items])
            avg_corr = np.mean([r["correlation"] for r in items])
            merged_results.append({
                "short_name": f"L{layer_id}_{merged_type}",
                "rel_mse": avg_rel_mse,
                "correlation": avg_corr,
                "weight_type": merged_type,
                "layer_id": layer_id,
                "sort_key": (layer_id, 0 if merged_type in ("qkv",) else 1 if merged_type == "wo" else 2 if merged_type == "w1w3" else 3),
            })

        merged_results.sort(key=lambda x: x["sort_key"])
        plot_results = merged_results
    else:
        plot_results = results

    return results, plot_results


def _scatter_on_ax(ax, plot_results, merge_inputs: bool = True):
    """Draw scatter plot on a single axes object."""
    names = [r["short_name"] for r in plot_results]
    rel_mses = [r["rel_mse"] * 100 for r in plot_results]
    weight_types = [r["weight_type"] for r in plot_results]
    x = np.arange(len(names))

    weight_type_order = ["qkv", "wo", "w1w3", "w2"] if merge_inputs else ["wq", "wk", "wv", "wo", "w1", "w3", "w2"]

    for wt in weight_type_order:
        indices = [i for i, w in enumerate(weight_types) if w == wt]
        if indices:
            color = get_weight_type_color(wt)
            ax.scatter(
                [x[i] for i in indices],
                [rel_mses[i] for i in indices],
                c=color, s=55, alpha=0.8, label=wt, edgecolors='none', zorder=3,
            )

    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=1, color='lightgray')
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.tick_params(axis='y', labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')


def _print_summary(results, plot_results, merge_inputs, output_path, run_dir):
    """Print summary stats and save JSON."""
    rel_mses = [r["rel_mse"] * 100 for r in plot_results]
    correlations = [r["correlation"] for r in plot_results]

    print("\nSummary:")
    print(f"  Data points: {len(plot_results)}" + (f" (merged from {len(results)} layers)" if merge_inputs else ""))
    print(f"  Rel MSE: min={min(rel_mses):.2f}%, max={max(rel_mses):.2f}%, mean={np.mean(rel_mses):.2f}%")
    print(f"  Correlation: min={min(correlations):.4f}, max={max(correlations):.4f}, mean={np.mean(correlations):.4f}")

    high_mse = [(r["short_name"], r["rel_mse"]*100) for r in plot_results if r["rel_mse"] > 0.1]
    if high_mse:
        print("\n  High MSE layers (>10%):")
        for name, mse in sorted(high_mse, key=lambda x: -x[1])[:10]:
            print(f"    {name}: {mse:.2f}%")

    low_corr = [(r["short_name"], r["correlation"]) for r in plot_results if r["correlation"] < 0.99]
    if low_corr:
        print("\n  Low correlation layers (<0.99):")
        for name, corr in sorted(low_corr, key=lambda x: x[1])[:10]:
            print(f"    {name}: {corr:.4f}")

    json_path = output_path.with_suffix('.json') if output_path else run_dir / "activation_mse.json"
    results_json = {
        "run_dir": str(run_dir),
        "merged": merge_inputs,
        "layers": [{
            "name": r.get("name", r["short_name"]),
            "short_name": r["short_name"],
            "rel_mse_pct": r["rel_mse"] * 100,
            "correlation": r["correlation"],
        } for r in results],
        "plotted": [{
            "short_name": r["short_name"],
            "rel_mse_pct": r["rel_mse"] * 100,
            "correlation": r["correlation"],
        } for r in plot_results],
        "summary": {
            "num_layers": len(results),
            "num_plotted": len(plot_results),
            "rel_mse_min_pct": min(rel_mses),
            "rel_mse_max_pct": max(rel_mses),
            "rel_mse_mean_pct": float(np.mean(rel_mses)),
            "correlation_min": min(correlations),
            "correlation_max": max(correlations),
            "correlation_mean": float(np.mean(correlations)),
        }
    }
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved JSON data to {json_path}")


def plot_activation_mse(
    run_dir: Path,
    output_path: Path = None,
    title: str = None,
    figsize: Tuple[float, float] = (22, 6.5),
    show_correlation: bool = False,  # Default to single panel
    merge_inputs: bool = True,  # Merge weights with identical inputs (qkv, w1w3)
):
    """Plot activation MSE for all layers in a run (single panel).

    If merge_inputs=True (default), weights with identical inputs are merged:
      - wq, wk, wv -> qkv (average)
      - w1, w3 -> w1w3 (average)
    """
    results, plot_results = _prepare_plot_data(run_dir, merge_inputs)
    if results is None:
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    _scatter_on_ax(ax, plot_results, merge_inputs)

    ax.set_ylabel("Relative Activation MSE (%)", fontsize=13)
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_title(
        title or f"Activation Drift: ||X - X̂|| / ||X|| per Layer  —  {run_dir.name}",
        fontsize=15, fontweight='normal',
    )
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.95,
              edgecolor='lightgray', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close(fig)

    _print_summary(results, plot_results, merge_inputs, output_path, run_dir)
    return results


def plot_activation_mse_split(
    run_dir: Path,
    output_path: Path = None,
    title: str = None,
    n_panels: int = 4,
    merge_inputs: bool = True,
):
    """Plot activation MSE split across multiple panels (for models with many layers).

    Layers are divided evenly across n_panels subplots stacked vertically.
    Each panel shares the same y-axis range for easy comparison.
    """
    results, plot_results = _prepare_plot_data(run_dir, merge_inputs)
    if results is None:
        return None

    # Determine layer range
    layer_ids = sorted(set(r["layer_id"] for r in plot_results))
    n_layers = len(layer_ids)
    layers_per_panel = (n_layers + n_panels - 1) // n_panels  # ceil division

    # Split plot_results into panels by layer_id
    panels = []
    for p in range(n_panels):
        lo = p * layers_per_panel
        hi = min((p + 1) * layers_per_panel, n_layers)
        panel_layer_ids = set(layer_ids[lo:hi])
        panel_data = [r for r in plot_results if r["layer_id"] in panel_layer_ids]
        if panel_data:
            panels.append(panel_data)

    n_panels_actual = len(panels)
    fig, axes = plt.subplots(n_panels_actual, 1, figsize=(22, 5 * n_panels_actual),
                             facecolor='white', squeeze=False)

    # Compute global y-range for shared axis
    all_rel_mses = [r["rel_mse"] * 100 for r in plot_results]
    y_max = max(all_rel_mses) * 1.08

    for i, panel_data in enumerate(panels):
        ax = axes[i, 0]
        ax.set_facecolor('white')
        _scatter_on_ax(ax, panel_data, merge_inputs)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Relative MSE (%)", fontsize=12)

        # Panel range label
        lo_layer = min(r["layer_id"] for r in panel_data)
        hi_layer = max(r["layer_id"] for r in panel_data)
        ax.set_title(f"Layers {lo_layer}–{hi_layer}", fontsize=13, fontweight='normal')

        # Only show legend on first panel
        if i == 0:
            ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.95,
                      edgecolor='lightgray', fontsize=10)

    axes[-1, 0].set_xlabel("Layer", fontsize=12)

    # Suptitle
    fig.suptitle(
        title or f"Activation Drift: ||X - X̂|| / ||X|| per Layer  —  {run_dir.name}",
        fontsize=15, fontweight='normal', y=1.0,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close(fig)

    _print_summary(results, plot_results, merge_inputs, output_path, run_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot activation MSE per layer from Qronos stats")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to quantization run directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot (default: {run_dir}/activation_mse.png)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title")
    parser.add_argument("--show_correlation", action="store_true",
                        help="Show correlation subplot (default: single panel)")
    parser.add_argument("--no_merge", action="store_true",
                        help="Don't merge weights with identical inputs (show all 7 weight types)")
    parser.add_argument("--figsize", type=str, default="22,6.5",
                        help="Figure size as 'width,height' (default: 22,6.5)")
    parser.add_argument("--split", type=int, default=0, metavar="N",
                        help="Split into N vertical panels (e.g. --split 4 for 80-layer models)")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else run_dir / "activation_mse.png"

    if args.split > 1:
        plot_activation_mse_split(
            run_dir=run_dir,
            output_path=output_path,
            title=args.title,
            n_panels=args.split,
            merge_inputs=not args.no_merge,
        )
    else:
        figsize = tuple(map(float, args.figsize.split(",")))
        plot_activation_mse(
            run_dir=run_dir,
            output_path=output_path,
            title=args.title,
            figsize=figsize,
            show_correlation=args.show_correlation,
            merge_inputs=not args.no_merge,
        )


if __name__ == "__main__":
    main()
