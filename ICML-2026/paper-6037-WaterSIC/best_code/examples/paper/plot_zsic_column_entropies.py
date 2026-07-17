"""Plot histograms of per-column entropies from ZSIC quantization.

Shows the distribution of compression rates across columns, illustrating
where the rate savings come from in adaptive quantization.

Usage:
    python scripts/plot_zsic_column_entropies.py \
        --run_dir /path/to/zsic_run \
        --output entropy_hist.png \
        --layer_idx 18  # middle layer for single-layer plot
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_layerwise.storage import LayerArtifact, RunManifest


def compute_column_entropies(Z: torch.Tensor) -> np.ndarray:
    """Compute entropy for each column of the Z tensor (vectorized).

    Args:
        Z: Integer quantization tensor, shape (out_features, in_features)

    Returns:
        Array of entropies, one per column (in_features,)
    """
    n_rows, n_cols = Z.shape
    Z_flat = Z.T.reshape(n_cols, -1)  # (n_cols, n_rows)

    # Vectorized: compute entropy for all columns at once
    entropies = np.zeros(n_cols)
    for col in range(n_cols):
        unique_vals, counts = torch.unique(Z_flat[col], return_counts=True)
        probs = counts.float() / n_rows
        entropies[col] = -torch.sum(probs * torch.log2(probs)).item()

    return entropies


def load_layer_artifact(run_dir: Path, module_name: str) -> Optional[LayerArtifact]:
    """Load a layer artifact from run directory."""
    manifest = RunManifest.load(run_dir / "manifest.json")
    if module_name not in manifest.artifacts:
        return None
    artifact_path = run_dir / manifest.artifacts[module_name]
    return LayerArtifact.load(artifact_path)


def get_all_zsic_layers(run_dir: Path) -> List[str]:
    """Get all ZSIC layer names from a run."""
    manifest = RunManifest.load(run_dir / "manifest.json")
    return sorted(manifest.artifacts.keys())


def plot_entropy_histograms(
    run_dir: Path,
    output_path: Path,
    layer_idx: int = 18,
    figsize: Tuple[float, float] = (14, 5),
    log_scale: bool = False,
    bins: int = 50,
    sample_layers: int = 0,  # Sample N layers for "all layers" plot (0 = all)
):
    """Plot entropy histograms: single layer + all layers."""

    run_dir = Path(run_dir)
    manifest = RunManifest.load(run_dir / "manifest.json")

    print(f"Run: {run_dir.name}")
    print(f"Model: {manifest.model_name}")
    print(f"Config: rate={manifest.config.get('target_rate_bits', 'N/A')}")

    all_layers = get_all_zsic_layers(run_dir)
    print(f"Found {len(all_layers)} quantized layers")

    # Find a middle layer (attention.wo or feed_forward.w2 to show output projection)
    middle_layer_name = None
    for name in all_layers:
        if f"layers.{layer_idx}." in name and ("wo" in name or "w2" in name):
            middle_layer_name = name
            break

    if middle_layer_name is None:
        # Fallback to any layer at layer_idx
        for name in all_layers:
            if f"layers.{layer_idx}." in name:
                middle_layer_name = name
                break

    if middle_layer_name is None:
        middle_layer_name = all_layers[len(all_layers) // 2]

    print(f"Single layer: {middle_layer_name}")

    # Load single layer
    artifact_single = load_layer_artifact(run_dir, middle_layer_name)
    if artifact_single is None or artifact_single.payload is None:
        raise ValueError(f"Could not load artifact for {middle_layer_name}")

    Z_single = artifact_single.payload.get("Z")
    if Z_single is None:
        raise ValueError(f"No Z tensor in artifact for {middle_layer_name}")

    print(f"  Shape: {Z_single.shape}")
    entropies_single = compute_column_entropies(Z_single)
    print(f"  Column entropies: min={entropies_single.min():.3f}, "
          f"mean={entropies_single.mean():.3f}, max={entropies_single.max():.3f}")

    # Load layers and aggregate entropies
    if sample_layers > 0 and sample_layers < len(all_layers):
        indices = np.linspace(0, len(all_layers) - 1, sample_layers, dtype=int)
        sampled_layers = [all_layers[i] for i in indices]
        print(f"Sampling {len(sampled_layers)} layers...")
    else:
        sampled_layers = all_layers
        print(f"Loading all {len(sampled_layers)} layers...")

    all_entropies = []
    n_total = len(sampled_layers)
    print_every = max(1, n_total // 20)  # Print ~20 progress updates
    for i, layer_name in enumerate(sampled_layers):
        artifact = load_layer_artifact(run_dir, layer_name)
        if artifact is None or artifact.payload is None:
            continue
        Z = artifact.payload.get("Z")
        if Z is None:
            continue
        entropies = compute_column_entropies(Z)
        all_entropies.extend(entropies.tolist())
        if (i + 1) % print_every == 0 or i == n_total - 1:
            print(f"  [{i + 1}/{n_total}] {layer_name}", flush=True)

    all_entropies = np.array(all_entropies)
    print(f"\nSampled layers ({len(sampled_layers)}/{len(all_layers)} layers, {len(all_entropies)} columns):")
    print(f"  Column entropies: min={all_entropies.min():.3f}, "
          f"mean={all_entropies.mean():.3f}, max={all_entropies.max():.3f}")

    # Compute target rate for reference (try multiple locations)
    target_rate = manifest.config.get("target_rate_bits", None)
    if target_rate is None and "zsic" in manifest.config:
        target_rate = manifest.config["zsic"].get("target_rate_bits", None)
    # Also try parsing from run name (e.g., "qwen3-8B.zsic.rescomp.r2.12")
    if target_rate is None:
        import re
        match = re.search(r'\.r(\d+\.\d+)', run_dir.name)
        if match:
            target_rate = float(match.group(1))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white')

    # Left: single layer histogram
    ax1 = axes[0]
    ax1.set_facecolor('white')

    counts1, bin_edges1, _ = ax1.hist(
        entropies_single, bins=bins, color='tab:blue', alpha=0.7,
        edgecolor='black', linewidth=0.5
    )

    ax1.axvline(entropies_single.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {entropies_single.mean():.2f} bits')
    if target_rate:
        ax1.axvline(target_rate, color='green', linestyle=':', linewidth=2,
                    label=f'Target: {target_rate:.2f} bits')

    ax1.set_xlabel("Column Entropy (bits)", fontsize=12)
    ax1.set_ylabel("Count" if not log_scale else "Count (log)", fontsize=12)
    ax1.set_title(f"Single Layer: {middle_layer_name}\n({Z_single.shape[1]} columns)", fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    if log_scale:
        ax1.set_yscale('log')

    # Right: all layers histogram
    ax2 = axes[1]
    ax2.set_facecolor('white')

    counts2, bin_edges2, _ = ax2.hist(
        all_entropies, bins=bins, color='tab:orange', alpha=0.7,
        edgecolor='black', linewidth=0.5
    )

    ax2.axvline(all_entropies.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {all_entropies.mean():.2f} bits')
    if target_rate:
        ax2.axvline(target_rate, color='green', linestyle=':', linewidth=2,
                    label=f'Target: {target_rate:.2f} bits')

    ax2.set_xlabel("Column Entropy (bits)", fontsize=12)
    ax2.set_ylabel("Count" if not log_scale else "Count (log)", fontsize=12)
    n_sampled = len(sampled_layers)
    if n_sampled == len(all_layers):
        ax2.set_title(f"All Layers ({n_sampled} layers)\n({len(all_entropies)} columns)", fontsize=11)
    else:
        ax2.set_title(f"Sampled Layers ({n_sampled}/{len(all_layers)})\n({len(all_entropies)} columns)", fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    if log_scale:
        ax2.set_yscale('log')

    # Main title
    rate_str = f"Rate={target_rate:.2f}" if target_rate else ""
    fig.suptitle(f"ZSIC Column Entropy Distribution - {manifest.model_name} {rate_str}",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved: {output_path}")
    else:
        plt.show()

    plt.close(fig)

    # Save stats to JSON
    stats = {
        "run_dir": str(run_dir),
        "model_name": manifest.model_name,
        "target_rate": target_rate,
        "single_layer": {
            "name": middle_layer_name,
            "shape": list(Z_single.shape),
            "n_columns": int(Z_single.shape[1]),
            "entropy_min": float(entropies_single.min()),
            "entropy_mean": float(entropies_single.mean()),
            "entropy_max": float(entropies_single.max()),
            "entropy_std": float(entropies_single.std()),
        },
        "all_layers": {
            "n_layers_total": len(all_layers),
            "n_layers_sampled": len(sampled_layers),
            "n_columns": len(all_entropies),
            "entropy_min": float(all_entropies.min()),
            "entropy_mean": float(all_entropies.mean()),
            "entropy_max": float(all_entropies.max()),
            "entropy_std": float(all_entropies.std()),
        },
    }

    json_path = output_path.with_suffix('.json') if output_path else run_dir / "column_entropy_stats.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {json_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Plot ZSIC column entropy histograms")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to ZSIC quantized run directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output plot path (default: run_dir/column_entropies.png)")
    parser.add_argument("--layer_idx", type=int, default=18,
                        help="Layer index for single-layer plot (default: 18 = middle for 36-layer model)")
    parser.add_argument("--log_scale", action="store_true",
                        help="Use logarithmic y-axis scale")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of histogram bins")
    parser.add_argument("--figsize", type=str, default="14,5",
                        help="Figure size as 'width,height'")
    parser.add_argument("--sample_layers", type=int, default=0,
                        help="Sample N layers for all-layers histogram (0=all)")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "column_entropies.png"
    figsize = tuple(map(float, args.figsize.split(",")))

    plot_entropy_histograms(
        run_dir=run_dir,
        output_path=output_path,
        layer_idx=args.layer_idx,
        figsize=figsize,
        log_scale=args.log_scale,
        bins=args.bins,
        sample_layers=args.sample_layers,
    )


if __name__ == "__main__":
    main()
