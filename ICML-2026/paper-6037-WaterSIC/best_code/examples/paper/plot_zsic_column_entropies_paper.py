#!/usr/bin/env python3
"""Plot ZSIC column entropy distribution for paper figures.

Two-panel histogram: single layer (left) and all layers (right).
Outputs SVG + PDF for LaTeX inclusion.

Usage:
    python scripts/plot_zsic_column_entropies_paper.py \
        --run_dir /path/to/zsic_run \
        --layer layers.18.attention.wo \
        --output zsic_column_entropies_qwen3_r2.svg

    # With custom model label:
    python scripts/plot_zsic_column_entropies_paper.py \
        --run_dir /path/to/zsic_run \
        --layer layers.18.attention.wo \
        --model_label "Qwen3-8B" \
        --output figures/zsic_column_entropies.svg
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_layerwise.storage import LayerArtifact, RunManifest


def compute_column_entropies(Z: torch.Tensor) -> np.ndarray:
    """Compute entropy (bits) for each column of Z tensor."""
    n_rows, n_cols = Z.shape
    entropies = np.zeros(n_cols)
    for col in range(n_cols):
        unique_vals, counts = torch.unique(Z[:, col], return_counts=True)
        probs = counts.float() / n_rows
        entropies[col] = -torch.sum(probs * torch.log2(probs)).item()
    return entropies


def main():
    parser = argparse.ArgumentParser(description="Plot ZSIC column entropy histograms (paper quality)")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--layer", type=str, default="layers.18.attention.wo",
                        help="Module name for single-layer panel (e.g. layers.18.attention.wo)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: run_dir/zsic_column_entropies.svg)")
    parser.add_argument("--model_label", type=str, default=None,
                        help="Model label for suptitle (default: from manifest)")
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = RunManifest.load(run_dir / "manifest.json")
    model_label = args.model_label or manifest.model_name

    all_layer_names = sorted(manifest.artifacts.keys())
    print(f"Run: {run_dir.name}")
    print(f"Model: {model_label}, {len(all_layer_names)} layers")

    # --- Single layer ---
    layer_name = args.layer
    if layer_name not in manifest.artifacts:
        print(f"ERROR: '{layer_name}' not in manifest. Available layers with 'wo':")
        for n in all_layer_names:
            if "wo" in n:
                print(f"  {n}")
        sys.exit(1)

    relpath = manifest.artifact_relpath_for_rank(layer_name, rank=0)
    art = LayerArtifact.load(run_dir / relpath, map_location="cpu")
    Z_single = art.payload["Z"]
    print(f"Single layer: {layer_name}  Z shape: {list(Z_single.shape)}")

    entropies_single = compute_column_entropies(Z_single)
    mean_single = entropies_single.mean()
    print(f"  {len(entropies_single)} columns, mean={mean_single:.2f} bits")

    # --- All layers ---
    all_entropies = []
    for i, name in enumerate(all_layer_names):
        relpath = manifest.artifact_relpath_for_rank(name, rank=0)
        a = LayerArtifact.load(run_dir / relpath, map_location="cpu")
        Z = a.payload.get("Z")
        if Z is None:
            continue
        ent = compute_column_entropies(Z)
        all_entropies.append(ent)
        if (i + 1) % 25 == 0 or i == len(all_layer_names) - 1:
            print(f"  [{i+1}/{len(all_layer_names)}]", flush=True)

    all_entropies = np.concatenate(all_entropies)
    mean_all = all_entropies.mean()
    print(f"All layers: {len(all_entropies):,} columns, mean={mean_all:.2f} bits")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: single layer
    ax1.hist(entropies_single, bins=args.bins, color="steelblue", edgecolor="black", linewidth=0.5)
    ax1.axvline(mean_single, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {mean_single:.2f} bits")
    ax1.set_xlabel("Column Entropy (bits)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title(f"{layer_name}\n({len(entropies_single):,} columns)", fontsize=10.5)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="gray")
    ax1.grid(True, alpha=0.25, color="gray")
    ax1.set_axisbelow(True)

    # Right: all layers
    ax2.hist(all_entropies, bins=args.bins, color="orange", edgecolor="black", linewidth=0.5)
    ax2.axvline(mean_all, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {mean_all:.2f} bits")
    ax2.set_xlabel("Column Entropy (bits)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title(f"All Layers ({len(all_layer_names)} layers)\n({len(all_entropies):,} columns)", fontsize=10.5)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="gray")
    ax2.grid(True, alpha=0.25, color="gray")
    ax2.set_axisbelow(True)

    fig.suptitle(f"ZSIC Column Entropy Distribution \u2014 {model_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save
    output = Path(args.output) if args.output else run_dir / "zsic_column_entropies.svg"
    output.parent.mkdir(parents=True, exist_ok=True)

    svg_path = output.with_suffix(".svg")
    pdf_path = output.with_suffix(".pdf")

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"Saved: {svg_path}")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
