"""Analyze per-dimension activation outliers at specific layers.

Finds which hidden dimensions have outlier activations at L6 and L16.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.analyze_activation_outliers \
        --model_name Qwen/Qwen3-8B \
        --layers 6,16 \
        --nsamples 4 \
        --init_dist
"""

import argparse
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_layerwise.data import get_wikitext2, split_dataset, take_nseq
from quant_layerwise.pipeline import ensure_single_process_distributed, load_model_and_tokenizer


class ActivationCapture:
    """Capture activation at a specific point."""

    def __init__(self):
        self.activation = None
        self.handle = None

    def hook(self, module, input, output):
        # Capture input to the module (w2 input = gated activation)
        self.activation = input[0].detach().clone()

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook)

    def remove(self):
        if self.handle:
            self.handle.remove()


def analyze_per_dimension(X: torch.Tensor, name: str, top_k: int = 20):
    """Analyze activation statistics per hidden dimension.

    X shape: [batch, seq, hidden_dim] or [batch * seq, hidden_dim]
    """
    X = X.float()

    # Flatten batch and seq dimensions
    if X.dim() == 3:
        batch, seq, hidden = X.shape
        X_flat = X.reshape(-1, hidden)  # [batch*seq, hidden]
    else:
        X_flat = X
        hidden = X_flat.shape[-1]

    n_tokens = X_flat.shape[0]

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Shape: {list(X.shape)}, n_tokens={n_tokens}, hidden_dim={hidden}")

    # Per-dimension statistics
    dim_mean = X_flat.mean(dim=0)  # [hidden]
    dim_std = X_flat.std(dim=0)    # [hidden]
    dim_abs_max = X_flat.abs().max(dim=0).values  # [hidden]
    dim_abs_mean = X_flat.abs().mean(dim=0)  # [hidden]

    # Per-dimension Frobenius norm contribution
    # ||X[:, j]||^2 for each column j
    dim_sq_sum = (X_flat ** 2).sum(dim=0)  # [hidden]
    total_sq_sum = dim_sq_sum.sum().item()
    dim_frob_pct = dim_sq_sum / total_sq_sum * 100  # percentage of Frobenius norm squared

    # Overall statistics
    global_mean = X_flat.mean().item()
    global_std = X_flat.std().item()
    global_abs_max = X_flat.abs().max().item()
    global_frob_norm = X_flat.norm().item()

    print(f"\nGlobal stats: mean={global_mean:.4f}, std={global_std:.4f}, abs_max={global_abs_max:.4f}, frob_norm={global_frob_norm:.4f}")

    # Find dimensions with largest absolute max values
    top_dims_by_max = torch.argsort(dim_abs_max, descending=True)[:top_k]

    print(f"\nTop {top_k} dimensions by abs_max:")
    print(f"{'Dim':>6} {'abs_max':>12} {'abs_mean':>12} {'mean':>12} {'std':>12} {'max/global':>12}")
    print("-" * 70)

    for dim_idx in top_dims_by_max:
        dim_idx = dim_idx.item()
        ratio = dim_abs_max[dim_idx].item() / global_abs_max if global_abs_max > 0 else 0
        print(f"{dim_idx:>6} {dim_abs_max[dim_idx].item():>12.4f} {dim_abs_mean[dim_idx].item():>12.4f} "
              f"{dim_mean[dim_idx].item():>12.4f} {dim_std[dim_idx].item():>12.4f} {ratio:>12.2%}")

    # Find dimensions with largest std (most variable)
    top_dims_by_std = torch.argsort(dim_std, descending=True)[:top_k]

    print(f"\nTop {top_k} dimensions by std (most variable):")
    print(f"{'Dim':>6} {'std':>12} {'abs_max':>12} {'abs_mean':>12} {'std/global':>12}")
    print("-" * 70)

    for dim_idx in top_dims_by_std:
        dim_idx = dim_idx.item()
        ratio = dim_std[dim_idx].item() / global_std if global_std > 0 else 0
        print(f"{dim_idx:>6} {dim_std[dim_idx].item():>12.4f} {dim_abs_max[dim_idx].item():>12.4f} "
              f"{dim_abs_mean[dim_idx].item():>12.4f} {ratio:>12.2f}x")

    # Histogram of dimension abs_max values
    print("\nHistogram of per-dimension abs_max values:")
    abs_max_np = dim_abs_max.cpu().numpy()

    # Define bins based on percentiles
    percentiles = [0, 50, 90, 95, 99, 99.9, 100]
    pct_values = np.percentile(abs_max_np, percentiles)

    print(f"{'Percentile':>12} {'Value':>12}")
    print("-" * 26)
    for p, v in zip(percentiles, pct_values):
        print(f"{p:>11}% {v:>12.4f}")

    # Count dimensions above various thresholds
    thresholds = [1, 10, 100, 1000, 5000]
    print("\nDimensions with abs_max above threshold:")
    print(f"{'Threshold':>12} {'Count':>8} {'Percentage':>12}")
    print("-" * 34)
    for thresh in thresholds:
        count = (dim_abs_max > thresh).sum().item()
        pct = count / hidden * 100
        print(f"{thresh:>12} {count:>8} {pct:>11.2f}%")

    # Frobenius norm contribution analysis
    top_dims_by_frob = torch.argsort(dim_frob_pct, descending=True)[:top_k]

    print(f"\nTop {top_k} dimensions by Frobenius norm contribution:")
    print(f"{'Dim':>6} {'Frob %':>12} {'Cumul %':>12} {'abs_max':>12} {'abs_mean':>12}")
    print("-" * 58)

    cumulative = 0.0
    for dim_idx in top_dims_by_frob:
        dim_idx = dim_idx.item()
        pct = dim_frob_pct[dim_idx].item()
        cumulative += pct
        print(f"{dim_idx:>6} {pct:>11.4f}% {cumulative:>11.2f}% "
              f"{dim_abs_max[dim_idx].item():>12.4f} {dim_abs_mean[dim_idx].item():>12.4f}")

    # How many dimensions needed to reach X% of Frobenius norm
    sorted_frob_pct, sorted_indices = torch.sort(dim_frob_pct, descending=True)
    cumsum = torch.cumsum(sorted_frob_pct, dim=0)

    print("\nDimensions needed to reach X% of Frobenius norm:")
    for target in [50, 80, 90, 95, 99]:
        n_dims = (cumsum < target).sum().item() + 1
        print(f"  {target}%: {n_dims} dims ({n_dims/hidden*100:.2f}% of all dims)")

    # Return the outlier dimensions for further analysis
    return {
        "top_dims_by_max": top_dims_by_max.tolist(),
        "top_dims_by_std": top_dims_by_std.tolist(),
        "top_dims_by_frob": top_dims_by_frob.tolist(),
        "dim_abs_max": dim_abs_max,
        "dim_std": dim_std,
        "dim_frob_pct": dim_frob_pct,
    }


def analyze_weight_per_row(W: torch.Tensor, name: str, top_k: int = 20):
    """Analyze weight matrix per-row statistics.

    W shape: [out_features, in_features]
    Each row corresponds to one output dimension.
    """
    W = W.float()
    out_features, in_features = W.shape

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Shape: [{out_features}, {in_features}]")

    # Per-row statistics
    row_frob_norm = W.norm(dim=1)  # [out_features] - Frobenius norm of each row
    row_abs_max = W.abs().max(dim=1).values  # [out_features]
    row_abs_mean = W.abs().mean(dim=1)  # [out_features]

    # Total Frobenius norm
    total_frob = W.norm().item()
    row_frob_sq = row_frob_norm ** 2
    total_frob_sq = row_frob_sq.sum().item()
    row_frob_pct = row_frob_sq / total_frob_sq * 100  # % contribution

    print(f"\nTotal Frobenius norm: {total_frob:.4f}")
    print(f"Row norm stats: mean={row_frob_norm.mean().item():.4f}, "
          f"std={row_frob_norm.std().item():.4f}, "
          f"min={row_frob_norm.min().item():.4f}, "
          f"max={row_frob_norm.max().item():.4f}")

    # Top rows by Frobenius norm
    top_rows_by_frob = torch.argsort(row_frob_norm, descending=True)[:top_k]

    print(f"\nTop {top_k} rows by Frobenius norm:")
    print(f"{'Row':>6} {'Frob norm':>12} {'Frob %':>10} {'abs_max':>12} {'abs_mean':>12}")
    print("-" * 56)

    for row_idx in top_rows_by_frob:
        row_idx = row_idx.item()
        print(f"{row_idx:>6} {row_frob_norm[row_idx].item():>12.4f} "
              f"{row_frob_pct[row_idx].item():>9.4f}% "
              f"{row_abs_max[row_idx].item():>12.4f} {row_abs_mean[row_idx].item():>12.4f}")

    # How many rows needed to reach X% of Frobenius norm
    sorted_frob_pct, sorted_indices = torch.sort(row_frob_pct, descending=True)
    cumsum = torch.cumsum(sorted_frob_pct, dim=0)

    print("\nRows needed to reach X% of Frobenius norm:")
    for target in [50, 80, 90, 95, 99]:
        n_rows = (cumsum < target).sum().item() + 1
        print(f"  {target}%: {n_rows} rows ({n_rows/out_features*100:.2f}% of all rows)")

    return {
        "top_rows_by_frob": top_rows_by_frob.tolist(),
        "row_frob_norm": row_frob_norm,
        "row_frob_pct": row_frob_pct,
        "row_abs_max": row_abs_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze per-dimension activation outliers")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--layers", type=str, default="6,16", help="Comma-separated layer indices")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=20, help="Number of top dimensions to show")
    parser.add_argument("--output", type=str, default="activation_outliers.png", help="Output PNG path")
    parser.add_argument("--init_dist", action="store_true")

    args = parser.parse_args()

    layers_to_analyze = [int(x) for x in args.layers.split(",")]

    if args.init_dist:
        ensure_single_process_distributed(local_rank=0, master_port=29500)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, local_rank=0)
    model.eval()

    device = next(model.parameters()).device
    n_layers = len(model.layers)

    print(f"Model has {n_layers} layers")
    print(f"Analyzing layers: {layers_to_analyze}")

    # Prepare data
    print(f"Preparing evaluation data (nsamples={args.nsamples})...")
    eval_tokens = split_dataset(get_wikitext2(tokenizer, split="test"), args.seqlen)
    eval_tokens = take_nseq(eval_tokens, args.nsamples)
    tokens = eval_tokens.to(device)

    # Resize KV caches
    batch_size = tokens.shape[0]
    for layer in model.layers:
        attn = layer.attention
        if hasattr(attn, "cache_k") and attn.cache_k.shape[0] < batch_size:
            old_shape = attn.cache_k.shape
            new_shape = (batch_size, old_shape[1], old_shape[2], old_shape[3])
            attn.cache_k = torch.zeros(new_shape, device=device, dtype=attn.cache_k.dtype)
            attn.cache_v = torch.zeros(new_shape, device=device, dtype=attn.cache_v.dtype)

    # Register hooks for w2 input (gated activation) at specified layers
    captures = {}
    for layer_idx in layers_to_analyze:
        layer = model.layers[layer_idx]
        cap = ActivationCapture()
        cap.register(layer.feed_forward.w2)
        captures[layer_idx] = cap

    # Also capture w1 and w3 outputs to understand where outliers come from
    captures_w1 = {}
    captures_w3 = {}
    for layer_idx in layers_to_analyze:
        layer = model.layers[layer_idx]

        # For w1 output
        cap_w1 = ActivationCapture()
        cap_w1.handle = layer.feed_forward.w1.register_forward_hook(
            lambda m, inp, out, idx=layer_idx: setattr(captures_w1[idx], 'activation', out.detach().clone())
        )
        captures_w1[layer_idx] = cap_w1
        captures_w1[layer_idx].activation = None

        # For w3 output
        cap_w3 = ActivationCapture()
        cap_w3.handle = layer.feed_forward.w3.register_forward_hook(
            lambda m, inp, out, idx=layer_idx: setattr(captures_w3[idx], 'activation', out.detach().clone())
        )
        captures_w3[layer_idx] = cap_w3
        captures_w3[layer_idx].activation = None

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        _ = model(tokens, start_pos=0)

    # Remove hooks
    for cap in captures.values():
        cap.remove()
    for cap in captures_w1.values():
        if cap.handle:
            cap.handle.remove()
    for cap in captures_w3.values():
        if cap.handle:
            cap.handle.remove()

    # Collect weight matrices for analysis
    weight_data = {}
    for layer_idx in layers_to_analyze:
        layer = model.layers[layer_idx]
        weight_data[layer_idx] = {
            "w1": layer.feed_forward.w1.weight.data.clone(),
            "w3": layer.feed_forward.w3.weight.data.clone(),
            "w2": layer.feed_forward.w2.weight.data.clone(),
        }

    # Analyze each layer
    all_results = {}
    for layer_idx in layers_to_analyze:
        print(f"\n\n{'#'*70}")
        print(f"# LAYER {layer_idx}")
        print(f"{'#'*70}")

        # Analyze w2 input (gated activation = SiLU(w1) * w3)
        w2_input = captures[layer_idx].activation
        if w2_input is not None:
            results = analyze_per_dimension(w2_input, f"L{layer_idx} ffn_w2_input (SiLU(w1)*w3)", args.top_k)
            all_results[f"L{layer_idx}_w2_input"] = results

        # Analyze w1 output
        w1_output = captures_w1[layer_idx].activation
        if w1_output is not None:
            results = analyze_per_dimension(w1_output, f"L{layer_idx} ffn_w1_output", args.top_k)
            all_results[f"L{layer_idx}_w1_output"] = results

        # Analyze w3 output
        w3_output = captures_w3[layer_idx].activation
        if w3_output is not None:
            results = analyze_per_dimension(w3_output, f"L{layer_idx} ffn_w3_output", args.top_k)
            all_results[f"L{layer_idx}_w3_output"] = results

        # Analyze weight matrices (per-row Frobenius norm)
        if layer_idx in weight_data:
            w1_weight = weight_data[layer_idx]["w1"]
            w3_weight = weight_data[layer_idx]["w3"]
            w2_weight = weight_data[layer_idx]["w2"]

            results_w1 = analyze_weight_per_row(w1_weight, f"L{layer_idx} w1 weight", args.top_k)
            all_results[f"L{layer_idx}_w1_weight"] = results_w1

            results_w3 = analyze_weight_per_row(w3_weight, f"L{layer_idx} w3 weight", args.top_k)
            all_results[f"L{layer_idx}_w3_weight"] = results_w3

            results_w2 = analyze_weight_per_row(w2_weight, f"L{layer_idx} w2 weight", args.top_k)
            all_results[f"L{layer_idx}_w2_weight"] = results_w2

    # Compare outlier dimensions between layers
    if len(layers_to_analyze) >= 2:
        print(f"\n\n{'#'*70}")
        print("# COMPARISON: Shared outlier dimensions between layers")
        print(f"{'#'*70}")

        for i, l1 in enumerate(layers_to_analyze):
            for l2 in layers_to_analyze[i+1:]:
                key1 = f"L{l1}_w2_input"
                key2 = f"L{l2}_w2_input"
                if key1 in all_results and key2 in all_results:
                    dims1 = set(all_results[key1]["top_dims_by_max"][:10])
                    dims2 = set(all_results[key2]["top_dims_by_max"][:10])
                    shared = dims1 & dims2
                    print(f"\nL{l1} vs L{l2} top-10 outlier dims:")
                    print(f"  L{l1}: {sorted(dims1)}")
                    print(f"  L{l2}: {sorted(dims2)}")
                    print(f"  Shared: {sorted(shared)} ({len(shared)} dims)")

    # Generate plots
    print("\n\nGenerating plots...")
    create_plots(all_results, layers_to_analyze, args.output)
    print(f"Plots saved to {args.output}")


def create_plots(all_results: dict, layers: list, output_path: str):
    """Create visualization plots for activation outliers."""

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 16))

    # Plot 1: Per-dimension abs_max for w2_input across layers
    ax1 = fig.add_subplot(3, 2, 1)
    for layer_idx in layers:
        key = f"L{layer_idx}_w2_input"
        if key in all_results:
            dim_abs_max = all_results[key]["dim_abs_max"].cpu().numpy()
            # Sort for visualization
            sorted_vals = np.sort(dim_abs_max)[::-1]
            ax1.semilogy(sorted_vals, label=f"L{layer_idx}", alpha=0.8)

    ax1.set_xlabel("Dimension (sorted by abs_max)")
    ax1.set_ylabel("abs_max (log scale)")
    ax1.set_title("Per-dimension abs_max of w2_input (sorted)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of abs_max values
    ax2 = fig.add_subplot(3, 2, 2)
    for layer_idx in layers:
        key = f"L{layer_idx}_w2_input"
        if key in all_results:
            dim_abs_max = all_results[key]["dim_abs_max"].cpu().numpy()
            ax2.hist(dim_abs_max, bins=50, alpha=0.5, label=f"L{layer_idx}", log=True)

    ax2.set_xlabel("abs_max value")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Distribution of per-dimension abs_max")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Top-20 dimensions comparison (bar chart)
    ax3 = fig.add_subplot(3, 2, 3)
    width = 0.35
    x = np.arange(20)

    for i, layer_idx in enumerate(layers[:2]):  # Compare first two layers
        key = f"L{layer_idx}_w2_input"
        if key in all_results:
            dim_abs_max = all_results[key]["dim_abs_max"]
            top_dims = all_results[key]["top_dims_by_max"][:20]
            top_vals = [dim_abs_max[d].item() for d in top_dims]
            offset = (i - 0.5) * width
            bars = ax3.bar(x + offset, top_vals, width, label=f"L{layer_idx}", alpha=0.8)
            # Add dimension labels on top of bars
            for j, (bar, dim) in enumerate(zip(bars, top_dims)):
                if j < 5:  # Only label top 5 to avoid clutter
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'd{dim}', ha='center', va='bottom', fontsize=6, rotation=45)

    ax3.set_xlabel("Rank")
    ax3.set_ylabel("abs_max value")
    ax3.set_title("Top-20 outlier dimensions by abs_max")
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Dimension index vs abs_max (to see if outliers are at specific positions)
    ax4 = fig.add_subplot(3, 2, 4)
    for layer_idx in layers:
        key = f"L{layer_idx}_w2_input"
        if key in all_results:
            dim_abs_max = all_results[key]["dim_abs_max"].cpu().numpy()
            ax4.scatter(range(len(dim_abs_max)), dim_abs_max, s=1, alpha=0.5, label=f"L{layer_idx}")

    ax4.set_xlabel("Dimension index")
    ax4.set_ylabel("abs_max value")
    ax4.set_title("abs_max by dimension index (looking for patterns)")
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Frobenius norm percentage per dimension (sorted)
    ax5 = fig.add_subplot(3, 2, 5)
    for layer_idx in layers:
        key = f"L{layer_idx}_w2_input"
        if key in all_results:
            dim_frob_pct = all_results[key]["dim_frob_pct"].cpu().numpy()
            sorted_vals = np.sort(dim_frob_pct)[::-1]
            ax5.semilogy(sorted_vals, label=f"L{layer_idx}", alpha=0.8)

    ax5.set_xlabel("Dimension (sorted by Frob %)")
    ax5.set_ylabel("% of Frobenius norm (log scale)")
    ax5.set_title("Per-dimension Frobenius norm contribution (sorted)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Cumulative Frobenius norm percentage
    ax6 = fig.add_subplot(3, 2, 6)
    for layer_idx in layers:
        key = f"L{layer_idx}_w2_input"
        if key in all_results:
            dim_frob_pct = all_results[key]["dim_frob_pct"].cpu().numpy()
            sorted_vals = np.sort(dim_frob_pct)[::-1]
            cumsum = np.cumsum(sorted_vals)
            ax6.plot(cumsum, label=f"L{layer_idx}", alpha=0.8)

    ax6.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
    ax6.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='90%')
    ax6.axhline(y=99, color='b', linestyle='--', alpha=0.5, label='99%')
    ax6.set_xlabel("Number of dimensions (sorted by contribution)")
    ax6.set_ylabel("Cumulative % of Frobenius norm")
    ax6.set_title("Cumulative Frobenius norm (how many dims for X%?)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 500)  # Zoom in to first 500 dims

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Create a second figure for w1 and w3 analysis
    fig2 = plt.figure(figsize=(16, 6))

    # Plot w1 output
    ax5 = fig2.add_subplot(1, 2, 1)
    for layer_idx in layers:
        key = f"L{layer_idx}_w1_output"
        if key in all_results:
            dim_abs_max = all_results[key]["dim_abs_max"].cpu().numpy()
            sorted_vals = np.sort(dim_abs_max)[::-1]
            ax5.semilogy(sorted_vals, label=f"L{layer_idx}", alpha=0.8)

    ax5.set_xlabel("Dimension (sorted)")
    ax5.set_ylabel("abs_max (log scale)")
    ax5.set_title("Per-dimension abs_max of w1_output (sorted)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot w3 output
    ax6 = fig2.add_subplot(1, 2, 2)
    for layer_idx in layers:
        key = f"L{layer_idx}_w3_output"
        if key in all_results:
            dim_abs_max = all_results[key]["dim_abs_max"].cpu().numpy()
            sorted_vals = np.sort(dim_abs_max)[::-1]
            ax6.semilogy(sorted_vals, label=f"L{layer_idx}", alpha=0.8)

    ax6.set_xlabel("Dimension (sorted)")
    ax6.set_ylabel("abs_max (log scale)")
    ax6.set_title("Per-dimension abs_max of w3_output (sorted)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path2 = output_path.replace('.png', '_w1w3.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path2}")

    # Create a third figure for weight matrix analysis
    fig3 = plt.figure(figsize=(16, 12))

    # Plot w1 weight per-row Frobenius norm
    ax7 = fig3.add_subplot(2, 3, 1)
    for layer_idx in layers:
        key = f"L{layer_idx}_w1_weight"
        if key in all_results:
            row_frob_norm = all_results[key]["row_frob_norm"].cpu().numpy()
            sorted_vals = np.sort(row_frob_norm)[::-1]
            ax7.semilogy(sorted_vals, label=f"L{layer_idx}", alpha=0.8)

    ax7.set_xlabel("Row (sorted by Frob norm)")
    ax7.set_ylabel("Row Frobenius norm (log)")
    ax7.set_title("w1 weight: per-row Frobenius norm")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot w3 weight per-row Frobenius norm
    ax8 = fig3.add_subplot(2, 3, 2)
    for layer_idx in layers:
        key = f"L{layer_idx}_w3_weight"
        if key in all_results:
            row_frob_norm = all_results[key]["row_frob_norm"].cpu().numpy()
            sorted_vals = np.sort(row_frob_norm)[::-1]
            ax8.semilogy(sorted_vals, label=f"L{layer_idx}", alpha=0.8)

    ax8.set_xlabel("Row (sorted by Frob norm)")
    ax8.set_ylabel("Row Frobenius norm (log)")
    ax8.set_title("w3 weight: per-row Frobenius norm")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot w2 weight per-row Frobenius norm
    ax9 = fig3.add_subplot(2, 3, 3)
    for layer_idx in layers:
        key = f"L{layer_idx}_w2_weight"
        if key in all_results:
            row_frob_norm = all_results[key]["row_frob_norm"].cpu().numpy()
            sorted_vals = np.sort(row_frob_norm)[::-1]
            ax9.semilogy(sorted_vals, label=f"L{layer_idx}", alpha=0.8)

    ax9.set_xlabel("Row (sorted by Frob norm)")
    ax9.set_ylabel("Row Frobenius norm (log)")
    ax9.set_title("w2 weight: per-row Frobenius norm")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Plot w1 weight row index vs Frobenius norm (to see patterns)
    ax10 = fig3.add_subplot(2, 3, 4)
    for layer_idx in layers:
        key = f"L{layer_idx}_w1_weight"
        if key in all_results:
            row_frob_norm = all_results[key]["row_frob_norm"].cpu().numpy()
            ax10.scatter(range(len(row_frob_norm)), row_frob_norm, s=1, alpha=0.5, label=f"L{layer_idx}")

    ax10.set_xlabel("Row index")
    ax10.set_ylabel("Row Frobenius norm")
    ax10.set_title("w1 weight: row index vs Frob norm")
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Plot w3 weight row index vs Frobenius norm
    ax11 = fig3.add_subplot(2, 3, 5)
    for layer_idx in layers:
        key = f"L{layer_idx}_w3_weight"
        if key in all_results:
            row_frob_norm = all_results[key]["row_frob_norm"].cpu().numpy()
            ax11.scatter(range(len(row_frob_norm)), row_frob_norm, s=1, alpha=0.5, label=f"L{layer_idx}")

    ax11.set_xlabel("Row index")
    ax11.set_ylabel("Row Frobenius norm")
    ax11.set_title("w3 weight: row index vs Frob norm")
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # Cumulative Frobenius norm for w1 weights
    ax12 = fig3.add_subplot(2, 3, 6)
    for layer_idx in layers:
        key = f"L{layer_idx}_w1_weight"
        if key in all_results:
            row_frob_pct = all_results[key]["row_frob_pct"].cpu().numpy()
            sorted_vals = np.sort(row_frob_pct)[::-1]
            cumsum = np.cumsum(sorted_vals)
            ax12.plot(cumsum, label=f"L{layer_idx} w1", alpha=0.8)

        key = f"L{layer_idx}_w3_weight"
        if key in all_results:
            row_frob_pct = all_results[key]["row_frob_pct"].cpu().numpy()
            sorted_vals = np.sort(row_frob_pct)[::-1]
            cumsum = np.cumsum(sorted_vals)
            ax12.plot(cumsum, label=f"L{layer_idx} w3", alpha=0.8, linestyle='--')

    ax12.axhline(y=50, color='r', linestyle=':', alpha=0.5)
    ax12.axhline(y=90, color='g', linestyle=':', alpha=0.5)
    ax12.set_xlabel("Number of rows (sorted by contribution)")
    ax12.set_ylabel("Cumulative % of Frobenius norm")
    ax12.set_title("Cumulative row Frob norm (w1 solid, w3 dashed)")
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)
    ax12.set_xlim(0, 1000)

    plt.tight_layout()
    output_path3 = output_path.replace('.png', '_weights.png')
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path3}")


if __name__ == "__main__":
    main()
