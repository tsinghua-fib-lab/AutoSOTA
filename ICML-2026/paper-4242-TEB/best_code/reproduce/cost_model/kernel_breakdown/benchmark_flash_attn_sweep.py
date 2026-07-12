"""
FlashAttention Comprehensive Sweep Benchmark

Measures the slope of kernel time vs. token count under different scenarios:
1. Pure Prefill: varying prefill lengths (128, 256, 512, 1024, 2048)
2. Pure Decode: varying context lengths (128, 256, 512, 1024, 2048)
3. Mixed (10%, 20%, 40%, 80%): varying total token counts

Usage:
    python reproduce/cost_model/kernel_breakdown/benchmark_flash_attn_sweep.py \
        --batch-size 512 \
        --output results_sweep.json
"""
import torch
import argparse
from typing import List, Tuple, Dict
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def create_attention_inputs(
    batch_size: int,
    query_lens: List[int],
    context_lens: List[int],
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, ...]:
    """Create input tensors for FlashAttention."""

    total_tokens = sum(query_lens)
    max_context = max(context_lens)
    max_blocks = (max_context + block_size - 1) // block_size

    query = torch.randn(total_tokens, num_heads, head_size,
                       device=device, dtype=dtype)

    num_blocks = batch_size * max_blocks
    kv_cache = torch.randn(2, num_blocks, block_size, num_kv_heads, head_size,
                          device=device, dtype=dtype)

    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(torch.tensor(query_lens, device=device), dim=0)

    seqused_k = torch.tensor(context_lens, dtype=torch.int32, device=device)

    block_table = torch.arange(batch_size * max_blocks, device=device, dtype=torch.int32)
    block_table = block_table.view(batch_size, max_blocks)

    output = torch.empty(total_tokens, num_heads, head_size,
                        device=device, dtype=dtype)

    return (query, kv_cache, cu_seqlens_q, seqused_k, block_table, output,
            max(query_lens), max_context)


def benchmark_single(
    query_lens: List[int],
    context_lens: List[int],
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    block_size: int = 16,
    warmup: int = 5,
    repeat: int = 50,
) -> float:
    """Run a single benchmark; return average time (ms)."""

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    batch_size = len(query_lens)
    query, kv_cache, cu_seqlens_q, seqused_k, block_table, output, max_q, max_k = \
        create_attention_inputs(
            batch_size, query_lens, context_lens,
            num_heads, num_kv_heads, head_size, block_size
        )

    key_cache = kv_cache[0]
    value_cache = kv_cache[1]

    # Warmup
    for _ in range(warmup):
        flash_attn_varlen_func(
            q=query, k=key_cache, v=value_cache, out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_k,
            softmax_scale=1.0 / (head_size ** 0.5),
            causal=True,
            block_table=block_table,
        )

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start_event.record()
        flash_attn_varlen_func(
            q=query, k=key_cache, v=value_cache, out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_k,
            softmax_scale=1.0 / (head_size ** 0.5),
            causal=True,
            block_table=block_table,
        )
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    return sum(times) / len(times)


def run_sweep(args) -> Dict:
    """Run a full sweep test."""

    results = {
        "pure_prefill": [],    # pure prefill, varying lengths
        "pure_decode": [],     # pure decode, varying context lengths
        "mixed": {}            # varying prefill ratios
    }

    # Length sequence under test
    seq_lengths = [128, 256, 512, 1024, 2048]
    prefill_percentages = [10, 20, 40, 80]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Prefill percentages: {prefill_percentages}")

    # ========== 1. Pure Prefill Sweep ==========
    print(f"\n{'='*60}")
    print("1. PURE PREFILL SWEEP")
    print(f"{'='*60}")

    for seq_len in seq_lengths:
        # Keep the total token count reasonable: 1 prefill sequence
        num_seqs = 1
        query_lens = [seq_len] * num_seqs
        context_lens = [seq_len] * num_seqs  # for prefill, context == query

        time_ms = benchmark_single(
            query_lens, context_lens,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
        )

        results["pure_prefill"].append({
            "seq_len": seq_len,
            "total_tokens": seq_len * num_seqs,
            "time_ms": time_ms,
        })
        print(f"  seq_len={seq_len:4d}: {time_ms:.3f} ms")

    # ========== 2. Pure Decode Sweep ==========
    print(f"\n{'='*60}")
    print("2. PURE DECODE SWEEP (varying context length)")
    print(f"{'='*60}")

    for context_len in seq_lengths:
        # Pure decode: each sequence has query_len=1 but different context
        query_lens = [1] * args.batch_size
        context_lens = [context_len] * args.batch_size

        time_ms = benchmark_single(
            query_lens, context_lens,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
        )

        results["pure_decode"].append({
            "context_len": context_len,
            "total_query_tokens": args.batch_size,
            "time_ms": time_ms,
        })
        print(f"  context_len={context_len:4d}: {time_ms:.3f} ms")

    # ========== 3. Mixed Sweep (varying ratios x varying total tokens) ==========
    print(f"\n{'='*60}")
    print("3. MIXED SWEEP (varying prefill % x total tokens)")
    print(f"{'='*60}")

    for pct in prefill_percentages:
        results["mixed"][f"{pct}pct"] = []
        print(f"\n  --- {pct}% Prefill ---")

        for prefill_len in seq_lengths:
            # Compute batch composition
            num_prefill = max(1, args.batch_size * pct // 100)
            num_decode = args.batch_size - num_prefill

            # Prefill sequences use prefill_len, decode sequences use 1
            query_lens = [prefill_len] * num_prefill + [1] * num_decode
            # Context: prefill uses prefill_len, decode also uses prefill_len (assume same length)
            context_lens = [prefill_len] * args.batch_size

            total_query_tokens = prefill_len * num_prefill + num_decode

            time_ms = benchmark_single(
                query_lens, context_lens,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
            )

            results["mixed"][f"{pct}pct"].append({
                "prefill_len": prefill_len,
                "num_prefill": num_prefill,
                "num_decode": num_decode,
                "total_query_tokens": total_query_tokens,
                "time_ms": time_ms,
            })
            print(f"    prefill_len={prefill_len:4d}, total_q_tokens={total_query_tokens:5d}: {time_ms:.3f} ms")

    return results


def calculate_slopes(results: Dict) -> Dict:
    """Compute slopes for each scenario."""
    slopes = {}

    # Pure Prefill slope
    prefill_data = results["pure_prefill"]
    if len(prefill_data) >= 2:
        x = np.array([d["seq_len"] for d in prefill_data])
        y = np.array([d["time_ms"] for d in prefill_data])
        slope, intercept = np.polyfit(x, y, 1)
        slopes["pure_prefill"] = {
            "slope_ms_per_token": slope,
            "intercept_ms": intercept,
        }

    # Pure Decode slope
    decode_data = results["pure_decode"]
    if len(decode_data) >= 2:
        x = np.array([d["context_len"] for d in decode_data])
        y = np.array([d["time_ms"] for d in decode_data])
        slope, intercept = np.polyfit(x, y, 1)
        slopes["pure_decode"] = {
            "slope_ms_per_context": slope,
            "intercept_ms": intercept,
        }

    # Mixed slopes (per ratio)
    slopes["mixed"] = {}
    for pct_key, mixed_data in results["mixed"].items():
        if len(mixed_data) >= 2:
            x = np.array([d["prefill_len"] for d in mixed_data])
            y = np.array([d["time_ms"] for d in mixed_data])
            slope, intercept = np.polyfit(x, y, 1)
            slopes["mixed"][pct_key] = {
                "slope_ms_per_prefill_len": slope,
                "intercept_ms": intercept,
            }

    return slopes


def plot_sweep_results(results: Dict, slopes: Dict, output_path: str, gpu_name: str):
    """Plot sweep results as a multi-line chart."""

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # ========== Fig 1: Time vs Seq Length, all scenarios ==========
    ax1 = axes[0]

    # Pure Prefill
    prefill_data = results["pure_prefill"]
    x_prefill = [d["seq_len"] for d in prefill_data]
    y_prefill = [d["time_ms"] for d in prefill_data]
    ax1.plot(x_prefill, y_prefill, 'o-', color=colors[0], linewidth=2,
             markersize=8, label=f'Pure Prefill (slope={slopes["pure_prefill"]["slope_ms_per_token"]*1000:.3f} μs/tok)')

    # Pure Decode
    decode_data = results["pure_decode"]
    x_decode = [d["context_len"] for d in decode_data]
    y_decode = [d["time_ms"] for d in decode_data]
    ax1.plot(x_decode, y_decode, 's-', color=colors[1], linewidth=2,
             markersize=8, label=f'Pure Decode (slope={slopes["pure_decode"]["slope_ms_per_context"]*1000:.3f} μs/ctx)')

    # Mixed scenarios
    mixed_colors = [colors[3], colors[4], colors[5], colors[6]]
    markers = ['^', 'v', 'D', 'p']
    for i, (pct_key, mixed_data) in enumerate(results["mixed"].items()):
        x_mixed = [d["prefill_len"] for d in mixed_data]
        y_mixed = [d["time_ms"] for d in mixed_data]
        slope = slopes["mixed"][pct_key]["slope_ms_per_prefill_len"]
        ax1.plot(x_mixed, y_mixed, f'{markers[i]}-', color=mixed_colors[i], linewidth=2,
                 markersize=8, label=f'Mixed {pct_key} (slope={slope*1000:.3f} μs/len)')

    ax1.set_xlabel('Sequence Length / Context Length', fontsize=12)
    ax1.set_ylabel('Kernel Time (ms)', fontsize=12)
    ax1.set_title(f'FlashAttention Kernel Time vs Sequence Length\n({gpu_name})', fontsize=14)
    ax1.legend(loc='upper left', fontsize=19)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('linear')

    # ========== Fig 2: slope bar comparison ==========
    ax2 = axes[1]

    labels = ['Pure\nPrefill', 'Pure\nDecode']
    slope_values = [
        slopes["pure_prefill"]["slope_ms_per_token"] * 1000,  # convert to us
        slopes["pure_decode"]["slope_ms_per_context"] * 1000,
    ]
    bar_colors = [colors[0], colors[1]]

    for i, (pct_key, slope_data) in enumerate(slopes["mixed"].items()):
        labels.append(f'Mixed\n{pct_key}')
        slope_values.append(slope_data["slope_ms_per_prefill_len"] * 1000)
        bar_colors.append(mixed_colors[i])

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, slope_values, color=bar_colors, edgecolor='black', linewidth=1.2)

    # Annotate bar values
    for bar, val in zip(bars, slope_values):
        ax2.annotate(f'{val:.3f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Slope (μs per unit length increase)', fontsize=12)
    ax2.set_title(f'Kernel Time Growth Rate Comparison\n({gpu_name})', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


def plot_comparison_bar(json_file1: str, json_file2: str,
                        gpu_name1: str, gpu_name2: str,
                        output_path: str):
    """
    Plot a side-by-side bar chart comparing two GPUs.

    Args:
        json_file1: result JSON path for the first GPU
        json_file2: result JSON path for the second GPU
        gpu_name1: name of the first GPU (e.g., "H200")
        gpu_name2: name of the second GPU (e.g., "A6000")
        output_path: output image path
    """
    # Load both JSON files
    with open(json_file1, 'r') as f:
        data1 = json.load(f)
    with open(json_file2, 'r') as f:
        data2 = json.load(f)

    slopes1 = data1["slopes"]
    slopes2 = data2["slopes"]

    # Build labels and data
    labels = ['Pure\nPrefill', 'Pure\nDecode']
    values1 = [
        slopes1["pure_prefill"]["slope_ms_per_token"] * 1000,  # μs
        slopes1["pure_decode"]["slope_ms_per_context"] * 1000,
    ]
    values2 = [
        slopes2["pure_prefill"]["slope_ms_per_token"] * 1000,
        slopes2["pure_decode"]["slope_ms_per_context"] * 1000,
    ]

    # Add Mixed scenarios
    for pct_key in slopes1["mixed"].keys():
        labels.append(f'Mixed\n{pct_key}')
        values1.append(slopes1["mixed"][pct_key]["slope_ms_per_prefill_len"] * 1000)
        values2.append(slopes2["mixed"][pct_key]["slope_ms_per_prefill_len"] * 1000)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(labels))
    width = 0.35  # bar width

    # Use two colors
    color1 = '#2E86AB'  # dark blue for H200
    color2 = '#E94F37'  # red for A6000

    bars1 = ax.bar(x - width/2, values1, width, label=gpu_name1,
                   color=color1, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, values2, width, label=gpu_name2,
                   color=color2, edgecolor='black', linewidth=1.2)

    # Annotate bar values
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=15, fontweight='bold')

    add_labels(bars1)
    add_labels(bars2)

    # Configure axes
    ax.set_xlabel('Scenario', fontsize=20)
    ax.set_ylabel('Slope (μs per unit length increase)', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=30, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Start y-axis from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")


def print_slope_summary(slopes: Dict):
    """Print slope summary."""
    print(f"\n{'='*70}")
    print("SLOPE SUMMARY (μs per unit increase)")
    print(f"{'='*70}")
    print(f"{'Scenario':<25} {'Slope (μs/unit)':<20} {'Description'}")
    print(f"{'-'*70}")

    print(f"{'Pure Prefill':<25} {slopes['pure_prefill']['slope_ms_per_token']*1000:<20.4f} μs per prefill token")
    print(f"{'Pure Decode':<25} {slopes['pure_decode']['slope_ms_per_context']*1000:<20.4f} μs per context token")

    for pct_key, slope_data in slopes["mixed"].items():
        print(f"{'Mixed ' + pct_key:<25} {slope_data['slope_ms_per_prefill_len']*1000:<20.4f} μs per prefill_len increase")


def main():
    parser = argparse.ArgumentParser(description="FlashAttention Comprehensive Sweep Benchmark")

    # Comparison-mode parameters
    parser.add_argument("--compare", action="store_true",
                       help="Compare two GPU results (requires --json1, --json2)")
    parser.add_argument("--json1", type=str, default=None,
                       help="First GPU result JSON file (for --compare mode)")
    parser.add_argument("--json2", type=str, default=None,
                       help="Second GPU result JSON file (for --compare mode)")
    parser.add_argument("--gpu1", type=str, default="H200",
                       help="Name of first GPU (default: H200)")
    parser.add_argument("--gpu2", type=str, default="RTX PRO 6000",
                       help="Name of second GPU (default: A6000)")
    parser.add_argument("--compare-output", type=str, default="comparison.pdf",
                       help="Output path for comparison plot")

    # Benchmark parameters
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size for decode/mixed scenarios")
    parser.add_argument("--num-heads", type=int, default=32,
                       help="Number of attention heads (Qwen3-8B: 32)")
    parser.add_argument("--num-kv-heads", type=int, default=8,
                       help="Number of KV heads (Qwen3-8B: 8)")
    parser.add_argument("--head-size", type=int, default=128,
                       help="Head dimension (Qwen3-8B: 128)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        if not args.json1 or not args.json2:
            parser.error("--compare requires --json1 and --json2")
        plot_comparison_bar(
            json_file1=args.json1,
            json_file2=args.json2,
            gpu_name1=args.gpu1,
            gpu_name2=args.gpu2,
            output_path=args.compare_output
        )
        return

    # Run sweep
    results = run_sweep(args)

    # Compute slope
    slopes = calculate_slopes(results)

    # Print slope summary
    print_slope_summary(slopes)

    # Save results
    gpu_name = torch.cuda.get_device_name()
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "gpu": gpu_name,
            "results": results,
            "slopes": slopes,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

        # Generate plots
        plot_path = args.output.replace('.json', '.png')
        plot_sweep_results(results, slopes, plot_path, gpu_name)


if __name__ == "__main__":
    main()
