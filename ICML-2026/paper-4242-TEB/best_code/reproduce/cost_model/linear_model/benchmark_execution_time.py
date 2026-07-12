#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark end-to-end execution time for different batch compositions.

This script measures the actual iteration execution time (wall-clock time)
for different batch compositions (decode percentages) using torch.cuda.synchronize().

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_execution_time.py run \
        --model Qwen/Qwen3-4B \
        --total-tokens 256,512,1024,2048 \
        --decode-percentages 0,20,40,60,80,100 \
        --output-json execution_time_results.json

    python benchmark_execution_time.py plot \
        --input-json execution_time_results.json \
        --output-dir ./plots
"""

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Add vllm project root to path
VLLM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VLLM_ROOT))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    decode_percentage: int
    total_tokens: int
    num_decode: int
    num_prefill_tokens: int
    execution_time_ms: float
    execution_time_std: float = 0.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    model: str = "Qwen/Qwen3-4B"
    dtype: str = "float16"
    total_tokens_list: list[int] = field(
        default_factory=lambda: [256, 512, 1024, 2048]
    )
    decode_percentages: list[int] = field(
        default_factory=lambda: [0, 20, 40, 60, 80, 100]
    )
    decode_context_len: int = 16
    num_warmup: int = 3
    num_iterations: int = 10
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 5000
    max_model_len: int | None = None
    output_json: str = "execution_time_results.json"


class ExecutionTimeBenchmark:
    """Benchmark for measuring end-to-end execution time."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine_core = None
        self.results: list[BenchmarkResult] = []

    def setup(self):
        """Initialize EngineCore."""
        import torch
        from vllm import SamplingParams
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor.abstract import Executor

        print(f"Initializing model: {self.config.model}")

        engine_args = EngineArgs(
            model=self.config.model,
            dtype=self.config.dtype,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=0.9,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            block_size=16,
        )

        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        # Set torch threads to 1 for consistent benchmarking
        old_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.engine_core = EngineCore(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )
        finally:
            torch.set_num_threads(old_num_threads)

        print("Setup complete!")

    def _add_request(self, prompt_len: int, max_tokens: int = 1000) -> str:
        """Add a new request to the engine."""
        from vllm import SamplingParams
        from vllm.v1.engine import EngineCoreRequest

        req_id = f"req_{uuid.uuid4().hex[:8]}"
        prompt_token_ids = list(np.random.randint(100, 10000, size=prompt_len))

        request = EngineCoreRequest(
            request_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        self.engine_core.add_request(
            *self.engine_core.preprocess_add_request(request)
        )
        return req_id

    def _cleanup_all_requests(self):
        """Clean up all requests."""
        if self.engine_core is None:
            return

        running_ids = [req.request_id for req in self.engine_core.scheduler.running]
        waiting_ids = [req.request_id for req in self.engine_core.scheduler.waiting]
        all_ids = running_ids + waiting_ids

        if all_ids:
            self.engine_core.abort_requests(all_ids)

        for _ in range(100):
            if not self.engine_core.scheduler.has_requests():
                break
            try:
                self.engine_core.step_fn()
            except Exception:
                break

        self.engine_core.scheduler.running.clear()
        self.engine_core.scheduler.waiting.clear()
        self.engine_core.scheduler.requests.clear()

        try:
            input_batch = self.engine_core.model_executor.driver_worker.worker.model_runner.input_batch
            input_batch.clear()
        except Exception:
            pass

    def _prepare_decode_requests(self, num_decode: int, context_len: int) -> bool:
        """Prepare decode requests by completing their prefill phase."""
        if num_decode == 0:
            return True

        for _ in range(num_decode):
            self._add_request(context_len, max_tokens=100000)

        tokens_to_process = context_len * num_decode
        max_steps = (tokens_to_process // self.config.max_num_batched_tokens) + 100

        for _ in range(max_steps):
            num_running = len(self.engine_core.scheduler.running)
            if num_running >= num_decode:
                all_in_decode = all(
                    req.num_computed_tokens >= req.num_prompt_tokens
                    for req in self.engine_core.scheduler.running
                )
                if all_in_decode:
                    return True
            if not self.engine_core.scheduler.has_requests():
                break
            try:
                self.engine_core.step_fn()
            except Exception as e:
                print(f"  Error during prefill: {e}")
                return False

        return False

    def _run_single_step_timed(self) -> Optional[float]:
        """Run a single step and measure execution time."""
        import torch

        if not self.engine_core.scheduler.has_requests():
            return None

        try:
            scheduler_output = self.engine_core.scheduler.schedule()
        except Exception as e:
            print(f"  Schedule failed: {e}")
            return None

        if scheduler_output.total_num_scheduled_tokens == 0:
            return None

        # Measure end-to-end execution time
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        model_output = self.engine_core.model_executor.execute_model(
            scheduler_output, non_block=False
        )
        if model_output is None:
            model_output = self.engine_core.model_executor.sample_tokens(None)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        self.engine_core.scheduler.update_from_output(scheduler_output, model_output)

        execution_time_ms = (end_time - start_time) * 1000
        return execution_time_ms

    def run_benchmark(
        self,
        decode_percentage: int,
        total_tokens: int,
    ) -> Optional[BenchmarkResult]:
        """Run benchmark for specific decode percentage and total tokens."""
        num_decode = int(total_tokens * decode_percentage / 100)
        num_prefill_tokens = total_tokens - num_decode

        if num_decode > self.config.max_num_seqs:
            print(f"  Skipping: {num_decode} decode requests exceeds max_num_seqs")
            return None

        desc = f"{decode_percentage}% decode ({num_decode}D + {num_prefill_tokens}P)"
        print(f"  Testing {desc}, total={total_tokens}...")

        execution_times = []

        for i in range(self.config.num_warmup + self.config.num_iterations):
            self._cleanup_all_requests()

            # Prepare decode requests
            if num_decode > 0:
                success = self._prepare_decode_requests(
                    num_decode, self.config.decode_context_len
                )
                if not success:
                    print(f"    Warning: Failed to prepare decode requests")
                    return None

            # Add prefill request
            if num_prefill_tokens > 0:
                self._add_request(num_prefill_tokens, max_tokens=1)

            # Run one step with timing
            exec_time = self._run_single_step_timed()

            if i >= self.config.num_warmup and exec_time is not None:
                execution_times.append(exec_time)

        if not execution_times:
            return None

        result = BenchmarkResult(
            decode_percentage=decode_percentage,
            total_tokens=total_tokens,
            num_decode=num_decode,
            num_prefill_tokens=num_prefill_tokens,
            execution_time_ms=float(np.mean(execution_times)),
            execution_time_std=float(np.std(execution_times)),
        )

        print(f"    Execution time: {result.execution_time_ms:.3f}±{result.execution_time_std:.3f}ms")
        return result

    def run_all_benchmarks(self, stop_on_incomplete: bool = False):
        """Run all benchmark configurations.

        Args:
            stop_on_incomplete: If True, stop when a total_tokens value cannot
                generate all decode percentages.
        """
        import torch

        print(f"\n{'='*60}")
        print("Execution Time Benchmark")
        print(f"{'='*60}")
        print(f"Total tokens: {self.config.total_tokens_list}")
        print(f"Decode percentages: {self.config.decode_percentages}")
        if stop_on_incomplete:
            print("Mode: Stop on first incomplete total_tokens")

        # Global warmup
        print("\nGlobal warmup...")
        for _ in range(5):
            self._cleanup_all_requests()
            self._add_request(512, max_tokens=5)
            for _ in range(3):
                if not self.engine_core.scheduler.has_requests():
                    break
                try:
                    self.engine_core.step_fn()
                except Exception:
                    break
        self._cleanup_all_requests()
        torch.cuda.synchronize()
        print("Warmup complete.")

        # Run benchmarks
        for total_tokens in self.config.total_tokens_list:
            print(f"\n=== Total tokens: {total_tokens} ===")
            batch_results = []
            batch_complete = True

            for decode_pct in self.config.decode_percentages:
                result = self.run_benchmark(decode_pct, total_tokens)
                if result:
                    batch_results.append(result)
                else:
                    batch_complete = False
                    print(f"  Failed at {decode_pct}% decode for total_tokens={total_tokens}")
                    break

            if batch_complete:
                self.results.extend(batch_results)
            else:
                if stop_on_incomplete:
                    print(f"\nStopping: total_tokens={total_tokens} incomplete")
                    break
                else:
                    self.results.extend(batch_results)

        print(f"\n{'='*60}")
        print(f"Completed {len(self.results)} benchmarks!")

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {filepath}")


def _get_model_short_name(model: str) -> str:
    """Extract short model name from full path (e.g. 'Qwen/Qwen3-4B' -> 'Qwen3-4B')."""
    return model.split("/")[-1]


def plot_results(input_json: str, output_dir: str, title: str | None = None, total_tokens_filter: list[int] | None = None):
    """Generate execution time plots from results."""
    import matplotlib.pyplot as plt

    with open(input_json, 'r') as f:
        data = json.load(f)

    results = data["results"]

    # Filter by total_tokens if specified
    if total_tokens_filter:
        results = [r for r in results if r["total_tokens"] in total_tokens_filter]

    if title is None:
        model_name = _get_model_short_name(data.get("config", {}).get("model", ""))
        title = f"{model_name} (NVIDIA RTX PRO 6000)" if model_name else "NVIDIA RTX PRO 6000"

    os.makedirs(output_dir, exist_ok=True)

    def _save_png_and_pdf(fig, filename_png: str) -> None:
        """Save figure as PNG and PDF (same stem)."""
        out_png = os.path.join(output_dir, filename_png)
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {out_png}")

        stem, _ = os.path.splitext(filename_png)
        out_pdf = os.path.join(output_dir, f"{stem}.pdf")
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"Plot saved to {out_pdf}")

    # Group by decode percentage
    by_decode_pct = {}
    for r in results:
        pct = r["decode_percentage"]
        if pct not in by_decode_pct:
            by_decode_pct[pct] = {
                "total_tokens": [],
                "execution_time_ms": [],
                "execution_time_std": [],
            }
        by_decode_pct[pct]["total_tokens"].append(r["total_tokens"])
        by_decode_pct[pct]["execution_time_ms"].append(r["execution_time_ms"])
        by_decode_pct[pct]["execution_time_std"].append(r.get("execution_time_std", 0))

    all_tokens = sorted(set(r["total_tokens"] for r in results))
    token_to_idx = {t: i for i, t in enumerate(all_tokens)}

    colors = plt.cm.viridis(np.linspace(0, 1, len(by_decode_pct)))

    # Plot execution time
    fig, ax = plt.subplots(figsize=(10, 6))

    for (pct, pct_data), color in zip(sorted(by_decode_pct.items()), colors):
        sorted_idx = np.argsort(pct_data["total_tokens"])
        tokens = np.array(pct_data["total_tokens"])[sorted_idx]
        y = np.array(pct_data["execution_time_ms"])[sorted_idx]
        yerr = np.array(pct_data["execution_time_std"])[sorted_idx]
        x = np.array([token_to_idx[t] for t in tokens])

        if pct == 0:
            label = "Pure Prefill"
        elif pct == 100:
            label = "Pure Decode"
        else:
            label = f"{pct}% Decode"

        if yerr.sum() > 0:
            ax.errorbar(x, y, yerr=yerr, marker='o', label=label, color=color,
                       capsize=3, linewidth=2, markersize=6)
        else:
            ax.plot(x, y, marker='o', label=label, color=color, linewidth=2, markersize=6)

    ax.set_xticks(range(len(all_tokens)))
    ax.set_xticklabels([str(t) for t in all_tokens])
    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title(f"{title}: Execution Time", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_png_and_pdf(fig, "execution_time.png")
    plt.close()

    # Plot marginal cost per token
    fig, ax = plt.subplots(figsize=(10, 6))

    for (pct, pct_data), color in zip(sorted(by_decode_pct.items()), colors):
        sorted_idx = np.argsort(pct_data["total_tokens"])
        tokens = np.array(pct_data["total_tokens"])[sorted_idx]
        times = np.array(pct_data["execution_time_ms"])[sorted_idx]
        x = np.array([token_to_idx[t] for t in tokens])

        # Calculate marginal cost (time per token)
        marginal_cost = times / tokens

        if pct == 0:
            label = "Pure Prefill"
        elif pct == 100:
            label = "Pure Decode"
        else:
            label = f"{pct}% Decode"

        ax.plot(x, marginal_cost, marker='o', label=label, color=color, linewidth=2, markersize=6)

    ax.set_xticks(range(len(all_tokens)))
    ax.set_xticklabels([str(t) for t in all_tokens])
    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("Marginal Cost (ms/token)", fontsize=12)
    ax.set_title(f"{title}: Marginal Cost", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_png_and_pdf(fig, "marginal_cost.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark execution time")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmark")
    run_parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    run_parser.add_argument("--dtype", type=str, default="float16")
    run_parser.add_argument("--total-tokens", type=str, default="256,512,1024,2048")
    run_parser.add_argument("--decode-percentages", type=str, default="0,20,40,60,80,100")
    run_parser.add_argument("--decode-context-len", type=int, default=16)
    run_parser.add_argument("--num-warmup", type=int, default=3)
    run_parser.add_argument("--num-iterations", type=int, default=10)
    run_parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run_parser.add_argument("--max-num-seqs", type=int, default=5000)
    run_parser.add_argument("--max-model-len", type=int, default=None)
    run_parser.add_argument("--output-json", type=str, default="execution_time_results.json")
    run_parser.add_argument("--stop-on-incomplete", action="store_true",
                           help="Stop when a total_tokens cannot generate all decode ratios")

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate plots from results")
    plot_parser.add_argument("--input-json", type=str, required=True)
    plot_parser.add_argument("--output-dir", type=str, default="./plots")
    plot_parser.add_argument("--title", type=str, default=None, help="Plot title (auto-detected from JSON if not set)")
    plot_parser.add_argument("--total-tokens", type=str, default=None, help="Filter total_tokens (e.g. 1024,2048,4096)")

    args = parser.parse_args()

    if args.command == "run":
        config = BenchmarkConfig(
            model=args.model,
            dtype=args.dtype,
            total_tokens_list=[int(x) for x in args.total_tokens.split(",")],
            decode_percentages=[int(x) for x in args.decode_percentages.split(",")],
            decode_context_len=args.decode_context_len,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            output_json=args.output_json,
        )

        benchmark = ExecutionTimeBenchmark(config)
        benchmark.setup()
        benchmark.run_all_benchmarks(stop_on_incomplete=args.stop_on_incomplete)
        benchmark.save_results(config.output_json)

    elif args.command == "plot":
        total_tokens_filter = None
        if args.total_tokens:
            total_tokens_filter = [int(x) for x in args.total_tokens.split(",")]
        plot_results(args.input_json, args.output_dir, title=args.title, total_tokens_filter=total_tokens_filter)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
