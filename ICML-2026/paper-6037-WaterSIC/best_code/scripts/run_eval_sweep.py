"""Run evaluation sweep and generate comparison plots.

Usage:
    # Evaluate a specific sweep by manifest
    python -m scripts.run_eval_sweep --sweep quant_runs/3-8B/sweeps/sweep_zsic_20260120.json --eval --plot

    # Compare ZSIC vs GPTQ on the same plot (pass both sweep manifests)
    python -m scripts.run_eval_sweep \
        --sweep quant_runs/3-8B/sweeps/sweep_zsic_20260120.json \
        --sweep quant_runs/3-8B/sweeps/sweep_gptq_20260120.json \
        --eval --plot

    # Auto-discover all sweeps for a model (finds both ZSIC and GPTQ)
    python -m scripts.run_eval_sweep --model 3-8B --run_root quant_runs --eval --plot

    # Just plot without re-running eval (if eval.json files already exist)
    python -m scripts.run_eval_sweep \
        --sweep quant_runs/3-8B/sweeps/sweep_zsic_20260120.json \
        --sweep quant_runs/3-8B/sweeps/sweep_gptq_20260120.json \
        --plot
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import GPUtil

from quant_layerwise.bucket import get_bucket_path
from scripts.run_eval_job import run_eval_job


def _worker_fn(gpu: int, n_concurrent: int, func_module: str, func_name: str, args: tuple, kwargs: dict):
    """Worker function that runs in spawned process."""
    import importlib
    import multiprocessing
    import os
    import sys
    import traceback
    import torch

    # Limit CPU threads to avoid oversubscription when running multiple jobs.
    n_cpus = multiprocessing.cpu_count()
    threads_per_task = str(max(4, n_cpus // max(n_concurrent, 1)))
    os.environ["OMP_NUM_THREADS"] = threads_per_task
    os.environ["MKL_NUM_THREADS"] = threads_per_task
    os.environ["OPENBLAS_NUM_THREADS"] = threads_per_task

    try:
        # Set LOCAL_RANK so that parallel/start.py uses the correct GPU
        os.environ["LOCAL_RANK"] = str(gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        torch.cuda.set_device(0)  # After CUDA_VISIBLE_DEVICES, device 0 is our assigned GPU
        torch.set_num_threads(int(threads_per_task))

        # Find a free port to avoid conflicts with concurrent quant sweeps
        port = random.randint(40000, 60000)
        for _ in range(5000):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    break
            except OSError:
                port = 40000 + (port - 40000 + 1) % 20000
        kwargs["master_port_base"] = port

        module = importlib.import_module(func_module)
        func = getattr(module, func_name)
        func(*args, **kwargs)
    except Exception as e:
        print(f"[worker GPU {gpu}] ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)


def run_task(task, gpu: int, n_concurrent: int = 1):
    """Run a task on a specific GPU using spawn-safe approach."""
    f, args, kwargs = task
    func_module = f.__module__
    func_name = f.__name__

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_worker_fn, args=(gpu, n_concurrent, func_module, func_name, args, kwargs))
    p.start()
    return p


def run_tasks(tasks: List, gpu_list: Optional[List[int]] = None):
    """Run tasks on available GPUs using a simple scheduler."""
    if gpu_list is None:
        gpu_list = GPUtil.getAvailable(
            order="first",
            maxLoad=0.05,
            maxMemory=0.05,
            limit=1000,
        )
    if not gpu_list:
        raise RuntimeError("No free GPUs available")

    print(f"Detected free GPUs: {gpu_list}")
    gpu_status = {i: None for i in gpu_list}
    next_task = 0
    finished_tasks = 0

    while finished_tasks < len(tasks):
        for gpu_id in gpu_status.keys():
            process = gpu_status[gpu_id]
            if process is not None and not process.is_alive():
                print(f"GPU {gpu_id} is free now")
                gpu_status[gpu_id] = None
                finished_tasks += 1
            if gpu_status[gpu_id] is None and next_task < len(tasks):
                print(f"Allocating task {next_task}/{len(tasks)} to GPU {gpu_id}")
                gpu_status[gpu_id] = run_task(tasks[next_task], gpu_id, n_concurrent=len(gpu_list))
                next_task += 1
                time.sleep(1.0)  # Small delay between task launches to avoid port conflicts
        time.sleep(0.5)

    print(f"All {len(tasks)} tasks completed")


def run_tasks_multigpu(tasks: List[Dict[str, Any]], gpu_list: List[int]):
    """Run eval tasks sequentially using torchrun for multi-GPU tensor parallelism.

    Each task uses ALL GPUs in gpu_list simultaneously (one process per GPU).
    Tasks run one at a time since all GPUs are occupied per task.
    """
    nproc = len(gpu_list)
    cuda_devices = ",".join(str(g) for g in gpu_list)
    port = random.randint(40000, 60000)

    print(f"Multi-GPU eval: nproc={nproc}, GPUs={gpu_list}")

    for i, task_kwargs in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Evaluating {task_kwargs['run_dir']}")

        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc-per-node={nproc}",
            f"--master-port={port + i}",
            "-m", "scripts.run_eval_job",
            "--run_dir", str(task_kwargs["run_dir"]),
            "--seqlen", str(task_kwargs.get("seqlen", 2048)),
        ]
        if task_kwargs.get("eval_nsamples") is not None:
            cmd += ["--eval_nsamples", str(task_kwargs["eval_nsamples"])]
        if task_kwargs.get("ppl_only"):
            cmd.append("--ppl_only")
        if task_kwargs.get("sequential"):
            cmd.append("--sequential")
        if task_kwargs.get("force_bos"):
            cmd.append("--force_bos")
        if task_kwargs.get("zero_out_rows"):
            cmd += ["--zero_out_rows", str(task_kwargs["zero_out_rows"])]
        # Do NOT pass --init_dist: torchrun sets up distributed env

        env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": cuda_devices}
        print(f"  cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  WARNING: task failed with exit code {result.returncode}")

    print(f"\nAll {len(tasks)} multi-GPU eval tasks processed")


def load_sweep_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load a sweep manifest file."""
    with open(manifest_path) as f:
        return json.load(f)


def find_sweep_manifests(run_root: Path, model: str) -> List[Path]:
    """Find all sweep manifests for a model in $QUANT_BUCKET/run_root/model/sweeps/."""
    bucket = get_bucket_path()
    sweeps_dir = bucket / run_root / model / "sweeps"
    if not sweeps_dir.exists():
        return []
    return sorted(sweeps_dir.glob("sweep_*.json"))


def build_eval_tasks(
    manifests: List[Dict[str, Any]],
    seqlen: int = 2048,
    eval_nsamples: Optional[int] = None,
    ppl_only: bool = False,
    sequential: bool = False,
    force_bos: bool = False,
    skip_existing: bool = True,
    zero_out_rows: str = "",
    multigpu: bool = False,
) -> List:
    """Build list of evaluation tasks from sweep manifests.

    When multigpu=False: returns list of (func, args, kwargs) tuples for single-GPU mode.
    When multigpu=True: returns list of kwargs dicts for multi-GPU torchrun mode.
    """
    tasks = []
    eval_filename = "eval_bos.json" if force_bos else "eval.json"

    for manifest in manifests:
        # Use zero_out_rows from manifest if not explicitly provided
        manifest_zero = manifest.get("zero_out_rows", "")
        effective_zero = zero_out_rows or manifest_zero

        for run_info in manifest["runs"]:
            run_dir = Path(run_info["run_dir"])

            # Skip if eval file already exists
            eval_path = run_dir / eval_filename
            if skip_existing and eval_path.exists():
                print(f"Skipping {run_dir.name} ({eval_filename} exists)")
                continue

            # Check if manifest exists (run completed)
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                print(f"Skipping {run_dir.name} (no manifest.json - run not complete)")
                continue

            kwargs = {
                "run_dir": str(run_dir),
                "seqlen": seqlen,
                "eval_nsamples": eval_nsamples,
                "ppl_only": ppl_only,
                "sequential": sequential,
                "force_bos": force_bos,
                "zero_out_rows": effective_zero,
            }

            if multigpu:
                tasks.append(kwargs)
            else:
                kwargs["init_dist"] = True
                tasks.append((run_eval_job, (), kwargs))

    return tasks


def collect_results(manifests: List[Dict[str, Any]], force_bos: bool = False) -> Dict[str, List[Dict]]:
    """Collect evaluation results from eval.json files, organized by method."""
    results: Dict[str, List[Dict]] = {}
    eval_filename = "eval_bos.json" if force_bos else "eval.json"

    for manifest in manifests:
        method = manifest["method"]
        if method not in results:
            results[method] = []

        for run_info in manifest["runs"]:
            run_dir = Path(run_info["run_dir"])
            eval_path = run_dir / eval_filename

            if not eval_path.exists():
                print(f"Warning: no {eval_filename} for {run_dir.name}")
                continue

            with open(eval_path) as f:
                data = json.load(f)

            # Get actual rate from rate_summary.json if available
            rate_summary_path = run_dir / "rate_summary.json"
            actual_rate = run_info["rate"]  # fallback to target rate
            if rate_summary_path.exists():
                with open(rate_summary_path) as f:
                    rate_summary = json.load(f)
                    actual_rate = rate_summary.get("avg_rate_bits_per_param", actual_rate)

            results[method].append({
                "target_rate": run_info["rate"],
                "actual_rate": actual_rate,
                "run_dir": str(run_dir),
                "sweep_id": manifest["sweep_id"],
                **data.get("eval", {}),
            })

    # Sort by rate
    for method in results:
        results[method].sort(key=lambda x: x["target_rate"])

    return results


def generate_plots(results: Dict[str, List[Dict]], output_dir: Path, model: str):
    """Generate comparison plots for PPL and KL divergence."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Colors and markers for each method
    styles = {
        "zsic": {"color": "blue", "marker": "o", "label": "ZSIC"},
        "gptq": {"color": "red", "marker": "s", "label": "GPTQ"},
    }

    # Plot 1: PPL vs Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ppl_cutoff = 40.0  # Ignore points with PPL > this value
    for method, data_list in results.items():
        if not data_list:
            continue
        style = styles.get(method, {"color": "gray", "marker": "x", "label": method})
        # Filter out points with PPL > cutoff
        filtered = [(d["actual_rate"], d.get("ppl_quant")) for d in data_list
                    if d.get("ppl_quant") is not None and d.get("ppl_quant") <= ppl_cutoff]
        if filtered:
            rates, ppls = zip(*filtered)
            ax.plot(rates, ppls, **style, linestyle="-", markersize=8)

    ax.set_xlabel("Rate (bits/param)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(f"Perplexity vs Rate - {model}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add reference PPL line if available
    ref_ppl = None
    for method, data_list in results.items():
        for d in data_list:
            if "ppl_ref" in d:
                ref_ppl = d["ppl_ref"]
                break
        if ref_ppl:
            break
    if ref_ppl:
        ax.axhline(y=ref_ppl, color="green", linestyle="--", label=f"Reference PPL: {ref_ppl:.2f}")
        ax.legend()

    ppl_path = output_dir / f"{model}_ppl_vs_rate.png"
    fig.savefig(ppl_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {ppl_path}")
    plt.close(fig)

    # Plot 2: KL Divergence vs Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    has_kl = False
    for method, data_list in results.items():
        if not data_list:
            continue
        style = styles.get(method, {"color": "gray", "marker": "x", "label": method})
        rates = [d["actual_rate"] for d in data_list]
        kls = [d.get("kl_ref_to_quant") for d in data_list]
        if all(k is not None for k in kls):
            ax.plot(rates, kls, **style, linestyle="-", markersize=8)
            has_kl = True

    if has_kl:
        ax.set_xlabel("Rate (bits/param)", fontsize=12)
        ax.set_ylabel("KL Divergence", fontsize=12)
        ax.set_title(f"KL Divergence vs Rate - {model}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        kl_path = output_dir / f"{model}_kl_vs_rate.png"
        fig.savefig(kl_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {kl_path}")
    else:
        print("No KL divergence data available (run with --ppl_only=False)")
    plt.close(fig)

    # Save results as JSON for later use
    json_path = output_dir / f"{model}_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")


def main():
    p = argparse.ArgumentParser(description="Run evaluation sweep and generate plots")
    p.add_argument("--sweep", action="append", dest="sweeps", default=[],
                   help="Path to sweep manifest file (can specify multiple)")
    p.add_argument("--model", type=str, default=None,
                   help="Model name to auto-discover sweeps (e.g., 3-8B)")
    p.add_argument("--run_root", type=str, default="quant_runs",
                   help="Root directory for quantization runs")
    p.add_argument("--eval", action="store_true",
                   help="Run evaluations on sweep runs")
    p.add_argument("--plot", action="store_true",
                   help="Generate comparison plots")
    p.add_argument("--seqlen", type=int, default=2048,
                   help="Sequence length for evaluation")
    p.add_argument("--eval_nsamples", type=int, default=None,
                   help="Number of eval samples (default: all)")
    p.add_argument("--ppl_only", action="store_true",
                   help="Only compute PPL (skip KL divergence)")
    p.add_argument("--sequential", action="store_true",
                   help="Load models sequentially to save GPU memory (slower)")
    p.add_argument("--force_bos", action="store_true",
                   help="Replace first token of each sequence with BOS token")
    p.add_argument("--force_reeval", action="store_true",
                   help="Re-run evaluations even if eval.json exists")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated list of GPU IDs to use")
    p.add_argument("--nproc", type=int, default=1,
                   help="Number of GPUs per eval task. >1 enables multi-GPU eval via torchrun "
                        "(required for sharded models like 70B). Default: 1 (single-GPU per task)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for plots (default: run_root/plots)")
    p.add_argument("--zero_out_rows", type=str, default="",
                   help="Zero out rows in models. Auto-detected from sweep manifest if not provided.")

    args = p.parse_args()

    if not args.eval and not args.plot:
        print("Nothing to do. Specify --eval and/or --plot")
        return

    run_root = Path(args.run_root)
    bucket = get_bucket_path()

    # Collect manifest paths
    manifest_paths: List[Path] = []

    # From explicit --sweep arguments
    for sweep_path in args.sweeps:
        manifest_paths.append(Path(sweep_path))

    # Auto-discover if --model specified
    if args.model:
        discovered = find_sweep_manifests(run_root, args.model)
        print(f"Discovered {len(discovered)} sweep manifests for model {args.model}")
        manifest_paths.extend(discovered)

    if not manifest_paths:
        print("No sweep manifests found. Specify --sweep or --model")
        return

    # Load all manifests
    manifests = []
    model = None
    for path in manifest_paths:
        if not path.exists():
            print(f"Warning: manifest not found: {path}")
            continue
        manifest = load_sweep_manifest(path)
        manifests.append(manifest)
        print(f"Loaded sweep: {manifest['sweep_id']} ({manifest['method']}, rates={manifest['rates']})")
        if model is None:
            model = manifest["model"]
        elif model != manifest["model"]:
            print(f"Warning: mixing models ({model} vs {manifest['model']})")

    if not manifests:
        print("No valid manifests loaded")
        return

    output_dir = Path(args.output_dir) if args.output_dir else bucket / run_root / "plots"

    if args.eval:
        multigpu = args.nproc > 1
        tasks = build_eval_tasks(
            manifests,
            seqlen=args.seqlen,
            eval_nsamples=args.eval_nsamples,
            ppl_only=args.ppl_only,
            sequential=args.sequential,
            force_bos=args.force_bos,
            skip_existing=not args.force_reeval,
            zero_out_rows=args.zero_out_rows,
            multigpu=multigpu,
        )

        if tasks:
            print(f"\nBuilt {len(tasks)} evaluation tasks")
            gpu_list = None
            if args.gpus:
                gpu_list = [int(g.strip()) for g in args.gpus.split(",")]

            if multigpu:
                if gpu_list is None:
                    gpu_list = list(range(args.nproc))
                elif len(gpu_list) < args.nproc:
                    raise RuntimeError(
                        f"--nproc={args.nproc} but only {len(gpu_list)} GPUs specified. "
                        f"Need at least {args.nproc} GPUs for multi-GPU eval."
                    )
                run_tasks_multigpu(tasks, gpu_list[:args.nproc])
            else:
                run_tasks(tasks, gpu_list=gpu_list)
        else:
            print("\nNo evaluation tasks to run (all already evaluated)")

    if args.plot:
        results = collect_results(manifests, force_bos=args.force_bos)
        generate_plots(results, output_dir, model or "unknown")


if __name__ == "__main__":
    main()
