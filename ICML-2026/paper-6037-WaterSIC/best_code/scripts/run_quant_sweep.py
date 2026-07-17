"""Run quantization sweep over multiple rates.

Usage:
    # Single-GPU (one rate per GPU, parallel across GPUs):
    python -m scripts.run_quant_sweep --model 3-8B --method zsic --rate_min 0.5 --rate_max 3.5 --rate_step 0.5

    # Multi-GPU (one rate across N GPUs via torchrun, sequential):
    python -m scripts.run_quant_sweep --model 3-70B --method zsic --rates 2.0 --nproc 8

Outputs a sweep manifest file that can be used by run_eval_sweep.py:
    quant_runs/{model}/sweeps/sweep_{method}_{timestamp}.json
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import GPUtil


def find_free_port(start: int = 30000, end: int = 65000) -> int:
    """Find a free port in the given range."""
    # Randomize starting point to reduce collisions between concurrent sweeps
    port = random.randint(start, end)
    for _ in range(end - start):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            port = start + (port - start + 1) % (end - start)
    raise RuntimeError(f"Could not find a free port in range {start}-{end}")

# NOTE: We intentionally avoid importing torch or any module that imports torch
# at module load time. This ensures CUDA_VISIBLE_DEVICES isolation works correctly
# in spawned processes.


def _get_bucket_path() -> Path:
    """Get bucket path from environment (inlined to avoid torch import)."""
    import os
    p = os.environ.get("QUANT_BUCKET", None)
    if not p:
        raise RuntimeError(
            "QUANT_BUCKET environment variable is not set. "
            "Example: export QUANT_BUCKET=/home/ubuntu/quant-bucket"
        )
    return Path(p)


# Model configs: model_name -> num_layers
MODEL_CONFIGS = {
    "3.2-1B": 16,
    "3.2-1B-4s": 16,
    "2-7B": 32,
    "3-8B": 32,
    "2-13B": 40,
    "3-70B": 80,
    "qwen3-8B": 36,
}


def generate_sweep_id(method: str, *, qronos: bool = False, residual_compensation: bool = False, attn_weighted_qkv: bool = False, attn_weighted_adapt_eps_joint: bool = False, qronos_adapt: bool = False, w1w3_qronos_adapt: bool = False) -> str:
    """Generate a unique sweep ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = ["sweep", method]
    if qronos:
        parts.append("qronos")
    if residual_compensation:
        parts.append("rescomp")
    if attn_weighted_qkv:
        if attn_weighted_adapt_eps_joint:
            parts.append("attnw_joint")
        else:
            parts.append("attnw")
    if qronos_adapt:
        parts.append("qadapt")
    if w1w3_qronos_adapt:
        parts.append("w1w3adapt")
    parts.append(timestamp)
    return "_".join(parts)


def _worker_fn(gpu: int, task_id: int, port_base: int, n_concurrent: int, func_module: str, func_name: str, args: tuple, kwargs: dict):
    """Worker function that runs in spawned process."""
    import importlib
    import os
    import sys
    import traceback

    # CRITICAL: Set CUDA_VISIBLE_DEVICES so that this process only sees one GPU
    # This makes "cuda:0" in this process map to the actual GPU we want.
    # This is necessary because parallel/start.py does torch.cuda.set_device(LOCAL_RANK)
    # and silences output for LOCAL_RANK > 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Limit CPU threads to avoid oversubscription when running multiple jobs.
    # Without this, each process uses ALL cores for BLAS (Cholesky, matmul),
    # causing severe contention with 4+ concurrent jobs.
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    threads_per_task = str(max(4, n_cpus // max(n_concurrent, 1)))
    os.environ["OMP_NUM_THREADS"] = threads_per_task
    os.environ["MKL_NUM_THREADS"] = threads_per_task
    os.environ["OPENBLAS_NUM_THREADS"] = threads_per_task

    # Force unbuffered output for spawned processes
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Compute task-specific port
    task_port = port_base + task_id * 10
    print(f"[worker GPU {gpu}, task {task_id}] starting (CUDA_VISIBLE_DEVICES={gpu}, port={task_port})", flush=True)

    exit_code = 0
    try:
        import torch
        # Now cuda:0 maps to our target GPU
        torch.cuda.set_device(0)
        # Also limit PyTorch interop threads to match env vars
        torch.set_num_threads(int(threads_per_task))

        # Print GPU diagnostics to confirm correct assignment
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"[worker GPU {gpu}, task {task_id}] torch sees {device_count} device(s), using: {device_name} (cpu_threads={threads_per_task})", flush=True)

        # Override master_port_base in kwargs to use task-specific port
        # This avoids NCCL port collisions between tasks
        kwargs["master_port_base"] = task_port

        module = importlib.import_module(func_module)
        func = getattr(module, func_name)
        func(*args, **kwargs)
        print(f"[worker GPU {gpu}, task {task_id}] completed successfully", flush=True)
    except Exception as e:
        # Print error to both stdout and stderr to ensure visibility
        error_msg = f"\n{'='*60}\n[worker GPU {gpu}, task {task_id}] ERROR: {e}\n{'='*60}\n"
        print(error_msg, flush=True)
        print(error_msg, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        exit_code = 1
    finally:
        # Clean up distributed process group and CUDA resources
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                print(f"[worker GPU {gpu}, task {task_id}] destroyed process group", flush=True)
        except Exception:
            pass  # Ignore cleanup errors

        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass  # Ignore cleanup errors

    if exit_code != 0:
        sys.exit(exit_code)


def run_task(task, gpu: int, task_id: int, port_base: int, n_concurrent: int = 1):
    """Run a task on a specific GPU using spawn-safe approach."""
    func_module, func_name, args, kwargs = task

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_worker_fn, args=(gpu, task_id, port_base, n_concurrent, func_module, func_name, args, kwargs))
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

    # Find a free port range for this sweep (avoids collisions with previous crashed processes)
    # Each task needs ~10 ports, so reserve enough for all tasks
    port_base = find_free_port(start=30000, end=60000)
    print(f"Detected free GPUs: {gpu_list}")
    print(f"Using port base: {port_base} (range {port_base}-{port_base + len(tasks) * 10})")

    # Track: gpu_id -> (process, task_id)
    gpu_status: Dict[int, Optional[Tuple[mp.Process, int]]] = {i: None for i in gpu_list}
    next_task = 0
    finished_tasks = 0
    failed_tasks: List[int] = []

    def cleanup_all():
        """Terminate and join all running processes."""
        for gpu_id, status in gpu_status.items():
            if status is not None:
                proc, tid = status
                if proc.is_alive():
                    print(f"Terminating task {tid} on GPU {gpu_id}...")
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        print(f"Force killing task {tid} on GPU {gpu_id}...")
                        proc.kill()
                        proc.join(timeout=2)
                else:
                    proc.join(timeout=1)  # Reap zombie

    try:
        while finished_tasks < len(tasks):
            for gpu_id in list(gpu_status.keys()):
                status = gpu_status[gpu_id]
                if status is not None:
                    proc, task_id = status
                    if not proc.is_alive():
                        # Process finished - join it to clean up resources
                        proc.join(timeout=5)
                        exitcode = proc.exitcode
                        if exitcode != 0:
                            print(f"WARNING: Task {task_id} on GPU {gpu_id} failed with exitcode {exitcode}")
                            failed_tasks.append(task_id)
                        else:
                            print(f"Task {task_id} on GPU {gpu_id} completed successfully")
                        gpu_status[gpu_id] = None
                        finished_tasks += 1

                if gpu_status[gpu_id] is None and next_task < len(tasks):
                    print(f"Allocating task {next_task}/{len(tasks)} to GPU {gpu_id}")
                    proc = run_task(tasks[next_task], gpu_id, task_id=next_task, port_base=port_base, n_concurrent=len(gpu_list))
                    gpu_status[gpu_id] = (proc, next_task)
                    next_task += 1
                    # Don't start another task in the same loop iteration
                    # Wait for model loading before starting more tasks
                    break
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted! Cleaning up processes...")
        cleanup_all()
        raise

    # Final cleanup - join any remaining processes
    for gpu_id, status in gpu_status.items():
        if status is not None:
            proc, _ = status
            proc.join(timeout=5)

    if failed_tasks:
        print(f"\nWARNING: {len(failed_tasks)} task(s) failed: {failed_tasks}")
    print(f"All {len(tasks)} tasks processed ({len(tasks) - len(failed_tasks)} succeeded, {len(failed_tasks)} failed)")


def run_tasks_multigpu(tasks: List[tuple], gpu_list: List[int]):
    """Run quant tasks sequentially using torchrun for multi-GPU tensor parallelism.

    Each task uses ALL GPUs in gpu_list simultaneously (one process per GPU).
    Tasks run one at a time since all GPUs are occupied per task.
    """
    nproc = len(gpu_list)
    cuda_devices = ",".join(str(g) for g in gpu_list)
    port = random.randint(40000, 60000)

    print(f"Multi-GPU quant: nproc={nproc}, GPUs={gpu_list}")

    for i, (func_module, func_name, _args, kwargs) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Quantizing rate={kwargs.get('target_rate', '?')}")

        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc-per-node={nproc}",
            f"--master-port={port + i}",
            "-m", "scripts.run_pipeline_job",
            "--model", str(kwargs["model_name"]),
            "--method", str(kwargs["method"]),
            "--target_rate", str(kwargs["target_rate"]),
            "--layer_begin", str(kwargs.get("layer_begin", 0)),
            "--layer_end", str(kwargs.get("layer_end", 32)),
            "--hessian_batch_size", str(kwargs.get("hessian_batch_size", 1)),
            "--calib_dataset", str(kwargs.get("calib_dataset", "redpajama")),
            "--calib_seed", str(kwargs.get("calib_seed", 42)),
            "--run_root", str(kwargs.get("run_root", "quant_runs")),
            "--run_id", str(kwargs.get("run_id", "")),
        ]
        if kwargs.get("nsamples") is not None:
            cmd += ["--nsamples", str(kwargs["nsamples"])]
        if kwargs.get("replay_batch_size") is not None:
            cmd += ["--replay_batch_size", str(kwargs["replay_batch_size"])]
        if kwargs.get("seqlen", 2048) != 2048:
            cmd += ["--seqlen", str(kwargs["seqlen"])]
        if kwargs.get("calib_stride") is not None:
            cmd += ["--calib_stride", str(kwargs["calib_stride"])]
        # Do NOT pass --init_dist: torchrun sets up distributed env

        # Boolean flags
        if kwargs.get("resume", True):
            cmd.append("--resume")
        if kwargs.get("collect_qronos_stats"):
            cmd.append("--collect_qronos_stats")
        if kwargs.get("plot_activation_mse"):
            cmd.append("--plot_activation_mse")
        if kwargs.get("zero_out_rows"):
            cmd += ["--zero_out_rows", str(kwargs["zero_out_rows"])]
        # Method-specific
        if kwargs["method"] == "gptq":
            cmd += ["--percdamp", str(kwargs.get("percdamp", 0.1))]
            cmd += ["--groupsize", str(kwargs.get("groupsize", -1))]
            if kwargs.get("gptq_maxq") is not None:
                cmd += ["--gptq_maxq", str(kwargs["gptq_maxq"])]
        elif kwargs["method"] == "zsic":
            if kwargs.get("zsic_binary_search"):
                cmd.append("--zsic_binary_search")
            if kwargs.get("rate_control"):
                cmd.append("--rate_control")
            if kwargs.get("qronos"):
                cmd.append("--qronos")
            if kwargs.get("zsic_percdamp") is not None:
                cmd += ["--zsic_percdamp", str(kwargs["zsic_percdamp"])]
            if kwargs.get("residual_compensation"):
                cmd.append("--residual_compensation")
                cmd += ["--rescomp_skip_prefix", str(kwargs.get("rescomp_skip_prefix", 0))]
            if kwargs.get("rate_weight_budgets"):
                cmd += ["--rate_weight_budgets", str(kwargs["rate_weight_budgets"])]
            if kwargs.get("attn_weighted_qkv"):
                cmd.append("--attn_weighted_qkv")
                cmd += ["--attn_weighted_qkv_eps", str(kwargs.get("attn_weighted_qkv_eps", 0.0))]
                cmd += ["--attn_weighted_weights", str(kwargs.get("attn_weighted_weights", "wq,wk,wv"))]
                if kwargs.get("attn_weighted_adapt_eps_joint"):
                    cmd.append("--attn_weighted_adapt_eps_joint")
            if kwargs.get("qronos_adapt"):
                cmd.append("--qronos_adapt")
            if kwargs.get("w1w3_qronos_adapt"):
                cmd.append("--w1w3_qronos_adapt")
            if kwargs.get("adapt_search_sample_ratio", 1.0) < 1.0:
                cmd += ["--adapt_search_sample_ratio", str(kwargs["adapt_search_sample_ratio"])]
            if kwargs.get("coord_adapt_a_eps_steps", 10) != 10:
                cmd += ["--coord_adapt_a_eps_steps", str(kwargs["coord_adapt_a_eps_steps"])]
            if kwargs.get("coord_adapt_q_eps_steps", 10) != 10:
                cmd += ["--coord_adapt_q_eps_steps", str(kwargs["coord_adapt_q_eps_steps"])]
            if kwargs.get("qronos_layer_min") is not None:
                cmd += ["--qronos_layer_min", str(kwargs["qronos_layer_min"])]
            if kwargs.get("qronos_layer_max") is not None:
                cmd += ["--qronos_layer_max", str(kwargs["qronos_layer_max"])]

        env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": cuda_devices}
        print(f"  cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  WARNING: task failed with exit code {result.returncode}")

    print(f"\nAll {len(tasks)} multi-GPU quant tasks processed")


def make_run_id(model: str, method: str, rate: float, *, calib_dataset: str = "redpajama", calib_seed: int = 42, seqlen: int = 2048, qronos: bool = False, residual_compensation: bool = False, rescomp_skip_prefix: int = 0, rate_weight_budgets: str = "", zero_out_rows: str = "", attn_weighted_qkv: bool = False, attn_weighted_qkv_eps: float = 0.0, attn_weighted_weights: str = "wq,wk,wv", attn_weighted_adapt_eps_joint: bool = False, qronos_adapt: bool = False, w1w3_qronos_adapt: bool = False) -> str:
    """Generate a run ID for a quantization run."""
    parts = [model, method]
    # Mark calibration dataset in run_id
    if "," in calib_dataset:
        parts.append("mix")
    elif calib_dataset == "wikitext2":
        parts.append("wiki")
    elif calib_dataset == "c4":
        parts.append("c4")
    else:
        parts.append("rp")
    if calib_seed != 42:
        parts.append(f"seed{calib_seed}")
    if seqlen != 2048:
        parts.append(f"s{seqlen}")
    if method == "zsic" and qronos:
        parts.append("qronos")
    if method == "zsic" and residual_compensation:
        if rescomp_skip_prefix > 0:
            # e.g., "rescomp_from8" means skip layers 0-7, apply from layer 8+
            parts.append(f"rescomp_from{rescomp_skip_prefix}")
        else:
            parts.append("rescomp")
    if attn_weighted_qkv:
        if attn_weighted_adapt_eps_joint:
            parts.append("attnw_joint")
        elif attn_weighted_qkv_eps != 0.01:
            parts.append(f"attnw_eps{attn_weighted_qkv_eps}")
        else:
            parts.append("attnw")
        # Tag non-default weight sets (e.g., "attnw_kv" for wk,wv only)
        aw_set = set(w.strip().lower() for w in attn_weighted_weights.split(",") if w.strip())
        if aw_set != {"wq", "wk", "wv"}:
            parts.append("_".join(sorted(aw_set)))
    if qronos_adapt:
        parts.append("qadapt")
    if w1w3_qronos_adapt:
        parts.append("w1w3adapt")
    # Add zero_out_rows tag (e.g., "zero6_16" for layers 6 and 16)
    if zero_out_rows:
        zero_layers = set()
        for item in zero_out_rows.split(";"):
            item = item.strip()
            if ":" in item:
                key = item.split(":")[0].strip()
                if "." in key:
                    zero_layers.add(key.split(".")[0])
        if zero_layers:
            parts.append("zero" + "_".join(sorted(zero_layers, key=int)))
    # Add weight budget info to run_id (e.g., "wb_wo_w2" for wo and w2 budgets)
    if rate_weight_budgets:
        # Parse "wo:1.15,w2:1.15" -> "wb_wo_w2"
        weight_names = []
        for item in rate_weight_budgets.split(","):
            item = item.strip()
            if ":" in item:
                wname = item.split(":")[0].strip()
                weight_names.append(wname)
        if weight_names:
            parts.append("wb_" + "_".join(sorted(set(weight_names))))
    parts.append(f"r{rate:.2f}")
    return ".".join(parts)


def build_quant_tasks(
    model: str,
    method: str,
    rates: List[float],
    run_root: str,
    hessian_batch_size: int = 10,
    nsamples: int | None = None,
    replay_batch_size: int | None = None,
    seqlen: int = 2048,
    calib_stride: int | None = None,
    # Calibration
    calib_dataset: str = "redpajama",
    calib_seed: int = 42,
    # GPTQ options
    groupsize: int = -1,
    percdamp: float = 0.1,
    gptq_maxq: int | None = None,
    # ZSIC options
    qronos: bool = False,
    zsic_percdamp: float | None = None,
    qronos_layer_min: int | None = None,
    qronos_layer_max: int | None = None,
    rate_weight_budgets: str = "",  # e.g., "wo:1.15,w2:1.15"
    zero_out_rows: str = "",  # e.g., "6.w1:5723,8518;6.w3:5723,8518"
    # Diagnostics: collect stats and plot activation MSE
    collect_qronos_stats: bool = False,
    plot_activation_mse: bool = False,
    # Residual stream compensation for wo/w2 layers
    residual_compensation: bool = False,
    rescomp_skip_prefix: int = 0,  # Skip first N layers
    # Attention-weighted QKV
    attn_weighted_qkv: bool = False,
    attn_weighted_qkv_eps: float = 0.0,
    attn_weighted_weights: str = "wq,wk,wv",
    attn_weighted_adapt_eps_joint: bool = False,
    # Qronos adapt
    qronos_adapt: bool = False,
    # w1w3 qronos adapt
    w1w3_qronos_adapt: bool = False,
    # Subsample calibration during golden-section search
    adapt_search_sample_ratio: float = 1.0,
    # Number of golden-section steps for a_eps in coord-adapt (0 = skip)
    coord_adapt_a_eps_steps: int = 10,
    # Number of golden-section steps for q_eps in coord-adapt (0 = skip)
    coord_adapt_q_eps_steps: int = 10,
    # Resume
    resume: bool = True,
) -> Tuple[List[Tuple], List[Dict[str, Any]]]:
    """Build list of quantization tasks.

    Returns:
        tasks: List of (func_module, func_name, args, kwargs) tuples
        run_infos: List of dicts with run metadata for sweep manifest
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Supported: {list(MODEL_CONFIGS.keys())}")

    num_layers = MODEL_CONFIGS[model]
    tasks = []
    run_infos = []

    # Get bucket path for run_dir
    bucket = _get_bucket_path()

    # Use string references to avoid importing torch in main process
    func_module = "scripts.run_pipeline_job"
    func_name = "run_pipeline_job"

    for rate in rates:
        run_id = make_run_id(model, method, rate, calib_dataset=calib_dataset, calib_seed=calib_seed, seqlen=seqlen, qronos=qronos, residual_compensation=residual_compensation, rescomp_skip_prefix=rescomp_skip_prefix, rate_weight_budgets=rate_weight_budgets, zero_out_rows=zero_out_rows, attn_weighted_qkv=attn_weighted_qkv, attn_weighted_qkv_eps=attn_weighted_qkv_eps, attn_weighted_weights=attn_weighted_weights, attn_weighted_adapt_eps_joint=attn_weighted_adapt_eps_joint, qronos_adapt=qronos_adapt, w1w3_qronos_adapt=w1w3_qronos_adapt)
        run_dir = str(bucket / run_root / model / run_id)

        kwargs = {
            "model_name": model,
            "method": method,
            "target_rate": rate,
            "layer_begin": 0,
            "layer_end": num_layers,
            "calib_dataset": calib_dataset,
            "calib_seed": calib_seed,
            "hessian_batch_size": hessian_batch_size,
            "nsamples": nsamples,
            "replay_batch_size": replay_batch_size,
            "seqlen": seqlen,
            "calib_stride": calib_stride,
            "run_root": run_root,
            "run_id": run_id,
            "resume": resume,
            "init_dist": True,
        }

        # Diagnostics options (apply to all methods)
        if collect_qronos_stats:
            kwargs["collect_qronos_stats"] = True
        if plot_activation_mse:
            kwargs["plot_activation_mse"] = True

        # Zero out rows (applies to all methods)
        if zero_out_rows:
            kwargs["zero_out_rows"] = zero_out_rows

        # Method-specific params
        if method == "zsic":
            kwargs.update({
                "zsic_binary_search": True,
                "rate_control": True,
                "qronos": qronos,
                "residual_compensation": residual_compensation,
                "rescomp_skip_prefix": rescomp_skip_prefix,
            })
            if zsic_percdamp is not None:
                kwargs["zsic_percdamp"] = zsic_percdamp
            if qronos_layer_min is not None:
                kwargs["qronos_layer_min"] = qronos_layer_min
            if qronos_layer_max is not None:
                kwargs["qronos_layer_max"] = qronos_layer_max
            if rate_weight_budgets:
                kwargs["rate_weight_budgets"] = rate_weight_budgets
            if attn_weighted_qkv:
                kwargs["attn_weighted_qkv"] = True
                kwargs["attn_weighted_qkv_eps"] = attn_weighted_qkv_eps
                kwargs["attn_weighted_weights"] = attn_weighted_weights
                if attn_weighted_adapt_eps_joint:
                    kwargs["attn_weighted_adapt_eps_joint"] = True
            if qronos_adapt:
                kwargs["qronos_adapt"] = True
            if w1w3_qronos_adapt:
                kwargs["w1w3_qronos_adapt"] = True
            if adapt_search_sample_ratio < 1.0:
                kwargs["adapt_search_sample_ratio"] = adapt_search_sample_ratio
            if coord_adapt_a_eps_steps != 10:
                kwargs["coord_adapt_a_eps_steps"] = coord_adapt_a_eps_steps
            if coord_adapt_q_eps_steps != 10:
                kwargs["coord_adapt_q_eps_steps"] = coord_adapt_q_eps_steps
        elif method == "gptq":
            kwargs.update({
                "percdamp": percdamp,
                "groupsize": groupsize,
            })
            if gptq_maxq is not None:
                kwargs["gptq_maxq"] = gptq_maxq

        tasks.append((func_module, func_name, (), kwargs))
        run_infos.append({
            "rate": rate,
            "run_id": run_id,
            "run_dir": run_dir,
        })

    return tasks, run_infos


def save_sweep_manifest(
    sweep_id: str,
    model: str,
    method: str,
    rates: List[float],
    run_infos: List[Dict[str, Any]],
    run_root: str,
    *,
    calib_dataset: str = "redpajama",
    calib_seed: int = 42,
    groupsize: int = -1,
    percdamp: float = 0.1,
    gptq_maxq: int | None = None,
    qronos: bool = False,
    zsic_percdamp: float | None = None,
    qronos_layer_min: int | None = None,
    qronos_layer_max: int | None = None,
    rate_weight_budgets: str = "",
    zero_out_rows: str = "",
    collect_qronos_stats: bool = False,
    plot_activation_mse: bool = False,
    residual_compensation: bool = False,
    rescomp_skip_prefix: int = 0,
    attn_weighted_qkv: bool = False,
    attn_weighted_qkv_eps: float = 0.0,
    attn_weighted_weights: str = "wq,wk,wv",
    attn_weighted_adapt_eps_joint: bool = False,
    qronos_adapt: bool = False,
    w1w3_qronos_adapt: bool = False,
    adapt_search_sample_ratio: float = 1.0,
    coord_adapt_a_eps_steps: int = 10,
    coord_adapt_q_eps_steps: int = 10,
) -> Path:
    """Save sweep manifest file to $QUANT_BUCKET/run_root/model/sweeps/."""
    manifest = {
        "sweep_id": sweep_id,
        "model": model,
        "method": method,
        "calib_dataset": calib_dataset,
        "calib_seed": calib_seed,
        "num_layers": MODEL_CONFIGS[model],
        "rates": rates,
        "runs": run_infos,
        "created_at": datetime.now().isoformat(),
    }

    # Add method-specific options
    if method == "gptq":
        manifest["groupsize"] = groupsize
        manifest["percdamp"] = percdamp
        if gptq_maxq is not None:
            manifest["gptq_maxq"] = gptq_maxq
    elif method == "zsic":
        manifest["qronos"] = qronos
        if zsic_percdamp is not None:
            manifest["zsic_percdamp"] = zsic_percdamp
        if qronos_layer_min is not None:
            manifest["qronos_layer_min"] = qronos_layer_min
        if qronos_layer_max is not None:
            manifest["qronos_layer_max"] = qronos_layer_max
        if rate_weight_budgets:
            manifest["rate_weight_budgets"] = rate_weight_budgets
        if residual_compensation:
            manifest["residual_compensation"] = residual_compensation
            if rescomp_skip_prefix > 0:
                manifest["rescomp_skip_prefix"] = rescomp_skip_prefix
        if attn_weighted_qkv:
            manifest["attn_weighted_qkv"] = True
            manifest["attn_weighted_qkv_eps"] = attn_weighted_qkv_eps
            manifest["attn_weighted_weights"] = attn_weighted_weights
            if attn_weighted_adapt_eps_joint:
                manifest["attn_weighted_adapt_eps_joint"] = True
        if qronos_adapt:
            manifest["qronos_adapt"] = True
        if w1w3_qronos_adapt:
            manifest["w1w3_qronos_adapt"] = True
        if adapt_search_sample_ratio < 1.0:
            manifest["adapt_search_sample_ratio"] = adapt_search_sample_ratio
        if coord_adapt_a_eps_steps != 10:
            manifest["coord_adapt_a_eps_steps"] = coord_adapt_a_eps_steps
        if coord_adapt_q_eps_steps != 10:
            manifest["coord_adapt_q_eps_steps"] = coord_adapt_q_eps_steps

    if zero_out_rows:
        manifest["zero_out_rows"] = zero_out_rows

    # Add diagnostics options if enabled
    if collect_qronos_stats or plot_activation_mse:
        manifest["diagnostics"] = {
            "collect_qronos_stats": collect_qronos_stats,
            "plot_activation_mse": plot_activation_mse,
        }



    # Save to $QUANT_BUCKET/run_root/model/sweeps/
    bucket = _get_bucket_path()
    sweeps_dir = bucket / run_root / model / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = sweeps_dir / f"{sweep_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def main():
    p = argparse.ArgumentParser(description="Run quantization sweep over multiple rates")
    p.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                   help="Model to quantize")
    p.add_argument("--method", required=True, choices=["zsic", "gptq"],
                   help="Quantization method")
    p.add_argument("--rate_min", type=float, default=1.0,
                   help="Minimum target rate (default: 1.0)")
    p.add_argument("--rate_max", type=float, default=4.0,
                   help="Maximum target rate (default: 4.0)")
    p.add_argument("--rate_step", type=float, default=0.5,
                   help="Rate step (default: 0.5)")
    p.add_argument("--rates", type=str, default=None,
                   help="Explicit list of rates, comma-separated (overrides min/max/step)")
    p.add_argument("--run_root", type=str, default="quant_runs",
                   help="Output directory for runs")
    p.add_argument("--hessian_batch_size", type=int, default=32,
                   help="Batch size for Hessian computation")

    p.add_argument("--seqlen", type=int, default=2048,
                   help="Sequence length for calibration (default: 2048)")
    p.add_argument("--calib_stride", type=int, default=None,
                   help="Stride for overlapping calib windows (default: seqlen = no overlap)")

    # Calibration
    p.add_argument("--calib_dataset", type=str, default="redpajama",
                   help="Calibration dataset: redpajama, c4, wikitext2, or mix like 'redpajama:0.5,c4:0.4,wikitext2:0.1'")
    p.add_argument("--calib_seed", type=int, default=42,
                   help="Seed for RedPajama shuffling (default: 42)")
    p.add_argument("--nsamples", type=int, default=None,
                   help="Max calibration samples (default: all available)")
    p.add_argument("--replay_batch_size", type=int, default=None,
                   help="Batch size for Qronos/residual replay (default: hessian_batch_size)")
    p.add_argument("--rate_control", action="store_true",
                   help="Enable global rate budget tracking")

    # GPTQ options
    p.add_argument("--groupsize", type=int, default=-1,
                   help="GPTQ group size (-1 = per-channel, default: -1)")
    p.add_argument("--percdamp", type=float, default=0.1,
                   help="GPTQ Hessian damping (default: 0.1)")
    p.add_argument("--gptq_maxq", type=int, default=None,
                   help="GPTQ maxq override (default: compute from target_rate as 2^(rate+1)-1)")
    # ZSIC options
    p.add_argument("--qronos", action="store_true",
                   help="Enable Qronos mode for ZSIC (default: off)")
    p.add_argument("--zsic_percdamp", type=float, default=None,
                   help="ZSIC Hessian damping (default: 0.0001)")
    p.add_argument("--qronos_layer_min", type=int, default=None,
                   help="Only apply Qronos targeting to layers >= this (default: all)")
    p.add_argument("--qronos_layer_max", type=int, default=None,
                   help="Only apply Qronos targeting to layers < this (default: all)")
    p.add_argument("--rate_weight_budgets", type=str, default="",
                   help="Weight-type budget multipliers. Format: 'wo:1.15,w2:1.15' gives wo/w2 15%% more bits")

    # Diagnostics options
    p.add_argument("--collect_qronos_stats", action="store_true",
                   help="Collect Qronos stats for diagnostics (no Qronos targeting)")
    p.add_argument("--plot_activation_mse", action="store_true",
                   help="Plot activation MSE at end of each run (requires --qronos or --collect_qronos_stats)")
    p.add_argument("--residual_compensation", action="store_true",
                   help="Enable residual stream compensation for wo/w2 layers (automatically enables Qronos mode for wo/w2)")
    p.add_argument("--rescomp_skip_prefix", type=int, default=0,
                   help="Skip residual compensation on the first N layers (0 = apply to all)")
    p.add_argument("--zero_out_rows", type=str, default="",
                   help="Zero out specific rows after quantization. Format: '6.w1:5723,8518;6.w3:5723,8518'")

    # Attention-weighted QKV
    p.add_argument("--attn_weighted_qkv", action="store_true",
                   help="Weight QKV calibration stats by per-token attention importance (BOS fix)")
    p.add_argument("--attn_weighted_qkv_eps", type=float, default=0.01,
                   help="Mixing: Sig = (1-eps)*weighted + eps*unweighted (default: 0.01)")
    p.add_argument("--attn_weighted_weights", type=str, default="wq,wk,wv",
                   help="Which QKV weights to apply attention weighting to (default: 'wq,wk,wv')")
    p.add_argument("--attn_weighted_adapt_eps_joint", action="store_true",
                   help="Joint eps search for wq/wk/wv (minimize wo_in relMSE via forward pass)")

    p.add_argument("--qronos_adapt", action="store_true",
                   help="Adaptive qronos/standard blending for QKV (minimize wo_in relMSE)")
    p.add_argument("--w1w3_qronos_adapt", action="store_true",
                   help="Adaptive qronos/standard blending for w1/w3 jointly (minimize w2 input relMSE)")
    p.add_argument("--adapt_search_sample_ratio", type=float, default=1.0,
                   help="Fraction of calibration samples for golden-section search (default: 1.0 = all)")
    p.add_argument("--coord_adapt_a_eps_steps", type=int, default=10,
                   help="Golden-section steps for a_eps in coord-adapt (0 = skip a_eps search, default: 10)")
    p.add_argument("--coord_adapt_q_eps_steps", type=int, default=10,
                   help="Golden-section steps for q_eps in coord-adapt (0 = skip q_eps search, default: 10)")

    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated list of GPU IDs to use (default: auto-detect free GPUs)")
    p.add_argument("--nproc", type=int, default=1,
                   help="Number of GPUs per quant task. >1 enables multi-GPU quant via torchrun "
                        "(required for sharded models like 70B). Tasks run sequentially. Default: 1")
    p.add_argument("--no_resume", action="store_true",
                   help="Force fresh runs (don't resume from existing artifacts)")

    args = p.parse_args()

    # Build rate list
    if args.rates:
        rates = [float(r.strip()) for r in args.rates.split(",")]
    else:
        rates = []
        r = args.rate_min
        while r <= args.rate_max + 1e-9:
            rates.append(round(r, 2))
            r += args.rate_step

    # Generate sweep ID
    sweep_id = generate_sweep_id(args.method, qronos=args.qronos, residual_compensation=args.residual_compensation, attn_weighted_qkv=args.attn_weighted_qkv, attn_weighted_adapt_eps_joint=args.attn_weighted_adapt_eps_joint, qronos_adapt=args.qronos_adapt, w1w3_qronos_adapt=args.w1w3_qronos_adapt)

    print("Sweep config:")
    print(f"  Sweep ID: {sweep_id}")
    print(f"  Model: {args.model} ({MODEL_CONFIGS[args.model]} layers)")
    print(f"  Method: {args.method}")
    print(f"  Rates: {rates}")
    print(f"  Calib dataset: {args.calib_dataset} (seed={args.calib_seed})")
    if args.method == "gptq":
        print(f"  Groupsize: {args.groupsize}")
        print(f"  Percdamp: {args.percdamp}")
        if args.gptq_maxq is not None:
            print(f"  Maxq override: {args.gptq_maxq}")
    if args.method == "zsic":
        print(f"  Qronos: {args.qronos}")
        if args.zsic_percdamp is not None:
            print(f"  ZSIC Percdamp: {args.zsic_percdamp}")
        if args.qronos and (args.qronos_layer_min is not None or args.qronos_layer_max is not None):
            print(f"  Qronos layer range: [{args.qronos_layer_min}, {args.qronos_layer_max})")
        if args.rate_weight_budgets:
            print(f"  Weight budgets: {args.rate_weight_budgets}")
    if args.collect_qronos_stats or args.plot_activation_mse:
        print(f"  Collect Qronos stats: {args.collect_qronos_stats}")
        print(f"  Plot activation MSE: {args.plot_activation_mse}")
    if args.residual_compensation:
        skip_str = f" (skip first {args.rescomp_skip_prefix} layers)" if args.rescomp_skip_prefix > 0 else ""
        print(f"  Residual compensation: {args.residual_compensation}{skip_str}")
    if args.attn_weighted_qkv:
        if args.attn_weighted_adapt_eps_joint:
            print(f"  Attention-weighted QKV: joint eps (wo_in relMSE), weights={args.attn_weighted_weights}")
        else:
            print(f"  Attention-weighted QKV: eps={args.attn_weighted_qkv_eps}, weights={args.attn_weighted_weights}")
    if args.qronos_adapt:
        print(f"  Qronos adapt: {args.qronos_adapt}")
    if args.w1w3_qronos_adapt:
        print(f"  W1W3 qronos adapt: {args.w1w3_qronos_adapt}")
    if args.zero_out_rows:
        print(f"  Zero out rows: {args.zero_out_rows}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Output: {args.run_root}")

    # Build tasks
    tasks, run_infos = build_quant_tasks(
        model=args.model,
        method=args.method,
        rates=rates,
        run_root=args.run_root,
        hessian_batch_size=args.hessian_batch_size,
        nsamples=args.nsamples,
        replay_batch_size=args.replay_batch_size,
        seqlen=args.seqlen,
        calib_stride=args.calib_stride,
        calib_dataset=args.calib_dataset,
        calib_seed=args.calib_seed,
        groupsize=args.groupsize,
        percdamp=args.percdamp,
        gptq_maxq=args.gptq_maxq,
        qronos=args.qronos,
        zsic_percdamp=args.zsic_percdamp,
        qronos_layer_min=args.qronos_layer_min,
        qronos_layer_max=args.qronos_layer_max,
        rate_weight_budgets=args.rate_weight_budgets,
        zero_out_rows=args.zero_out_rows,
        collect_qronos_stats=args.collect_qronos_stats,
        plot_activation_mse=args.plot_activation_mse,
        residual_compensation=args.residual_compensation,
        rescomp_skip_prefix=args.rescomp_skip_prefix,
        attn_weighted_qkv=args.attn_weighted_qkv,
        attn_weighted_qkv_eps=args.attn_weighted_qkv_eps,
        attn_weighted_weights=args.attn_weighted_weights,
        attn_weighted_adapt_eps_joint=args.attn_weighted_adapt_eps_joint,
        qronos_adapt=args.qronos_adapt,
        w1w3_qronos_adapt=args.w1w3_qronos_adapt,
        adapt_search_sample_ratio=args.adapt_search_sample_ratio,
        coord_adapt_a_eps_steps=args.coord_adapt_a_eps_steps,
        coord_adapt_q_eps_steps=args.coord_adapt_q_eps_steps,
        resume=not args.no_resume,
    )

    print(f"\nBuilt {len(tasks)} quantization tasks")

    # Save sweep manifest before running (so eval can find it even if interrupted)
    manifest_path = save_sweep_manifest(
        sweep_id=sweep_id,
        model=args.model,
        method=args.method,
        rates=rates,
        run_infos=run_infos,
        run_root=args.run_root,
        calib_dataset=args.calib_dataset,
        calib_seed=args.calib_seed,
        groupsize=args.groupsize,
        percdamp=args.percdamp,
        gptq_maxq=args.gptq_maxq,
        qronos=args.qronos,
        zsic_percdamp=args.zsic_percdamp,
        qronos_layer_min=args.qronos_layer_min,
        qronos_layer_max=args.qronos_layer_max,
        rate_weight_budgets=args.rate_weight_budgets,
        zero_out_rows=args.zero_out_rows,
        collect_qronos_stats=args.collect_qronos_stats,
        plot_activation_mse=args.plot_activation_mse,
        residual_compensation=args.residual_compensation,
        rescomp_skip_prefix=args.rescomp_skip_prefix,
        attn_weighted_qkv=args.attn_weighted_qkv,
        attn_weighted_qkv_eps=args.attn_weighted_qkv_eps,
        attn_weighted_weights=args.attn_weighted_weights,
        attn_weighted_adapt_eps_joint=args.attn_weighted_adapt_eps_joint,
        qronos_adapt=args.qronos_adapt,
        w1w3_qronos_adapt=args.w1w3_qronos_adapt,
        adapt_search_sample_ratio=args.adapt_search_sample_ratio,
        coord_adapt_a_eps_steps=args.coord_adapt_a_eps_steps,
        coord_adapt_q_eps_steps=args.coord_adapt_q_eps_steps,
    )
    print(f"Saved sweep manifest: {manifest_path}")

    # Parse GPU list
    gpu_list = None
    if args.gpus:
        gpu_list = [int(g.strip()) for g in args.gpus.split(",")]

    # Run tasks
    if args.nproc > 1:
        if gpu_list is None:
            gpu_list = list(range(args.nproc))
        elif len(gpu_list) < args.nproc:
            raise RuntimeError(
                f"--nproc={args.nproc} but only {len(gpu_list)} GPUs specified. "
                f"Need at least {args.nproc} GPUs for multi-GPU quant."
            )
        run_tasks_multigpu(tasks, gpu_list[:args.nproc])
    else:
        run_tasks(tasks, gpu_list=gpu_list)

    print("\nSweep complete. To evaluate and plot:")
    eval_nproc = f" --nproc {args.nproc}" if args.nproc > 1 else ""
    print(f"  python -m scripts.run_eval_sweep --sweep {manifest_path} --eval --plot{eval_nproc}")


if __name__ == "__main__":
    main()
