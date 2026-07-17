#!/usr/bin/env python3
"""Finetune ZSIC continuous parameters (t_vec, g_vec) on WikiText-2.

Usage (single checkpoint):
    python scripts/finetune_zsic.py \
        --model_name 3-8B \
        --run_dir /path/to/zsic_run \
        --gpu 0

Usage (sweep over multiple checkpoints, one GPU each):
    python scripts/finetune_zsic.py \
        --model_name 3-8B \
        --run_dirs /path/to/run1 /path/to/run2 /path/to/run3 \
        --gpus 0 1 2

Usage (multi-GPU tensor parallel, e.g. 70B on 4 GPUs):
    torchrun --nproc_per_node=4 scripts/finetune_zsic.py \
        --model_name 3-70B \
        --run_dir /path/to/zsic_run \
        --batch_size 1

Output is saved to <run_dir>_tuned/ (configurable via --suffix).
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def finetune_one(
    model_name: str,
    run_dir: str,
    gpu: int,
    suffix: str,
    seqlen: int,
    batch_size: int,
    lr: float,
    epochs: int,
    max_steps: int | None,
    log_interval: int,
    max_seq_len: int,
    master_port: int,
    dataset: str,
    nsamples: int | None,
    min_lr: float,
    eval_ppl_each_epoch: bool,
    eval_datasets: list[tuple[str, int | None]] | None = None,
    calib_stride: int | None = None,
    zero_out_rows: str = "",
):
    """Finetune a single checkpoint on a single or multi-GPU setup.

    Single-GPU: sets CUDA_VISIBLE_DEVICES and uses ensure_single_process_distributed.
    Multi-GPU (torchrun): uses LOCAL_RANK/RANK/WORLD_SIZE from environment.
    """
    import torch

    # Detect multi-GPU (torchrun sets WORLD_SIZE > 1)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_tp = world_size > 1

    if is_tp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist_rank = int(os.environ.get("RANK", 0))
        device = f"cuda:{local_rank}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Isolate NCCL so concurrent single-GPU processes don't interfere
        os.environ["NCCL_COMM_ID"] = ""
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_SHM_DISABLE"] = "1"
        local_rank = 0
        dist_rank = 0
        device = "cuda:0"

    run_dir = Path(run_dir)
    output_dir = run_dir.parent / f"{run_dir.name}_{suffix}"

    if dist_rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"[finetune] run_dir:    {run_dir}", flush=True)
        print(f"[finetune] output_dir: {output_dir}", flush=True)
        if is_tp:
            print(f"[finetune] tensor parallel: {world_size} GPUs", flush=True)
        else:
            print(f"[finetune] GPU:        {gpu}", flush=True)
        print(f"[finetune] lr={lr}, epochs={epochs}, batch_size={batch_size}, seqlen={seqlen}", flush=True)
        print(f"{'='*60}\n", flush=True)

    from quant_layerwise.finetune import build_finetunable_model, finetune

    t0 = time.time()

    student, teacher, manifest, tokenizer = build_finetunable_model(
        model_name, run_dir, device=device, max_seq_len=max_seq_len,
        master_port=master_port,
    )

    # Zero out specified rows in the teacher so hidden states match quantized behavior.
    # The student already handles this via dead_row_indices in QuantizedLinear artifacts.
    if zero_out_rows:
        from scripts.run_eval_job import _apply_zero_out_rows
        _apply_zero_out_rows(teacher, zero_out_rows)

    steps = finetune(
        student, teacher, tokenizer, manifest, run_dir, output_dir,
        device=device,
        seqlen=seqlen,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        log_interval=log_interval,
        max_steps=max_steps,
        gradient_checkpointing=True,
        dataset=dataset,
        nsamples=nsamples,
        min_lr=min_lr,
        eval_ppl_each_epoch=eval_ppl_each_epoch,
        eval_datasets=eval_datasets,
        calib_stride=calib_stride,
        dist_rank=dist_rank,
        dist_world_size=world_size,
    )

    elapsed = time.time() - t0
    if dist_rank == 0:
        print(f"\n[finetune] DONE {run_dir.name} -> {output_dir.name} "
              f"({steps} steps, {elapsed:.1f}s)", flush=True)

    # Clean up
    del student, teacher
    torch.cuda.empty_cache()


def _worker(args_tuple):
    """Multiprocessing worker for sweep."""
    finetune_one(*args_tuple)


def main():
    parser = argparse.ArgumentParser(description="Finetune ZSIC t_vec/g_vec on WikiText-2")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g. 3-8B, 3.2-1B)")

    # Single checkpoint
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to ZSIC run directory (single checkpoint)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index for single checkpoint")

    # Sweep over multiple checkpoints
    parser.add_argument("--run_dirs", type=str, nargs="+", default=None,
                        help="Multiple run directories for sweep")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU indices for sweep (one per run_dir, or cycled)")

    # Output
    parser.add_argument("--suffix", type=str, default=None,
                        help="Output directory suffix (default: 'tuned' or 'tuned_{seqlen}' if seqlen != 2048)")
    parser.add_argument("--master_port", type=int, default=29500,
                        help="Distributed master port (use different ports for concurrent runs)")

    # Training hyperparameters
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Model max_seq_len (for KV cache sizing, but we don't use KV cache)")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "redpajama", "redpajama_sample", "c4"],
                        help="Training dataset (default: wikitext2)")
    parser.add_argument("--nsamples", type=int, default=None,
                        help="Number of seqlen-sized chunks for redpajama/c4 (default: all)")
    parser.add_argument("--min_lr", type=float, default=5e-6,
                        help="Minimum LR for cosine annealing (default: 5e-6)")
    parser.add_argument("--eval_ppl_each_epoch", action="store_true",
                        help="Evaluate WikiText-2 test PPL after each epoch")
    parser.add_argument("--eval_extra", type=str, default=None,
                        help="Extra eval datasets. Format: 'c4:141,redpajama:200' (dataset:nsamples pairs)")
    parser.add_argument("--calib_stride", type=int, default=None,
                        help="Stride for overlapping training windows (default: seqlen = no overlap)")
    parser.add_argument("--zero_out_rows", type=str, default="",
                        help="Zero out rows in teacher model. Format: '6.w1:5723,8518;16.w1:2271,1875'")

    args = parser.parse_args()

    # Auto-generate suffix if not provided
    if args.suffix is None:
        args.suffix = f"tuned_{args.seqlen}" if args.seqlen != 2048 else "tuned"

    # Ensure max_seq_len >= seqlen so RoPE freqs_cis is large enough
    if args.max_seq_len < args.seqlen:
        args.max_seq_len = args.seqlen

    # Parse --eval_extra "c4:141,redpajama:200" -> [("c4", 141), ("redpajama", 200)]
    eval_datasets = None
    if args.eval_extra:
        eval_datasets = []
        for item in args.eval_extra.split(","):
            item = item.strip()
            if ":" in item:
                ds, ns = item.split(":", 1)
                eval_datasets.append((ds.strip(), int(ns.strip())))
            else:
                eval_datasets.append((item, None))

    # Multi-GPU tensor parallel mode: torchrun sets WORLD_SIZE > 1.
    # In this mode, only --run_dir is supported (no sweep), and we call
    # finetune_one directly (torchrun manages all processes).
    tp_world_size = int(os.environ.get("WORLD_SIZE", 1))
    if tp_world_size > 1:
        if args.run_dir is None:
            parser.error("Multi-GPU (torchrun) mode requires --run_dir (no sweep)")
        finetune_one(
            args.model_name, args.run_dir, 0, args.suffix,
            args.seqlen, args.batch_size, args.lr, args.epochs,
            args.max_steps, args.log_interval, args.max_seq_len,
            master_port=args.master_port,
            dataset=args.dataset, nsamples=args.nsamples,
            min_lr=args.min_lr,
            eval_ppl_each_epoch=args.eval_ppl_each_epoch,
            eval_datasets=eval_datasets,
            calib_stride=args.calib_stride,
            zero_out_rows=args.zero_out_rows,
        )
        return

    # Determine run_dirs and gpus
    if args.run_dirs is not None:
        run_dirs = args.run_dirs
        if args.gpus is not None:
            gpus = args.gpus
        else:
            gpus = list(range(len(run_dirs)))
    elif args.run_dir is not None:
        run_dirs = [args.run_dir]
        gpus = [args.gpu]
    else:
        parser.error("Must specify --run_dir or --run_dirs")

    # Cycle GPUs if fewer gpus than run_dirs
    if len(gpus) < len(run_dirs):
        gpus = [gpus[i % len(gpus)] for i in range(len(run_dirs))]

    # For a single run, just call directly (no multiprocessing overhead)
    if len(run_dirs) == 1:
        finetune_one(
            args.model_name, run_dirs[0], gpus[0], args.suffix,
            args.seqlen, args.batch_size, args.lr, args.epochs,
            args.max_steps, args.log_interval, args.max_seq_len,
            master_port=args.master_port,
            dataset=args.dataset, nsamples=args.nsamples,
            min_lr=args.min_lr,
            eval_ppl_each_epoch=args.eval_ppl_each_epoch,
            eval_datasets=eval_datasets,
            calib_stride=args.calib_stride,
            zero_out_rows=args.zero_out_rows,
        )
        return

    # Multiple runs: spawn one process per unique GPU to avoid contention.
    # If multiple run_dirs share a GPU, they run sequentially on that GPU.
    # If they have different GPUs, they run in parallel.
    gpu_groups: dict[int, list[int]] = {}  # gpu -> [indices into run_dirs]
    for i, g in enumerate(gpus):
        gpu_groups.setdefault(g, []).append(i)

    processes = []
    for gpu_id, indices in gpu_groups.items():
        for idx in indices:
            port = 29500 + idx  # unique port per process
            proc_args = (
                args.model_name, run_dirs[idx], gpu_id, args.suffix,
                args.seqlen, args.batch_size, args.lr, args.epochs,
                args.max_steps, args.log_interval, args.max_seq_len,
                port, args.dataset, args.nsamples,
                args.min_lr,
                args.eval_ppl_each_epoch,
                eval_datasets,
                args.calib_stride,
                args.zero_out_rows,
            )
            p = mp.Process(target=_worker, args=(proc_args,))
            processes.append((p, run_dirs[idx], gpu_id))

    # Start processes that use different GPUs in parallel;
    # for same-GPU processes, start sequentially.
    started: dict[int, mp.Process] = {}  # gpu -> currently running process
    pending = list(processes)

    while pending or started:
        # Start any process whose GPU is free
        still_pending = []
        for p, rd, g in pending:
            if g not in started:
                print(f"[sweep] starting {rd} on GPU {g}", flush=True)
                p.start()
                started[g] = p
            else:
                still_pending.append((p, rd, g))
        pending = still_pending

        # Wait for any process to finish
        if started:
            for g, p in list(started.items()):
                p.join(timeout=1.0)
                if not p.is_alive():
                    if p.exitcode != 0:
                        print(f"[sweep] WARNING: process on GPU {g} exited with code {p.exitcode}", flush=True)
                    del started[g]

    print("\n[sweep] all done!", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
