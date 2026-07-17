#!/usr/bin/env python3
"""Sweep ZSIC finetuning across multiple rate checkpoints.

Usage:
    # Finetune all rates on 2 GPUs:
    python scripts/sweep_finetune_zsic.py \
        --model_name 3.2-1B \
        --base 3.2-1B.zsic.wiki.qronos.rescomp.attnw_joint.qadapt \
        --rates 1.00 1.50 2.00 2.50 3.00 3.50 4.00 \
        --gpus 0 1

    # Auto-discover all rates matching r* pattern:
    python scripts/sweep_finetune_zsic.py \
        --model_name 3.2-1B \
        --base 3.2-1B.zsic.wiki.qronos.rescomp.attnw_joint.qadapt \
        --gpus 0 1 2 3

    # Custom hyperparams:
    python scripts/sweep_finetune_zsic.py \
        --model_name 3.2-1B \
        --base 3.2-1B.zsic.wiki.qronos.rescomp.attnw_joint.qadapt \
        --rates 1.00 2.00 4.00 \
        --gpus 0 1 \
        --lr 5e-4 --epochs 100 --batch_size 8

Checkpoints are expected at:
    $QUANT_BUCKET/quant_runs/{model_name}/{base}.r{rate}/

Output saved to:
    $QUANT_BUCKET/quant_runs/{model_name}/{base}.r{rate}_tuned/
"""

import argparse
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path


def find_free_port(start: int = 30000, end: int = 65000) -> int:
    port = random.randint(start, end)
    for _ in range(end - start):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            port = start + (port - start + 1) % (end - start)
    raise RuntimeError(f"No free port in {start}-{end}")


def discover_rates(runs_dir: Path, base: str) -> list[str]:
    """Find all rate directories matching {base}.r*"""
    pattern = f"{base}.r*"
    matches = sorted(runs_dir.glob(pattern))
    # Filter out _tuned dirs
    matches = [m for m in matches if m.is_dir() and "_tuned" not in m.name]
    rates = []
    for m in matches:
        # Extract rate from e.g. "base.r1.00" -> "1.00"
        suffix = m.name[len(base) + 2:]  # skip "{base}.r"
        rates.append(suffix)
    return rates


def main():
    parser = argparse.ArgumentParser(description="Sweep ZSIC finetuning across rates")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base", type=str, required=True,
                        help="Base name without .r{rate} (e.g. 3.2-1B.zsic.wiki.qronos.rescomp.attnw_joint.qadapt)")
    parser.add_argument("--rates", type=str, nargs="*", default=None,
                        help="Rate values (e.g. 1.00 2.00 4.00). If omitted, auto-discovers all r* dirs.")
    parser.add_argument("--gpus", type=int, nargs="+", required=True,
                        help="GPU indices to use")

    # Output
    parser.add_argument("--suffix", type=str, default=None,
                        help="Output dir suffix (default: 'tuned' or 'tuned_{seqlen}' if seqlen != 2048)")

    # Training hyperparams (passed through to finetune_zsic.py)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "redpajama", "redpajama_sample", "c4"])
    parser.add_argument("--nsamples", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=5e-6)
    parser.add_argument("--eval_ppl_each_epoch", action="store_true")
    parser.add_argument("--eval_extra", type=str, default=None,
                        help="Extra eval datasets. Format: 'c4:141,redpajama:200'")
    parser.add_argument("--zero_out_rows", type=str, default="",
                        help="Zero out rows spec passed to finetune_zsic.py")

    # Runs root
    parser.add_argument("--run_root", type=str, default=None,
                        help="Override run root (default: $QUANT_BUCKET/quant_runs/{model_name})")

    args = parser.parse_args()

    # Auto-generate suffix if not provided
    if args.suffix is None:
        args.suffix = f"tuned_{args.seqlen}" if args.seqlen != 2048 else "tuned"

    # Determine runs directory
    if args.run_root:
        runs_dir = Path(args.run_root)
    else:
        bucket = os.environ.get("QUANT_BUCKET")
        if not bucket:
            print("ERROR: QUANT_BUCKET not set and --run_root not provided", file=sys.stderr)
            sys.exit(1)
        runs_dir = Path(bucket) / "quant_runs" / args.model_name

    # Determine rates
    if args.rates:
        rates = args.rates
    else:
        rates = discover_rates(runs_dir, args.base)
        if not rates:
            print(f"ERROR: no runs matching {args.base}.r* in {runs_dir}", file=sys.stderr)
            sys.exit(1)

    # Build list of (rate, run_dir) pairs
    jobs = []
    for rate in rates:
        run_dir = runs_dir / f"{args.base}.r{rate}"
        if not run_dir.exists():
            print(f"WARNING: {run_dir} not found, skipping rate {rate}", flush=True)
            continue
        output_dir = runs_dir / f"{args.base}.r{rate}_{args.suffix}"
        if output_dir.exists():
            print(f"SKIP: {output_dir} already exists, skipping rate {rate}", flush=True)
            continue
        jobs.append((rate, str(run_dir)))

    if not jobs:
        print("No jobs to run.", flush=True)
        return

    print(f"[sweep] {len(jobs)} jobs: rates {[j[0] for j in jobs]}", flush=True)
    print(f"[sweep] GPUs: {args.gpus}", flush=True)
    print(f"[sweep] lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}", flush=True)
    print(flush=True)

    # Run jobs, scheduling one per GPU
    script = str(Path(__file__).resolve().parent / "finetune_zsic.py")
    active: dict[int, tuple[subprocess.Popen, str, str]] = {}  # gpu -> (proc, rate, run_dir)
    pending = list(jobs)

    def _launch(gpu: int, rate: str, run_dir: str):
        port = find_free_port()
        cmd = [
            sys.executable, script,
            "--model_name", args.model_name,
            "--run_dir", run_dir,
            "--gpu", str(gpu),
            "--suffix", args.suffix,
            "--master_port", str(port),
            "--lr", str(args.lr),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seqlen", str(args.seqlen),
            "--dataset", args.dataset,
            "--min_lr", str(args.min_lr),
        ]
        if args.nsamples is not None:
            cmd += ["--nsamples", str(args.nsamples)]
        if args.eval_ppl_each_epoch:
            cmd += ["--eval_ppl_each_epoch"]
        if args.eval_extra:
            cmd += ["--eval_extra", args.eval_extra]
        if args.zero_out_rows:
            cmd += ["--zero_out_rows", args.zero_out_rows]

        print(f"[sweep] launching rate={rate} on GPU {gpu} (port {port})", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return proc

    while pending or active:
        # Launch jobs on free GPUs
        free_gpus = [g for g in args.gpus if g not in active]
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            rate, run_dir = pending.pop(0)
            proc = _launch(gpu, rate, run_dir)
            active[gpu] = (proc, rate, run_dir)

        # Poll active processes
        for gpu in list(active.keys()):
            proc, rate, run_dir = active[gpu]
            ret = proc.poll()
            if ret is not None:
                if ret == 0:
                    print(f"[sweep] rate={rate} on GPU {gpu} DONE", flush=True)
                else:
                    print(f"[sweep] rate={rate} on GPU {gpu} FAILED (exit {ret})", flush=True)
                del active[gpu]

        if active:
            time.sleep(5)

    print(f"\n[sweep] all {len(jobs)} jobs finished!", flush=True)


if __name__ == "__main__":
    main()
