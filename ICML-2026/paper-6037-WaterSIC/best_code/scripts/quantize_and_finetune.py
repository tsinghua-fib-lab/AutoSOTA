"""One-shot quantize-then-finetune wrapper.

Runs ``scripts.run_pipeline_job`` with all forwarded args, then on success
optionally runs ``scripts.finetune_zsic`` on the produced run directory.

Usage::

    python -m scripts.quantize_and_finetune \\
        --model 3-8B --method zsic --target_rate 3.0 \\
        --run_id 3-8B.zsic.r3.00 \\
        --zsic_binary_search --rate_control \\
        --qronos --residual_compensation \\
        --qronos_adapt --attn_weighted_qkv --attn_weighted_adapt_eps_joint \\
        --finetune

When ``--finetune`` is passed, ``--run_id`` is required so the wrapper can
locate the directory produced by the quantize step.
"""
from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Argument splitting: wrapper owns the --ft_* knobs, everything else passes
# through unmodified to run_pipeline_job.
# ---------------------------------------------------------------------------

def _build_wrapper_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--finetune", action="store_true",
                   help="After quantization, run scripts.finetune_zsic on the produced run_dir.")
    # Finetune knobs (forwarded to scripts.finetune_zsic with the ft_ prefix stripped)
    p.add_argument("--ft_epochs", type=int, default=4)
    p.add_argument("--ft_lr", type=float, default=5e-4)
    p.add_argument("--ft_min_lr", type=float, default=5e-6)
    p.add_argument("--ft_batch_size", type=int, default=8)
    p.add_argument("--ft_seqlen", type=int, default=2048)
    p.add_argument("--ft_dataset", type=str, default="wikitext2")
    p.add_argument("--ft_eval_each_epoch", action="store_true",
                   help="Evaluate WikiText-2 test PPL after each finetune epoch.")
    return p


def _extract_quantize_paths(passthrough: list[str]) -> tuple[str, str, str]:
    """Re-parse just the args needed to derive run_dir for the finetune step."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", required=True)
    p.add_argument("--run_root", default="quant_runs")
    p.add_argument("--run_id", default="")
    args, _ = p.parse_known_args(passthrough)
    return args.model, args.run_root, args.run_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    wrap_parser = _build_wrapper_parser()
    wrap_args, passthrough = wrap_parser.parse_known_args()

    # Validate --run_id up front when finetune is requested — fail before
    # spending compute on the quantize step.
    model = run_root = run_id = ""
    if wrap_args.finetune:
        model, run_root, run_id = _extract_quantize_paths(passthrough)
        if not run_id:
            sys.stderr.write(
                "[quantize_and_finetune] --finetune requires --run_id so the wrapper "
                "can locate the quantize step's output directory.\n"
            )
            sys.exit(2)

    saved_argv = sys.argv

    # --- Step 1: quantize ---
    print("[quantize_and_finetune] === quantize step ===", flush=True)
    sys.argv = ["run_pipeline_job", *passthrough]
    try:
        from scripts.run_pipeline_job import main as quantize_main
        quantize_main()
    finally:
        sys.argv = saved_argv

    if not wrap_args.finetune:
        return

    # --- Step 2: finetune ---
    from quant_layerwise.bucket import get_bucket_path
    run_dir = str(get_bucket_path() / run_root / model / run_id)

    ft_argv: list[str] = [
        "finetune_zsic",
        "--model_name", model,
        "--run_dir", run_dir,
        "--epochs", str(wrap_args.ft_epochs),
        "--lr", str(wrap_args.ft_lr),
        "--min_lr", str(wrap_args.ft_min_lr),
        "--batch_size", str(wrap_args.ft_batch_size),
        "--seqlen", str(wrap_args.ft_seqlen),
        "--dataset", wrap_args.ft_dataset,
    ]
    if wrap_args.ft_eval_each_epoch:
        ft_argv.append("--eval_ppl_each_epoch")

    print("[quantize_and_finetune] === finetune step ===", flush=True)
    print(f"[quantize_and_finetune]   run_dir={run_dir}", flush=True)
    sys.argv = ft_argv
    try:
        from scripts.finetune_zsic import main as finetune_main
        finetune_main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
