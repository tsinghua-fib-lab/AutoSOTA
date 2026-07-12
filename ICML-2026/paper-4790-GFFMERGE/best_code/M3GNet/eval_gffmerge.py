#!/usr/bin/env python3
"""Evaluate GFFMERGE: merge, fine-tune, and evaluate on MD17 test set."""
import argparse, os, subprocess, sys, re
from pathlib import Path

os.environ.setdefault("MATGL_BACKEND", "DGL")
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")
os.environ.setdefault("TMPDIR", "/repo/tmp")
Path("/repo/tmp").mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parent

def run(cmd, desc=""):
    print(f"[{desc}] " + " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        print("STDERR:", p.stderr)
        raise RuntimeError(f"Command failed: {desc}")
    print(p.stdout)
    return p.stdout

def main():
    p = argparse.ArgumentParser(description="GFFMERGE evaluation")
    p.add_argument("--ckpt-a", required=True, help="Checkpoint for dataset A")
    p.add_argument("--ckpt-b", required=True, help="Checkpoint for dataset B")
    p.add_argument("--config-a", required=True, help="Config for dataset A")
    p.add_argument("--config-b", required=True, help="Config for dataset B")
    p.add_argument("--combined-config", default="configs/combined_quick.yaml")
    p.add_argument("--eval-config", default="configs/combined_val_eval.yaml")
    p.add_argument("--label-a", default="aspirin")
    p.add_argument("--label-b", default="uracil")
    p.add_argument("--output-dir", default="runs/repro_eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-ft", type=int, default=10)
    p.add_argument("--limit-ft", type=int, default=1000)
    p.add_argument("--lr-ft", type=float, default=1e-4)
    p.add_argument("--force-weight", type=float, default=0.1)
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--last-n-blocks", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--adaptive-reg", action="store_true", default=False, help="Use per-layer adaptive regularization in merge.")
    p.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled).")
    p.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max norm (0=disabled).")
    p.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"], help="LR schedule.")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout probability (0=disabled).")
    p.add_argument("--huber-delta", type=float, default=0.0, help="Huber loss delta for forces (0=MSE, 0.1 recommended).")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    merged_ckpt = out / "merged_closed_form.ckpt"
    ft_ckpt = out / "merged_ft.ckpt"
    switch_result = out / "switch_test_result.txt"

    # Step 1: GFFMERGE closed-form merge
    merge_cmd = [
        sys.executable, "scripts/merge_closed_form_individual_m3gnet.py",
        "--checkpoint", args.ckpt_a, "--checkpoint", args.ckpt_b,
        "--config", args.config_a, "--config", args.config_b,
        "--batch-size", str(args.batch_size),
        "--output-ckpt", str(merged_ckpt),
    ]
    if args.adaptive_reg:
        merge_cmd.append("--adaptive-reg")
    run(merge_cmd, "GFFMERGE merge")

    # Step 2: Fine-tune last N blocks
    ft_cmd = [
        sys.executable, "scripts/switch_finetune_energy_readout_last_block_m3gnet.py",
        "--config", args.combined_config,
        "--checkpoint", str(merged_ckpt),
        "--epochs", str(args.epochs_ft), "--limit", str(args.limit_ft),
        "--lr", str(args.lr_ft),
        "--force-weight", str(args.force_weight),
        "--energy-weight", str(args.energy_weight),
        "--seed", str(args.seed), "--last-n-blocks", str(args.last_n_blocks),
        "--output", str(ft_ckpt),
        "--source-checkpoint", f"{args.label_a}={args.ckpt_a}",
        "--source-checkpoint", f"{args.label_b}={args.ckpt_b}",
    ]
    if args.patience > 0:
        ft_cmd.extend(["--patience", str(args.patience)])
    if args.grad_clip > 0:
        ft_cmd.extend(["--grad-clip", str(args.grad_clip)])
    if args.lr_schedule != "constant":
        ft_cmd.extend(["--lr-schedule", args.lr_schedule])
    if args.dropout > 0:
        ft_cmd.extend(["--dropout", str(args.dropout)])
    if args.huber_delta > 0:
        ft_cmd.extend(["--huber-delta", str(args.huber_delta)])
    run(ft_cmd, "Fine-tune")

    # Step 3: Switch embedding evaluation
    run([
        sys.executable, "scripts/evaluate_switch_embeddings_m3gnet.py",
        "--split", "test",
        "--config", args.eval_config,
        "--checkpoint", str(ft_ckpt),
        "--source-checkpoint", f"{args.label_a}={args.ckpt_a}",
        "--source-checkpoint", f"{args.label_b}={args.ckpt_b}",
        "--save", str(switch_result),
    ], "Evaluate (switch)")

    # Parse and report key metrics
    metrics = {}
    for line in switch_result.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                metrics[k.strip()] = float(v.strip())
            except ValueError:
                pass
    
    print("\n=== GFFMERGE Final Results (seed=%d) ===" % args.seed)
    for k in sorted(metrics):
        print(f"  {k} = {metrics[k]:.6f}")
    energy_mae = metrics.get("test_Energy_MAE", "N/A")
    print(f"\n  test_Energy_MAE = {energy_mae}")
    force_mae = metrics.get("test_Force_MAE", "N/A")
    print(f"  test_Force_MAE = {force_mae}")

if __name__ == "__main__":
    main()
