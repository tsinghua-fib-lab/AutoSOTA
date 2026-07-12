#!/usr/bin/env python3
"""
Run the full GFFMERGE experiment pipeline for M3GNet on MD17 Aspirin+Uracil.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MATGL_BACKEND", "DGL")
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")

import yaml

M3GNET_ROOT = Path(__file__).resolve().parents[1]

def run_cmd(cmd: List[str], description: str = "") -> str:
    """Run a command and return stdout. Raise on failure."""
    cmd_str = " ".join(str(x) for x in cmd)
    print(f"[RUN] {cmd_str}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(M3GNET_ROOT))
    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {description}")
    return "".join(output_lines)


def pick_checkpoint(run_dir: Path) -> Path:
    best = run_dir / "best_chk.ckpt"
    if best.exists():
        return best
    ckpt_dir = run_dir / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return ckpts[0]


def update_config_seed(config_path: Path, seed: int, epochs: Optional[int] = None, lr: Optional[float] = None):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.setdefault("train", {})["seed"] = seed
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    if lr is not None:
        cfg["train"]["lr"] = lr
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def train_model(config: Path, workdir: Path, seed: int, epochs: int, lr: Optional[float] = None) -> float:
    update_config_seed(config, seed, epochs=epochs, lr=lr)
    cmd = [
        sys.executable, "scripts/train_m3gnet.py",
        "--config", str(config),
        "--workdir", str(workdir),
        "--device", "auto",
        "--plot",
        "--epochs", str(epochs),
    ]
    if lr is not None:
        cmd += ["--lr", str(lr)]
    output = run_cmd(cmd, f"Training {config} -> {workdir}")
    match = re.search(r"Training time: ([0-9.]+)s", output)
    training_time = float(match.group(1)) if match else None
    if training_time:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "training_time.txt").write_text(str(training_time))
    return training_time


def eval_checkpoint(checkpoint: Path, config: Path, split: str, output_csv: Optional[Path] = None) -> Dict[str, float]:
    cmd = [
        sys.executable, "scripts/evaluate_m3gnet.py",
        "--split", split,
        "--checkpoint", str(checkpoint),
        "--config", str(config),
    ]
    if output_csv:
        cmd += ["--output-csv", str(output_csv)]
    output = run_cmd(cmd, f"Evaluating {checkpoint} on {split}")
    
    # Parse metrics from output
    metrics = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        for part in line.strip().split("|"):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    pass
    return metrics


def parse_kv_metrics(path: Path) -> Dict[str, float]:
    metrics = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        try:
            metrics[key.strip()] = float(val.strip())
        except ValueError:
            continue
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run GFFMERGE experiment pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--epochs-indiv", type=int, default=75, help="Epochs for individual models")
    parser.add_argument("--epochs-combined", type=int, default=50, help="Epochs for combined model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--force-weight", type=float, default=0.1, help="Force loss weight")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Energy loss weight")
    parser.add_argument("--skip-training", action="store_true", help="Skip training steps")
    parser.add_argument("--only-merge", action="store_true", help="Only run merge and eval steps")
    parser.add_argument("--dataset-a", type=str, default="aspirin", help="First dataset name")
    parser.add_argument("--dataset-b", type=str, default="uracil", help="Second dataset name")
    parser.add_argument("--quick-test", action="store_true", help="Run with 3 epochs for quick testing")
    args = parser.parse_args()

    if args.quick_test:
        args.epochs_indiv = 3
        args.epochs_combined = 3

    ds_a = args.dataset_a
    ds_b = args.dataset_b
    seed = args.seed
    
    config_a = M3GNET_ROOT / "configs" / f"{ds_a}_quick.yaml"
    config_b = M3GNET_ROOT / "configs" / f"{ds_b}_quick.yaml"
    config_combined = M3GNET_ROOT / "configs" / "combined_quick.yaml"
    config_combined_eval = M3GNET_ROOT / "configs" / "combined_val_eval.yaml"
    
    run_a = M3GNET_ROOT / "runs" / f"{ds_a}_seed{seed}"
    run_b = M3GNET_ROOT / "runs" / f"{ds_b}_seed{seed}"
    run_combined = M3GNET_ROOT / "runs" / f"combined_seed{seed}"
    merged_dir = M3GNET_ROOT / "runs" / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    mean_ckpt = merged_dir / f"mean_seed{seed}.ckpt"
    closed_form_ckpt = merged_dir / f"closed_form_individual_seed{seed}.ckpt"

    print(f"=== GFFMERGE Pipeline: {ds_a}+{ds_b}, seed={seed} ===")
    
    # Write combined_val_eval config
    cfg_eval = {
        "model": {"pretrained_name": "M3GNet-ANI-1x-Subset-PES", "cutoff": 5.0},
        "train": {
            "seed": seed, "batch_size": args.batch_size, "lr": args.lr,
            "energy_weight": args.energy_weight, "force_weight": args.force_weight,
            "stress_weight": 0.0, "decay_steps": 1000, "decay_alpha": 0.01, "num_workers": 0,
        },
        "data": {
            "val_path": "data/prepared/combined_val.extxyz",
            "test_path": "data/prepared/combined_test.extxyz",
            "cache_dir": "data/cache/combined_val",
        },
        "output": {"run_dir": "runs/combined_val_eval"},
    }
    config_combined_eval.parent.mkdir(parents=True, exist_ok=True)
    config_combined_eval.write_text(yaml.safe_dump(cfg_eval, sort_keys=False), encoding="utf-8")
    print(f"Wrote eval config to {config_combined_eval}")

    if not args.only_merge:
        # Step 1: Train individual models
        print(f"\n=== Step 1: Training {ds_a} (seed={seed}, epochs={args.epochs_indiv}) ===")
        train_model(config_a, run_a, seed, args.epochs_indiv, args.lr)

        print(f"\n=== Step 2: Training {ds_b} (seed={seed}, epochs={args.epochs_indiv}) ===")
        train_model(config_b, run_b, seed, args.epochs_indiv, args.lr)

        # Step 3: Train combined model
        print(f"\n=== Step 3: Training combined (seed={seed}, epochs={args.epochs_combined}) ===")
        update_config_seed(config_combined, seed, epochs=args.epochs_combined, lr=args.lr)
        train_model(config_combined, run_combined, seed, args.epochs_combined, args.lr)

        # Step 4: Evaluate individual models
        print(f"\n=== Step 4: Evaluate individual models ===")
        ckpt_a = pick_checkpoint(run_a)
        ckpt_b = pick_checkpoint(run_b)
        ckpt_combined = pick_checkpoint(run_combined)
        
        eval_individual = eval_checkpoint(ckpt_a, (run_a / "config.yaml"), "test",
                                          M3GNET_ROOT / "runs" / f"metrics_{ds_a}_test_seed{seed}.csv")
        eval_individual_b = eval_checkpoint(ckpt_b, (run_b / "config.yaml"), "test",
                                            M3GNET_ROOT / "runs" / f"metrics_{ds_b}_test_seed{seed}.csv")
        eval_combined = eval_checkpoint(ckpt_combined, (run_combined / "config.yaml"), "test",
                                        M3GNET_ROOT / "runs" / f"metrics_combined_test_seed{seed}.csv")
    else:
        ckpt_a = pick_checkpoint(run_a)
        ckpt_b = pick_checkpoint(run_b)
        ckpt_combined = pick_checkpoint(run_combined)
        print(f"Using checkpoints: {ckpt_a}, {ckpt_b}")

    # Step 5: Mean merge
    print(f"\n=== Step 5: Mean merge + eval ===")
    run_cmd([
        sys.executable, "scripts/merge_m3gnet_checkpoints.py",
        "--ckpt", str(ckpt_a), "--ckpt", str(ckpt_b),
        "--output-ckpt", str(mean_ckpt),
    ], "Mean merge")
    eval_mean = eval_checkpoint(mean_ckpt, config_combined_eval, "test",
                                M3GNET_ROOT / "runs" / f"metrics_mean_merge_test_seed{seed}.csv")

    # Step 6: GFFMERGE closed-form individual merge
    print(f"\n=== Step 6: GFFMERGE closed-form merge + eval ===")
    merge_output = run_cmd([
        sys.executable, "scripts/merge_closed_form_individual_m3gnet.py",
        "--checkpoint", str(ckpt_a), "--checkpoint", str(ckpt_b),
        "--config", str(run_a / "config.yaml"), "--config", str(run_b / "config.yaml"),
        "--batch-size", "64",
        "--output-ckpt", str(closed_form_ckpt),
    ], "GFFMERGE closed-form merge")
    merge_time_match = re.search(r"Closed-form individual merge compute time: ([0-9.]+)s", merge_output)
    merge_time = float(merge_time_match.group(1)) if merge_time_match else None
    eval_closed_form = eval_checkpoint(closed_form_ckpt, config_combined_eval, "test",
                                       M3GNET_ROOT / "runs" / f"metrics_closed_form_individual_test_seed{seed}.csv")

    # Step 7: Switch embedding evaluation
    print(f"\n=== Step 7: Switch embedding evaluation ===")
    switch_metrics_path = M3GNET_ROOT / "runs" / f"closed_form_individual_switch_test_seed{seed}.txt"
    run_cmd([
        sys.executable, "scripts/evaluate_switch_embeddings_m3gnet.py",
        "--split", "test",
        "--config", str(config_combined_eval),
        "--checkpoint", str(closed_form_ckpt),
        "--source-checkpoint", f"{ds_a}={ckpt_a}",
        "--source-checkpoint", f"{ds_b}={ckpt_b}",
        "--save", str(switch_metrics_path),
    ], "Switch embedding evaluation")
    switch_metrics = parse_kv_metrics(switch_metrics_path) if switch_metrics_path.exists() else {}

    # Step 8: Fine-tune last 3 blocks
    print(f"\n=== Step 8: Fine-tune last 3 blocks ===")
    ft_results_dir = M3GNET_ROOT / "runs" / f"finetune_last_block_grid_seed{seed}"
    ft_results_dir.mkdir(parents=True, exist_ok=True)
    
    EPOCH_LIMIT_PAIRS = [
        (10, 125), (10, 250), (10, 500), (10, 1000),
        (20, 125), (20, 250), (20, 500), (20, 750),
        (30, 125), (30, 250), (30, 500), (40, 250),
    ]
    LR_LIST = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    
    source_args = [
        "--source-checkpoint", f"{ds_a}={ckpt_a}",
        "--source-checkpoint", f"{ds_b}={ckpt_b}",
    ]
    
    ft_summary_rows = []
    for epochs_ft, limit in EPOCH_LIMIT_PAIRS:
        best_loss = (float("inf"), None, None)
        for lr_ft in LR_LIST:
            out_ckpt = ft_results_dir / f"closed_form_ft_last3_ep{epochs_ft}_lim{limit}_lr{lr_ft}.ckpt"
            metrics_path = ft_results_dir / f"val_metrics_ep{epochs_ft}_lim{limit}_lr{lr_ft}.txt"
            
            print(f"  FT: epochs={epochs_ft} limit={limit} lr={lr_ft}")
            run_cmd([
                sys.executable, "scripts/switch_finetune_energy_readout_last_block_m3gnet.py",
                "--config", str(run_combined / "config.yaml"),
                "--checkpoint", str(closed_form_ckpt),
                "--epochs", str(epochs_ft), "--limit", str(limit), "--lr", str(lr_ft),
                "--force-weight", str(args.force_weight), "--energy-weight", str(args.energy_weight),
                "--seed", str(seed), "--last-n-blocks", "3",
                "--output", str(out_ckpt),
            ] + source_args, f"FT {epochs_ft}/{limit}/{lr_ft}")
            
            run_cmd([
                sys.executable, "scripts/evaluate_switch_embeddings_m3gnet.py",
                "--split", "val",
                "--config", str(config_combined_eval),
                "--checkpoint", str(out_ckpt),
                "--save", str(metrics_path),
            ] + source_args, f"Validate FT {epochs_ft}/{limit}/{lr_ft}")
            
            val_metrics = parse_kv_metrics(metrics_path)
            val_loss = val_metrics.get("val_Total_Loss", float("inf"))
            if val_loss < best_loss[0]:
                best_loss = (val_loss, lr_ft, out_ckpt)
        
        if best_loss[2] is not None:
            test_metrics_path = ft_results_dir / f"test_best_val_loss_ep{epochs_ft}_lim{limit}_lr{best_loss[1]}.txt"
            run_cmd([
                sys.executable, "scripts/evaluate_switch_embeddings_m3gnet.py",
                "--split", "test",
                "--config", str(config_combined_eval),
                "--checkpoint", str(best_loss[2]),
                "--save", str(test_metrics_path),
            ] + source_args, f"Test eval FT best {epochs_ft}/{limit}")
            test_m = parse_kv_metrics(test_metrics_path)
            ft_summary_rows.append({
                "epochs": epochs_ft, "limit": limit, "best_lr": best_loss[1],
                "test_Energy_MAE": test_m.get("test_Energy_MAE"),
                "test_Energy_RMSE": test_m.get("test_Energy_RMSE"),
                "test_Force_MAE": test_m.get("test_Force_MAE"),
                "test_Force_RMSE": test_m.get("test_Force_RMSE"),
            })

    if ft_summary_rows:
        ft_summary_path = ft_results_dir / f"grid_summary_seed{seed}.csv"
        with ft_summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ft_summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(ft_summary_rows)
        print(f"Wrote FT summary to {ft_summary_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for label, m in [
        (f"{ds_a} (individual)", eval_checkpoint(ckpt_a, run_a / "config.yaml", "test")),
        (f"{ds_b} (individual)", eval_checkpoint(ckpt_b, run_b / "config.yaml", "test")),
        ("Combined", eval_checkpoint(ckpt_combined, run_combined / "config.yaml", "test")),
        ("Mean Merge", eval_mean),
        ("GFFMERGE (standard)", eval_closed_form),
        ("GFFMERGE (switch)", switch_metrics),
    ]:
        print(f"\n{label}:")
        for k in sorted(m):
            print(f"  {k} = {m[k]:.6f}")

    if ft_summary_rows:
        print(f"\nBest FT result:")
        best_ft = min(ft_summary_rows, key=lambda r: r.get("test_Total_Loss", float("inf")) if r.get("test_Total_Loss") else float("inf"))
        for k, v in best_ft.items():
            print(f"  {k} = {v}")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
