#!/usr/bin/env python3
"""
scChord Multi-Run Evaluation Script for Paper Reproduction.
Runs training and evaluation for multiple seeds and reports mean/std.

Usage:
    python3 run_experiment.py --data_path /repo/data/GSE100866_CBMC.h5ad --seeds 0,10,20,30,40
"""
import os, sys, argparse, subprocess, json, numpy as np
from pathlib import Path

STAGE1_CMD = """python3 /repo/train_stage1_vae.py \
    --data_path {data_path} \
    --output_dir {output_dir} \
    --device {device} \
    --epochs 600 \
    --n_top_genes 1000 \
    --batch_size 512 \
    --lr 2e-4 \
    --dz 32 \
    --beta_kl 0.8 \
    --dist_type ZINB \
    --use_raw_for_nb \
    --seed {seed} \
    --split_seed 0 \
    --num_workers 4"""

STAGE2_CMD = """python3 /repo/train_stage2_cfm.py \
    --data_path {data_path} \
    --vae_path {vae_path} \
    --output_dir {output_dir} \
    --device {device} \
    --epochs 200 \
    --n_top_genes 1000 \
    --batch_size 512 \
    --lr 1e-4 \
    --dc 512 \
    --p_uncond 0.2 \
    --lambda_cons 0.2 \
    --n_steps 50 \
    --cfg_scale 3.0 \
    --ode_method dopri5 \
    --seed {seed} \
    --split_seed 0 \
    --num_workers 4"""

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"{'='*60}")
    print(f"CMD: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with code {result.returncode}")
        return False
    print(f"{desc} completed successfully!")
    return True

def parse_metrics_from_output(output_dir):
    """Parse metrics from stage2 output directory."""
    results = {}

    # Read predictions and ground truth
    pred_path = Path(output_dir) / 'predictions.npy'
    truth_path = Path(output_dir) / 'ground_truth.npy'

    if pred_path.exists() and truth_path.exists():
        import numpy as np
        from metrics import evaluate_predictions

        pred = np.load(pred_path)
        truth = np.load(truth_path)
        eval_results = evaluate_predictions(pred, truth, verbose=False)

        results['pcc_p'] = float(eval_results['pcc_protein_mean'])
        results['pcc_c'] = float(eval_results['pcc_cell_mean'])
        results['cmd_p'] = float(eval_results['cmd_protein'])
        results['cmd_c'] = float(eval_results['cmd_cell'])
        results['rmse'] = float(eval_results['rmse'])

        print(f"  PCC-P: {results['pcc_p']:.4f}, PCC-C: {results['pcc_c']:.4f}")
        print(f"  CMD-P: {results['cmd_p']:.4f}, CMD-C: {results['cmd_c']:.4f}")
        print(f"  RMSE:  {results['rmse']:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--seeds', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--base_dir', type=str, default='/repo/outputs')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    print(f"Running experiment with seeds: {seeds}")

    all_metrics = []

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        stage1_dir = f"{args.base_dir}/seed{seed}/stage1"
        stage2_dir = f"{args.base_dir}/seed{seed}/stage2"

        # Stage 1
        cmd = STAGE1_CMD.format(
            data_path=args.data_path, output_dir=stage1_dir,
            device=args.device, seed=seed)
        if not run_cmd(cmd, f"Stage 1 (seed={seed})"):
            print(f"SKIPPING seed {seed} due to Stage 1 failure")
            continue

        # Stage 2
        cmd = STAGE2_CMD.format(
            data_path=args.data_path, vae_path=f"{stage1_dir}/vae_best.pt",
            output_dir=stage2_dir, device=args.device, seed=seed)
        if not run_cmd(cmd, f"Stage 2 (seed={seed})"):
            print(f"SKIPPING seed {seed} due to Stage 2 failure")
            continue

        # Parse metrics
        metrics = parse_metrics_from_output(stage2_dir)
        if metrics:
            metrics['seed'] = seed
            all_metrics.append(metrics)

    # Report summary
    if all_metrics:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        for key in ['pcc_p', 'pcc_c', 'cmd_p', 'cmd_c', 'rmse']:
            vals = [m[key] for m in all_metrics]
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"{key}: {mean:.4f} ± {std:.4f}  (values: {vals})")

        # Save results
        with open(f"{args.base_dir}/results.json", 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nResults saved to {args.base_dir}/results.json")

if __name__ == '__main__':
    main()
