#!/usr/bin/env python3
"""Gradient-based post-processing for NeuroLDS output.
Applies projected gradient descent on the predicted points to minimize D2_star.
Usage: python3 grad_postproc.py --input results/run_name/run_name.txt --output results/run_name/
"""
import numpy as np
import torch
import argparse
import os
import sys

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import seqL2star, USE_LOSS_FP64

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the txt output file with predicted points")
    parser.add_argument("--steps", type=int, default=200, help="Number of gradient descent steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for point optimization")
    parser.add_argument("--output", default=None, help="Output directory for post-processed results")
    args = parser.parse_args()
    
    # Parse predicted points from the txt file
    points = []
    in_points_section = False
    in_discrepancy_section = False
    
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Predicted points"):
                in_points_section = True
                continue
            if in_points_section:
                if line.startswith("Index"):
                    continue
                if line.startswith("Discrepancy"):
                    in_points_section = False
                    continue
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        points.append(coords)
                    except ValueError:
                        continue
    
    if not points:
        print("ERROR: Could not parse predicted points from input file")
        sys.exit(1)
    
    N = len(points)
    d = len(points[0])
    print(f"Loaded {N} points in dimension {d}")
    
    # Convert to tensor
    pred_t = torch.tensor(points, dtype=torch.float32)
    
    # Compute initial D2_star
    if USE_LOSS_FP64:
        star_fn = lambda x: seqL2star(x.to(dtype=torch.float64)).to(dtype=x.dtype)
    else:
        star_fn = seqL2star
    
    with torch.no_grad():
        D_init_seq = star_fn(pred_t)
        D_init = D_init_seq[-1].item()
    print(f"Initial D2_star at N={N}: {D_init:.8f}")
    
    # Apply projected gradient descent
    pred_opt = pred_t.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([pred_opt], lr=args.lr)
    
    best_D = D_init
    best_points = pred_t.clone()
    
    for step in range(args.steps):
        opt.zero_grad()
        D_seq = star_fn(pred_opt)
        loss = D_seq[-1]  # D2_star at N
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred_opt.clamp_(0.0, 1.0)
        
        D_current = loss.item()
        if D_current < best_D:
            best_D = D_current
            best_points = pred_opt.detach().clone()
        
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  Step {step:4d}: D2_star = {D_current:.8f}  (best: {best_D:.8f})")
    
    improvement = (D_init - best_D) / D_init * 100.0
    print(f"\nFinal: D2_star = {best_D:.8f} (improvement: {improvement:+.2f}%)")
    
    # Save post-processed points if output dir specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(args.output, f"{base}_postproc.txt")
        best_np = best_points.numpy()
        with open(out_path, "w") as f:
            f.write(f"Post-processed points (gradient descent, {args.steps} steps, D2_star={best_D:.8f})\n")
            f.write(f"Improvement: {improvement:+.2f}%\n")
            f.write("Index  " + "  ".join([f"x{j}" for j in range(d)]) + "\n")
            for i in range(N):
                coords = "  ".join([f"{val:.8f}" for val in best_np[i]])
                f.write(f"{i}  {coords}\n")
        print(f"Saved post-processed points to {out_path}")
    
    return best_D, improvement

if __name__ == "__main__":
    main()
