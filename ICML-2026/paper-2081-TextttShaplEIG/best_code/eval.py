#!/autosota_cache/venv/bin/python3
"""
Reproduction evaluation script for ShaplEIG (paper 2081).
Evaluates ShaplEIG (EIGFunctionProperty) on Breast Cancer FI task.

Usage (inside container, from /repo):
    /autosota_cache/venv/bin/python3 eval.py --seeds 100 --dataset tabpfn_15 --eval-budget 200

Output: JSON summary at /autosota_cache/tmp/eval_results/summary.json
"""
import os, sys, argparse, json, logging
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TABPFN_DISABLE_PROGRESS"] = "1"
os.environ["TABPFN_CACHE_DIR"] = "/models/tabpfn"

import tabpfn.browser_auth
tabpfn.browser_auth.ensure_license_accepted = lambda *a, **kw: True

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "/repo/src")
os.chdir("/repo")

import torch
from xac.blackbox_functions import ShapIQGameBBF
from xac.applications import ShapiqShapleyApplication
from xac.surrogates import GPSurrogateConfig, HammingKernelConfig, MLMConfig, ConstantNoiseConfig
from xac.acquisition_functions import EIGFunctionProperty
from xac.acquisition_optimizers import Exhaustive
from xac.experimental_designs import ExperimentalDesignConfig
from xac.experimental_designs.experiment_runner import run_experiment
from xac.experiments import MetaExperimentConfig
from xac.utils.metrics import compute_mse
from functools import partial
from xac.surrogates.gp_surrogate import GPSurrogate
from pathlib import Path

def run_single_experiment(seed, dataset_name, eval_budget, output_dir,
                          init_design_factor=1, min_lengthscale=1e-6, amount_restarts=5,
                          warmstart=False, refit_interval=1):
    bbf = ShapIQGameBBF(name=dataset_name, seed=seed)
    n_players = bbf.dim
    init_design_size = max(init_design_factor * n_players, n_players + 1)
    iterations = eval_budget - init_design_size

    app = ShapiqShapleyApplication(init_design_factor=init_design_factor)

    surrogate_cfg = GPSurrogateConfig(
        kernel_config=HammingKernelConfig(min_lengthscale=min_lengthscale),
        missing_strategy="mean", warp=False,
        fit_config=MLMConfig(optimizer="default", amount_restarts=amount_restarts, run_all_attempts=False, warmstart=warmstart, refit_interval=refit_interval),
        noise_config=ConstantNoiseConfig(noise_level=1e-6),
    )
    
    acq_fn = EIGFunctionProperty()
    acq_opt = Exhaustive()
    ed_cfg = ExperimentalDesignConfig(iterations=iterations)
    meta_cfg = MetaExperimentConfig(seed=seed, debug_mode=True, skip_fitting=False, time_ops=False, scalability_mode=True)
    
    run_dir = Path(output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    ((prop_posts, _, _, _, prop_gt, _), _, _) = run_experiment(
        application=app, blackbox_fn=bbf, surrogate_cfg=surrogate_cfg,
        acquisition_fn=acq_fn, acquisition_optimizer=acq_opt,
        ed_cfg=ed_cfg, meta_cfg=meta_cfg, run_dir=str(run_dir),
    )
    
    results = {"seed": seed, "n_players": n_players, "init_design_size": init_design_size, "iterations": iterations}
    for i, prop_post in enumerate(prop_posts):
        mse = compute_mse(prop_post, prop_gt)
        total_evals = app.init_design_size + i + 1
        results[f"mse_at_{total_evals}"] = float(mse.item())
    
    return results

def main():
    parser = argparse.ArgumentParser(description="ShaplEIG Reproduction Evaluation")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--dataset", default="tabpfn_15", help="Dataset: tabpfn_15 or tabpfn_37")
    parser.add_argument("--eval-budget", type=int, default=200, help="Total evaluation budget")
    parser.add_argument("--output-dir", default="/autosota_cache/tmp/eval_results")
    parser.add_argument("--init-design-factor", type=int, default=1, help="Initial design factor (default: 1)")
    parser.add_argument("--min-lengthscale", type=float, default=1e-6, help="Min kernel lengthscale (default: 1e-6)")
    parser.add_argument("--amount-restarts", type=int, default=5, help="GP hyperparameter restarts (default: 5)")
    parser.add_argument("--warmstart", action="store_true", default=False, help="Enable GP warmstarting")
    parser.add_argument("--refit-interval", type=int, default=1, help="GP refit interval (default: 1)")
    args = parser.parse_args()

    all_results = []
    for seed in range(args.seeds):
        print(f"Seed {seed}/{args.seeds}...")
        result = run_single_experiment(seed, args.dataset, args.eval_budget, args.output_dir,
                                       init_design_factor=args.init_design_factor,
                                       min_lengthscale=args.min_lengthscale,
                                       amount_restarts=args.amount_restarts,
                                       warmstart=args.warmstart,
                                       refit_interval=args.refit_interval)
        all_results.append(result)
        mse_key = f"mse_at_{args.eval_budget}"
        print(f"  MSE@{args.eval_budget}: {result.get(mse_key, 'N/A'):.6e}" if mse_key in result else f"  Done")
    
    mse_key = f"mse_at_{args.eval_budget}"
    mse_values = [r[mse_key] for r in all_results if mse_key in r]
    
    summary = {
        "dataset": args.dataset,
        "eval_budget": args.eval_budget,
        "num_seeds": args.seeds,
        "config": {
            "init_design_factor": args.init_design_factor,
            "min_lengthscale": args.min_lengthscale,
            "amount_restarts": args.amount_restarts,
            "warmstart": args.warmstart,
            "refit_interval": args.refit_interval,
        },
        "mse_values": mse_values,
        "mse_mean": float(np.mean(mse_values)) if mse_values else None,
        "mse_std": float(np.std(mse_values)) if mse_values else None,
        "paper_value": 1.16e-08,
        "per_seed_results": all_results,
    }
    
    output_path = Path(args.output_dir) / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== RESULTS ===")
    print(f"Dataset: {args.dataset}, Seeds: {args.seeds}, Budget: {args.eval_budget}")
    if summary['mse_mean']:
        print(f"MSE Mean: {summary['mse_mean']:.6e}")
        print(f"MSE Std:  {summary['mse_std']:.6e}")
    print(f"Paper: ~1.16e-08")
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
