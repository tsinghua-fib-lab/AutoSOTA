#!/usr/bin/env python3
"""
Sweep over scale values to find optimal configuration for label generator evaluation.
Runs fast evaluation on a small subset of latents with different scale configurations.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Scale grid candidates
SCALE_GRIDS = {
    # Original paper config
    "paper_default": [0.5, 0.8, 1.3, 2.1, 3.4, 5.5],
    
    # Log-spaced grids with different ranges
    "log_wide_6": [0.3, 0.55, 1.0, 1.8, 3.3, 6.0],
    "log_wide_8": [0.2, 0.4, 0.7, 1.2, 2.0, 3.5, 6.0, 10.0],
    "log_wide_10": [0.2, 0.35, 0.6, 1.0, 1.6, 2.6, 4.2, 6.8, 11.0, 18.0],
    "log_dense_8": [0.3, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.9],
    "log_dense_10": [0.25, 0.4, 0.65, 1.0, 1.6, 2.5, 4.0, 6.3, 10.0, 16.0],
    
    # Higher range (more abstract labels)
    "high_range_6": [0.8, 1.5, 2.5, 4.0, 6.5, 10.5],
    "high_range_8": [0.5, 1.0, 1.8, 3.0, 5.0, 8.0, 13.0, 21.0],
    
    # Lower range (more specific labels)
    "low_range_6": [0.2, 0.35, 0.6, 1.0, 1.7, 2.8],
    "low_range_8": [0.15, 0.25, 0.4, 0.65, 1.0, 1.6, 2.5, 4.0],
    
    # Finer control: 12 values
    "ultra_fine_12": [0.15, 0.25, 0.4, 0.6, 0.9, 1.35, 2.0, 3.0, 4.5, 6.75, 10.0, 15.0],
    
    # Adapted: keep original base but add extremes
    "extended_8": [0.2, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.9],
    "extended_10": [0.1, 0.3, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.9, 14.4],
}

# num_labels_per_scale candidates
NUM_LABELS_CANDIDATES = [1, 2]

def run_sweep(config_template, scale_values, num_labels, latent_count, output_dir, sweep_name):
    """Run a single sweep configuration."""
    # Create config
    config = json.loads(json.dumps(config_template))
    config["scale_values"] = scale_values
    config["num_labels_per_scale"] = num_labels
    config["max_latents"] = latent_count
    config["run_id"] = f"sweep_{sweep_name}"
    config["num_reward_samples"] = 5  # Keep same for fairness
    
    config_path = os.path.join(output_dir, f"config_{sweep_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    result_path = os.path.join(output_dir, f"eval_results_layer_19_{sweep_name}.json")
    
    cmd = (
        f"cd /repo && "
        f"HF_HOME=/autosota_cache/hf WANDB_MODE=disabled "
        f"python3 evals/generation_scoring/run_eval.py "
        f"--config-file {config_path} "
        f"--output-dir {output_dir} "
        f"--no-wandb"
    )
    
    print(f"Running: {sweep_name}")
    print(f"  Scales: {scale_values}")
    print(f"  Labels per scale: {num_labels}")
    print(f"  Latents: {latent_count}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    
    # Find the actual result file (run_eval overrides output_volume_path)
    result_files = list(Path(output_dir).glob(f"eval_results_layer_19_sweep_{sweep_name}*"))
    if not result_files:
        result_files = list(Path(output_dir).glob("eval_results_layer_19_*"))
        result_files = [f for f in result_files if f"layer_19_fast_eval" not in str(f) and f"test_small" not in str(f)]
    
    if result_files:
        actual_result = str(result_files[0])
    else:
        actual_result = result_path
    
    return result, actual_result

def compute_metrics(result_path):
    """Compute hit rate and coverage from results file."""
    cmd_hit = f"cd /repo && python3 evals/generation_scoring/compute_mean_max_hit_rate.py {result_path}"
    result = subprocess.run(cmd_hit, shell=True, capture_output=True, text=True, timeout=60)
    
    hit_rate = None
    for line in result.stdout.split("\n"):
        if "FINAL METRIC - Mean of max hit rates:" in line:
            hit_rate = float(line.split(":")[-1].strip())
    
    cmd_cov = f"cd /repo && python3 evals/generation_scoring/compute_coverage.py {result_path}"
    result_cov = subprocess.run(cmd_cov, shell=True, capture_output=True, text=True, timeout=60)
    
    coverage = None
    for line in result_cov.stdout.split("\n"):
        if "Coverage:" in line:
            coverage = float(line.split(":")[-1].strip().replace("%", ""))
    
    return hit_rate, coverage, result.stdout

def main():
    config_template_path = "/repo/evals/generation_scoring/configs/fast_eval_50.json"
    with open(config_template_path) as f:
        config_template = json.load(f)
    
    output_dir = "/repo/evals/generation_scoring/results/sweep_scales"
    os.makedirs(output_dir, exist_ok=True)
    
    # Quick sweep on 20 latents for speed
    latent_count = 20
    
    results = []
    
    # Test selected grids
    grids_to_test = [
        "paper_default",
        "log_wide_6",
        "log_dense_8",
        "high_range_6",
        "low_range_6",
        "extended_8",
        "log_dense_10",
        "ultra_fine_12",
        "log_wide_8",
        "extended_10",
    ]
    
    # Also test num_labels_per_scale=2 for top configs
    for grid_name in grids_to_test[:6]:  # First 6 at num_labels=1
        scale_values = SCALE_GRIDS[grid_name]
        sweep_name = f"{grid_name}_n1"
        
        result, actual_result = run_sweep(
            config_template, scale_values, 1, latent_count, output_dir, sweep_name
        )
        
        if result.returncode == 0:
            hit_rate, coverage, stdout = compute_metrics(actual_result)
            results.append({
                "grid": grid_name,
                "scales": scale_values,
                "num_labels": 1,
                "hit_rate": hit_rate,
                "coverage": coverage,
            })
            print(f"  -> Hit Rate: {hit_rate}, Coverage: {coverage}")
        else:
            print(f"  -> FAILED: {result.stderr[-500:]}")
    
    # Test num_labels_per_scale=2 for top candidates
    results_sorted = sorted([r for r in results if r["hit_rate"] is not None], 
                           key=lambda x: x["hit_rate"], reverse=True)
    
    if results_sorted:
        top_grids = [r["grid"] for r in results_sorted[:3]]
        for grid_name in top_grids:
            scale_values = SCALE_GRIDS[grid_name]
            sweep_name = f"{grid_name}_n2"
            
            result, actual_result = run_sweep(
                config_template, scale_values, 2, latent_count, output_dir, sweep_name
            )
            
            if result.returncode == 0:
                hit_rate, coverage, stdout = compute_metrics(actual_result)
                results.append({
                    "grid": grid_name,
                    "scales": scale_values,
                    "num_labels": 2,
                    "hit_rate": hit_rate,
                    "coverage": coverage,
                })
                print(f"  -> Hit Rate: {hit_rate}, Coverage: {coverage}")
    
    # Print final ranking
    print("\n" + "="*80)
    print("FINAL RANKING (on 20-latent subset)")
    print("="*80)
    results_sorted = sorted([r for r in results if r["hit_rate"] is not None], 
                           key=lambda x: x["hit_rate"], reverse=True)
    for i, r in enumerate(results_sorted):
        print(f"{i+1}. {r[grid]} (n={r[num_labels]}): Hit={r[hit_rate]:.4f}, Cov={r[coverage]:.1f}% | scales={r[scales]}")
    
    # Save sweep results
    sweep_results_path = os.path.join(output_dir, "sweep_results.json")
    serializable = [{**r, "scales": list(r["scales"])} for r in results_sorted]
    with open(sweep_results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {sweep_results_path}")

if __name__ == "__main__":
    main()
