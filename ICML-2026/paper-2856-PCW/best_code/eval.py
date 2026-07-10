#!/usr/bin/env python3
"""Evaluation script for paper 2856: compute -log(DKL) and TPR from generation results."""
import os, sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.stats import norm

# Override HF endpoint - the mirror may be broken in some environments
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_config import create_config
from run_experiment_combined import run_single_experiment

def main():
    parser = argparse.ArgumentParser(description="Evaluate watermark metrics")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset", default="c4")
    parser.add_argument("--gamma", type=float, default=0.9462668142137298)
    parser.add_argument("--delta", type=float, default=8.830725909368594)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results_final")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for detection")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling top-p threshold (default: 1.0 = disabled)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--entropy-threshold", type=float, default=0.0,
                        help="Entropy gating threshold as fraction of max entropy (0.0 = disabled, e.g. 0.3)")
    args = parser.parse_args()

    # Build config for the watermarked experiment
    top_p_suffix = f"_tp{args.top_p:.2f}" if args.top_p < 1.0 else ""
    temp_suffix = f"_T{args.temperature:.1f}" if args.temperature != 0.7 else ""
    ent_suffix = f"_ent{args.entropy_threshold:.2f}" if args.entropy_threshold > 0 else ""
    exp_name = f"eval_{args.model}_{args.dataset}_g{args.gamma:.2f}_d{args.delta:.5f}{top_p_suffix}{temp_suffix}{ent_suffix}"
    config = create_config(
        model_key=args.model,
        dataset_key=args.dataset,
        num_samples=args.num_samples,
        truncate_at=50,
        max_new_tokens=200,
        gamma=args.gamma,
        delta=args.delta,
        top_p=args.top_p,
        temperature=args.temperature,
        experiment_name=exp_name,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    config["save_mode"] = "compact"
    config["cache_dir"] = os.environ.get("HF_HOME", "/autosota_cache/hf")
    if args.entropy_threshold > 0:
        config["entropy_threshold"] = args.entropy_threshold

    print(f"Running experiment: {exp_name}")
    success, path = run_single_experiment(config, verbose=True)

    if not success:
        print("Experiment failed!")
        sys.exit(1)

    # Load results and compute metrics
    with open(path) as f:
        data = json.load(f)

    summary = data["experiment_summary"]
    z_scores = [r["watermarked"]["summary"]["z_score"] for r in data["results"]]

    z_critical = norm.ppf(1 - args.alpha)
    mean_kl = summary["mean_kl"]
    neg_log_kl = -np.log(max(mean_kl, 1e-12))
    tpr = np.mean([z > z_critical for z in z_scores])

    metrics = {
        "-log(DKL)": round(float(neg_log_kl), 4),
        "TPR": round(float(tpr), 4),
        "mean_kl": round(float(mean_kl), 6),
        "mean_z_score": round(float(summary["mean_z_score"]), 4),
        "significance_level": args.alpha,
        "z_critical": round(float(z_critical), 4),
        "n_samples": len(z_scores),
        "num_tokens": args.num_samples,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    main()
