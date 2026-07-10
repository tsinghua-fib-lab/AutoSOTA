"""
Batched training runs for consistency bridges.

Repeats `train_bridge.py`'s training run several times (varying the seed),
collects per-run KL metrics, and writes mean ± std to an outer folder.

Usage:
    python run_repeats.py --config configs/ou.yaml --gpu 0 --num_repeats 5
    python run_repeats.py --config configs/ou.yaml --gpu 0 --num_repeats 5 --train.lr 3e-4
"""

import os
import argparse
import yaml
import json
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Run a consistency-bridge training multiple times")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num_repeats", type=int, required=True, help="Number of repeated runs")
    parser.add_argument("--outdir", type=str, default=None, help="Outer output directory (default: outputs/<problem_name>/<timestamp>_repeats)")
    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i].lstrip("-")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1
    return args, overrides


def main():
    args, overrides = parse_args()

    # GPU set before importing JAX
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Import after CUDA_VISIBLE_DEVICES is set
    from train_bridge import apply_cli_overrides, run_training

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    apply_cli_overrides(cfg, overrides)

    base_seed = int(cfg.get("train", {}).get("seed", 0))
    problem_name = cfg["problem"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.outdir is not None:
        outer_dir = args.outdir
    else:
        outer_dir = os.path.join("outputs", problem_name, f"{timestamp}_repeats")
    os.makedirs(outer_dir, exist_ok=True)

    # save outer config + meta
    with open(os.path.join(outer_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    with open(os.path.join(outer_dir, "meta.json"), "w") as f:
        json.dump({
            "num_repeats": args.num_repeats,
            "base_seed": base_seed,
            "timestamp": timestamp,
        }, f, indent=2)

    print("=" * 60)
    print(f"Outer directory: {outer_dir}")
    print(f"num_repeats: {args.num_repeats}, base_seed: {base_seed}")
    print("=" * 60)

    per_run_results = []
    for i in range(args.num_repeats):
        run_outdir = os.path.join(outer_dir, f"{timestamp}_run{i + 1}")
        cfg["train"]["seed"] = base_seed + i

        print()
        print("#" * 60)
        print(f"# Run {i + 1}/{args.num_repeats} — seed={cfg['train']['seed']}")
        print("#" * 60)

        metrics = run_training(cfg, run_outdir)
        per_run_results.append(metrics)

    # aggregate metrics
    metric_keys = sorted({k for r in per_run_results for k in r.keys()})
    aggregated = {"num_repeats": args.num_repeats}
    for k in metric_keys:
        values = [r[k] for r in per_run_results if k in r]
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        std = var ** 0.5
        aggregated[k] = {"mean": mean, "std": std, "values": values}

    with open(os.path.join(outer_dir, "results.json"), "w") as f:
        json.dump(aggregated, f, indent=2)

    print()
    print("=" * 60)
    print(f"Aggregated results ({args.num_repeats} runs) saved to {outer_dir}/results.json")
    for k in metric_keys:
        print(f"  {k}: {aggregated[k]['mean']:.6f} ± {aggregated[k]['std']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
