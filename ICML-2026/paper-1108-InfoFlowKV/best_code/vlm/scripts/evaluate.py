#!/usr/bin/env python3
"""
Run BLINK experiments based on config.

Usage:
    python scripts/evaluate.py --config configs/blink_counting.yaml
"""

import argparse
import subprocess
import sys
import tempfile
import yaml


def to_list(v):
    return v if isinstance(v, list) else [v]


def run(script, cfg):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    subprocess.run([sys.executable, script, "--config", f.name], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/blink_counting.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    datasets = to_list(cfg["dataset"])
    ratios = to_list(cfg.get("recompute_ratio", 0.15))
    chunks = to_list(cfg.get("chunk_k", [None]))
    num_samples = cfg.get("num_samples")
    run_baseline = cfg.get("run_baseline", False)
    run_recompute = cfg.get("run_recompute", True)

    for dataset in datasets:
        if run_baseline:
            print(f"\n>>> Baseline: {dataset}")
            tmp = {**cfg, "dataset": dataset}
            if num_samples:
                tmp["num_samples"] = num_samples
            run("scripts/run_blink.py", tmp)

        if run_recompute:
            for ratio in ratios:
                for chunk in chunks:
                    print(f"\n>>> Recompute: {dataset}, ratio={ratio}, chunk_k={chunk}")
                    tmp = {**cfg, "dataset": dataset, "recompute_ratio": ratio, "chunk_k": chunk}
                    if num_samples:
                        tmp["num_samples"] = num_samples
                    run("scripts/inference_with_recompute_kv.py", tmp)


if __name__ == "__main__":
    main()
