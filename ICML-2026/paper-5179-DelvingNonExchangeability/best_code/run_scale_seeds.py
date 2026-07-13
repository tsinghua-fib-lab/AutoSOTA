"""Run SCALE Stage 2 with multiple seeds and collect metrics."""
import json
import os
import sys
import copy
from pathlib import Path

ROOT = Path("/repo")
sys.path.insert(0, str(ROOT))

from experiments.run_config import _load_config, _import_entrypoint
from conformal_model.config_utils import ensure_run_fields
from omegaconf import OmegaConf
from experiments.run_scale import run_experiment

def run_single_seed(config_path: Path, src_dir: str, seed: int):
    """Run a single seed of SCALE."""
    config = _load_config(config_path)
    cfg_dict = copy.deepcopy(config["args"])

    # Override src_dir and seed
    cfg_dict["src_dir"] = src_dir
    cfg_dict.setdefault("run", {})
    cfg_dict["run"]["seed"] = seed

    # Adjust output directory
    run_template = cfg_dict.get("run", {}).get("dir",
        "logs/scale/${dataset.name}/${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    cfg_dict["run"]["dir"] = run_template

    cfg_dict = ensure_run_fields(cfg_dict, root=ROOT, default_dir=run_template)
    cfg = OmegaConf.create(cfg_dict)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Running seed={seed}, src_dir={src_dir}")
    print(f"Output: {cfg.run.dir}")
    print(f"{sep}\n")
    result = run_experiment(cfg)

    # Read saved metrics
    metrics_file = Path(cfg.run.dir) / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return {
        "seed": seed,
        "run_dir": cfg.run.dir,
        "result": result,
        "metrics": metrics,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conformal_model/scale/config/la.py")
    parser.add_argument("--src-dir", type=str, required=True,
                        help="Stage 1 output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=str, default="scale_results.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    all_results = []
    for seed in args.seeds:
        result = run_single_seed(config_path, args.src_dir, seed)
        all_results.append(result)

    output_file = ROOT / args.output
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"Ran {len(all_results)} seeds")

    # Print key metrics for alpha=0.1
    for r in all_results:
        metrics = r.get("metrics", {})
        per_alpha = metrics.get("per_alpha", {})
        for alpha_str, val in per_alpha.items():
            alpha = float(alpha_str)
            if abs(alpha - 0.1) < 0.001:
                cv = val.get("observed_coverage")
                piw = val.get("pi_width")
                wink = val.get("winkler")
                print(f"Seed {r[seed]}: Coverage={cv}, PI-Width={piw}, Winkler={wink}")

if __name__ == "__main__":
    main()
