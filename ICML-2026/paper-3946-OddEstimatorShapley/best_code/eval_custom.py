"""Custom evaluation script for OddSHAP SOTA optimization.

Usage: python3 eval_custom.py --tree-params '{"max_depth": 6}' --interaction-factor 10

If no overrides are given, reproduces the baseline exactly.
"""
from __future__ import annotations

import json, os, sys, tempfile, time, warnings, argparse
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

# Add local source paths
sys.path.insert(0, "/repo/src")
sys.path.insert(0, "/repo/shapiq-benchmark/src")

from oddshap.oddshap import OddSHAP
from oddshap.approx_utils import get_approximators
from shapiq_benchmark.load import GameFactory
from shapiq import InteractionValues
from shapiq_benchmark.metrics import get_all_metrics

# ── Default Configuration (matches reproduce_mse.py) ────────────
RANDOM_STATE = 40
PAIRING = True
REPLACEMENT = False
BUDGET = 1591
N_INSTANCES = 30
GAME_ID = "SentimentIMDBDistilBERT14_1"
CONFIG_PATH = "shapiq-benchmark/configurations_exhaustive/SentimentAnalysisLocalXAI.json"


def load_iv_safe(path):
    try:
        return InteractionValues.load(str(path))
    except (json.JSONDecodeError, ValueError):
        with open(path, "rb") as f:
            raw = f.read().rstrip(b"\x00")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            return InteractionValues.load(tmp_path)
        finally:
            os.unlink(tmp_path)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Custom OddSHAP evaluation")
    parser.add_argument("--tree-params", type=str, default=None,
                        help="JSON dict of LightGBM tree params, e.g. '{\"max_depth\": 6}'")
    parser.add_argument("--interaction-factor", type=int, default=10,
                        help="Interaction factor eta (default: 10)")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE,
                        help="Random state")
    parser.add_argument("--grid-search", action="store_true", default=False,
                        help="Enable LightGBM grid search")
    parser.add_argument("--n-instances", type=int, default=N_INSTANCES,
                        help="Number of instances to evaluate")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help="Evaluation budget")
    parser.add_argument("--force-recompute", action="store_true", default=False,
                        help="Force recompute all approximations")
    parser.add_argument("--output-mse-file", type=str, default=None,
                        help="Optional file to write per-instance MSE values")
    args = parser.parse_args()

    tree_params = None
    if args.tree_params:
        tree_params = json.loads(args.tree_params)
        print(f"Using custom tree_params: {tree_params}")
    else:
        print("Using default tree_params (max_depth=10)")

    print(f"interaction_factor={args.interaction_factor}")
    print(f"grid_search={args.grid_search}")
    print(f"budget={args.budget}")
    print(f"n_instances={args.n_instances}")
    print(f"random_state={args.random_state}")
    print(f"force_recompute={args.force_recompute}")
    print()

    # ── Phase 1: Ground truth ───────────────────────────────────
    print("=" * 60)
    print("Phase 1: Computing/Loading ground truth Shapley values")
    print("=" * 60)

    game_generator, _ = GameFactory.load_configuration_file_interactive(
        config_path=CONFIG_PATH,
        n_games=args.n_instances,
        check_pre_computed=True,
        only_pre_computed=True,
        return_config_id=True,
    )

    gt_dir = Path("ground_truth/exhaustive")
    ensure_dir(gt_dir)

    for idx, game in enumerate(game_generator):
        save_path = gt_dir / f"{GAME_ID}_{args.random_state}_{idx}_SV_1_exact_values.json"
        if save_path.exists():
            continue
        gt = game.exact_values(index="SV", order=1)
        gt.save(save_path)
        print(f"  GT [{idx+1}/{args.n_instances}] saved")
    print()

    # ── Phase 2: OddSHAP approximations ─────────────────────────
    print("=" * 60)
    print("Phase 2: Running OddSHAP approximations")
    print("=" * 60)

    # Unique suffix for this config
    config_suffix = f"_custom_md{tree_params.get('max_depth', 10) if tree_params else 10}_eta{args.interaction_factor}"
    approx_dir = Path(f"approximations/exhaustive_custom")
    ensure_dir(approx_dir)

    # Re-load games
    game_generator, _ = GameFactory.load_configuration_file_interactive(
        config_path=CONFIG_PATH,
        n_games=args.n_instances,
        check_pre_computed=True,
        only_pre_computed=True,
        return_config_id=True,
    )

    # Initialize sampling weights (leverage score - uniform)
    sampling_weights = np.ones(14 + 1)  # d=14 for SentimentIMDBDistilBERT

    total_time = 0.0
    n_computed = 0
    n_cached = 0

    for idx, game in enumerate(game_generator):
        save_path = approx_dir / f"{GAME_ID}_{args.random_state}_{idx}_OddSHAP_custom_{args.budget}_SV_1{config_suffix}.json"
        if save_path.exists() and not args.force_recompute:
            print(f"  Approx [{idx+1}/{args.n_instances}] (cached)")
            n_cached += 1
            continue

        approximator = OddSHAP(
            n=game.n_players,
            regression_basis="Fourier",
            interaction_detection="ProxySPEX",
            odd_only=True,
            sampling_weights=sampling_weights,
            pairing_trick=PAIRING,
            replacement=REPLACEMENT,
            random_state=args.random_state,
            grid_search=args.grid_search,
            interaction_factor=args.interaction_factor,
            tree_params=tree_params,
        )

        t0 = time.time()
        result = approximator.approximate(budget=args.budget, game=game)
        elapsed = time.time() - t0
        total_time += elapsed
        n_computed += 1
        result.save(save_path)
        print(f"  Approx [{idx+1}/{args.n_instances}] {elapsed:.1f}s")

    print(f"\n  Computed: {n_computed}, Cached: {n_cached}")
    if n_computed > 0:
        print(f"  Avg time per instance: {total_time/n_computed:.1f}s")
    print()

    # ── Phase 3: Compute MSE ────────────────────────────────────
    print("=" * 60)
    print("Phase 3: Computing MSE metric")
    print("=" * 60)

    mse_values = []
    for idx in range(args.n_instances):
        gt_path = gt_dir / f"{GAME_ID}_{args.random_state}_{idx}_SV_1_exact_values.json"
        ap_path = approx_dir / f"{GAME_ID}_{args.random_state}_{idx}_OddSHAP_custom_{args.budget}_SV_1{config_suffix}.json"
        if not gt_path.exists() or not ap_path.exists():
            print(f"  SKIP instance {idx}: missing files")
            continue
        gt = load_iv_safe(gt_path)
        ap = load_iv_safe(ap_path)
        for m in get_all_metrics(gt, ap):
            if m.metric_id == "MSE":
                mse_values.append(float(m.value))
                break

    if not mse_values:
        print("ERROR: No MSE values computed!")
        sys.exit(1)

    mean_mse = np.mean(mse_values)
    median_mse = np.median(mse_values)
    q1 = np.percentile(mse_values, 25)
    q3 = np.percentile(mse_values, 75)

    print(f"  Instances evaluated: {len(mse_values)}")
    print(f"  Mean MSE:   {mean_mse:.10f}")
    print(f"  Median MSE: {median_mse:.10f}")
    print(f"  Q1 MSE:     {q1:.10f}")
    print(f"  Q3 MSE:     {q3:.10f}")

    # ── Final summary ───────────────────────────────────────────
    baseline_mse = 5.02025e-05
    improvement_pct = (baseline_mse - mean_mse) / baseline_mse * 100

    print()
    print("=" * 60)
    print("CUSTOM EVALUATION RESULT")
    print(f"  Config: tree_params={tree_params}, eta={args.interaction_factor}")
    print(f"  Mean MSE = {mean_mse:.10f}")
    print(f"  Baseline  = {baseline_mse:.10f}")
    print(f"  Improvement: {improvement_pct:+.2f}%")
    print(f"  (negative means worse, positive means better)")
    print("=" * 60)

    # Save per-instance MSE for analysis
    if args.output_mse_file:
        with open(args.output_mse_file, "w") as f:
            json.dump({"mse_values": mse_values, "mean": float(mean_mse),
                       "median": float(median_mse), "q1": float(q1), "q3": float(q3)}, f, indent=2)
        print(f"\nPer-instance MSE saved to {args.output_mse_file}")

    # Print the final MSE value for easy parsing
    print(f"\nFINAL_MSE={mean_mse:.10f}")


if __name__ == "__main__":
    main()
