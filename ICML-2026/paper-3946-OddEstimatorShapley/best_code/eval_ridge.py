"""Custom evaluation with RidgeCV support."""
from __future__ import annotations

import json, os, sys, tempfile, time, warnings, argparse
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/repo/src")
sys.path.insert(0, "/repo/shapiq-benchmark/src")

from oddshap.oddshap import OddSHAP
from shapiq_benchmark.load import GameFactory
from shapiq import InteractionValues
from shapiq_benchmark.metrics import get_all_metrics

RANDOM_STATE = 40
PAIRING = True
REPLACEMENT = False
BUDGET = 1591
N_INSTANCES = 30
GAME_ID = "SentimentIMDBDistilBERT14_1"
CONFIG_PATH = "shapiq-benchmark/configurations_exhaustive/SentimentAnalysisLocalXAI.json"

# Use NFS for storage to save overlay space
BASE_DIR = "/autosota_cache/paper-3946-sota"

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

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-params", type=str, default=None)
    parser.add_argument("--interaction-factor", type=int, default=10)
    parser.add_argument("--ridge-alphas", type=str, default=None,
                        help="Comma-separated RidgeCV alphas, e.g. '0.1,1.0,10.0,100.0'")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--budget", type=int, default=1591)
    parser.add_argument("--force-recompute", action="store_true", default=False)
    args = parser.parse_args()

    tree_params = json.loads(args.tree_params) if args.tree_params else None
    ridge_alphas = [float(x) for x in args.ridge_alphas.split(",")] if args.ridge_alphas else None

    print(f"tree_params={tree_params}")
    print(f"interaction_factor={args.interaction_factor}")
    print(f"ridge_alphas={ridge_alphas}")

    config_tag = f"eta{args.interaction_factor}"
    if tree_params:
        config_tag += f"_md{tree_params.get('max_depth',10)}_ne{tree_params.get('n_estimators',100)}"
    if ridge_alphas:
        config_tag += "_ridge"

    gt_dir = Path(BASE_DIR) / "ground_truth"
    approx_dir = Path(BASE_DIR) / "approximations"
    ensure_dir(gt_dir)
    ensure_dir(approx_dir)

    # Phase 1: Ground truth
    print("Phase 1: Ground truth...")
    game_generator, _ = GameFactory.load_configuration_file_interactive(
        config_path=CONFIG_PATH, n_games=N_INSTANCES,
        check_pre_computed=True, only_pre_computed=True, return_config_id=True)
    for idx, game in enumerate(game_generator):
        sp = gt_dir / f"{GAME_ID}_{args.random_state}_{idx}_SV_1_exact_values.json"
        if not sp.exists():
            gt = game.exact_values(index="SV", order=1)
            gt.save(sp)
    print("Done")

    # Phase 2: Approximations
    print("Phase 2: Approximations...")
    sampling_weights = np.ones(14 + 1)
    game_generator, _ = GameFactory.load_configuration_file_interactive(
        config_path=CONFIG_PATH, n_games=N_INSTANCES,
        check_pre_computed=True, only_pre_computed=True, return_config_id=True)

    for idx, game in enumerate(game_generator):
        sp = approx_dir / f"{GAME_ID}_{args.random_state}_{idx}_OddSHAP_{BUDGET}_SV_1_{config_tag}.json"
        if sp.exists() and not args.force_recompute:
            continue
        approx = OddSHAP(
            n=game.n_players, regression_basis="Fourier",
            interaction_detection="ProxySPEX", odd_only=True,
            sampling_weights=sampling_weights, pairing_trick=PAIRING,
            replacement=REPLACEMENT, random_state=args.random_state,
            grid_search=False, interaction_factor=args.interaction_factor,
            tree_params=tree_params, ridge_alphas=ridge_alphas)
        t0 = time.time()
        result = approx.approximate(budget=args.budget, game=game)
        elapsed = time.time() - t0
        result.save(sp)
        print(f"  [{idx+1}/{N_INSTANCES}] {elapsed:.1f}s")
    print("Done")

    # Phase 3: MSE
    print("Phase 3: MSE...")
    mse_values = []
    for idx in range(N_INSTANCES):
        gt_path = gt_dir / f"{GAME_ID}_{args.random_state}_{idx}_SV_1_exact_values.json"
        ap_path = approx_dir / f"{GAME_ID}_{args.random_state}_{idx}_OddSHAP_{BUDGET}_SV_1_{config_tag}.json"
        if not gt_path.exists() or not ap_path.exists():
            print(f"  SKIP {idx}")
            continue
        gt = load_iv_safe(gt_path)
        ap = load_iv_safe(ap_path)
        for m in get_all_metrics(gt, ap):
            if m.metric_id == "MSE":
                mse_values.append(float(m.value))
                break
    if not mse_values:
        print("ERROR: No MSE values!")
        sys.exit(1)
    mean_mse = np.mean(mse_values)
    print(f"Instances: {len(mse_values)}")
    print(f"Mean MSE:  {mean_mse:.10f}")
    print(f"FINAL_MSE={mean_mse:.10f}")

if __name__ == "__main__":
    main()
