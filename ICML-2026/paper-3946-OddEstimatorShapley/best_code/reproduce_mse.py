"""Reproduce OddSHAP MSE on SentimentIMDBDistilBERT (d=14) benchmark.

Reproduces the rubric metric from paper 3946:
  OddSHAP MSE at budget m≈100d (1440), paired sampling, η=10,
  LightGBM proxy max_depth=10, leverage-score sampling,
  30 explained instances.

Usage: python3 reproduce_mse.py
"""
from __future__ import annotations

import json, os, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

# Add local source paths
sys.path.insert(0, "/repo/src")
sys.path.insert(0, "/repo/shapiq-benchmark/src")

from oddshap.approx_utils import get_approximators
from shapiq_benchmark.load import GameFactory
from shapiq import InteractionValues
from shapiq_benchmark.metrics import get_all_metrics

# ── Configuration ──────────────────────────────────────────────
RANDOM_STATE = 40
PAIRING = True       # paired_sampling=true
REPLACEMENT = False  # baseline_imputation
BUDGET = 1591        # closest logspace step to m=1440 ≈ 100d
CONFIG_ID = 37       # PAIRING=True, REPLACEMENT=False
METHOD = "OddSHAP-Fourier-ProxySPEX-NoCV"
N_INSTANCES = 30

# ── Helpers ─────────────────────────────────────────────────────
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

# ── Phase 1: Ground truth ───────────────────────────────────────
print("=" * 60)
print("Phase 1: Computing ground truth Shapley values")
print("=" * 60)

game_generator, _ = GameFactory.load_configuration_file_interactive(
    config_path="shapiq-benchmark/configurations_exhaustive/SentimentAnalysisLocalXAI.json",
    n_games=N_INSTANCES,
    check_pre_computed=True,
    only_pre_computed=True,
    return_config_id=True,
)

game_id = "SentimentIMDBDistilBERT14_1"
gt_dir = Path("ground_truth/exhaustive")
ensure_dir(gt_dir)

for idx, game in enumerate(game_generator):
    save_path = gt_dir / f"{game_id}_{RANDOM_STATE}_{idx}_SV_1_exact_values.json"
    if save_path.exists():
        continue
    gt = game.exact_values(index="SV", order=1)
    gt.save(save_path)
    print(f"  GT [{idx+1}/{N_INSTANCES}] saved")

print()

# ── Phase 2: OddSHAP approximations ─────────────────────────────
print("=" * 60)
print("Phase 2: Running OddSHAP approximations (budget ≈ m)")
print("=" * 60)

approx_dir = Path("approximations/exhaustive")
ensure_dir(approx_dir)

# Re-load games for approximation
game_generator, _ = GameFactory.load_configuration_file_interactive(
    config_path="shapiq-benchmark/configurations_exhaustive/SentimentAnalysisLocalXAI.json",
    n_games=N_INSTANCES,
    check_pre_computed=True,
    only_pre_computed=True,
    return_config_id=True,
)

for idx, game in enumerate(game_generator):
    save_path = approx_dir / f"{game_id}_{CONFIG_ID}_{idx}_{METHOD}_{BUDGET}_SV_1.json"
    if save_path.exists():
        print(f"  Approx [{idx+1}/{N_INSTANCES}] (cached)")
        continue

    approximators = get_approximators(
        [METHOD], game.n_players, RANDOM_STATE, PAIRING, REPLACEMENT
    )
    for approx in approximators:
        t0 = time.time()
        result = approx.approximate(budget=BUDGET, game=game)
        elapsed = time.time() - t0
        result.save(save_path)
        print(f"  Approx [{idx+1}/{N_INSTANCES}] {elapsed:.1f}s")

print()

# ── Phase 3: Compute MSE ────────────────────────────────────────
print("=" * 60)
print("Phase 3: Computing MSE metric")
print("=" * 60)

mse_values = []
for idx in range(N_INSTANCES):
    gt_path = gt_dir / f"{game_id}_{RANDOM_STATE}_{idx}_SV_1_exact_values.json"
    ap_path = approx_dir / f"{game_id}_{CONFIG_ID}_{idx}_{METHOD}_{BUDGET}_SV_1.json"
    if not gt_path.exists() or not ap_path.exists():
        print(f"  SKIP instance {idx}: missing files")
        continue
    gt = load_iv_safe(gt_path)
    ap = load_iv_safe(ap_path)
    for m in get_all_metrics(gt, ap):
        if m.metric_id == "MSE":
            mse_values.append(float(m.value))
            break

mean_mse = np.mean(mse_values)
median_mse = np.median(mse_values)
q1 = np.percentile(mse_values, 25)
q3 = np.percentile(mse_values, 75)

print(f"  Instances evaluated: {len(mse_values)}")
print(f"  Mean MSE:   {mean_mse:.10f}")
print(f"  Median MSE: {median_mse:.10f}")
print(f"  Q1 MSE:     {q1:.10f}")
print(f"  Q3 MSE:     {q3:.10f}")
print(f"  Paper MSE:  0.000046 (4.6e-5)")
print(f"  Paper Q1:   0.000016 | Median: 0.000038 | Q3: 0.000057")
print()

# ── Final summary ───────────────────────────────────────────────
print("=" * 60)
print("REPRODUCTION RESULT")
print(f"  OddSHAP Mean MSE = {mean_mse:.10f}")
print(f"  Rubric CI: [{0.000031}, {0.0000475}]")
in_ci = 0.000031 <= mean_mse <= 0.0000475
print(f"  Within CI: {in_ci}")
if not in_ci:
    print(f"  (slightly above by {mean_mse - 0.0000475:.10f})")
print("=" * 60)
