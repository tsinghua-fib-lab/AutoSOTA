#!/usr/bin/env bash
set -euo pipefail

# One-command reproduction of paper experiments + figures.
# Assumes you have created an environment (see README.md) and are running from repo root.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RESULTS_DIR="${RESULTS_DIR:-results/paper}"
FIGURES_DIR="${FIGURES_DIR:-figures/paper}"
SEED0="${SEED0:-123}"

mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

echo "[1/5] Synthetic (Fig. 1)"
python experiments/synthetic_hitl_causal_dpo.py \
  --D 20 \
  --S 10000 \
  --T 190 \
  --edge_prob_true 0.25 \
  --flip_prob 0.10 \
  --add_remove_prob 0.05 \
  --weight_noise 0.20 \
  --beta_edge 10.0 \
  --beta_dir 10.0 \
  --lam 0.0 \
  --screen_k 200 \
  --resample_threshold 0.6 \
  --policy eig \
  --rejuvenate_samples \
  --rejuvenate_steps 2 \
  --seed "$SEED0" \
  --save_prefix "eig"_seed"${SEED0}" \
  --outdir "$RESULTS_DIR/results_synthetic"

echo "[2/5] Sachs observational-only (Fig. 2 + Fig. 3 heatmaps)"
python experiments/sachs_hitl_causal_dpo.py \
  --download \
  --S 500 \
  --T 40 \
  --runs 10 \
  --policies eig,uncertainty,random,static_eig \
  --beta_edge 10.0 \
  --beta_dir 10.0 \
  --lam 0.0 \
  --screen_k 200 \
  --rejuvenate_samples \
  --rejuvenate_steps 2 \
  --seed0 "$SEED0" \
  --outdir "$RESULTS_DIR/results_sachs"

echo "[3/5] Sachs with DAG-GFN prior (Fig. 5)"
python experiments/sachs_hitl_causal_dpo.py \
  --download \
  --use_dag_gfn_prior \
  --S 500 \
  --T 40 \
  --runs 10 \
  --policies eig,uncertainty,random,static_eig \
  --beta_edge 10.0 \
  --beta_dir 10.0 \
  --lam 0.0 \
  --screen_k 200 \
  --rejuvenate_samples \
  --rejuvenate_steps 2 \
  --seed0 "$SEED0" \
  --outdir "$RESULTS_DIR/results_sachs_dag_gfn"

echo "[4/5] CausalBench K562-50 (Fig. 4)"
# Requires the pre-exported 50-gene NPZ in data/causalbench/exports/
DATASET_NPZ="${DATASET_NPZ:-data/causalbench/exports/weissmann_k562_50.npz}"
if [ ! -f "$DATASET_NPZ" ]; then
  echo "ERROR: Missing DATASET_NPZ: $DATASET_NPZ"
  echo "See docs/CAUSALBENCH.md for how to obtain/export it."
  exit 1
fi

python experiments/causalbench_hitl_causal_dpo.py \
  --dataset_npz "$DATASET_NPZ" \
  --outdir "$RESULTS_DIR/results_causalbench" \
  --policy eig \
  --seed "$SEED0" \
  --S 1000 \
  --T 200 \
  --screen_k 800 \
  --resample_threshold 0.5 \
  --rejuvenate_samples \
  --rejuvenate_steps 2 \
  --max_parents 3 \
  --corr_screen_k 8 \
  --ridge 1e-2 \
  --beta_edge 10.0 \
  --beta_dir 10.0 \
  --lam 0.0

echo "[5/5] Generate figures into $FIGURES_DIR"
bash scripts/generate_paper_figures.sh "$RESULTS_DIR" "$FIGURES_DIR"

echo "Done."
echo "Results:  $RESULTS_DIR"
echo "Figures:  $FIGURES_DIR"
