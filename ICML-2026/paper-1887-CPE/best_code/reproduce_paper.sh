#!/usr/bin/env bash
set -euo pipefail

# One-command reproduction of paper experiments + figures.
# Assumes you have created an environment (see README.md) and are running from repo root.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RESULTS_ROOT="${RESULTS_ROOT:-results/paper}"
FIGURES_DIR="${FIGURES_DIR:-figures/paper}"
SEED0="${SEED0:-123}"

mkdir -p "$RESULTS_ROOT" "$FIGURES_DIR"

###############################################################################
# 1) Synthetic
###############################################################################
SYN_POLICIES=(eig uncertainty random static_eig)
SYN_SEEDS=$(seq 1 10)
echo
echo "=== Synthetic experiments ==="
echo "[1/5] Synthetic (Fig. 1)"
for POLICY in "${SYN_POLICIES[@]}"; do
  OUTDIR="${RESULTS_ROOT}/showcase/${POLICY}"
  mkdir -p "$OUTDIR"

  for SEED in $SYN_SEEDS; do
    echo "[Synthetic] policy=$POLICY seed=$SEED"

    python -u experiments/synthetic_hitl_causal_dpo.py \
      --outdir "$OUTDIR" \
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
      --policy "$POLICY" \
      --rejuvenate_samples \
      --rejuvenate_steps 2 \
      --seed "$SEED" \
      --save_prefix "${POLICY}_seed${SEED}"
  done
done

###############################################################################
# 2-3) Sachs
###############################################################################
echo
echo "=== Sach experiments ==="
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
  --outdir "$RESULTS_ROOT/results_sachs"

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
  --outdir "$RESULTS_ROOT/results_sachs_dag_gfn"


###############################################################################
# 4) CausalBench-50
###############################################################################
CB_POLICIES=(eig uncertainty random static_eig)
CB_SEEDS=$(seq 1 10)
DATASET_NPZ="${DATASET_NPZ:-data/causalbench/exports/weissmann_k562_50.npz}"

if [ ! -f "$DATASET_NPZ" ]; then
  echo "ERROR: Missing CausalBench dataset: $DATASET_NPZ"
  echo "See docs/CAUSALBENCH.md for setup."
  exit 1
fi

echo
echo "=== CausalBench-50 experiments ==="
for POLICY in "${CB_POLICIES[@]}"; do
  OUTDIR="${RESULTS_ROOT}/cb50/${POLICY}"
  mkdir -p "$OUTDIR"

  for SEED in $CB_SEEDS; do
    echo "[CausalBench] policy=$POLICY seed=$SEED"

    python -u experiments/causalbench_hitl_causal_dpo.py \
      --dataset_npz "$DATASET_NPZ" \
      --outdir "$OUTDIR" \
      --policy "$POLICY" \
      --seed "$SEED" \
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
  done
done

###############################################################################
# 5) Generate plots
###############################################################################
echo "[5/5] Generate figures into $FIGURES_DIR"
bash scripts/generate_paper_figures.sh "$RESULTS_ROOT" "$FIGURES_DIR"

echo "Done."
echo "Results:  $RESULTS_ROOT"
echo "Figures:  $FIGURES_DIR"



