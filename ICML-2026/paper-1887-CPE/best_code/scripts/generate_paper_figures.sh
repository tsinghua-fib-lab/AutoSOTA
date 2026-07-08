#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-results/paper}"
FIGURES_DIR="${2:-figures/paper}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

mkdir -p "$FIGURES_DIR"

# Synthetic: Figure 1 (+ Figure 6 in appendix)
python experiments/compare_policies_synthetic.py \
  --indir "$RESULTS_DIR/results_synthetic" \
  --outdir "$FIGURES_DIR/results_synthetic" \
  --policies eig uncertainty random static_eig \
  --max_round 190 \

# Sachs: Figure 2 (+ Table 3 and Figure 7/3)
python experiments/summarize_sachs.py \
  --indir "$RESULTS_DIR/results_sachs" \
  --outdir "$FIGURES_DIR/results_sachs" \
  --policies eig uncertainty random static_eig \

# Sachs DAG-GFN prior: Figure 5
python experiments/summarize_sachs.py \
  --indir "$RESULTS_DIR/results_sachs_dag_gfn" \
  --outdir "$FIGURES_DIR/results_sachs_dag_gfn" \
  --policies eig uncertainty random static_eig \

# Posterior heatmaps (Figure 3-style)
if [ -f "$RESULTS_DIR/results_sachs/sachs_eig_seed123.json" ]; then
  python experiments/plot_posterior_sachs.py \
    --run_json "$RESULTS_DIR/results_sachs/sachs_eig_seed123.json" \
    --meta_json "$RESULTS_DIR/results_sachs/sachs_meta.json" \
    --outdir "$FIGURES_DIR/results_sachs/graphs" \
    --mode topk \
    --k 17 \
    --which final
fi

# CausalBench: Figure 4 (+ Figure 8 in appendix)
python experiments/summarize_cb50.py \
  --indir "$RESULTS_DIR/results_causalbench" \
  --outdir "$FIGURES_DIR/results_causalbench" \
  --policies eig \

echo "Figures written to: $FIGURES_DIR"
