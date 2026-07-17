#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/Transfer Error (Cora, Pubmed, Ogbn-Arxiv)"

DATASET="${1:-Cora}"
LAYERS="${2:-2}"
HIDDEN="${3:-32}"
TRIALS="${4:-20}"

echo "=== Training GCN: dataset=$DATASET layers=$LAYERS hidden=$HIDDEN ==="
python3 Stretched_GCN_train.py \
    --dataset "$DATASET" \
    --num_layers "$LAYERS" \
    --hidden_channels "$HIDDEN"

echo "=== Evaluating Transfer Error: dataset=$DATASET layers=$LAYERS hidden=$HIDDEN trials=$TRIALS ==="
python3 Stretched_GCN_test.py \
    --dataset "$DATASET" \
    --num_layers "$LAYERS" \
    --hidden_channels "$HIDDEN" \
    --num_trials "$TRIALS"

echo "=== Results ==="
CSV="${DATASET}_Test_${LAYERS}_${HIDDEN}.csv"
if [ -f "$CSV" ]; then
    cat "$CSV"
    echo ""
    echo "--- Scheme I, n=600 ---"
    grep "^1,600," "$CSV" || echo "Not found"
fi
