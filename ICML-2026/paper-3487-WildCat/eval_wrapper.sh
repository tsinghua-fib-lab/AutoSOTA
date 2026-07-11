#!/bin/bash
# Multi-seed evaluation wrapper for WildCat BigGAN optimization
# Usage: bash eval_wrapper.sh [--attention wildcat|exact] [--r R] [--bins B] [--subsample_ratio SR] [--phi PHI] [--seeds N] [--output FILE]
# Computes IS, FID, generation time, IS Degradation, FID Degradation, Speed-up

set -euo pipefail

CDIR=/repo/examples/biggan
ATTENTION="wildcat"
R=96
BINS=8
SUBSAMPLE_RATIO=""
PHI=""
SEEDS=5
OUTPUT=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attention) ATTENTION="$2"; shift 2 ;;
    --r) R="$2"; shift 2 ;;
    --bins) BINS="$2"; shift 2 ;;
    --subsample_ratio) SUBSAMPLE_RATIO="$2"; shift 2 ;;
    --phi) PHI="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --extra) EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; shift ;;
  esac
done

cd "$CDIR"
export PYTORCH_PRETRAINED_BIGGAN_CACHE=/models/biggan
export CUDA_VISIBLE_DEVICES=0,1

TOTAL_IS_EXACT=0
TOTAL_IS_WILDCAT=0
TOTAL_FID_EXACT=0
TOTAL_FID_WILDCAT=0
TOTAL_TIME_EXACT=0
TOTAL_TIME_WILDCAT=0
SUCCESS_SEEDS=0

echo "=== WildCat Evaluation Wrapper ==="
echo "Attention: $ATTENTION | r=$R | bins=$BINS | seeds=$SEEDS"
echo "Start: $(date)"

for seed in $(seq 1 $SEEDS); do
    echo ""
    echo "--- Seed $seed: Exact Attention ---"
    EXACT_OUT=$(python3 eval_biggan_attentions.py --fid --attention exact --seed $seed --data_per_class 5 --num_splits 10 2>&1)
    IS_EXACT=$(echo "$EXACT_OUT" | grep "Inception score" | grep -oP [d]+.[d]+ | head -1)
    FID_EXACT=$(echo "$EXACT_OUT" | grep "^FID  :" | grep -oP [d]+.[d]+ | head -1)
    TIME_EXACT=$(echo "$EXACT_OUT" | grep "generation time" | grep -oP [d]+.[d]+ | head -1)
    echo "  IS=$IS_EXACT FID=$FID_EXACT time=$TIME_EXACT"
    
    echo "--- Seed $seed: WildCat (r=$R, B=$BINS) ---"
    WILDCAT_CMD="python3 eval_biggan_attentions.py --fid --attention wildcat --seed $seed --data_per_class 5 --num_splits 10 --r $R --bins $BINS"
    [[ -n "$SUBSAMPLE_RATIO" ]] && WILDCAT_CMD="$WILDCAT_CMD --subsample_ratio $SUBSAMPLE_RATIO"
    [[ -n "$EXTRA_ARGS" ]] && WILDCAT_CMD="$WILDCAT_CMD $EXTRA_ARGS"
    
    WILDCAT_OUT=$(eval "$WILDCAT_CMD" 2>&1)
    IS_WILDCAT=$(echo "$WILDCAT_OUT" | grep "Inception score" | grep -oP [d]+.[d]+ | head -1)
    FID_WILDCAT=$(echo "$WILDCAT_OUT" | grep "^FID  :" | grep -oP [d]+.[d]+ | head -1)
    TIME_WILDCAT=$(echo "$WILDCAT_OUT" | grep "generation time" | grep -oP [d]+.[d]+ | head -1)
    echo "  IS=$IS_WILDCAT FID=$FID_WILDCAT time=$TIME_WILDCAT"
    
    if [[ -n "$IS_EXACT" && -n "$IS_WILDCAT" ]]; then
        TOTAL_IS_EXACT=$(python3 -c "print($TOTAL_IS_EXACT + $IS_EXACT)")
        TOTAL_IS_WILDCAT=$(python3 -c "print($TOTAL_IS_WILDCAT + $IS_WILDCAT)")
        TOTAL_FID_EXACT=$(python3 -c "print($TOTAL_FID_EXACT + $FID_EXACT)")
        TOTAL_FID_WILDCAT=$(python3 -c "print($TOTAL_FID_WILDCAT + $FID_WILDCAT)")
        TOTAL_TIME_EXACT=$(python3 -c "print($TOTAL_TIME_EXACT + $TIME_EXACT)")
        TOTAL_TIME_WILDCAT=$(python3 -c "print($TOTAL_TIME_WILDCAT + $TIME_WILDCAT)")
        SUCCESS_SEEDS=$((SUCCESS_SEEDS + 1))
    fi
done

echo ""
echo "=== Summary ==="
if [[ $SUCCESS_SEEDS -gt 0 ]]; then
    AVG_IS_EXACT=$(python3 -c "print(round($TOTAL_IS_EXACT / $SUCCESS_SEEDS, 4))")
    AVG_IS_WILDCAT=$(python3 -c "print(round($TOTAL_IS_WILDCAT / $SUCCESS_SEEDS, 4))")
    AVG_FID_EXACT=$(python3 -c "print(round($TOTAL_FID_EXACT / $SUCCESS_SEEDS, 4))")
    AVG_FID_WILDCAT=$(python3 -c "print(round($TOTAL_FID_WILDCAT / $SUCCESS_SEEDS, 4))")
    AVG_TIME_EXACT=$(python3 -c "print(round($TOTAL_TIME_EXACT / $SUCCESS_SEEDS, 4))")
    AVG_TIME_WILDCAT=$(python3 -c "print(round($TOTAL_TIME_WILDCAT / $SUCCESS_SEEDS, 4))")
    
    IS_DEGRADATION=$(python3 -c "print(round(($AVG_IS_EXACT - $AVG_IS_WILDCAT) / $AVG_IS_EXACT * 100, 4))")
    FID_DEGRADATION=$(python3 -c "print(round(($AVG_FID_WILDCAT - $AVG_FID_EXACT) / $AVG_FID_EXACT * 100, 4))")
    SPEEDUP=$(python3 -c "print(round($AVG_TIME_EXACT / $AVG_TIME_WILDCAT, 4))")
    
    echo "Seeds completed: $SUCCESS_SEEDS/$SEEDS"
    echo "Avg IS (Exact):    $AVG_IS_EXACT"
    echo "Avg IS (WildCat):  $AVG_IS_WILDCAT"
    echo "Avg FID (Exact):   $AVG_FID_EXACT"
    echo "Avg FID (WildCat): $AVG_FID_WILDCAT"
    echo "IS Degradation:    $IS_DEGRADATION%"
    echo "FID Degradation:   $FID_DEGRADATION%"
    echo "Speed-up:          ${SPEEDUP}x"
    echo "Avg Time (Exact):  ${AVG_TIME_EXACT}s"
    echo "Avg Time (WildCat): ${AVG_TIME_WILDCAT}s"
    
    # Write results file if output specified
    if [[ -n "$OUTPUT" ]]; then
        cat > "$OUTPUT" << EOF
IS_EXACT=$AVG_IS_EXACT
IS_WILDCAT=$AVG_IS_WILDCAT
FID_EXACT=$AVG_FID_EXACT
FID_WILDCAT=$AVG_FID_WILDCAT
IS_DEGRADATION=$IS_DEGRADATION
FID_DEGRADATION=$FID_DEGRADATION
SPEEDUP=$SPEEDUP
TIME_EXACT=$AVG_TIME_EXACT
TIME_WILDCAT=$AVG_TIME_WILDCAT
SEEDS=$SUCCESS_SEEDS
EOF
        echo "Results written to $OUTPUT"
    fi
else
    echo "ERROR: No successful seeds"
    exit 1
fi
echo "End: $(date)"
