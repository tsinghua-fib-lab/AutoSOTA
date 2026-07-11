#!/bin/bash
# Quick wildcat evaluation using cached exact baselines
# Usage: bash quick_eval.sh <r> <bins> [extra_args_for_python]
# Outputs: IS_DEGRADATION FID_DEGRADATION SPEEDUP

set -euo pipefail
R=${1:-96}
BINS=${2:-8}
shift 2 || true
EXTRA_ARGS="$@"

cd /repo/examples/biggan
export PYTORCH_PRETRAINED_BIGGAN_CACHE=/models/biggan
export CUDA_VISIBLE_DEVICES=0,1

# Cached exact baseline values (pre-computed)
declare -A IS_EXACT
IS_EXACT[1]=58.36433
IS_EXACT[2]=57.69869
IS_EXACT[3]=57.64122
IS_EXACT[4]=57.61149
IS_EXACT[5]=57.59332

declare -A FID_EXACT
FID_EXACT[1]=32.16951978032063
FID_EXACT[2]=31.880433002893255
FID_EXACT[3]=32.2209578562352
FID_EXACT[4]=32.176822213154196
FID_EXACT[5]=32.32195230418148

declare -A TIME_EXACT
TIME_EXACT[1]=38.4349
TIME_EXACT[2]=38.1778
TIME_EXACT[3]=38.2401
TIME_EXACT[4]=36.8398
TIME_EXACT[5]=38.5061

TOTAL_IS_WC=0
TOTAL_FID_WC=0
TOTAL_TIME_WC=0
TOTAL_IS_EX=0
TOTAL_FID_EX=0
TOTAL_TIME_EX=0
SUCCESS=0

for seed in 1 2 3 4 5; do
    WILDCAT_CMD="python3 eval_biggan_attentions.py --fid --attention wildcat --seed $seed --data_per_class 5 --num_splits 10 --r $R --bins $BINS $EXTRA_ARGS"
    WILDCAT_OUT=$(eval "$WILDCAT_CMD" 2>&1)
    IS_WC=$(echo "$WILDCAT_OUT" | grep "Inception score" | grep -oP "[\d]+\.[\d]+" | head -1)
    FID_WC=$(echo "$WILDCAT_OUT" | grep "^FID  :" | grep -oP "[\d]+\.[\d]+" | head -1)
    TIME_WC=$(echo "$WILDCAT_OUT" | grep "generation time" | grep -oP "[\d]+\.[\d]+" | head -1)
    
    if [[ -n "$IS_WC" ]]; then
        TOTAL_IS_WC=$(python3 -c "print($TOTAL_IS_WC + $IS_WC)")
        TOTAL_FID_WC=$(python3 -c "print($TOTAL_FID_WC + $FID_WC)")
        TOTAL_TIME_WC=$(python3 -c "print($TOTAL_TIME_WC + $TIME_WC)")
        TOTAL_IS_EX=$(python3 -c "print($TOTAL_IS_EX + ${IS_EXACT[$seed]})")
        TOTAL_FID_EX=$(python3 -c "print($TOTAL_FID_EX + ${FID_EXACT[$seed]})")
        TOTAL_TIME_EX=$(python3 -c "print($TOTAL_TIME_EX + ${TIME_EXACT[$seed]})")
        SUCCESS=$((SUCCESS + 1))
        echo "  seed=$seed IS=$IS_WC FID=$FID_WC time=$TIME_WC"
    fi
done

if [[ $SUCCESS -gt 0 ]]; then
    AVG_IS_EX=$(python3 -c "print(round($TOTAL_IS_EX / $SUCCESS, 4))")
    AVG_IS_WC=$(python3 -c "print(round($TOTAL_IS_WC / $SUCCESS, 4))")
    AVG_FID_EX=$(python3 -c "print(round($TOTAL_FID_EX / $SUCCESS, 4))")
    AVG_FID_WC=$(python3 -c "print(round($TOTAL_FID_WC / $SUCCESS, 4))")
    AVG_TIME_EX=$(python3 -c "print(round($TOTAL_TIME_EX / $SUCCESS, 4))")
    AVG_TIME_WC=$(python3 -c "print(round($TOTAL_TIME_WC / $SUCCESS, 4))")
    
    IS_DEG=$(python3 -c "print(round(($AVG_IS_EX - $AVG_IS_WC) / $AVG_IS_EX * 100, 4))")
    FID_DEG=$(python3 -c "print(round(($AVG_FID_WC - $AVG_FID_EX) / $AVG_FID_EX * 100, 4))")
    SPEEDUP=$(python3 -c "print(round($AVG_TIME_EX / $AVG_TIME_WC, 4))")
    
    echo "=== r=$R B=$BINS ==="
    echo "IS_DEGRADATION=$IS_DEG"
    echo "FID_DEGRADATION=$FID_DEG"
    echo "SPEEDUP=$SPEEDUP"
    echo "AVG_TIME_EXACT=$AVG_TIME_EX"
    echo "AVG_TIME_WILDCAT=$AVG_TIME_WC"
    echo "SEEDS=$SUCCESS"
    
    # Write structured output for parsing
    cat > /tmp/quick_eval_result.txt << EOF
IS_DEGRADATION=$IS_DEG
FID_DEGRADATION=$FID_DEG
SPEEDUP=$SPEEDUP
AVG_IS_EXACT=$AVG_IS_EX
AVG_IS_WILDCAT=$AVG_IS_WC
AVG_FID_EXACT=$AVG_FID_EX
AVG_FID_WILDCAT=$AVG_FID_WC
AVG_TIME_EXACT=$AVG_TIME_EX
AVG_TIME_WILDCAT=$AVG_TIME_WC
SEEDS=$SUCCESS
EOF
else
    echo "ERROR: No successful seeds"
    exit 1
fi
