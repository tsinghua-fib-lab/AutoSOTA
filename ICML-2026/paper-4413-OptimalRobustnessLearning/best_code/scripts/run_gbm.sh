#!/bin/bash
# Usage: scripts/run_gbm.sh [MODEL_FRACTION] [CAPACITY]
fraction="${1:-1}"
capacity="${2:-0}"
datasets=("astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")
MAX_JOBS=4

cap_flag=""
if [ "$capacity" -gt 0 ] 2>/dev/null; then
    cap_flag="--capacity $capacity"
fi

mkdir -p logs/benchmark/gbm
pids=()
for dataset in "${datasets[@]}"; do
    while [ ${#pids[@]} -ge $MAX_JOBS ]; do
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $MAX_JOBS ] && sleep 5
    done

    echo "Running dataset=$dataset fraction=$fraction"
    python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred gbm --model_fraction "$fraction" --dump_file --output_root_dir stat $cap_flag > "logs/benchmark/gbm/${dataset}_${fraction}.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} jobs to finish..."
wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name gbm --fraction "$fraction" --results_dir stat
