#!/bin/bash
# Usage: scripts/run_lrb.sh [MODEL_FRACTION] [CAPACITY]
# MODEL_FRACTION defaults to 1 (full training set).
fraction="${1:-1}"
capacity="${2:-0}"
datasets=("astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")
MAX_JOBS=4

cap_flag=""
if [ "$capacity" -gt 0 ] 2>/dev/null; then
    cap_flag="--capacity $capacity"
fi

mkdir -p logs/benchmark/lrb
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
    python -m benchmark --boost --boost_fr --dataset "$dataset" --real --pred lrb --model_fraction "$fraction" --dump_file --output_root_dir stat $cap_flag > "logs/benchmark/lrb/${dataset}_${fraction}.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} jobs to finish..."
wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name lrb --fraction "$fraction" --results_dir stat
