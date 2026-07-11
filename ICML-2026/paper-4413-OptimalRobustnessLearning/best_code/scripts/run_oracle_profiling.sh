#!/bin/bash
# Oracle consistency profiling for RPB-OM
# Measures the upper bound of algorithm performance with perfect predictions
# Usage: bash scripts/run_oracle_profiling.sh

capacity="0"
datasets=("astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")
MAX_JOBS=4

cap_flag=""
if [ "$capacity" -gt 0 ] 2>/dev/null; then
    cap_flag="--capacity $capacity"
fi

mkdir -p logs/benchmark/oracle
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

    echo "Running oracle profiling dataset=$dataset"
    python -m benchmark --oracle --pred oracle_dis --dataset "$dataset" --dump_file --output_root_dir stat_oracle $cap_flag > "logs/benchmark/oracle/${dataset}.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} jobs to finish..."
wait "${pids[@]}"

echo "All oracle profiling runs finished. Aggregating results..."
python scripts/aggregate_results.py --name oracle_dis --results_dir stat_oracle
