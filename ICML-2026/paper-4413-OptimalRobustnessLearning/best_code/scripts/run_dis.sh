#!/bin/bash
# Usage: scripts/run_dis.sh [CAPACITY]
capacity="${1:-0}"
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

    echo "Running dataset=$dataset"
    python -m benchmark --boost_fr --dataset "$dataset" --oracle --pred oracle_dis --noise_type dis --dump_file --output_root_dir stat $cap_flag > "logs/benchmark/oracle/${dataset}_dis.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} jobs to finish..."
wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name dis --results_dir stat
