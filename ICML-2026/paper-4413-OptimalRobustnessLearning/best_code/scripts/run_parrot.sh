#!/bin/bash
# Usage: scripts/run_parrot.sh [MODEL_FRACTION] [CAPACITY]
fraction="${1:-1}"
capacity="${2:-0}"
datasets=("astar" "bwaves" "bzip" "cactusadm" "gems" "lbm" "leslie3d" "libq" "mcf" "milc" "omnetpp" "sphinx3" "xalanc")
MAX_JOBS=4

cap_flag=""
if [ "$capacity" -gt 0 ] 2>/dev/null; then
    cap_flag="--capacity $capacity"
fi

mkdir -p logs/benchmark/parrot

cuda_devices=("cuda:0" "cuda:1")
cuda_index=0

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

    current_device=${cuda_devices[$cuda_index]}
    echo "Running dataset=$dataset fraction=$fraction device=$current_device"
    python -m benchmark --boost --boost_fr --dataset "$dataset" --device "$current_device" --real --pred parrot --model_fraction "$fraction" --dump_file --output_root_dir stat $cap_flag > "logs/benchmark/parrot/${dataset}_${fraction}.log" 2>&1 &
    pids+=($!)
    ((cuda_index=(cuda_index+1)%2))
done

echo "Waiting for ${#pids[@]} jobs to finish..."
wait "${pids[@]}"

echo "All runs finished. Aggregating results..."
python scripts/aggregate_results.py --name parrot --fraction "$fraction" --results_dir stat
