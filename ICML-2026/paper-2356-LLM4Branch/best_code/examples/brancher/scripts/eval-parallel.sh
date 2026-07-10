DATASET="setcover"
PHYSICAL_CORES=64
CORES=($(seq 1 $((PHYSICAL_CORES / 2 - 1))))
CORES=($(seq $((PHYSICAL_CORES / 2)) $((PHYSICAL_CORES - 1))))

MACHINE_ID="${MACHINE_ID:-default}"
LOCK_DIR="./tmp/cpu_locks/${MACHINE_ID}"
mkdir -p "$LOCK_DIR"

for core_num in "${CORES[@]}"
do
    lock_filepath="${LOCK_DIR}/lock_${core_num}"
    if [ -f "$lock_filepath" ]; then
        rm -f "$lock_filepath"
        echo "  - Removed old lock file: ${lock_filepath}"
    fi
done

export PYTHONPATH=$(pwd)
python ./examples/brancher/tester_parallel.py \
    --program  ./examples/brancher/program/${DATASET}/program.py \
    --dataset "${DATASET}" \
    --easy \
    --cores_list "${CORES[0]}-${CORES[-1]}" \
    --lock_dir "$LOCK_DIR" \
    --output "./log/eval/${DATASET}"   \
    --log_level INFO 
