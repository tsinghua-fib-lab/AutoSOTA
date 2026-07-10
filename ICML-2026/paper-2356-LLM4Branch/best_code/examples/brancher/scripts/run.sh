DATASET="setcover"
PHYSICAL_CORES=64
# CORES=($(seq 1 $((PHYSICAL_CORES / 2 - 1))))
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
python ./openevolve-run.py ./examples/brancher/initial_program.py ./examples/brancher/evaluator.py \
 --config ./examples/brancher/config.yaml \
 --evaluator-config ./examples/brancher/evaluator_config.yaml \
 --cores-list "${CORES[0]}-${CORES[-1]}" \
 --lock-dir "$LOCK_DIR" \
 -o "./examples/brancher/output/${DATASET}/${MACHINE_ID}/$(date +"%m%d-%H%M%S")" \
 -i 200 \
 -l INFO \
 -d "${DATASET}" \