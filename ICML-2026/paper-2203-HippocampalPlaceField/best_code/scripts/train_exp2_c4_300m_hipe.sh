#!/bin/bash

#SBATCH --job-name=exp2-c4-300M-hipe
#SBATCH --output=/data/xxxxxxxxx/03-proj/PE/logs/exp2_c4_300M/tmp_hipe_%j.out
#SBATCH --error=/data/xxxxxxxxx/03-proj/PE/logs/exp2_c4_300M/tmp_hipe_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
PYTHON_BIN="/home/xxxxxxxxx/anaconda3/bin/python"
SCRIPT="train_exp2_c4full.py" 


ROOT_DIR="/data/xxxxxxxxx/03-proj/PE"
LOG_DIR="${ROOT_DIR}/logs/exp2_c4_300M/$(date +%Y%m%d)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/c4_300M/hipe"
WANDB_DIR="${ROOT_DIR}/wandb/offline/exp2_c4_300M"

C4_DATA_ROOT="${ROOT_DIR}"
LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/hipe_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/hipe_${JOB_ID}.err"
exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f /data/xxxxxxxxx/03-proj/PE/logs/exp2_c4_300M/tmp_hipe_${JOB_ID}.out
    rm -f /data/xxxxxxxxx/03-proj/PE/logs/exp2_c4_300M/tmp_hipe_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"


GLOBAL_BS=64
SEEDS=(6198)
MAX_TOKENS=1000000000


TRAIN_SAMPLES=5000000
VAL_SAMPLES=10000


DEBUG_STEPS="" 


if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS --lr 3e-4"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS --lr 3e-4"
fi


SIGMAS=(700.0)  # (50.0 100.0 200.0 500.0 700.0 1000.0)
THRESHOLDS=(7 9) # (3 5 7 9 11 13)
MODELS=("300M")
LENGTHS=(512 1024 2048)


get_mbs() {
    local m_size=$1
    local seq_len=$2
    
    local mbs=16
    if [ "$seq_len" -ge 2048 ]; then mbs=8; fi

    echo $mbs
}


run_experiment() {
    local M_SIZE=$1
    local SEQ_LEN=$2
    local SIGMA=$3
    local THR=$4
    local SEED=$5
    
    CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN)
    RUN_ID="c4_hipe_${M_SIZE}_L${SEQ_LEN}_sig${SIGMA}_thr${THR}"
    OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"

    echo -e "\n>>> [C4 HIPE] Model: $M_SIZE | Len: $SEQ_LEN | Sigma: $SIGMA | Thr: $THR | MBS: $CUR_MICRO_BS | SEED: $SEED"
    echo ">>> Output Dir: $OUTPUT_DIR"
    
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUTPUT_DIR \
        --run_id $RUN_ID \
        --model_size $M_SIZE \
        --dataset_path $C4_DATA_ROOT \
        --local_tokenizer_path $LOCAL_TOKENIZER \
        --wandb_dir $WANDB_DIR \
        --wandb_mode $WANDB_MODE \
        --train_size $TRAIN_SAMPLES \
        --val_size $VAL_SAMPLES \
        --seq_len $SEQ_LEN \
        --global_batch_size $GLOBAL_BS \
        --micro_batch_size $CUR_MICRO_BS \
        --seed $SEED \
        --use_scaled_rope \
        --sigma $SIGMA \
        --rope_scaling_threshold $THR \
        $LIMIT_ARGS

    if [ $? -ne 0 ]; then
        echo ">>> [ERROR] C4 HIPE: $M_SIZE, Len: $SEQ_LEN, Sigma: $SIGMA, SEED: $SEED"
    fi
}

echo -e "\n>>> [BATCH START] Running C4 300M HIPE experiments..."

for SEED in "${SEEDS[@]}"; do
    for M_SIZE in "${MODELS[@]}"; do
        for SEQ_LEN in "${LENGTHS[@]}"; do
            for SIGMA in "${SIGMAS[@]}"; do
                for THR in "${THRESHOLDS[@]}"; do
                    run_experiment $M_SIZE $SEQ_LEN $SIGMA $THR $SEED
                done
            done
        done
    done
done

echo -e "\n>>> All C4 300M HIPE Experiments Completed."