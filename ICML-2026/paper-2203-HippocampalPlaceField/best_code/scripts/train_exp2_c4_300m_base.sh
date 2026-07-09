#!/bin/bash

#SBATCH --job-name=exp2-c4-300M-base
#SBATCH --output=/data/xxxxxxxxx/03-proj/PE/logs/exp2_c4_300M/tmp_base_%j.out
#SBATCH --error=/data/xxxxxxxxx/03-proj/PE/logs/exp2_c4_300M/tmp_base_%j.err
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
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/c4_300M/base"
WANDB_DIR="${ROOT_DIR}/wandb/offline/exp2_c4_300M"

C4_DATA_ROOT="${ROOT_DIR}"
LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR

JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/base_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/base_${JOB_ID}.err"

exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f /data/xxxxxxxxx/03-proj/PE/logs/exp2_c4_300M/tmp_base_${JOB_ID}.out
    rm -f /data/xxxxxxxxx/03-proj/PE/logs/exp2_c4_300M/tmp_base_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

GLOBAL_BS=64
SEEDS=(6198) # 1024 7 568 3427)


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

MODELS=("300M")
LENGTHS=(512 1024 2048)

get_mbs() {
    local m_size=$1
    local seq_len=$2
    local baseline_type=$3
    
    local mbs=16
    if [ "$seq_len" -ge 2048 ]; then mbs=8; fi

    if [ "$baseline_type" == "alibi" ]; then
        mbs=$((mbs / 2));
        if [ "$seq_len" -ge 1024 ]; then mbs=1; fi
    fi

    echo $mbs
}

run_baseline_experiment() {
    local baseline_type=$1
    local extra_args=$2
    
    echo -e "\n>>> [BATCH START] Running $baseline_type on C4..."
    
    for SEED in "${SEEDS[@]}"; do
        for M_SIZE in "${MODELS[@]}"; do
            for SEQ_LEN in "${LENGTHS[@]}"; do
                CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN $baseline_type)
                RUN_ID="c4_baseline_${baseline_type}_${M_SIZE}_L${SEQ_LEN}"
                OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"
                
                echo ">>> [$baseline_type] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | SEED: $SEED"
                
                $PYTHON_BIN $SCRIPT \
                    --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                    --dataset_path $C4_DATA_ROOT --local_tokenizer_path $LOCAL_TOKENIZER \
                    --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
                    --train_size $TRAIN_SAMPLES --val_size $VAL_SAMPLES \
                    --wandb_dir $WANDB_DIR --wandb_mode $WANDB_MODE \
                    $extra_args \
                    $LIMIT_ARGS --seed $SEED
                
                if [ $? -ne 0 ]; then
                    echo ">>> [ERROR] $baseline_type Model: $M_SIZE, Len: $SEQ_LEN, SEED: $SEED"
                fi
            done
        done
    done
}

run_baseline_experiment "rope" ""
run_baseline_experiment "xpos" "--xpos"
run_baseline_experiment "nope" "--nope"
run_baseline_experiment "alibi" "--alibi"

echo -e "\n>>> All C4 300M Baselines Completed."