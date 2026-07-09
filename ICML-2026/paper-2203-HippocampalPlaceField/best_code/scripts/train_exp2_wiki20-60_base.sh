#!/bin/bash

#SBATCH --job-name=exp2-wiki20-60-base
#SBATCH --output=/data/xxxxxxxxx/03-proj/PE/logs/exp2_wiki20-60/tmp_base_%j.out
#SBATCH --error=/data/xxxxxxxxx/03-proj/PE/logs/exp2_wiki20-60/tmp_base_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
PYTHON_BIN="/home/xxxxxxxxx/anaconda3/bin/python"
SCRIPT="train_exp2_wiki20-60.py" 

ROOT_DIR="/data/xxxxxxxxx/03-proj/PE"
LOG_DIR="${ROOT_DIR}/logs/exp2_wiki20-60/$(date +%Y%m%d)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/wiki20-60/base"
WANDB_DIR="${ROOT_DIR}/wandb/offline/exp2_wiki20-60"

LOCAL_DATA="/data/xxxxxxxxx/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/xxxxxxxxx/03-proj/PE/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR


JOB_ID=${SLURM_JOB_ID}

FINAL_OUT="${LOG_DIR}/base_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/base_${JOB_ID}.err"


exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f /data/xxxxxxxxx/03-proj/PE/logs/exp2_wiki20-60/tmp_base_${JOB_ID}.out
    rm -f /data/xxxxxxxxx/03-proj/PE/logs/exp2_wiki20-60/tmp_base_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"

GLOBAL_BS=64
SEEDS=(6198 1024 7 568 3427)
MAX_TOKENS=100000000


DEBUG_STEPS="" 

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS"
fi


MODELS=("20M" "60M")
LENGTHS=(512 1024 2048)


get_mbs() {
    local m_size=$1
    local seq_len=$2
    local baseline_type=$3
    

    local mbs=32
    if [ "$m_size" == "60M" ]; then mbs=16; fi
    if [ "$seq_len" -ge 2048 ]; then mbs=$((mbs / 2)); fi


    if [ "$baseline_type" == "alibi" ]; then
        if [ "$seq_len" -eq 2048 ]; then
            if [ "$m_size" == "20M" ]; then mbs=8; else mbs=4; fi
        fi
    fi

    echo $mbs
}


run_baseline_experiment() {
    local baseline_type=$1
    local extra_args=$2

    echo -e "\n>>> [BATCH START] Running $baseline_type..."
    

    for SEED in "${SEEDS[@]}"; do
        for M_SIZE in "${MODELS[@]}"; do
            for SEQ_LEN in "${LENGTHS[@]}"; do

                CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN $baseline_type)
                
                RUN_ID="baseline_${baseline_type}_${M_SIZE}_L${SEQ_LEN}"
                
                OUTPUT_DIR="${CHECKPOINT_ROOT}/${M_SIZE}/${RUN_ID}/seed_${SEED}"
                
                echo ">>> [$baseline_type] Model: $M_SIZE | Len: $SEQ_LEN | MBS: $CUR_MICRO_BS | SEED: $SEED"

                $PYTHON_BIN $SCRIPT \
                    --output_dir $OUTPUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
                    --local_data_path $LOCAL_DATA --local_tokenizer_path $LOCAL_TOKENIZER \
                    --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
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


echo -e "\n>>> All Baselines Completed."