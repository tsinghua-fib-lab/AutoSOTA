#!/bin/bash

#SBATCH --job-name=exp2-c4-syn-60m
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
export WANDB_MODE="offline"
PYTHON_BIN="/home/xxxxxxxxx/anaconda3/bin/python"
SCRIPT="train_exp2_c4_60m_syn.py" 

ROOT_DIR="/data/xxxxxxxxx/03-proj/PE"
LOG_DIR="${ROOT_DIR}/logs/synergy_60M/$(date +%Y%m%d)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints_exp2/synergy_60M"
WANDB_DIR="${ROOT_DIR}/wandb/offline/synergy_60M"
C4_DATA_ROOT="${ROOT_DIR}"
LOCAL_TOKENIZER="${ROOT_DIR}/wikitext/tokenizer"

mkdir -p $LOG_DIR $CHECKPOINT_ROOT $WANDB_DIR


JOB_ID=${SLURM_JOB_ID}
FINAL_OUT="${LOG_DIR}/syn_${JOB_ID}.out"
FINAL_ERR="${LOG_DIR}/syn_${JOB_ID}.err"

exec > >(tee -a ${FINAL_OUT}) 2> >(tee -a ${FINAL_ERR} >&2)

function cleanup {
    rm -f /data/xxxxxxxxx/03-proj/PE/logs/synergy_60M/tmp_${JOB_ID}.out
    rm -f /data/xxxxxxxxx/03-proj/PE/logs/synergy_60M/tmp_${JOB_ID}.err
}
trap cleanup EXIT

echo ">>> Experiment started at $(date)"
echo ">>> Log directory: ${LOG_DIR}"
echo ">>> SLURM Job ID: ${JOB_ID}"


MODELS=("60M")
GLOBAL_BS=64
SEEDS=(6198)


MAX_TOKENS=400000000 # 400M tokens
TRAIN_SAMPLES=2000000
VAL_SAMPLES=10000


DEBUG_STEPS="" 


if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE ENABLED] Steps: $DEBUG_STEPS"
    LIMIT_ARGS="--max_train_steps $DEBUG_STEPS --lr 6e-4"
else
    echo ">>> [FULL MODE] Max Tokens: $MAX_TOKENS"
    LIMIT_ARGS="--max_tokens $MAX_TOKENS --lr 6e-4"
fi

LENGTHS=(1024 2048)  # 2048)
WINDOW_SIZES=(128)  # 256)
SIGMAS=(200.0 300.0)
LOCAL_LAYERS=4  
HIPE_THRESHOLD=3

get_mbs() {
    local m_size=$1
    local seq_len=$2
    local exp_type=$3
    

    local mbs=32
    if [ "$m_size" == "60M" ]; then mbs=16; fi
    

    if [ "$seq_len" -ge 2048 ]; then mbs=$((mbs / 2)); fi


    if [[ "$exp_type" == "L2G" ]] || [[ "$exp_type" == "Syn" ]]; then
        if [ "$seq_len" -ge 2048 ]; then
            if [ "$m_size" == "20M" ]; then 
                mbs=8 
            else 
                mbs=4 
            fi
        fi
    fi

    echo $mbs
}

run_exp() {
    local TYPE=$1; local M_SIZE=$2; local SEQ_LEN=$3; local EXTRA_ARGS=$4

    local CUR_MICRO_BS=$(get_mbs $M_SIZE $SEQ_LEN $TYPE)
    
    local CLEAN_ARGS=$(echo "$EXTRA_ARGS" | sed 's/--//g' | sed 's/ /_/g' | sed 's/_local_window_size_/W/g' | sed 's/_num_local_layers_4//g' | sed 's/_use_scaled_rope//g' | sed 's/_sigma_/S/g' | sed 's/_rope_scaling_threshold_3//g')
    
    local RUN_ID="syn_${TYPE}_L${SEQ_LEN}_${CLEAN_ARGS}"
    local OUT_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
    
    echo ">>> Running: [${TYPE}] Model=${M_SIZE} | Len=${SEQ_LEN} | MBS=${CUR_MICRO_BS} | Args: ${EXTRA_ARGS}"
    
    $PYTHON_BIN $SCRIPT \
        --output_dir $OUT_DIR --run_id $RUN_ID --model_size $M_SIZE \
        --dataset_path $C4_DATA_ROOT --local_tokenizer_path $LOCAL_TOKENIZER \
        --wandb_dir $WANDB_DIR --wandb_mode $WANDB_MODE \
        --train_size $TRAIN_SAMPLES --val_size $VAL_SAMPLES \
        --seq_len $SEQ_LEN --global_batch_size $GLOBAL_BS --micro_batch_size $CUR_MICRO_BS \
        --seed ${SEEDS[0]} $LIMIT_ARGS $EXTRA_ARGS
}

echo ">>> Starting 60M Synergy Ablation Study..."

for M in "${MODELS[@]}"; do
    for L in "${LENGTHS[@]}"; do

        run_exp "Base" $M $L ""
        
        for W in "${WINDOW_SIZES[@]}"; do
            run_exp "L2G" $M $L "--local_window_size $W --num_local_layers $LOCAL_LAYERS"
        done
        
        for S in "${SIGMAS[@]}"; do
            run_exp "HIPE" $M $L "--use_scaled_rope --sigma $S --rope_scaling_threshold $HIPE_THRESHOLD"
        done
        
        for W in "${WINDOW_SIZES[@]}"; do
            for S in "${SIGMAS[@]}"; do
                run_exp "Syn" $M $L "--local_window_size $W --num_local_layers $LOCAL_LAYERS --use_scaled_rope --sigma $S --rope_scaling_threshold $HIPE_THRESHOLD"
            done
        done
    done
done

echo ">>> All Synergy Experiments Completed."