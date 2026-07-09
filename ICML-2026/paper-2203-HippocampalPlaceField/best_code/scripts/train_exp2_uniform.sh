#!/bin/bash

#SBATCH --job-name=exp2-uniform
#SBATCH --output=./logs/exp2-uniform_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/xxxxxxxxx/anaconda3/bin/python"
SCRIPT="train_exp2_layerwise.py"

CHECKPOINT_ROOT="/data/xxxxxxxxx/03-proj/PE/checkpoints_variable_len"
LOCAL_DATA="/data/xxxxxxxxx/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/xxxxxxxxx/03-proj/PE/wikitext/tokenizer"

mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs

GLOBAL_BS=64
SEED=6198
MAX_TOKENS=100000000

DEBUG_STEPS="" 


if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG] Running for $DEBUG_STEPS steps only."
    TRAIN_ARGS="--max_train_steps $DEBUG_STEPS"
else
    TRAIN_ARGS="--max_tokens $MAX_TOKENS"
fi


MODELS=("20M" "60M")
LENGTHS=(512 1024 2048)


SIGMAS=(50.0 80.0 100.0 150.0 200.0 10.0 1.0)


for M_SIZE in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do
        for SIGMA in "${SIGMAS[@]}"; do
        
            CUR_MICRO_BS=16
            
            if [ "$M_SIZE" == "60M" ]; then
                CUR_MICRO_BS=8
            fi

            if [ "$SEQ_LEN" -ge 2048 ]; then
                CUR_MICRO_BS=$((CUR_MICRO_BS / 2))
            fi


            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            RUN_ID="uniform_${M_SIZE}_L${SEQ_LEN}_sig${SIGMA}"
            OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}_${TIMESTAMP}"

            echo "----------------------------------------------------------------"
            echo ">>> [UNIFORM] Model: $M_SIZE | Len: $SEQ_LEN | Sigma: $SIGMA"
            echo ">>> Threshold: -1 (All Layers) | Micro BS: $CUR_MICRO_BS"
            
            $PYTHON_BIN $SCRIPT \
                --output_dir $OUTPUT_DIR \
                --run_id $RUN_ID \
                --model_size $M_SIZE \
                --local_data_path $LOCAL_DATA \
                --local_tokenizer_path $LOCAL_TOKENIZER \
                --seq_len $SEQ_LEN \
                --global_batch_size $GLOBAL_BS \
                --micro_batch_size $CUR_MICRO_BS \
                --seed $SEED \
                --use_scaled_rope \
                --sigma $SIGMA \
                --rope_scaling_threshold -1 \
                $TRAIN_ARGS

            if [ $? -ne 0 ]; then
                echo ">>> [ERROR] Uniform Run Failed!"
            fi
            
        done
    done
done

echo ">>> All Uniform Ablation Experiments Finished."