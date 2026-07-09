#!/bin/bash

#SBATCH --job-name=exp2-grad-fix
#SBATCH --output=./logs/exp2_grad_%j.out
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

export PYTHONPATH="$(pwd)/OLMo:$PYTHONPATH"
PYTHON_BIN="/home/xxxxxxxxx/anaconda3/bin/python"
SCRIPT="train_exp2_layerwise.py"

CHECKPOINT_ROOT="/data/xxxxxxxxx/03-proj/PE/checkpoints_gradient_full"
LOCAL_DATA="/data/xxxxxxxxx/03-proj/PE/wikitext/raw"
LOCAL_TOKENIZER="/data/xxxxxxxxx/03-proj/PE/wikitext/tokenizer"

mkdir -p $CHECKPOINT_ROOT
mkdir -p ./logs


GLOBAL_BS=64
SEED=6198
MAX_TOKENS=100000000


DEBUG_STEPS=""

if [ -n "$DEBUG_STEPS" ]; then
    echo ">>> [DEBUG MODE] Running for $DEBUG_STEPS steps only."
    TRAIN_ARGS="--max_train_steps $DEBUG_STEPS"
else
    echo ">>> [FULL MODE] Training for $MAX_TOKENS tokens."
    TRAIN_ARGS="--max_tokens $MAX_TOKENS"
fi



MODELS=("20M")
LENGTHS=(2048)



EXP_CONFIGS=(
    "grad_1:None None None 50.0 200.0 500.0 700.0 1000.0"
    "grad_2:None None 10.0 50.0 200.0 500.0 700.0 1000.0"
    "grad_3:None None None None 50.0 200.0 500.0 700.0"
    "grad_4:None 1.0 10.0 50.0 200.0 500.0 700.0 1000.0"
    "grad_5:None None None None None None 200.0 700.0"
)

echo ">>> Starting Experiments..."

for MODEL in "${MODELS[@]}"; do
    for SEQ_LEN in "${LENGTHS[@]}"; do

        
        if [ "$SEQ_LEN" -eq 512 ]; then
            MICRO_BS=64
        elif [ "$SEQ_LEN" -eq 1024 ]; then
            MICRO_BS=16
        elif [ "$SEQ_LEN" -ge 2048 ]; then
            MICRO_BS=8
        else
            MICRO_BS=8
        fi
        
        if [ "$MODEL" == "60M" ] && [ "$SEQ_LEN" -ge 2048 ]; then
            MICRO_BS=4
        fi
        
        for config in "${EXP_CONFIGS[@]}"; do
            
            EXP_SUFFIX="${config%%:*}" 
            SIGMA_LIST="${config#*:}" 
            
            RUN_ID="${MODEL}_L${SEQ_LEN}_${EXP_SUFFIX}"
            
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            if [ -n "$DEBUG_STEPS" ]; then
                OUTPUT_DIR="$CHECKPOINT_ROOT/debug_${RUN_ID}_${TIMESTAMP}"
            else
                OUTPUT_DIR="$CHECKPOINT_ROOT/${RUN_ID}_${TIMESTAMP}"
            fi

            echo "================================================================"
            echo ">>> Run: $RUN_ID"
            echo ">>> Len: $SEQ_LEN | Global BS: $GLOBAL_BS | Micro BS: $MICRO_BS"
            echo ">>> Accum Steps: $((GLOBAL_BS / MICRO_BS))"
            echo "================================================================"

            $PYTHON_BIN $SCRIPT \
                --output_dir $OUTPUT_DIR \
                --run_id $RUN_ID \
                --model_size $MODEL \
                --local_data_path $LOCAL_DATA \
                --local_tokenizer_path $LOCAL_TOKENIZER \
                --seq_len $SEQ_LEN \
                --global_batch_size $GLOBAL_BS \
                --micro_batch_size $MICRO_BS \
                --seed $SEED \
                --use_scaled_rope \
                --sigma_list $SIGMA_LIST \
                $TRAIN_ARGS
            
            if [ $? -ne 0 ]; then
                echo ">>> [ERROR] Run ${RUN_ID} Failed!"
                if [ -n "$DEBUG_STEPS" ]; then exit 1; fi
            fi
            
        done
    done
done

echo ">>> All Experiments Finished."