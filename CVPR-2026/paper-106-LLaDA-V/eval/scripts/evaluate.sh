# Define multiple model paths
# Ablation Models
MODEL_PATHS=(
    "GSAI-ML/LLaDA-V"
)

# Set output path
OUTPUT_PATH=exp/llava_v_eval
# Set task names
TASK_NAMES="mmmu_val,mmmu_pro_standard,mmstar,ai2d,seedbench,mmbench_en_dev,mmmu_pro_vision,muirbench,videomme,mlvu_dev,mme,realworldqa,chartqa,docvqa_val,infovqa_val,mathvista_testmini,mathverse_testmini_vision"

TOTAL_GPUS=8
declare -A GPU_STATUS  # Record GPU status: 0=idle, 1=busy
declare -A GPU_PIDS    # Record PIDs of tasks running on each GPU

# Initialize GPU status
for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
    GPU_STATUS[$gpu]=0
done

# Split task names into array
IFS=',' read -ra TASKS <<< "$TASK_NAMES"

# Create task queue
declare -a TASK_QUEUE
for model_path in "${MODEL_PATHS[@]}"; do
    # Set model parameters based on model path
    MODEL=llava_onevision_llada
    MODEL_NAME=llava_llada
    CONV_TEMPLATE=llava_llada

    # Use task-specific generation parameters
    for task in "${TASKS[@]}"; do
        case $task in
            mmmu_val|mmmu_pro_standard|mmstar|ai2d|seedbench|mmbench_en_dev|mmmu_pro_vision|muirbench|videomme|mlvu_dev|mme|realworldqa)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
                ;;
            chartqa)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":16,"block_length":16,"gen_steps":16,"stopping_criteria":["\n"],"think_mode":"no_think"}'
                ;;
            docvqa_val|infovqa_val)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":32,"block_length":32,"gen_steps":16,"think_mode":"no_think"}'
                ;;
            mathvista_testmini)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":96,"block_length":96,"gen_steps":48,"think_mode":"think"}'
                ;;
            mathverse_testmini_vision)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":64,"block_length":64,"gen_steps":32,"think_mode":"think"}'
                ;;
            *)
                GEN_KWARGS='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}'
                ;;
        esac

        # Add model parameters to task queue
        TASK_QUEUE+=("$model_path $task $MODEL $MODEL_NAME $CONV_TEMPLATE $GEN_KWARGS")
    done
done

# Total tasks
TOTAL_TASKS=${#TASK_QUEUE[@]}
COMPLETED_TASKS=0 # Count of started tasks
FINISHED_TASKS=0 # Count of finished tasks (used to determine if all tasks are done)

echo "Total $TOTAL_TASKS evaluation tasks to execute"

# Main loop until all tasks are completed
while [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; do
    # Check completed background tasks and release GPU
    for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
        if [[ ${GPU_STATUS[$gpu]} -eq 1 && -n "${GPU_PIDS[$gpu]}" ]]; then # If GPU is marked as busy and has PID
            if ! kill -0 ${GPU_PIDS[$gpu]} 2>/dev/null; then  # Check if task process is still running
                echo "Detected task on GPU $gpu (PID: ${GPU_PIDS[$gpu]}) has completed or exited abnormally."
                GPU_STATUS[$gpu]=0    # Mark GPU as idle
                unset GPU_PIDS[$gpu] # Clear PID record
                FINISHED_TASKS=$((FINISHED_TASKS + 1)) # Increment finished task count
                echo "GPU $gpu released. Completed tasks: $FINISHED_TASKS / $TOTAL_TASKS"
            fi
        fi
    done

    # Assign new tasks to idle GPUs (if there are tasks not started)
    if [ $COMPLETED_TASKS -lt $TOTAL_TASKS ]; then # Make sure there are still tasks to start
        for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
            if [[ ${GPU_STATUS[$gpu]} -eq 0 && $COMPLETED_TASKS -lt $TOTAL_TASKS ]]; then # If GPU is idle and there are tasks not started
                # Get current task
                CURRENT_TASK_STRING="${TASK_QUEUE[$COMPLETED_TASKS]}"
                read -r MODEL_PATH TASK_NAME CURRENT_MODEL CURRENT_MODEL_NAME CURRENT_CONV_TEMPLATE CURRENT_GEN_KWARGS <<< "$CURRENT_TASK_STRING"

                # Update GPU status
                GPU_STATUS[$gpu]=1

                # Prepare output path
                MODEL_PATH_LAST=$(basename "$MODEL_PATH")
                OUTPUT_PATH_LAST=$(basename "$OUTPUT_PATH")

                CURRENT_OUTPUT_PATH="$OUTPUT_PATH/$MODEL_PATH_LAST"
                
                # Create log file
                LOG_FILE_NAME="${TASK_NAME}_${CURRENT_GEN_KWARGS//,/_}.log"
                mkdir -p "$CURRENT_OUTPUT_PATH" # Ensure path exists
                # Clear or create log file
                echo "Task: $TASK_NAME" > "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
                echo "Output path: $CURRENT_OUTPUT_PATH" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
                if [[ -n "$CURRENT_GEN_KWARGS" ]]; then
                    echo "Generation parameters: $CURRENT_GEN_KWARGS" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
                fi
                echo "Model: $CURRENT_MODEL_NAME" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
                echo "Conversation template: $CURRENT_CONV_TEMPLATE" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
                echo "Model path: $MODEL_PATH" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
                echo "GPU ID: $gpu" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
                echo "Task number: $COMPLETED_TASKS / $TOTAL_TASKS" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"
                echo "----------------------------------------" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME"

                echo "Starting task (number $COMPLETED_TASKS / $TOTAL_TASKS): Evaluating $MODEL_PATH_LAST on $TASK_NAME task using GPU $gpu"

                # Execute evaluation command in background
                (
                    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 -m lmms_eval \
                        --model "$CURRENT_MODEL" \
                        ${CURRENT_GEN_KWARGS:+--gen_kwargs="$CURRENT_GEN_KWARGS"} \
                        --model_args "pretrained=$MODEL_PATH,conv_template=$CURRENT_CONV_TEMPLATE,model_name=$CURRENT_MODEL_NAME" \
                        --tasks "$TASK_NAME" \
                        --batch_size 1 \
                        --log_samples \
                        --log_samples_suffix "$TASK_NAME" \
                        --output_path "$CURRENT_OUTPUT_PATH" >> "$CURRENT_OUTPUT_PATH/$LOG_FILE_NAME" 2>&1
                ) &
                GPU_PIDS[$gpu]=$! # Get background task PID and record it
                echo "Task (PID: ${GPU_PIDS[$gpu]}) started on GPU $gpu."

                # Increment started task count
                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
            fi
        done
    fi

    # Short wait to avoid CPU spinning too frequently
    if [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; then # Only sleep if there are still tasks to complete
    # Brief wait to avoid CPU spinning too frequently
    if [ $FINISHED_TASKS -lt $TOTAL_TASKS ]; then # Only sleep when there are unfinished tasks
        sleep 10 # Can adjust wait time based on actual situation
    fi
done

echo "Waiting for all remaining background tasks to complete naturally..."
wait # Wait for any potentially still-running background tasks (shouldn't exist in theory, but as a safeguard)
echo "All evaluation tasks completed! Total of $TOTAL_TASKS tasks executed."