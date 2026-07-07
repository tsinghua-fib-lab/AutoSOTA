CONFIG_DIR=$1
REPEAT=$2
MAX_JOBS=${3:-8}
MAIN=${4:-main_pyg}
GPU_IDS=(${5:-0 1 2 3 4 5 6 7})

(
  trap 'kill 0' SIGINT
  CUR_JOBS=0
  NUM_GPUS=${#GPU_IDS[@]}  # Number of GPUs available
  GPU_STATUS=($(for ((i=0;i<$NUM_GPUS;i++)); do echo 0; done))  # Array to track GPU availability (0=free, 1=busy)
  RUNNING_JOBS=()  # Array to keep track of running jobs, their GPUs, and PIDs

  # Function to update job status and free up GPU when jobs finish
  check_jobs() {
    local updated_jobs=()
    for job in "${RUNNING_JOBS[@]}"; do
      IFS=':' read -r config_name gpu_id job_pid <<< "$job"
      
      # Check if the process is still running
      if kill -0 "$job_pid" 2>/dev/null; then
        updated_jobs+=("$job")  # Keep jobs that are still running
      else
        GPU_STATUS[$gpu_id]=0  # Mark the GPU as free
        ((--CUR_JOBS))  # Decrement the running job count
      fi
    done
    RUNNING_JOBS=("${updated_jobs[@]}")  # Update the list of running jobs
  }

  for CONFIG in "$CONFIG_DIR"/*.yaml; do
    # Skip already completed jobs
    if [[ "$CONFIG" == *_done.yaml ]]; then
      continue
    fi

    if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
      # Wait if we have reached the maximum number of jobs
      while [ "$CUR_JOBS" -ge "$MAX_JOBS" ]; do
        check_jobs  # Check if any jobs have finished and free up GPUs
        sleep 10  # Small delay to avoid tight looping
      done

      # Find an available GPU
      gpu_id=-1
      for ((i=0; i<NUM_GPUS; i++)); do
        if [ "${GPU_STATUS[$i]}" -eq 0 ]; then
          gpu_id=$i
          break
        fi
      done

      if [ "$gpu_id" -eq -1 ]; then
        echo "Error: No available GPU found, something went wrong."
        exit 1
      fi

      echo "Job launched: $CONFIG on GPU $gpu_id"

      # Launch the job and capture its PID
      python "$MAIN.py" --cfg "$CONFIG" --repeat "$REPEAT" --gpu_id "$gpu_id" --mark_done &
      job_pid=$!

      # Mark the GPU as busy and add the job to the list of running jobs with its PID
      GPU_STATUS[$gpu_id]=1
      RUNNING_JOBS+=("$CONFIG:$gpu_id:$job_pid")
      ((++CUR_JOBS))  # Increment the running job count
    fi
  done

  # Wait for all remaining jobs to finish
  while [ "$CUR_JOBS" -gt 0 ]; do
    check_jobs
    sleep 10
  done
)

  # Function to update job status and free up GPU when jobs finish
  # check_jobs() {
  #   local updated_jobs=()
  #   for job in "${RUNNING_JOBS[@]}"; do
  #     local config_name=$(echo "$job" | cut -d':' -f1)
  #     local gpu_id=$(echo "$job" | cut -d':' -f2)
      
  #     # If the job has finished (i.e., config file is marked as _done)
  #     if [[ -f "${config_name}_done" ]]; then
  #       GPU_STATUS[$gpu_id]=0  # Mark the GPU as free
  #       ((--CUR_JOBS))  # Decrement the running job count
  #     else
  #       updated_jobs+=("$job")  # Keep jobs that are still running
  #     fi
  #   done
  #   RUNNING_JOBS=("${updated_jobs[@]}")  # Update the list of running jobs
  # }

#   for CONFIG in "$CONFIG_DIR"/*.yaml; do
#     # Skip already completed jobs
#     if [[ "$CONFIG" == *_done.yaml ]]; then
#       continue
#     fi
    
#     if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
#       # Wait if we have reached the maximum number of jobs
#       while [ "$CUR_JOBS" -ge "$MAX_JOBS" ]; do
#         check_jobs  # Check if any jobs have finished and free up GPUs
#         sleep 10  # Small delay to avoid tight looping
#       done

#       # Find an available GPU
#       gpu_id=-1
#       for ((i=0; i<NUM_GPUS; i++)); do
#         if [ "${GPU_STATUS[$i]}" -eq 0 ]; then
#           gpu_id=$i
#           break
#         fi
#       done
      
#       if [ "$gpu_id" -eq -1 ]; then
#         echo "Error: No available GPU found, something went wrong."
#         exit 1
#       fi

#       echo "Job launched: $CONFIG on GPU $gpu_id"
#       python $MAIN.py --cfg $CONFIG --repeat $REPEAT --gpu_id $gpu_id --mark_done &

#       # Mark the GPU as busy and add the job to the list of running jobs
#       GPU_STATUS[$gpu_id]=1
#       RUNNING_JOBS+=("$CONFIG:$gpu_id")
#       ((++CUR_JOBS))  # Increment the running job count
#     fi
#   done

#   # Wait for all remaining jobs to finish
#   while [ "$CUR_JOBS" -gt 0 ]; do
#     check_jobs
#     sleep 10
#   done
# )
