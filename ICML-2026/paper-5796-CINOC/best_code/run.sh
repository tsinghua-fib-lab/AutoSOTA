#!/bin/bash

# Define the log directory and create it if it doesn't exist
LOG_DIR="/home/zanot/projects/Multi-Agent-DPC/logs"
mkdir -p "$LOG_DIR"

SCRIPTS=(
    # "/home/zanot/projects/Multi-Agent-DPC/examples/fkpp1d/decentralized/bench3.py"
    # "/home/zanot/projects/Multi-Agent-DPC/examples/heat1d/decentralized/bench3.py"
    # "/home/zanot/projects/Multi-Agent-DPC/examples/heat2D/decentralized/bench3.py"
    # "/home/zanot/projects/Multi-Agent-DPC/examples/heat2D_obstacles/decentralized/bench3.py"
    # "/home/zanot/projects/Multi-Agent-DPC/examples/ks1d/decentralized/bench3.py"
    # "/home/zanot/projects/Multi-Agent-DPC/examples/ks2d/decentralized/bench3.py"
    # "/home/zanot/projects/Multi-Agent-DPC/examples/turbulence2d/decentralized/bench3.py"
    # "/home/zanot/projects/Multi-Agent-DPC/examples/density/centralized/bench3.py"
    "/home/zanot/projects/Multi-Agent-DPC/examples/turbulence2d/decentralized/bench/train_deeponet_mappo.py"
    "/home/zanot/projects/Multi-Agent-DPC/examples/turbulence2d/decentralized/bench/train_deeponet_matd3.py"
)

NUM_SCRIPTS=${#SCRIPTS[@]}
CURRENT_SCRIPT=0

for SCRIPT_PATH in "${SCRIPTS[@]}"; do
    ((CURRENT_SCRIPT++))
    
    # Extract folder and script name
    FOLDER=$(dirname "$SCRIPT_PATH")
    SCRIPT_NAME=$(basename "$SCRIPT_PATH")
    
    # Create a unique log file name using the queue number and script name (stripping the .py)
    # Example: 1_train.txt, 2_train_mappo.txt
    LOG_FILE="$LOG_DIR/${CURRENT_SCRIPT}_${SCRIPT_NAME%.*}.txt"
    
    # Start writing to the specific log file for this run
    echo "========================================================" | tee "$LOG_FILE"
    echo "Processing script $CURRENT_SCRIPT of $NUM_SCRIPTS" | tee -a "$LOG_FILE"
    echo ">>> Running $SCRIPT_NAME from $FOLDER..." | tee -a "$LOG_FILE"
    
    # Just print this to the terminal so you know where the log is going
    echo ">>> Logs are being saved to: $LOG_FILE"
    
    # Switch to the directory
    cd "$FOLDER" || { echo "⚠️ Directory not found! Skipping $SCRIPT_NAME..." | tee -a "$LOG_FILE"; continue; }

    # Run python, capture both normal prints and errors (2>&1), and send to screen + log file
    python "$SCRIPT_NAME" 2>&1 | tee -a "$LOG_FILE"
    
    # Check if python crashed by checking the exit status
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "⚠️ ERROR: $SCRIPT_NAME failed. Continuing to the next..." | tee -a "$LOG_FILE"
    else
        echo "✅ SUCCESS: $SCRIPT_NAME finished successfully." | tee -a "$LOG_FILE"
    fi

    # Apply 90-second cooldown timer (skip if it's the final script)
    if [ "$CURRENT_SCRIPT" -lt "$NUM_SCRIPTS" ]; then
        echo ">>> Run finished (or failed). Cooling down GPU for 90 seconds..." | tee -a "$LOG_FILE"
        sleep 1
    fi

done

echo "========================================================"
echo "Queue finished! All listed scripts have been processed."
echo "You can check your logs in: $LOG_DIR"