# Choose GPU
export CUDA_VISIBLE_DEVICES=5

# Log file name (timestamped)
LOGFILE="logs/sweep_cd0.log"

# Run the job detached with nohup
nohup python3 -u sweeps/sweep_cd0.py > "$LOGFILE" 2>&1 &

# Print info
echo "Started sweep_cd0.py on GPU $CUDA_VISIBLE_DEVICES"
echo "Logging to: $LOGFILE"
echo "PID: $!"
