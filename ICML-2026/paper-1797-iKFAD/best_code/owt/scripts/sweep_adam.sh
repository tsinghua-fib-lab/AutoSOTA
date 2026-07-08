# Choose GPU
export CUDA_VISIBLE_DEVICES=0

# Log file name (timestamped)
LOGFILE="logs/sweep_adam.log"

# Run the job detached with nohup
nohup python3 -u sweeps/sweep_adam.py > "$LOGFILE" 2>&1 &
echo "Logging to: $LOGFILE"
echo "PID: $!"
