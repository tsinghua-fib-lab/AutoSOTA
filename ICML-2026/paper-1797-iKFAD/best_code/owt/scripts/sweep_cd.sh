# Choose GPU
export CUDA_VISIBLE_DEVICES=3

# Log file name (timestamped)
LOGFILE="logss/sweep_cd.log"

# Run the job detached with nohup
nohup python3 -u sweeps/sweep_cd.py > "$LOGFILE" 2>&1 &

# Print info
echo "Started sweep_cd.py on GPU $CUDA_VISIBLE_DEVICES"
echo "Logging to: $LOGFILE"
echo "PID: $!"
