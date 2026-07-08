# Choose GPU
export CUDA_VISIBLE_DEVICES=4

# Log file name (timestamped)
LOGFILE="logs/sweep_ikfad.log"

# Run the job detached with nohup
nohup python3 -u sweeps/sweep_ikfad.py > "$LOGFILE" 2>&1 &

# Print info
echo "Logging to: $LOGFILE"
echo "PID: $!"
