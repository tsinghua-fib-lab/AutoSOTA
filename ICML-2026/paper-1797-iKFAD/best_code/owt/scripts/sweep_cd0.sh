# Choose GPU
export CUDA_VISIBLE_DEVICES=7

# Log file name (timestamped)
LOGFILE="logs_best_hparams/sweep_cd0.log"

# Run the job detached with nohup
nohup python3 -u best_hparams/sweep_cd0.py > "$LOGFILE" 2>&1 &

# Print info
echo "Started sweep_cd0.py on GPU $CUDA_VISIBLE_DEVICES"
echo "Logging to: $LOGFILE"
echo "PID: $!"
