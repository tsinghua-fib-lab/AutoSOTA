# Choose GPU
export CUDA_VISIBLE_DEVICES=0

# Log file name (timestamped)
LOGFILE="logs_best_hparams/sweep_cadam0.log"

# Run the job detached with nohup
nohup python3 -u best_hparams/sweep_cadam0.py > "$LOGFILE" 2>&1 &

# Print info
echo "Logging to: $LOGFILE"
echo "PID: $!"
