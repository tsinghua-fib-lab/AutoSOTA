# Choose GPU
export CUDA_VISIBLE_DEVICES=5

# Log file name (timestamped)
LOGFILE="logs_besthparams/sweep_cadam.log"

# Run the job detached with nohup
nohup python3 -u best_hparams/sweep_cadam.py > "$LOGFILE" 2>&1 &

# Print info
echo "Logging to: $LOGFILE"
echo "PID: $!"
