# Choose GPU
export CUDA_VISIBLE_DEVICES=5

# Log file name (timestamped)
LOGFILE="logs_besthparams/sweep_adam.log"

# Run the job detached with nohup
nohup python3 -u best_hparams/sweep_adam.py > "$LOGFILE" 2>&1 &
echo "Logging to: $LOGFILE"
echo "PID: $!"
