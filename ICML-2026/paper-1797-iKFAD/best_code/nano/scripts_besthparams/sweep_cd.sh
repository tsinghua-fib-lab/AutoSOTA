# Choose GPU
export CUDA_VISIBLE_DEVICES=1

# Log file name (timestamped)
LOGFILE="logs_besthparams/sweep_cd.log"

# Run the job detached with nohup
nohup python3 -u best_hparams/sweep_cd.py > "$LOGFILE" 2>&1 &

# Print info
echo "Started sweep_cd.py on GPU $CUDA_VISIBLE_DEVICES"
echo "Logging to: $LOGFILE"
echo "PID: $!"
