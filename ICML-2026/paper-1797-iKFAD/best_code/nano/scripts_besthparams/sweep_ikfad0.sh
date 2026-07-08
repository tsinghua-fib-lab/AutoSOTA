# Log file name (timestamped)
LOGFILE="logs_besthparams/sweep_ikfad0.log"

# Run the job detached with nohup
# We define CUDA_VISIBLE_DEVICES=6 RIGHT HERE to force it strictly for this command
CUDA_VISIBLE_DEVICES=4 nohup python3 -u best_hparams/sweep_ikfad0.py > "$LOGFILE" 2>&1 &

# Print info
echo "Job launched on GPU 6."
echo "Logging to: $LOGFILE"
echo "PID: $!"