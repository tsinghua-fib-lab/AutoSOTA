# Define seeds
seeds="128 64 66 72 136 75 76 13 145 19"

# Start a new tmux session (detached)
tmux new-session -d -s mysession

# Run commands for each seed in a new tmux window
for seed in $seeds; do
    tmux new-window -t mysession -n "seed_$seed" "conda activate iperi"
    tmux new-window -t mysession -n "seed_$seed" "sh baselines/FedDAG/test.sh $seed"
done
