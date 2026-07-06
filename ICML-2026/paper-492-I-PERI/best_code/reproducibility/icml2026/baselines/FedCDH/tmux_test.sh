# Define seeds
seeds="84752 96542"

# Start a new tmux session (detached)
tmux new-session -d -s mysession

# Run commands for each seed in a new tmux window
for seed in $seeds; do
    tmux new-window -t mysession -n "seed_$seed" "conda activate peri+"
    tmux new-window -t mysession -n "seed_$seed" "sh baselines/FedCDH/test.sh $seed"
done
