# Define seeds
seeds="48291 17305 96542 20877 73119 59483 12640 84752 31908 67024"

# Start a new tmux session (detached)
tmux new-session -d -s mysession

# Run commands for each seed in a new tmux window
for seed in $seeds; do
    tmux new-window -t mysession -n "seed_$seed" "conda activate peri+"
    tmux new-window -t mysession -n "seed_$seed" "sh baselines/notears-admm/test.sh $seed"
done
