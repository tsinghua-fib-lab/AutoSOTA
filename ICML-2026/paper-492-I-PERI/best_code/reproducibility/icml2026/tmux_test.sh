# Start a new tmux session (detached)
tmux new-session -d -s mysession
seeds="48291 17305 96542 20877 73119 59483 12640 84752 31908 67024"

# Run commands for each seed in a new tmux window
for seed in {21..52}; do
    tmux new-window -t mysession -n "seed_$seed" "conda activate peri+"
    tmux new-window -t mysession -n "seed_$seed" "sh test.sh $seed"
done
