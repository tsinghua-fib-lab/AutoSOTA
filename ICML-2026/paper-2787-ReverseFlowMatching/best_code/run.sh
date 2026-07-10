#!/bin/bash

ENVS=(
	"cheetah-run"
#	"cartpole-swingup"
#	"fish-swim"
#	"fish-upright"
#	"walker-walk"
#	"walker-run"
#	"finger-spin"
#	"finger-turn_hard"
)

ALGOS=(
	"rfm"
#	"dqs"
#	"maxentdp"
#	"qsm"
#	"qvpo"
#	"sac"
)

SEED_START=1
SEED_END=1

for env_name in "${ENVS[@]}"; do
	for algo in "${ALGOS[@]}"; do
		for seed in $(seq $SEED_START $SEED_END); do

			echo "=========================================="
			echo "Training: $algo on $env_name (seed=$seed)"
			echo "=========================================="

			config_args=()
			if [[ -f "configs/$algo.yaml" ]]; then
				config_args=(--config "configs/$algo.yaml")
			fi

			python main.py \
				"${config_args[@]}" \
				--override "env.name=$env_name" \
				--override "algo=$algo" \
				--override "seed=$seed"
		done
	done
done

echo "=========================================="
echo "All training runs completed!"
echo "=========================================="
