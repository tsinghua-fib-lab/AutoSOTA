#!/usr/bin/env bash
# Run a best-of-N simulator experiment (--mode best_of_n).
#
# At every turn, all assistants generate candidate responses; the highest-reward
# one is committed to the conversation, and every candidate (with its score) is
# recorded under per_turn_candidates for downstream DPO/GRPO dataset construction.
#
# Usage:
#   bash scripts/simulate/run_synth.sh \
#     <artifacts.json> <output_dir> <assistants.json> [user.json] [reward.json] [max_turns] [workers]
#
# After this completes, materialize a training dataset with:
#   python -m discoverllm.data.build_dataset \
#     --input_dir <output_dir> --output <dataset_dir> --save_format hf --score_type multiturn

source "$(dirname "$0")/../_common.sh"
parse_simulate_args "$@"
MAX_TURNS="${6:-5}"  # best_of_n typically converges faster than best_of_1

python -m discoverllm.simulate.run \
    "$ARTIFACTS" "$OUTPUT_DIR" \
    -a "$ASSISTANTS" -u "$USER_CFG" -r "$REWARD_CFG" \
    --mode best_of_n \
    --max-turns "$MAX_TURNS" \
    --parallel-workers "$WORKERS" \
    --window-size 0
