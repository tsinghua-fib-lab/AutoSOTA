#!/usr/bin/env bash
# Run an EVALUATION simulator experiment.
#
# Each assistant in --assistant-configs-file is run independently against each
# artifact: a fresh user simulator with hidden criteria, up to --max-turns of
# back-and-forth, scored by the reward assistant.
#
# Usage:
#   bash scripts/simulate/run_eval.sh \
#     <artifacts.json> <output_dir> <assistants.json> [user.json] [reward.json] [max_turns] [workers]

source "$(dirname "$0")/../_common.sh"
parse_simulate_args "$@"

python -m discoverllm.simulate.run \
    "$ARTIFACTS" "$OUTPUT_DIR" \
    -a "$ASSISTANTS" -u "$USER_CFG" -r "$REWARD_CFG" \
    --mode best_of_1 \
    --max-turns "$MAX_TURNS" \
    --parallel-workers "$WORKERS"
