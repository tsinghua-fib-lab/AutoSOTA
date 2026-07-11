#!/bin/bash
# run_petri_all_targets.sh — run Petri (conversation-only) auditing on the ANCHOR target models.
#
# Prereqs:
#   pip install git+https://github.com/safety-research/petri   # Anthropic/Safety Research's Petri
#   export OPENROUTER_API_KEY=...   ANTHROPIC_API_KEY=...
#
# Then score the resulting trajectories with evaluate_petri_trajectories.py.

AUDITOR="openrouter/qwen/qwen3-235b-a22b"
JUDGE="openrouter/qwen/qwen3-8b"
MAX_TURNS="${MAX_TURNS:-100}"
SAVE_DIR="${SAVE_DIR:-./petri_runs}"      # where transcripts are written
mkdir -p "$SAVE_DIR"

TARGETS=(
    "anthropic/claude-haiku-4-5-20251001"
    "openrouter/qwen/qwen3-235b-a22b-2507"
    "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
    "openrouter/qwen/qwen3-14b"
    "openrouter/qwen/qwen3-8b"
    "openrouter/xiaomi/mimo-v2-flash"
    "openrouter/z-ai/glm-4.5"
)

count=0
for target in "${TARGETS[@]}"; do
    short_name=$(echo "$target" | sed 's|.*/||')
    log_file="${SAVE_DIR}/petri_${short_name}.log"
    echo "Starting Petri eval: target=${target} -> ${log_file}"

    inspect eval petri/audit \
        --model-role auditor="$AUDITOR" \
        --model-role target="$target" \
        --model-role judge="$JUDGE" \
        -T max_turns="$MAX_TURNS" \
        -T transcript_save_dir="$SAVE_DIR" \
        > "$log_file" 2>&1 &

    count=$((count + 1))
    if [ $((count % 3)) -eq 0 ]; then echo "Waiting for batch of 3..."; wait; fi
done
wait
echo "All done."
