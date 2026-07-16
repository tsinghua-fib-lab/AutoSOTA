#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <config.yaml> [output_dir]"
    exit 1
fi

CONFIG_FILE=$1
OUTPUT_DIR=${2:-"modified_scripts"}
TASKS=('arc_challenge' 'arc_easy' 'boolq' 'csqa' 'hellaswag' 'MMLU' 'openbookqa' 'piqa' 'socialiqa' 'winogrande')

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Read the YAML config file
CONFIG=$(cat $CONFIG_FILE)
for TASK in "${TASKS[@]}"; do
    MODIFIED_CONFIG=$(echo "$CONFIG" | sed "s/task_aggregation:.*/task_aggregation: $TASK/")
    MODIFIED_CONFIG=$(echo "$MODIFIED_CONFIG" | sed "s|out_dir:.*|out_dir: ./per_task_out/${TASK}_out|")
    MODIFIED_CONFIG=$(echo "$MODIFIED_CONFIG" | sed "s|plots_dir:.*|plots_dir: ./per_task_out/${TASK}_out/figures|")
    echo "$MODIFIED_CONFIG" > "${OUTPUT_DIR}/${TASK}_config.yaml"
done