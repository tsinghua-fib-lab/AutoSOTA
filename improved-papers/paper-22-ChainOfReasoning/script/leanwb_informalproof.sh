# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

export OPENAI_API_KEY=...

INPUT_FILE="/path/to/Lean-Workbook"
OUTPUT_FILE="/path/to/save/output"

OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

python3 ./utils/generate_informalproof.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"
