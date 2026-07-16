#!/bin/bash
# Extract numeric indices from all TinyBenchmark calibration response matrices

cd baseline_compare

for benchmark in gsm8k hellaswag truthfulqa winogrande; do
  echo "Extracting indices for ${benchmark}..."
  
  input_file="tinybenchmark_calibration/response_matrix_tiny${benchmark}.csv"
  output_file="tiny${benchmark}_numeric_indices.csv"
  
  if [ ! -f "$input_file" ]; then
    echo "  ERROR: $input_file not found"
    continue
  fi
  
  # Extract header, transpose to column, remove quotes, skip first (model) column, add header
  head -1 "$input_file" | tr ',' '\n' | sed 's/"//g' | tail -n +2 > /tmp/indices_${benchmark}.txt
  echo "item_index" > "$output_file"
  cat /tmp/indices_${benchmark}.txt >> "$output_file"
  
  count=$(wc -l < "$output_file")
  echo "  Created $output_file with $((count-1)) items"
done

echo ""
echo "Done! Created numeric indices for all datasets."
