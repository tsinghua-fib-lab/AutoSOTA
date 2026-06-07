#!/bin/bash

# List of tar.gz files that do NOT have a root directory
no_root_path_pair_list=("pair00_1" "pair19_1" "pair19_2" "pair32_1" "pair32_2")

input_dir="data/Hi4D/data"
destination_dir="data/Hi4D/data"

# If destination is not specified, use input directory
if [ -z "$destination_dir" ]; then
  destination_dir="$input_dir"
fi

# Check input directory exists
if [ ! -d "$input_dir" ]; then
  echo "Error: Input directory '$input_dir' does not exist."
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Loop through all .tar.gz files in the input directory
for file_path in "$input_dir"/*.tar.gz; do
  [ -e "$file_path" ] || continue  # Skip if no files found

  file_name=$(basename "$file_path")
  pair_name="${file_name%.tar.gz}"

  echo "Processing $file_name..."

  # Check if this file is in the no_root_path list
  if [[ " ${no_root_path_pair_list[@]} " =~ " $pair_name " ]]; then
    extract_path="$destination_dir/$pair_name"
    mkdir -p "$extract_path"
    echo "  Extracting to $extract_path..."
    tar -xzf "$file_path" -C "$extract_path"
  else
    echo "  Extracting to $destination_dir..."
    tar -xzf "$file_path" -C "$destination_dir"
  fi

  if [ $? -eq 0 ]; then
    echo "  Successfully extracted $file_name"
  else
    echo "  Failed to extract $file_name"
  fi
done