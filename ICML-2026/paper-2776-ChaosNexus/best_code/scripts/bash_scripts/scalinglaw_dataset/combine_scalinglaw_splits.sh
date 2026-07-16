#!/bin/bash

# Combine scalinglaw splits into a single dataset.
# For example, say we have the following splits:
# split_0-163 with 128 ICs
# split_163-327 with 64 ICs
# split_327-655 with 32 ICs
# split_655-1311 with 16 ICs
# split_1311-2622 with 8 ICs
# split_2622-5244 with 4 ICs
# split_5244-10489 with 2 ICs

# Where the smaller splits are subsets of the larger splits. We saved compute time by using the fact that
# split_0-163 is a subset of split_0-327, which is a subset of split_0-655, which is a subset of split_0-1311, etc.

# This script will combine splits from the bottom up, so it will start with split_0-163 and move up to split_0-10489.
# e.g. the first 64 ICs from all systems in split_0-163 will be copied to split_163-327, etc.

SRC_DIR="$1"
DEST_DIR="$2"
n_files_to_move="$3"

if [[ -z "$SRC_DIR" || -z "$DEST_DIR" ]]; then
  echo "Usage: $0 <source_directory> <destination_directory>"
  exit 1
fi

n_overlap_subdirs=0
# Iterate over each subdirectory in the source directory
for subdir in "$SRC_DIR"/*/ ; do
  # Ensure it's a directory
  if [ -d "$subdir" ]; then
    subdirName=$(basename "$subdir")

    echo "Processing $subdirName"
    
    # Create the corresponding subdirectory in the destination directory
    if [ -d "$DEST_DIR/$subdirName" ]; then
      echo "Warning: Directory $DEST_DIR/$subdirName already exists"
      n_overlap_subdirs=$((n_overlap_subdirs + 1))
    fi
    mkdir -p "$DEST_DIR/$subdirName"
    
    # Find files matching the pattern and sort by the numeric prefix before the underscore.
    # Then select the first two files.
    files=$(ls "$subdir"*"_T-4096.arrow" 2>/dev/null | sort -V | head -n "$n_files_to_move")
    # echo "Found:"
    # echo "$files"

    # Move each selected file into the destination subdirectory.
    for file in $files; do
    #   echo "Moving $file"
      if [ -f "$file" ]; then
        cp "$file" "$DEST_DIR/$subdirName/"
      fi
    done
  fi
done

echo "Move completed."
echo "Number of overlapping subdirectories: $n_overlap_subdirs"