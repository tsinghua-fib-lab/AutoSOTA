#!/bin/bash

# Root directory where layout 1 is located
ROOT_DIR=$1

# Loop over each subdirectory in ROOT_DIR
for dir in "$ROOT_DIR"/*/; do
    dir=${dir%/}  # remove trailing slash
    base_dir=$(basename "$dir")

    # Match directories like systemA_systemB-pp12
    if [[ "$base_dir" =~ ^([a-zA-Z0-9]+_[a-zA-Z0-9]+)-pp[0-9]+$ ]]; then
        prefix=$(echo "$base_dir" | grep -o 'pp[0-9]\+')
        target_dirname="${BASH_REMATCH[1]}"

        echo "prefix: $prefix"
        echo "target_dirname: $target_dirname"

        # Create target directory if it doesn't exist
        target_dir="$ROOT_DIR/$target_dirname"
        mkdir -p "$target_dir"

        # Loop through all files in current subdirectory
        for file in "$dir"/*; do
            [[ -f "$file" ]] || continue  # skip if not a file
            filename=$(basename "$file")
            new_filename="${prefix}_${filename}"
            mv "$file" "$target_dir/$new_filename"
        done

        # Check if directory is empty before removing
        if [ -z "$(ls -A "$dir")" ]; then
            rmdir "$dir"
        fi
    fi
done

echo "Restructuring complete."
