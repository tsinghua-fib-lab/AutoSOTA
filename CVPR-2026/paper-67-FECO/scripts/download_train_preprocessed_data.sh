#!/bin/bash

# Set directories
TEMP_DIR="temp_feco_train_data"
TARGET_DIR="data"

# Initialize git-lfs and clone dataset repo
git lfs install
git clone https://huggingface.co/datasets/dqj5182/feco-data "$TEMP_DIR"

# --------- Extract all .tar.gz in TEMP_DIR BEFORE moving ---------
echo "Extracting .tar.gz files inside $TEMP_DIR..."

find "$TEMP_DIR" -type f -name "*.tar.gz" | while read -r file; do
    echo "Extracting: $file"
    dir=$(dirname "$file")
    tar -xzf "$file" -C "$dir"
    if [ $? -eq 0 ]; then
        echo "Successfully extracted: $file"
        rm "$file"
    else
        echo "Failed to extract: $file"
    fi
done

# Create target directory if needed
mkdir -p "$TARGET_DIR"

# Now sync only the extracted contents (excluding .tar.gz)
rsync -av --exclude='*.tar.gz' "$TEMP_DIR/train/data/" "$TARGET_DIR/"

# Clean up temporary cloned repo
rm -rf "$TEMP_DIR"

echo "All extracted data moved to $TARGET_DIR"