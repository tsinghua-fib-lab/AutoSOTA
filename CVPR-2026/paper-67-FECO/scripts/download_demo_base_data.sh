#!/bin/bash

TARGET_DIR="data"
FILE_URL="https://huggingface.co/datasets/dqj5182/feco-data/resolve/main/demo/data/base_data.tar.gz"
ARCHIVE_NAME="$TARGET_DIR/base_data.tar.gz"

mkdir -p "$TARGET_DIR"

echo "Downloading base_data.tar.gz..."
wget -c "$FILE_URL" -O "$ARCHIVE_NAME"

echo "Decompressing into $TARGET_DIR..."
tar -xvzf "$ARCHIVE_NAME" -C "$TARGET_DIR"

rm "$ARCHIVE_NAME"

echo "Done. Extracted to $TARGET_DIR"