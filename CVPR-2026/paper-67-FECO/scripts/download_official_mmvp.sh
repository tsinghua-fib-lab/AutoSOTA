#!/bin/bash

# Set save location explicitly
SAVE_DIR="data/MMVP/data"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

echo "Saving all files to: $SAVE_DIR"

# Download files
gdown https://drive.google.com/uc?id=15nlNVIqjd7PAmqH0soa-l-z3_emTv7SG -O "$SAVE_DIR/images.7z"
gdown https://drive.google.com/uc?id=1dy5jR0h4QcLViVS1pbELub-yluUtkz3R -O "$SAVE_DIR/annotations.7z"

echo "Download completed. Starting extraction..."

# Extract images
7z x "$SAVE_DIR/images.7z" -o"$SAVE_DIR"

# Extract annotations
7z x "$SAVE_DIR/annotations.7z" -o"$SAVE_DIR"

echo "Extraction completed."

Optional: remove compressed files
rm "$SAVE_DIR/images.7z"
rm "$SAVE_DIR/annotations.7z"

echo "All files are ready in: $SAVE_DIR"