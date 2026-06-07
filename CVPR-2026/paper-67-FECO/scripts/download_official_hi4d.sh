#!/bin/bash

# Write Hi4D URL
read -p "Enter your personal Hi4D download URL: " URL

# Set target directory
TARGET_DIR="data/Hi4D/data"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

BASE_URL="https://hi4d.ait.ethz.ch"

echo "Fetching page content from $URL..."
html=$(curl -s "$URL")

echo "Extracting .tar.gz links..."
echo "$html" | grep -oP 'href="\K[^"]+\.tar\.gz' > hrefs.txt

if [ ! -s hrefs.txt ]; then
  echo "No .tar.gz links found. Please check your URL or access permissions."
  exit 1
fi

echo "Downloading files into $TARGET_DIR..."
while read -r href; do
  filename=$(basename "$href")
  full_url="$BASE_URL/$href"
  echo "Downloading $filename..."
  wget -c "$full_url" -O "$filename"
done < hrefs.txt

echo "Done. Files downloaded to $TARGET_DIR"