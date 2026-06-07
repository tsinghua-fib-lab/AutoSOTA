#!/bin/bash

# Set the save directory
save_dir="data/UTZap50K/data"
mkdir -p "$save_dir"

# Download files
wget https://vision.cs.utexas.edu/projects/finegrained/utzap50k/readme.txt -O "$save_dir/readme.txt"
wget https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-data.zip -O "$save_dir/ut-zap50k-data.zip"
wget https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-feats.zip -O "$save_dir/ut-zap50k-feats.zip"
wget https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-lexi.zip -O "$save_dir/ut-zap50k-lexi.zip"
wget https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip -O "$save_dir/ut-zap50k-images.zip"
wget https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip -O "$save_dir/ut-zap50k-images-square.zip"

echo "All files downloaded to $save_dir"