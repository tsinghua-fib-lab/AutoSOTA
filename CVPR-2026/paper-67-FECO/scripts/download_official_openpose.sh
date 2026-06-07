#!/bin/bash

# Set save directory
SAVE_DIR="data/OpenPose/data"
mkdir -p "$SAVE_DIR"

# OpenPose foot keypoints annotations
echo "Downloading OpenPose foot keypoints annotations..."
wget -c https://raw.githubusercontent.com/Eva20150932/coco-foot-and-leg/main/person_keypoints_train2017_foot_v1.json -P "$SAVE_DIR"
wget -c https://raw.githubusercontent.com/Eva20150932/coco-foot-and-leg/main/person_keypoints_val2017_foot_v1.json -P "$SAVE_DIR"

# COCO train/val images
echo "Downloading COCO 2017 train/val images..."
wget -c http://images.cocodataset.org/zips/train2017.zip -P "$SAVE_DIR"
wget -c http://images.cocodataset.org/zips/val2017.zip -P "$SAVE_DIR"

echo "All downloads completed in: $SAVE_DIR"