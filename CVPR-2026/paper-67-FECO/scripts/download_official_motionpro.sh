#!/bin/bash

# Set save directory
SAVE_DIR="data/MotionPro/data"
mkdir -p "$SAVE_DIR"

echo "Downloading MotionPro..."

# Go to https://shenqiu.njucite.cn/download and get download link for MotionPRO_1.zip, MotionPRO_2.zip, MotionPRO_3.zip, MotionPRO_4.zip, MotionPRO_5.zip, MotionPRO_6.zip, MotionPRO_7.zip

# MotionPRO_1 with Pressure data, RGB data, SMPL parameters
wget -c https://box.nju.edu.cn/seafhttp/files/fd727214-e1c6-4de6-8352-7ec3cf57d590/MotionPRO_1.zip -P "$SAVE_DIR"
unzip $SAVE_DIR/MotionPRO_1.zip -d $SAVE_DIR

# MotionPRO_2 with Pressure data, RGB data, SMPL parameters
wget -c https://box.nju.edu.cn/seafhttp/files/5b3aa96a-2e17-4b89-9d98-49a3d8b845c1/MotionPRO_2.zip -P "$SAVE_DIR"
unzip $SAVE_DIR/MotionPRO_2.zip -d $SAVE_DIR

# MotionPRO_3 with Pressure data, RGB data, SMPL parameters
wget -c https://box.nju.edu.cn/seafhttp/files/b224936f-6c4e-4cbc-8d35-f36677f7febf/MotionPRO_3.zip -P "$SAVE_DIR"
unzip $SAVE_DIR/MotionPRO_3.zip -d $SAVE_DIR

# MotionPRO_4 with Pressure data, RGB data, SMPL parameters
wget -c https://box.nju.edu.cn/seafhttp/files/4c04e84e-ecc8-49ad-a884-952b309373df/MotionPRO_4.zip -P "$SAVE_DIR"
unzip $SAVE_DIR/MotionPRO_4.zip -d $SAVE_DIR

# MotionPRO_5 with Pressure data, RGB data, SMPL parameters
wget -c https://box.nju.edu.cn/seafhttp/files/5e57db2a-99d9-468a-a127-123914dc7c51/MotionPRO_5.zip -P "$SAVE_DIR"
unzip $SAVE_DIR/MotionPRO_5.zip -d $SAVE_DIR

# MotionPRO_6 with Pressure data, RGB data, SMPL parameters
wget -c https://box.nju.edu.cn/seafhttp/files/30f8e503-4b2d-420f-b0ef-0a3b4af26252/MotionPRO_6.zip -P "$SAVE_DIR"
unzip $SAVE_DIR/MotionPRO_6.zip -d $SAVE_DIR

# MotionPRO_7 with Pressure data, RGB data, SMPL parameters
wget -c https://box.nju.edu.cn/seafhttp/files/f2b3bb36-112f-4327-aaac-7a325491606f/MotionPRO_7.zip -P "$SAVE_DIR"
unzip $SAVE_DIR/MotionPRO_7.zip -d $SAVE_DIR

echo "All downloads completed in: $SAVE_DIR"