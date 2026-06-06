"""
VRS MP4 video extraction code for SAVVY-Bench data preprocessing - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""
import sys
import os
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

def load_timestamps(timestamp_file):
    timestamps = {}
    with open(timestamp_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                timestamps[parts[0]] = (float(parts[1]), float(parts[2]))
    return timestamps

def get_source_vrs_id(video_id):
    for seq_num in ['31', '32', '33', '34']:
        if f'seq{seq_num}' in video_id:
            return video_id.replace(f'seq{seq_num}', 'seq3')
    return video_id

def convert_frames_to_video(source_video_id, output_video_id, start_time, end_time,
                            aea_processed_dir, output_dir, fps):
    source_dir = Path(aea_processed_dir) / source_video_id
    transforms_file = source_dir / "transforms.json"

    with open(transforms_file, 'r') as f:
        transforms = json.load(f)

    filtered_frames = [f for f in transforms['frames']
                        if start_time <= f.get('timestamp', 0) <= end_time]

    temp_dir = Path(f"/tmp/{output_video_id}_frames")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(filtered_frames):
        src_path = source_dir / frame['image_path'].lstrip('./')
        dst_path = temp_dir / f"frame_{i:06d}.jpg"
        image = np.array(Image.open(src_path).rotate(270))
        Image.fromarray(image).save(dst_path)
    video_output_dir = Path(output_dir) / output_video_id / 'video'
    video_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = video_output_dir / f"{output_video_id}.mp4"

    subprocess.run([
        'python', 'egolifter/scripts/frames_to_mp4.py',
        str(temp_dir), str(output_video_path), '--fps', str(fps)
    ], check=True)

    shutil.rmtree(temp_dir)

aea_processed_dir = sys.argv[1]
output_dir = sys.argv[2]
fps = 20

os.makedirs(output_dir, exist_ok=True)

timestamps = load_timestamps('aea/video_timestamps.txt')

for video_id in timestamps.keys():
    source_id = get_source_vrs_id(video_id)
    start, end = timestamps[video_id]
    start_ns, end_ns = int(start * 1e9), int(end * 1e9)
    convert_frames_to_video(source_id, video_id, start_ns, end_ns,
                           aea_processed_dir, output_dir, fps)
