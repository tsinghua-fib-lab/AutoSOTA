"""
VRS audio extraction code for SAVVY-Bench data preprocessing - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""
import sys
import os
from projectaria_tools.core import data_provider
from scipy.io import wavfile
import numpy as np

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

def extract_audio_segment(vrs_file_path, output_dir, video_id, start_time, end_time):
    audio_dir = os.path.join(output_dir, video_id, 'audio')
    os.makedirs(audio_dir, exist_ok=True)

    provider = data_provider.create_vrs_data_provider(vrs_file_path)
    stream_id = provider.get_stream_id_from_label("mic")

    audio_channels = [[] for _ in range(7)]

    for index in range(provider.get_num_data(stream_id)):
        audio_data = provider.get_audio_data_by_index(stream_id, index)
        audio_signal_block = audio_data[0].data
        timestamps_ns = audio_data[1].capture_timestamps_ns

        for i, ts_ns in enumerate(timestamps_ns):
            ts_sec = ts_ns * 1e-9
            if start_time <= ts_sec <= end_time:
                sample_idx = i * 7
                for c in range(7):
                    if sample_idx + c < len(audio_signal_block):
                        audio_channels[c].append(audio_signal_block[sample_idx + c])

    channels_5_6 = np.column_stack([audio_channels[5], audio_channels[6]])
    output_path = os.path.join(audio_dir, f"{video_id}.wav")
    wavfile.write(output_path, 48000, channels_5_6.astype(np.int32))

vrs_base_dir = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

timestamps = load_timestamps('aea/video_timestamps.txt')

for video_id in timestamps.keys():
    source_id = get_source_vrs_id(video_id)
    vrs_file_path = os.path.join(vrs_base_dir, source_id, "recording.vrs")
    start, end = timestamps[video_id]
    extract_audio_segment(vrs_file_path, output_dir, video_id, start, end)
