#!/usr/bin/env python3
"""Setup CREMA-D dataset for MMIA pipeline.

Creates train.csv/test.csv and extracts video frames.
"""
import os
import sys
import csv
import random
import subprocess
import multiprocessing as mp
from tqdm import tqdm

SRC = "/autosota_cache/tmp/CREMA-D_download"
DST = os.environ.get("CREMAD_DATA_ROOT", "/datasets/CREMA-D")

EMOTIONS = {'NEU', 'HAP', 'SAD', 'FEA', 'DIS', 'ANG'}
SEED = 42
TRAIN_RATIO = 0.9
FPS = 1  # frames per second

def get_filename_and_label(wav_file):
    """Extract filename (no ext) and emotion label from WAV filename."""
    name = wav_file.replace('.wav', '')
    parts = name.split('_')
    if len(parts) >= 3 and parts[2] in EMOTIONS:
        return name, parts[2]
    return None, None

def create_csv_files():
    """Create train.csv and test.csv with 90/10 split."""
    audio_dir = os.path.join(SRC, "AudioWAV")
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    items = []
    for wf in wav_files:
        name, label = get_filename_and_label(wf)
        if name:
            items.append((name, label))

    random.seed(SEED)
    random.shuffle(items)

    split_idx = int(len(items) * TRAIN_RATIO)
    train_items = sorted(items[:split_idx])
    test_items = sorted(items[split_idx:])

    os.makedirs(DST, exist_ok=True)

    for fname, data in [("train.csv", train_items), ("test.csv", test_items)]:
        path = os.path.join(DST, fname)
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            for name, label in data:
                writer.writerow([name, label])
        print(f"Created {path}: {len(data)} entries")

def symlink_audio():
    """Symlink audio files to destination."""
    src_audio = os.path.join(SRC, "AudioWAV")
    dst_audio = os.path.join(DST, "AudioWAV")
    if os.path.exists(dst_audio):
        print(f"AudioWAV already exists at {dst_audio}")
        return
    os.symlink(src_audio, dst_audio)
    print(f"Symlinked {src_audio} -> {dst_audio}")

def extract_frame(args):
    """Extract 1 frame from a video at the middle timestamp."""
    video_path, output_dir = args
    os.makedirs(output_dir, exist_ok=True)

    # Get video duration
    probe_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
    except Exception:
        duration = 3.0  # default ~3 seconds

    # Extract 1 frame per second
    for sec in range(int(duration)):
        out_file = os.path.join(output_dir, f"frame_{sec:04d}.jpg")
        if os.path.exists(out_file):
            continue
        extract_cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-ss', str(sec),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            out_file
        ]
        subprocess.run(extract_cmd, capture_output=True, timeout=30)

def extract_all_frames():
    """Extract frames from all FLV videos."""
    video_dir = os.path.join(SRC, "VideoFlash")
    image_dir = os.path.join(DST, "Image-{:02d}-FPS".format(FPS))
    os.makedirs(image_dir, exist_ok=True)

    flv_files = [f for f in os.listdir(video_dir) if f.endswith('.flv')]

    tasks = []
    for flv in flv_files:
        name = flv.replace('.flv', '')
        video_path = os.path.join(video_dir, flv)
        output_dir = os.path.join(image_dir, name)

        # Check if already done
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            continue

        tasks.append((video_path, output_dir))

    if not tasks:
        print("All frames already extracted")
        return

    print(f"Extracting frames for {len(tasks)} videos (using {min(8, mp.cpu_count())} workers)...")

    with mp.Pool(min(8, mp.cpu_count())) as pool:
        list(tqdm(pool.imap_unordered(extract_frame, tasks), total=len(tasks)))

def verify():
    """Verify dataset integrity."""
    errors = []

    # Check CSV files
    for csv_file in ['train.csv', 'test.csv']:
        path = os.path.join(DST, csv_file)
        if not os.path.exists(path):
            errors.append(f"Missing {csv_file}")
            continue

        with open(path, encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if len(row) != 2:
                    errors.append(f"{csv_file}:{i} bad row: {row}")
                    continue
                name, label = row

                # Check audio
                audio_path = os.path.join(DST, "AudioWAV", name + '.wav')
                if not os.path.exists(audio_path):
                    errors.append(f"Missing audio: {audio_path}")

                # Check frames
                img_dir = os.path.join(DST, f"Image-{FPS:02d}-FPS", name)
                if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
                    errors.append(f"Missing frames: {img_dir}")

    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} errors")
        for e in errors[:20]:
            print(f"  {e}")
    else:
        print("VERIFICATION PASSED: Dataset is ready")

    return len(errors) == 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['csv', 'audio', 'frames', 'verify', 'all'], default='all')
    args = parser.parse_args()

    if args.step in ('csv', 'all'):
        create_csv_files()
    if args.step in ('audio', 'all'):
        symlink_audio()
    if args.step in ('frames', 'all'):
        extract_all_frames()
    if args.step in ('verify', 'all'):
        verify()
