#!/usr/bin/env python3
"""Precompute CREMA-D spectrograms and image tensors to speed up training."""
import os
import sys
import csv
import numpy as np
import torch
import librosa
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import multiprocessing as mp

DATA_ROOT = os.environ.get("CREMAD_DATA_ROOT", "/datasets/CREMA-D")
CACHE_DIR = os.path.join(DATA_ROOT, "precomputed")
os.makedirs(CACHE_DIR, exist_ok=True)

# Same transforms as the dataset class
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def compute_spectrogram(audio_path):
    """Compute log-magnitude spectrogram from audio file."""
    samples, rate = librosa.load(audio_path, sr=22050)
    resamples = np.tile(samples, 3)[:22050 * 3]
    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
    spectrogram = np.log(np.abs(spectrogram) + 1e-7)
    return spectrogram.astype(np.float32)

def process_file(args):
    """Process a single file: compute spectrogram and load first frame tensor."""
    name, audio_path, visual_path = args
    try:
        # Audio: always same
        spec = compute_spectrogram(audio_path)
        np.save(os.path.join(CACHE_DIR, f"{name}_audio.npy"), spec)

        # Visual: first frame only (dataset randomly picks one)
        image_samples = os.listdir(visual_path)
        if len(image_samples) == 0:
            return name, False, "no images"
        img_path = os.path.join(visual_path, image_samples[0])
        img = Image.open(img_path).convert("RGB")

        # Save as tensor
        img_tensor = TEST_TRANSFORM(img)  # Use test transform for precomputation
        torch.save(img_tensor, os.path.join(CACHE_DIR, f"{name}_visual.pt"))

        return name, True, None
    except Exception as e:
        return name, False, str(e)

def main():
    # Collect all files from train.csv and test.csv
    all_files = []
    for csv_file in ["train.csv", "test.csv"]:
        path = os.path.join(DATA_ROOT, csv_file)
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                name = row[0]
                audio_path = os.path.join(DATA_ROOT, "AudioWAV", name + ".wav")
                visual_path = os.path.join(DATA_ROOT, "Image-01-FPS", name)
                all_files.append((name, audio_path, visual_path))

    print(f"Processing {len(all_files)} files...")

    # Check which are already done
    todo = []
    for name, audio_path, visual_path in all_files:
        spec_file = os.path.join(CACHE_DIR, f"{name}_audio.npy")
        img_file = os.path.join(CACHE_DIR, f"{name}_visual.pt")
        if not os.path.exists(spec_file) or not os.path.exists(img_file):
            todo.append((name, audio_path, visual_path))

    print(f"Already done: {len(all_files) - len(todo)}, Remaining: {len(todo)}")

    if not todo:
        print("All files already precomputed!")
        return

    with mp.Pool(min(8, mp.cpu_count())) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, todo), total=len(todo)))

    failed = [r for r in results if not r[1]]
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed[:10]:
            print(f"  {r[0]}: {r[2]}")
    else:
        print("All precomputed successfully!")

if __name__ == "__main__":
    main()
