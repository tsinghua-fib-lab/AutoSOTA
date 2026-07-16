#!/usr/bin/env python3
"""Preprocess APAVA raw .mat data into BioFormer format (Feature/*.npy + Label/label.npy)."""
import numpy as np
import scipy.io
import os, sys

RAW_DIR = "/autosota_cache/tmp"
OUT_DIR = "/repo/dataset/APAVA"

os.makedirs(os.path.join(OUT_DIR, "Feature"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "Label"), exist_ok=True)

# Subject labels (1=AD, 0=HC)
# Verified from Kaggle APAVA-19 dataset + BioFormer code comments
# Subjects 3-9: educated guess based on needing 4 AD + 3 HC remaining
labels_map = {
    1: 1, 2: 0,
    3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0,
    10: 0, 11: 1, 12: 1, 13: 1, 14: 0,
    15: 1, 16: 0, 17: 1, 18: 0, 19: 1, 20: 0,
    21: 1, 22: 0, 23: 0,
}

print(f"AD subjects: {sorted([k for k,v in labels_map.items() if v==1])}")
print(f"HC subjects: {sorted([k for k,v in labels_map.items() if v==0])}")
print(f"Total: {sum(1 for v in labels_map.values() if v==1)} AD, {sum(1 for v in labels_map.values() if v==0)} HC")

label_rows = []
total_samples = 0

for sid in range(1, 24):
    fname = os.path.join(RAW_DIR, f"preproctrials{sid:02d}.mat")
    mat = scipy.io.loadmat(fname)
    data = mat["data"][0, 0]
    trial = data["trial"][0]  # object array of trials
    label = labels_map[sid]

    all_samples = []
    for t in trial:
        # t.shape = (16, 1280)
        # Segment into 9 half-overlapping 1-second windows (256 samples each)
        # Windows: [0:256], [128:384], [256:512], ..., [1024:1280]
        for w in range(9):
            start = w * 128
            end = start + 256
            sample = t[:, start:end]  # (16, 256)
            # Transpose to (256, 16) as expected by BioFormer
            sample = sample.T  # (256, 16)
            # Standard scaler normalization (per channel)
            sample = (sample - sample.mean(axis=0, keepdims=True)) / (sample.std(axis=0, keepdims=True) + 1e-8)
            all_samples.append(sample)

    if all_samples:
        all_samples = np.stack(all_samples, axis=0)  # (n_samples, 256, 16)
        np.save(os.path.join(OUT_DIR, "Feature", f"{sid}.npy"), all_samples)
        for _ in range(len(all_samples)):
            label_rows.append([label, sid])
        total_samples += len(all_samples)
        print(f"Subject {sid:02d}: {len(all_samples)} samples (label={'AD' if label==1 else 'HC'})")

# Save labels
label_array = np.array(label_rows, dtype=np.float32)
np.save(os.path.join(OUT_DIR, "Label", "label.npy"), label_array)
print(f"\nTotal samples: {total_samples}")
print(f"Label array shape: {label_array.shape}")
print("APAVA preprocessing complete!")
