#!/usr/bin/env python3
"""Set up ImageNet dataset for CSR evaluation.

Creates the CSR-expected directory structure:
  {data_root}/General/ImageNet/images/  (flat symlinks)
  {data_root}/General/ImageNet/labels.csv
"""

import os
import sys
import csv
import json
import urllib.request
from torchvision.models import ResNet50_Weights

DATA_ROOT = "/repo/data"
SOURCE_VAL = "/datasets/imagenet/images/val"
IMAGENET_DIR = os.path.join(DATA_ROOT, "General/ImageNet")
IMAGES_DIR = os.path.join(IMAGENET_DIR, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

# Get synset directories from source
if not os.path.isdir(SOURCE_VAL):
    print(f"ERROR: Source val not found at {SOURCE_VAL}")
    sys.exit(1)

synsets = sorted(os.listdir(SOURCE_VAL))
print(f"Found {len(synsets)} synset directories")

# Get class names from ResNet50 (standard ILSVRC2012 order 0-999)
cat_names = ResNet50_Weights.DEFAULT.meta["categories"]
print(f"ResNet50 categories: {len(cat_names)}")

# Try to download the standard imagenet class index mapping
# Format: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
synset_to_name = {}
try:
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with urllib.request.urlopen(url, timeout=15) as response:
        class_idx_data = json.load(response)
    for idx_str, (synset, name) in class_idx_data.items():
        synset_to_name[synset] = name
    print(f"Downloaded mapping: {len(synset_to_name)} entries")
except Exception as e:
    print(f"Cannot download mapping: {e}")
    print("Using fallback: alphabetical synset ordering")
    # Fallback: map synset to class name using sorted order
    # WARNING: Alphabetical order may NOT match the standard class index order
    for i, synset in enumerate(synsets):
        if i < len(cat_names):
            synset_to_name[synset] = cat_names[i]
    print(f"Created fallback mapping: {len(synset_to_name)} entries")

# Create symlinks and labels.csv
labels_data = []
symlink_count = 0
max_per_class = 50  # limit images per class

for synset in synsets:
    class_name = synset_to_name.get(synset)
    if class_name is None:
        print(f"WARNING: No class name for synset {synset}, skipping")
        continue

    source_dir = os.path.join(SOURCE_VAL, synset)
    if not os.path.isdir(source_dir):
        continue

    img_files = sorted([f for f in os.listdir(source_dir)
                        if f.lower().endswith((".jpeg", ".jpg", ".png"))])
    img_files = img_files[:max_per_class]

    for img_file in img_files:
        src = os.path.join(source_dir, img_file)
        dest_name = f"{synset}_{img_file}"
        dest = os.path.join(IMAGES_DIR, dest_name)

        if not os.path.exists(dest):
            try:
                os.symlink(src, dest)
                symlink_count += 1
            except OSError:
                pass

        labels_data.append([dest_name, class_name])

# Save labels.csv
csv_path = os.path.join(IMAGENET_DIR, "labels.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(labels_data)

print(f"Created {symlink_count} symlinks")
print(f"Saved labels.csv with {len(labels_data)} entries")
print(f"First 5 entries: {labels_data[:5]}")
print("Done!")
