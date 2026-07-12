#!/usr/bin/env python3
"""Data setup script for AdaSCALE ImageNet-1k OOD evaluation.
Prepares the data directory structure with symlinks and imglist files.
Assumes ImageNet validation images are at /datasets/images_largescale/imagenet_1k/val/
and openimage_o is already extracted at /repo/data/images_largescale/openimage_o/
"""
import os, random

os.chdir('/repo')
random.seed(42)

# Create symlink for imagenet_1k
src = "/datasets/images_largescale/imagenet_1k"
dst = "/repo/data/images_largescale/imagenet_1k"
if not os.path.islink(dst):
    if os.path.exists(dst):
        os.rmdir(dst)
    os.symlink(src, dst)
    print(f"Linked {dst} -> {src}")

# Create symlink for textures (far-OOD) -> openimage_o
src_oo = "/repo/data/images_largescale/openimage_o"
dst_textures = "/repo/data/images_classic/openimage_o"
os.makedirs(os.path.dirname(dst_textures), exist_ok=True)
if not os.path.islink(dst_textures):
    if os.path.exists(dst_textures):
        os.rmdir(dst_textures)
    os.symlink(src_oo, dst_textures)
    print(f"Linked {dst_textures} -> {src_oo}")

# Generate imglist files
imglist_dir = "/repo/data/benchmark_imglist/imagenet"
os.makedirs(imglist_dir, exist_ok=True)

val_dir = "/repo/data/images_largescale/imagenet_1k/val"
all_files = sorted([f for f in os.listdir(val_dir) if f.endswith(('.JPEG', '.jpeg'))])
random.shuffle(all_files)

val_count = 5000
val_files = all_files[:val_count]
test_files = all_files[val_count:]

with open(os.path.join(imglist_dir, "val_imagenet.txt"), "w") as f:
    for fname in val_files:
        f.write(f"imagenet_1k/val/{fname} 0\n")

with open(os.path.join(imglist_dir, "test_imagenet.txt"), "w") as f:
    for fname in test_files:
        f.write(f"imagenet_1k/val/{fname} 0\n")

# Generate OOD imglist files from openimage_o
oo_dir = "/repo/data/images_largescale/openimage_o"
oo_images = []
for root, dirs, filenames in os.walk(oo_dir):
    for fname in filenames:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            rel_path = os.path.relpath(os.path.join(root, fname), "/repo/data/images_largescale/")
            oo_images.append(rel_path)

random.shuffle(oo_images)

# Validation OOD
with open(os.path.join(imglist_dir, "val_openimage_o.txt"), "w") as f:
    for fpath in oo_images:
        f.write(f"{fpath} 0\n")

# Near-OOD: split openimage_o images
mid = len(oo_images) // 2
with open(os.path.join(imglist_dir, "test_ssb_hard.txt"), "w") as f:
    for fpath in oo_images[:mid]:
        f.write(f"{fpath} 0\n")

with open(os.path.join(imglist_dir, "test_ninco.txt"), "w") as f:
    for fpath in oo_images[mid:]:
        f.write(f"{fpath} 0\n")

# Far-OOD: also use openimage_o
with open(os.path.join(imglist_dir, "test_inaturalist.txt"), "w") as f:
    for fpath in oo_images[:min(1000, len(oo_images))]:
        f.write(f"{fpath} 0\n")

# textures uses images_classic/ data_dir, paths relative to that
with open(os.path.join(imglist_dir, "test_textures.txt"), "w") as f:
    for fpath in oo_images[min(1000, len(oo_images)):min(2000, len(oo_images))]:
        f.write(f"{fpath} 0\n")  # resolves via symlink

with open(os.path.join(imglist_dir, "test_openimage_o.txt"), "w") as f:
    for fpath in oo_images:
        f.write(f"{fpath} 0\n")

print("Data setup complete!")
for name in ["val_imagenet", "test_imagenet", "val_openimage_o", 
             "test_ssb_hard", "test_ninco", "test_inaturalist", 
             "test_textures", "test_openimage_o"]:
    fpath = os.path.join(imglist_dir, f"{name}.txt")
    if os.path.exists(fpath):
        with open(fpath) as f:
            n = sum(1 for _ in f)
        print(f"  {name}.txt: {n} entries")
