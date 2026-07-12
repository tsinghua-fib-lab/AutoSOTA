#!/usr/bin/env python3
"""
Prepare demo data from an unzipped BackdoorBench folder.

Downloads CIFAR-10 test set via torchvision, splits it class-balanced into
trusted (2500), sampling (2500), and test (5000) subsets, then pairs test
images with their backdoor counterparts from BackdoorBench.

Usage:
    python prepare_data.py --bbench_dir unzipped/
    python prepare_data.py --bbench_dir unzipped/ --output_dir data/ --seed 0
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torchvision.datasets as tv_datasets


def save_image_to_class_folder(img, label, idx, dest_dir):
    """Save a PIL image as <dest_dir>/<label>/<idx>.png."""
    class_dir = dest_dir / str(label)
    class_dir.mkdir(parents=True, exist_ok=True)
    img.save(class_dir / f"{idx}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare demo data from unzipped BackdoorBench folder")
    parser.add_argument("--bbench_dir", type=str, required=True,
                        help="Path to unzipped BackdoorBench folder "
                             "(contains attack_result.pt, bd_test_dataset/)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for split (default: 0)")
    parser.add_argument("--cifar_download_dir", type=str, default=None,
                        help="Directory to download CIFAR-10 to "
                             "(default: ./cifar10_raw)")
    args = parser.parse_args()

    bbench_dir = Path(args.bbench_dir)
    output_dir = Path(args.output_dir)
    cifar_dir = Path(args.cifar_download_dir) if args.cifar_download_dir \
        else Path("cifar10_raw")

    rng = np.random.RandomState(args.seed)

    # Validate BackdoorBench folder
    ckpt_path = bbench_dir / "attack_result.pt"
    bd_test_path = bbench_dir / "bd_test_dataset"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not bd_test_path.exists():
        raise FileNotFoundError(f"bd_test_dataset not found: {bd_test_path}")

    # Download CIFAR-10 test set
    print("Downloading CIFAR-10 test set ...")
    cifar_test = tv_datasets.CIFAR10(
        root=str(cifar_dir), train=False, download=True)
    print(f"  {len(cifar_test)} images loaded")

    # Group indices by class
    n_classes = 10
    class_indices = {c: [] for c in range(n_classes)}
    for idx in range(len(cifar_test)):
        _, label = cifar_test[idx]
        class_indices[label].append(idx)

    for c in range(n_classes):
        assert len(class_indices[c]) == 1000, \
            f"Expected 1000 images for class {c}, got {len(class_indices[c])}"

    # Class-balanced split: 250 trusted + 250 sampling + 500 test per class
    trusted_indices = []
    sampling_indices = []
    test_indices = []

    for c in range(n_classes):
        idxs = np.array(class_indices[c])
        rng.shuffle(idxs)
        trusted_indices.extend(idxs[:250].tolist())
        sampling_indices.extend(idxs[250:500].tolist())
        test_indices.extend(idxs[500:].tolist())

    print(f"\nSplit (seed={args.seed}):")
    print(f"  Trusted:  {len(trusted_indices)}")
    print(f"  Sampling: {len(sampling_indices)}")
    print(f"  Test:     {len(test_indices)}")

    # Build lookup: CIFAR-10 test index -> path in bd_test_dataset
    bd_test_lookup = {}
    for class_dir in bd_test_path.iterdir():
        if not class_dir.is_dir():
            continue
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() == ".png":
                bd_test_lookup[int(img_file.stem)] = img_file

    test_with_bd = [idx for idx in test_indices if idx in bd_test_lookup]
    test_without_bd = [idx for idx in test_indices if idx not in bd_test_lookup]
    print(f"  Test with backdoor counterpart: {len(test_with_bd)}")
    print(f"  Test without (target class): {len(test_without_bd)}")

    # Clean output directory
    if output_dir.exists():
        print(f"\nRemoving existing {output_dir}/ ...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # 1. Copy checkpoint
    print("\n[1/5] Copying attack_result.pt ...")
    shutil.copy2(ckpt_path, output_dir / "attack_result.pt")

    # 2. Save trusted/
    print("[2/5] Saving trusted/ ...")
    trusted_dir = output_dir / "trusted"
    for idx in trusted_indices:
        img, label = cifar_test[idx]
        save_image_to_class_folder(img, label, idx, trusted_dir)

    # 3. Save sampling/
    print("[3/5] Saving sampling/ ...")
    sampling_dir = output_dir / "sampling"
    for idx in sampling_indices:
        img, label = cifar_test[idx]
        save_image_to_class_folder(img, label, idx, sampling_dir)

    # 4. Save clean_test/
    print("[4/5] Saving clean_test/ ...")
    clean_test_dir = output_dir / "clean_test"
    for idx in test_indices:
        img, label = cifar_test[idx]
        save_image_to_class_folder(img, label, idx, clean_test_dir)

    # 5. Save bd_test/ (copy backdoor counterparts for test indices)
    print("[5/5] Saving bd_test/ ...")
    bd_test_dir = output_dir / "bd_test"
    bd_copied = 0
    for idx in test_indices:
        if idx not in bd_test_lookup:
            continue
        src_file = bd_test_lookup[idx]
        dest_class_dir = bd_test_dir / src_file.parent.name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_class_dir / f"{idx}.png")
        bd_copied += 1

    print(f"  Copied {bd_copied} backdoor test images")

    # Summary
    print(f"\nDone. Contents of {output_dir}/:")
    print(f"  attack_result.pt: {(output_dir / 'attack_result.pt').stat().st_size / 1e6:.1f} MB")
    for folder_name in ["trusted", "sampling", "clean_test", "bd_test"]:
        folder = output_dir / folder_name
        count = sum(1 for _ in folder.rglob("*.png"))
        print(f"  {folder_name}/: {count} images")


if __name__ == "__main__":
    main()
