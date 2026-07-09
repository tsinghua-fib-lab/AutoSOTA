"""
download_data.py — Download all datasets supported by Grad-Mimic.

Usage
-----
python scripts/download_data.py [--data_dir ./data] [--datasets cifar10 cifar100 ...]

Tiny ImageNet must be downloaded manually (see instructions below).
"""

import argparse
import os

from torchvision import datasets

SUPPORTED = ["cifar10", "cifar100", "dtd", "stl10", "flower102", "pet"]

TINY_IMAGENET_INSTRUCTIONS = """
Tiny ImageNet must be downloaded manually:

  1. Download from https://image-net.org/download-images (account required), or
     use the Kaggle mirror: https://www.kaggle.com/c/tiny-imagenet
  2. Extract so that the layout matches:

       {data_dir}/tiny-imagenet-200/
           train/<wnid>/images/*.JPEG
           val/images/*.JPEG
           val/val_annotations.txt
           wnids.txt
"""


def download(data_dir, dataset_names):
    os.makedirs(data_dir, exist_ok=True)

    for name in dataset_names:
        print(f"\n--- Downloading {name} ---")

        if name == "cifar10":
            datasets.CIFAR10(root=data_dir, train=True,  download=True)
            datasets.CIFAR10(root=data_dir, train=False, download=True)

        elif name == "cifar100":
            datasets.CIFAR100(root=data_dir, train=True,  download=True)
            datasets.CIFAR100(root=data_dir, train=False, download=True)

        elif name == "dtd":
            datasets.DTD(root=data_dir, split="train", download=True)
            datasets.DTD(root=data_dir, split="test",  download=True)

        elif name == "stl10":
            datasets.STL10(root=data_dir, split="train", download=True)
            datasets.STL10(root=data_dir, split="test",  download=True)

        elif name == "flower102":
            datasets.Flowers102(root=data_dir, split="train", download=True)
            datasets.Flowers102(root=data_dir, split="test",  download=True)

        elif name == "pet":
            datasets.OxfordIIITPet(root=data_dir, split="trainval", target_types="category", download=True)
            datasets.OxfordIIITPet(root=data_dir, split="test",     target_types="category", download=True)

        else:
            print(f"  Skipping unknown dataset: {name!r}")
            continue

        print(f"  Done: {name}")


def main():
    parser = argparse.ArgumentParser(description="Download Grad-Mimic datasets.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store downloaded data.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=SUPPORTED,
        choices=SUPPORTED,
        help="Datasets to download (default: all supported).",
    )
    args = parser.parse_args()

    print(f"Downloading to: {os.path.abspath(args.data_dir)}")
    download(args.data_dir, args.datasets)
    print("\nAll downloads complete.")
    print(TINY_IMAGENET_INSTRUCTIONS.format(data_dir=args.data_dir))


if __name__ == "__main__":
    main()
