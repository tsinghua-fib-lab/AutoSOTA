# -*- coding: utf-8 -*-
"""
Download all datasets used in UltraLIF experiments.

Dataset sizes (approximate):
    Static:        MNIST ~11MB, FashionMNIST ~30MB, CIFAR-10 ~170MB
    Neuromorphic:  N-MNIST ~1GB, DVS-Gesture ~4GB, DVS-CIFAR10 ~6GB,
                   SHD ~500MB, SSC ~5GB

Usage:
    python scripts/download_datasets.py                # download all
    python scripts/download_datasets.py --static       # static datasets only
    python scripts/download_datasets.py --neuromorphic # neuromorphic only
"""

import argparse
from pathlib import Path


def download_static(data_dir: Path) -> None:
    from torchvision import datasets
    print("=== Static Datasets ===")

    print("MNIST (~11MB)...")
    datasets.MNIST(data_dir, train=True, download=True)
    datasets.MNIST(data_dir, train=False, download=True)

    print("FashionMNIST (~30MB)...")
    datasets.FashionMNIST(data_dir, train=True, download=True)
    datasets.FashionMNIST(data_dir, train=False, download=True)

    print("CIFAR-10 (~170MB)...")
    datasets.CIFAR10(data_dir, train=True, download=True)
    datasets.CIFAR10(data_dir, train=False, download=True)

    print("Static datasets: done.")


def download_neuromorphic(data_dir: Path) -> None:
    try:
        import tonic
    except ImportError:
        print("ERROR: tonic not installed. Run: pip install tonic")
        return

    print("\n=== Neuromorphic Datasets ===")

    print("N-MNIST (~1GB)...")
    tonic.datasets.NMNIST(save_to=str(data_dir), train=True)
    tonic.datasets.NMNIST(save_to=str(data_dir), train=False)

    print("DVS-Gesture (~4GB)...")
    tonic.datasets.DVSGesture(save_to=str(data_dir), train=True)
    tonic.datasets.DVSGesture(save_to=str(data_dir), train=False)

    print("DVS-CIFAR10 (~6GB)...")
    tonic.datasets.CIFAR10DVS(save_to=str(data_dir))

    print("SHD (~500MB)...")
    tonic.datasets.SHD(save_to=str(data_dir), train=True)
    tonic.datasets.SHD(save_to=str(data_dir), train=False)

    print("SSC (~5GB)...")
    tonic.datasets.SSC(save_to=str(data_dir), split="train")
    tonic.datasets.SSC(save_to=str(data_dir), split="test")

    print("Neuromorphic datasets: done.")


def main():
    parser = argparse.ArgumentParser(description="Download UltraLIF datasets")
    parser.add_argument("--data-dir", default="data", help="Download directory (default: data/)")
    parser.add_argument("--static", action="store_true", help="Download static datasets only")
    parser.add_argument("--neuromorphic", action="store_true", help="Download neuromorphic datasets only")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    if args.static:
        download_static(data_dir)
    elif args.neuromorphic:
        download_neuromorphic(data_dir)
    else:
        download_static(data_dir)
        download_neuromorphic(data_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
