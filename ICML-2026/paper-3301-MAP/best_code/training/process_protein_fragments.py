#!/usr/bin/env python3
"""
Process SidechainNet protein coordinates into backbone fragments and save them to disk.

Usage:
  python training/process_protein_fragments.py --name casp12 --fragment-length 10 --max-data-length 20000

The script will create the output directory (default: `data/protein/`) and write:
  - a compressed NumPy archive containing `fragments` with shape `(N, L, 3, 3)`;
  - a sidecar JSON manifest recording SidechainNet/package provenance and hashes.

SidechainNet imports OpenMM at module import time, so both packages must be
installed before running this script.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(package_name):
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Extract backbone fragments from SidechainNet and save to disk")
    p.add_argument("--name", type=str, default="casp12", help="SidechainNet dataset name (e.g. casp12)")
    p.add_argument("--fragment-length", type=int, default=10, help="Residue length of fragments to extract")
    p.add_argument("--max-data-length", type=int, default=20000, help="Maximum number of fragments to extract")
    p.add_argument("--out-dir", type=str, default="data/protein", help="Directory to save processed fragments")
    p.add_argument("--out-fname", type=str, default=None, help="Output filename (overrides default)")
    p.add_argument("--manifest-path", type=str, default=None, help="Output manifest filename/path (default: archive stem + .manifest.json)")
    return p.parse_args()


def main():
    args = parse_args()

    # Import sidechainnet and the helper from the repo
    try:
        import sidechainnet as scn
    except ModuleNotFoundError as e:
        if getattr(e, "name", None) == "openmm":
            print(
                "Failed to import sidechainnet because dependency 'openmm' is missing. "
                "Install project requirements or run: python -m pip install openmm==8.5.1"
            )
        else:
            print(
                "Failed to import sidechainnet or one of its dependencies. "
                "Install project requirements, including sidechainnet==1.0.1 and openmm==8.5.1."
            )
        raise
    except Exception:
        print(
            "Failed to import sidechainnet. Install project requirements, including "
            "sidechainnet==1.0.1 and openmm==8.5.1."
        )
        raise

    # Import the extraction helper from the repository
    try:
        from datasets import extract_backbone_fragments
    except Exception as e:
        print("Failed to import extract_backbone_fragments from datasets.py. Make sure you're running from the project root.")
        raise

    print(f"Loading SidechainNet dataset '{args.name}' (this may download files the first time)...")
    data = scn.load(name=args.name, with_coordinates=True)
    print(f"Loaded {len(data)} proteins (SidechainNet container). Extracting fragments...")

    fragments = extract_backbone_fragments(data, fragment_length=args.fragment_length, max_data_length=args.max_data_length)
    fragments = np.asarray(fragments, dtype=np.float32)

    print(f"Extracted fragments array with shape: {getattr(fragments, 'shape', None)} and dtype={getattr(fragments, 'dtype', None)}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_fname is None:
        fname = f"{args.name}_fragments_L{args.fragment_length}_N{args.max_data_length}.npz"
    else:
        fname = args.out_fname

    out_path = os.path.join(args.out_dir, fname)
    # Save as compressed numpy archive
    np.savez_compressed(out_path, fragments=fragments)
    print(f"Saved fragments to: {out_path}")

    if args.manifest_path is None:
        manifest_path = os.path.splitext(out_path)[0] + ".manifest.json"
    elif os.path.isabs(args.manifest_path):
        manifest_path = args.manifest_path
    else:
        manifest_path = os.path.join(args.out_dir, args.manifest_path)

    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dataset_source": "sidechainnet",
        "sidechainnet_dataset": args.name,
        "sidechainnet_version": package_version("sidechainnet"),
        "sidechainnet_module": getattr(scn, "__file__", None),
        "openmm_version": package_version("openmm"),
        "with_coordinates": True,
        "fragment_length": args.fragment_length,
        "max_data_length": args.max_data_length,
        "fragments": {
            "path": out_path,
            "shape": list(fragments.shape),
            "dtype": str(fragments.dtype),
            "bytes": os.path.getsize(out_path),
            "sha256": sha256_file(out_path),
        },
        "notes": [
            "The .npz contains the exact processed fragments consumed by driver.py.",
            "Keep this manifest with the .npz file or publish both with checksums.",
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
