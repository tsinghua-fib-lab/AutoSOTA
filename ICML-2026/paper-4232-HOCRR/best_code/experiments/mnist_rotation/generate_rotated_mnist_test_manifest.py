#!/usr/bin/env python3
"""
Generate a reusable rotated-MNIST test-set manifest for certification experiments.

Why:
- Avoid regenerating rotated images/angles in every pipeline run
- Ensure all methods (bounded certifiers, alpha-trimming, pseudo-radius, soundness)
  use EXACTLY the same rotated inputs and ground-truth angles

Input:
- A JSON file that contains the chosen test indices, e.g.
  - mnist_rotation_full_cert_rotated_n100_sigma*.json (estimation outputs)
  - alpha-trimming outputs that include `test_indices`

Output:
- <output_prefix>.json   (metadata, indices, angles, labels, constants)
- <output_prefix>.npz    (images_raw: float32 [N,28,28] in [0,1])

Rotation determinism:
- Matches experiments/mnist_rotation/dataset_generator.py:
  seed=42, rotation_range=(0,360), augmentation_factor=1,
  background_color=33, expand=True, resize=(28,28) bilinear
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_source_indices(source_json: str) -> List[int]:
    """Extract chosen test indices from a results JSON."""
    with open(source_json, "r") as f:
        data = json.load(f)

    # Preferred: per-sample indices
    samples = data.get("samples", [])
    if samples:
        idxs = []
        for s in samples:
            idx = s.get("test_dataset_idx", s.get("image_idx", s.get("sample_idx", None)))
            if idx is not None:
                idxs.append(int(idx))
        if idxs:
            # Keep order, but de-dup if needed
            seen = set()
            ordered = []
            for i in idxs:
                if i not in seen:
                    ordered.append(i)
                    seen.add(i)
            return ordered

    # Fallback: explicit list fields
    for key in ("selected_test_indices", "test_indices", "selected_indices"):
        if key in data and isinstance(data[key], list) and data[key]:
            return [int(x) for x in data[key]]

    raise ValueError(f"Could not find test indices in {source_json}")


def rotate_image_like_dataset_generator(image_raw: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 28x28 raw [0,1] image like MNISTRotationDataset."""
    background_color = 33  # same as dataset_generator.py
    pil_img = Image.fromarray((np.clip(image_raw, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    rotated = pil_img.rotate(angle_deg, fillcolor=background_color, expand=True)
    rotated = rotated.resize((28, 28), Image.BILINEAR)
    out = np.asarray(rotated).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def build_rotation_angles(max_idx: int, seed: int, rotation_range: Tuple[float, float]) -> List[float]:
    """Deterministically generate angles[0..max_idx] matching dataset_generator.py for augmentation_factor=1."""
    rng = random.Random(seed)
    angles = []
    for _ in range(max_idx + 1):
        angles.append(rng.uniform(*rotation_range))
    return angles


def main():
    parser = argparse.ArgumentParser(description="Generate rotated MNIST test-set manifest")
    parser.add_argument("--source_json", type=str, required=True, help="Source JSON containing chosen test indices")
    parser.add_argument("--output_prefix", type=str, required=True, help="Output prefix (without extension)")
    parser.add_argument("--seed", type=int, default=42, help="Rotation seed (must match other pipelines)")
    parser.add_argument("--rotation_min_deg", type=float, default=0.0, help="Rotation min degrees (default 0)")
    parser.add_argument("--rotation_max_deg", type=float, default=360.0, help="Rotation max degrees (default 360)")
    args = parser.parse_args()

    source_json = args.source_json
    output_prefix = args.output_prefix
    rotation_range = (args.rotation_min_deg, args.rotation_max_deg)

    idxs = load_source_indices(source_json)
    if not idxs:
        raise ValueError("No indices found")

    max_idx = max(idxs)
    print(f"Found {len(idxs)} indices (max idx={max_idx}) from {source_json}")

    # Load original MNIST test set (raw [0,1])
    import torchvision
    from torchvision import transforms

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    if max_idx >= len(test_dataset):
        raise ValueError(f"Index {max_idx} out of MNIST test range (max {len(test_dataset)-1})")

    # Precompute angles up to max idx (augmentation_factor=1)
    angles_all = build_rotation_angles(max_idx=max_idx, seed=args.seed, rotation_range=rotation_range)

    images_raw = np.zeros((len(idxs), 28, 28), dtype=np.float32)
    angles_deg = np.zeros((len(idxs),), dtype=np.float32)
    digit_labels = np.zeros((len(idxs),), dtype=np.int64)

    for j, idx in enumerate(idxs):
        img_tensor, label = test_dataset[idx]
        img_raw = img_tensor.squeeze(0).numpy().astype(np.float32)
        img_raw = np.clip(img_raw, 0.0, 1.0)
        ang = float(angles_all[idx])
        img_rot = rotate_image_like_dataset_generator(img_raw, ang)

        images_raw[j] = img_rot
        angles_deg[j] = ang
        digit_labels[j] = int(label)

        if (j + 1) % 10 == 0:
            print(f"  processed {j+1}/{len(idxs)}")

    out_json = f"{output_prefix}.json"
    out_npz = f"{output_prefix}.npz"

    # Save NPZ payload
    np.savez_compressed(
        out_npz,
        images_raw=images_raw,
        test_dataset_idx=np.array(idxs, dtype=np.int64),
        true_angle_deg=angles_deg,
        digit_label=digit_labels,
    )

    meta: Dict = {
        "format": "rotated_mnist_test_manifest_v1",
        "source_json": str(source_json),
        "seed": int(args.seed),
        "rotation_range_deg": [float(rotation_range[0]), float(rotation_range[1])],
        "augmentation_factor": 1,
        "background_color": 33,
        "mnist_mean": MNIST_MEAN,
        "mnist_std": MNIST_STD,
        "n_samples": int(len(idxs)),
        "npz_path": str(Path(out_npz).name),
        "arrays": {
            "images_raw": {"shape": list(images_raw.shape), "dtype": "float32", "range": "[0,1]"},
            "test_dataset_idx": {"shape": [len(idxs)], "dtype": "int64"},
            "true_angle_deg": {"shape": [len(idxs)], "dtype": "float32"},
            "digit_label": {"shape": [len(idxs)], "dtype": "int64"},
        },
    }

    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved manifest metadata: {out_json}")
    print(f"✓ Saved manifest arrays:   {out_npz}")


if __name__ == "__main__":
    main()

