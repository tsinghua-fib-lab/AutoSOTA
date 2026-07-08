"""
FILE: images.py
INPUT: Local Extended Yale B CroppedYalePNG image directory and split seeds.
OUTPUT: Downsampled image matrices, labels, and source-matching subject splits.
POS: Face-experiment data loader for reproducible ICML experiments.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from dro_subspace.synthetic import normalize_design

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

DEFAULT_FACE_WIDTH = 21
DEFAULT_FACE_HEIGHT = 24
YALE_FILENAME_LENGTH = 24
STANDARD_SPLIT_SIZE = 10
STANDARD_SPLIT_COUNT = 3


def load_cropped_yale_faces(
    folder_path: Path,
    new_width: int = DEFAULT_FACE_WIDTH,
    new_height: int = DEFAULT_FACE_HEIGHT,
) -> tuple[FloatArray, IntArray]:
    """Load and downsample CroppedYalePNG files using the notebook preprocessing logic."""
    if not folder_path.exists():
        raise FileNotFoundError(f"Cannot proceed - required data directory not found: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Cannot proceed - required path is not a directory: {folder_path}")

    image_paths = sorted(path for path in folder_path.iterdir() if path.is_file())
    image_paths = [path for path in image_paths if len(path.name) == YALE_FILENAME_LENGTH]
    if not image_paths:
        raise ValueError(f"No CroppedYalePNG images with filename length {YALE_FILENAME_LENGTH} found in {folder_path}.")

    with Image.open(image_paths[0]) as first_raw_image:
        first_image = first_raw_image.convert("L")
        target_width = first_image.width if new_width == -1 else new_width
        target_height = first_image.height if new_height == -1 else new_height
    if target_width <= 0 or target_height <= 0:
        raise ValueError("new_width and new_height must be positive, or -1 to preserve the original dimension.")

    images = np.empty((len(image_paths), target_width * target_height), dtype=float)
    labels = np.empty(len(image_paths), dtype=int)
    current_subject = 1

    for row_idx, image_path in enumerate(image_paths):
        subject = int(image_path.name[5:7])
        if subject > current_subject:
            current_subject = subject
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("L")
            resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        images[row_idx] = np.asarray(resized_image, dtype=float).reshape(-1)
        labels[row_idx] = current_subject

    return images, labels


def random_subject_combination(subjects: list[int], size: int, seed: int) -> IntArray:
    """Select subjects with the same sorted random.sample rule used in the face notebook."""
    if size <= 0 or size > len(subjects):
        raise ValueError("size must be positive and no larger than the number of subjects.")
    random.seed(seed)
    return np.array(sorted(random.sample(subjects, size)), dtype=int)


def standard_subject_splits(labels: IntArray) -> list[IntArray]:
    """Return the three 10-subject standard splits used by the paper and notebook."""
    labels_unique = sorted(int(label) for label in set(labels))
    if len(labels_unique) < STANDARD_SPLIT_SIZE * STANDARD_SPLIT_COUNT:
        raise ValueError("At least 30 unique labels are required for the three standard face splits.")
    return [
        np.array(labels_unique[start : start + STANDARD_SPLIT_SIZE], dtype=int)
        for start in range(0, STANDARD_SPLIT_SIZE * STANDARD_SPLIT_COUNT, STANDARD_SPLIT_SIZE)
    ]


def indices_for_subjects(labels: IntArray, subjects: IntArray) -> IntArray:
    """Return image row indexes for a subject set."""
    return np.where(np.isin(labels, subjects))[0].astype(int)


def face_design_for_indices(images: FloatArray, labels: IntArray, indices: IntArray) -> tuple[FloatArray, IntArray]:
    """Build the notebook's image design matrix: images selected, transposed, centered, and normalized."""
    if indices.size == 0:
        raise ValueError("indices must contain at least one image.")
    selected_images = images[indices]
    selected_labels = labels[indices].astype(int)
    design = normalize_design(selected_images.T, n_removed_pcs=0)
    return design, selected_labels
