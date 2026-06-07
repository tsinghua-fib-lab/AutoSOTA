"""
Preprocessing utilities for FocusUI patch scoring.
This module provides functions to generate patch-level importance scores for visual tokens.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Default patch configuration (matching Qwen2.5-VL)
PATCH_SIZE = 14  # Size of each visual patch in pixels
MERGE_SIZE = 2   # Spatial merge factor for patch tokens
IMAGE_FACTOR = PATCH_SIZE * MERGE_SIZE  # = 28, factor for smart resize

# Image size constraints
MIN_PIXELS = 4 * 28 * 28        # Minimum total pixels after resize
MAX_PIXELS = 16384 * 28 * 28    # Maximum total pixels after resize
MAX_RATIO = 200                  # Maximum aspect ratio allowed

def smart_resize_with_factor(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Improved smart_resize() from qwen_utils
    """
    def _round_by_factor(number: int, factor: int) -> int:
        return round(number / factor) * factor

    def _ceil_by_factor(number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def _floor_by_factor(number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"Aspect ratio must be <= {MAX_RATIO}, got {max(height, width) / min(height, width):.1f}"
        )

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        # Scale down to fit within max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        # Scale up to meet min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


# =================================
# Bounding Box-Based Patch Scoring
# =================================

def build_patch_score_from_bbox(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    temporal_patch_size: int = 2,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
) -> np.ndarray:
    """
    Compute patch scores based on overlap with a bounding box.
    Each patch receives a score proportional to its overlap with the target bounding box region.

    Args:
        image: Input PIL image (determines output grid dimensions)
        bbox: Bounding box as [x1, y1, x2, y2], either in pixels or normalized [0,1]
        temporal_patch_size: Temporal dimension factor (default: 2)
        patch_size: Size of each patch in pixels (default: 14)
        merge_size: Spatial merge factor (default: 2)

    Returns:
        1D array of patch scores with shape (num_patches,), values in [0, 1]
    """
    width, height = image.size

    # Create binary mask for bounding box region
    mask_2d = np.zeros((height, width), dtype=np.float32)

    x1, y1, x2, y2 = bbox

    # Handle different bbox formats
    if x1 <= 1 and x2 <= 1 and y1 <= 1 and y2 <= 1:
        # Normalized coordinates [0, 1] -> convert to pixels
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
    elif x1 > x2 or y1 > y2:
        # Possibly xywh format -> convert to xyxy
        x2 = int(x1 + x2)
        y2 = int(y1 + y2)
    else:
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)

    # Clip to image boundaries
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)

    # Fill bounding box region
    mask_2d[y1:y2, x1:x2] = 1

    # Compute patch grid dimensions
    grid_h = height // patch_size
    grid_w = width // patch_size

    # Vectorized patch mean computation
    # Reshape: (H, W) -> (grid_h, patch_size, grid_w, patch_size)
    patches_reshaped = mask_2d.reshape(grid_h, patch_size, grid_w, patch_size)
    # Transpose: -> (grid_h, grid_w, patch_size, patch_size)
    patches_reshaped = patches_reshaped.transpose(0, 2, 1, 3)
    # Mean over patch pixels: -> (grid_h, grid_w)
    patch_means = np.mean(patches_reshaped, axis=(2, 3))

    # Reorder to match Qwen's patch token ordering
    # (grid_h, grid_w) -> (grid_h//merge, merge, grid_w//merge, merge)
    patch_means_reordered = patch_means.reshape(
        grid_h // merge_size, merge_size,
        grid_w // merge_size, merge_size
    )
    # Transpose to group sub-patches: -> (grid_h//merge, grid_w//merge, merge, merge)
    patch_means_reordered = patch_means_reordered.transpose(0, 2, 1, 3)

    # Flatten to 1D and apply temporal scaling
    patch_scores = patch_means_reordered.flatten()
    patch_scores = (patch_scores * temporal_patch_size / 2).astype(np.float32)

    return patch_scores


# =================================
# UI Graph-Based Patch Scoring
# =================================
class UnionFind:
    """Union-Find data structure for efficient connected component grouping."""

    def __init__(self, size: int):
        self.parent = np.arange(size)

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Merge two sets."""
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px


def _rerank_values(arr: np.ndarray) -> np.ndarray:
    """Rerank array values to consecutive integers starting from 0."""
    mapping = {}
    new_arr = np.empty_like(arr)
    next_value = 0

    for idx, x in enumerate(arr):
        if x not in mapping:
            mapping[x] = next_value
            next_value += 1
        new_arr[idx] = mapping[x]

    return new_arr


def _reweight_patch_scores(
    labels: np.ndarray,
    method: str = "log",
    min_weight_percentile: float = 0,
) -> np.ndarray:
    """
    Assign importance weights based on cluster frequency.

    Patches in smaller clusters (rarer visual patterns) receive higher weights,
    encouraging the model to preserve unique UI elements.

    Args:
        labels: Cluster assignment for each patch
        method: Weighting scheme
            - "inverse": weight = 1/count
            - "log": weight = 1/log(count+1)
            - "sqrt": weight = 1/sqrt(count)
        min_weight_percentile: Zero out weights below this percentile

    Returns:
        Array of importance weights, same shape as labels
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, counts))

    importance = np.zeros_like(labels, dtype=float)

    for label, count in label_to_count.items():
        if method == "inverse":
            weight = 1.0 / count
        elif method == "log":
            weight = 1.0 / max(1.0, np.log(count + 1))
        elif method == "sqrt":
            weight = 1.0 / np.sqrt(count)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        importance[labels == label] = weight

    # Zero out small weights if requested
    if min_weight_percentile > 0:
        threshold = np.percentile(importance, min_weight_percentile)
        importance = np.where(importance < threshold, 0, importance)

    return importance


def build_patch_score_from_uigraph(
    image: Image.Image,
    ui_graph_threshold: float = 2.0,
    mode: str = "log",
    temporal_patch_size: int = 2,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
) -> np.ndarray:
    """
    Compute patch scores based on UI graph clustering.

    Groups visually similar adjacent patches using Union-Find, then assigns
    higher scores to patches in smaller (more unique) clusters.

    Args:
        image: Input PIL image
        ui_graph_threshold: L2 distance threshold for merging adjacent patches
        mode: Reweighting method ("inverse", "log", or "sqrt")
        temporal_patch_size: Temporal dimension factor (default: 2)
        patch_size: Size of each patch in pixels (default: 14)
        merge_size: Spatial merge factor (default: 2)

    Returns:
        1D array of patch scores with shape (num_patches,)
    """
    width, height = image.size

    # Convert image to normalized numpy array
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Ensure 3 channels
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    # Reshape for patch processing: (H, W, C) -> (T, C, H, W)
    patches = image_array.transpose(2, 0, 1)[None, ...]  # (1, C, H, W)

    # Replicate for temporal dimension
    if patches.shape[0] == 1:
        patches = np.tile(patches, (temporal_patch_size, 1, 1, 1))

    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size

    # Reshape into hierarchical patch grid
    patches = patches.reshape(
        grid_t, temporal_patch_size, channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    # Reorder: (T, H1, W1, m, m, C, tp, p, p)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

    # Build UI graph using Union-Find
    grid_h_half = grid_h // merge_size
    grid_w_half = grid_w // merge_size
    num_patches = grid_t * grid_h_half * grid_w_half
    uf = UnionFind(num_patches)

    def patch_idx(t: int, i: int, j: int) -> int:
        return t * grid_h_half * grid_w_half + i * grid_w_half + j

    # Connect adjacent patches with similar appearance
    for t in range(grid_t):
        for i in range(grid_h_half):
            for j in range(grid_w_half):
                current_idx = patch_idx(t, i, j)
                current_patch = patches[t, i, j, ...]

                # Check right neighbor
                if j + 1 < grid_w_half:
                    right_patch = patches[t, i, j + 1, ...]
                    if np.linalg.norm(current_patch - right_patch) < ui_graph_threshold:
                        uf.union(current_idx, patch_idx(t, i, j + 1))

                # Check bottom neighbor
                if i + 1 < grid_h_half:
                    bottom_patch = patches[t, i + 1, j, ...]
                    if np.linalg.norm(current_patch - bottom_patch) < ui_graph_threshold:
                        uf.union(current_idx, patch_idx(t, i + 1, j))

    # Get cluster assignments and rerank
    cluster_ids = np.array([uf.find(x) for x in range(num_patches)])
    cluster_ids = _rerank_values(cluster_ids)

    # Compute importance weights
    patch_scores_merged = _reweight_patch_scores(cluster_ids, method=mode)

    # Expand to full patch grid (each merged block -> merge_size^2 patches)
    expanded_scores = []
    score_idx = 0

    for _ in range(grid_t):
        for _ in range(grid_h // merge_size):
            for _ in range(grid_w // merge_size):
                block_score = patch_scores_merged[score_idx]
                # Repeat for all sub-patches in this block
                for _ in range(merge_size * merge_size):
                    expanded_scores.append(block_score)
                score_idx += 1

    return np.array(expanded_scores, dtype=np.float32)


def merge_patches_mean(arr: np.ndarray, merge_size: int = MERGE_SIZE) -> np.ndarray:
    """
    Merge patches by taking the mean of consecutive groups.
    Used to adapt patch scores to Qwen's spatial merge factor.

    Args:
        arr: 1D array of patch values
        merge_size: Spatial merge factor (merges merge_size^2 patches)

    Returns:
        1D array with length reduced by merge_size^2
    """
    merge_factor = merge_size ** 2
    assert arr.shape[0] % merge_factor == 0, (
        f"Array length {arr.shape[0]} must be divisible by {merge_factor}"
    )

    arr_reshaped = arr.reshape(-1, merge_factor)
    return arr_reshaped.mean(axis=1)


# =================================
# Main Preprocessing Function
# =================================
def preprocess_focusui_data(
    ele_image: Image.Image,
    ele_bbox: Optional[Tuple[float, float, float, float]],
    gt_bbox_weight: float = 1.0,
    gt_uigraph_weight: float = 0.5,
    tokenizer=None,
    instruction: Optional[str] = None,
    image_grid_thw: Optional[Tuple[int, int, int]] = None,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
) -> Dict[str, torch.Tensor]:
    """
    Generate FocusUI training data for a single sample.
    Computes patch-level importance scores by combining:
    1. Bounding box overlap scores (where the target element is)
    2. UI graph uniqueness scores (visually distinctive regions)
    """
    width, height = ele_image.size
    patch_merge_size = patch_size * merge_size

    # Determine target image size
    if image_grid_thw is not None:
        # Use pre-computed grid dimensions
        smart_h = int(image_grid_thw[1] * patch_size)
        smart_w = int(image_grid_thw[2] * patch_size)
    else:
        # Compute smart resize dimensions
        smart_h, smart_w = smart_resize_with_factor(
            height, width,
            factor=patch_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    resized_image = ele_image.resize((smart_w, smart_h))

    # Compute bounding box-based patch scores
    if ele_bbox:
        patch_score_bbox = build_patch_score_from_bbox(
            resized_image,
            ele_bbox,
            patch_size=patch_size,
        )
    else:
        num_merged_patches = (smart_w // patch_merge_size) * (smart_h // patch_merge_size)
        patch_score_bbox = np.zeros(num_merged_patches, dtype=np.float32)

    # Compute UI graph-based patch scores
    patch_score_uigraph = build_patch_score_from_uigraph(
        resized_image,
        ui_graph_threshold=2.0,
        mode='log',
        patch_size=patch_size,
    )

    # Combine scores with weights
    patch_scores_label = (
        gt_bbox_weight * patch_score_bbox +
        gt_uigraph_weight * patch_score_uigraph
    )
    patch_scores_label = np.clip(patch_scores_label, 0, 1)
    # Scale from [0, 1] to [-1, 1] for training
    patch_scores_label = patch_scores_label * 2 - 1

    # Merge patches to match Qwen's spatial merge factor
    patch_scores_merged = merge_patches_mean(patch_scores_label)

    # Tokenize instruction if provided
    if tokenizer is not None and instruction:
        focus_inputs = tokenizer(instruction, return_tensors="pt")
        focus_input_ids = focus_inputs['input_ids'][0]
        focus_attention_mask = focus_inputs['attention_mask'][0]
    else:
        focus_input_ids = None
        focus_attention_mask = None

    return {
        "patch_score_bbox": torch.from_numpy(patch_score_bbox),
        "patch_score_uigraph": torch.from_numpy(patch_score_uigraph),
        "patch_scores_label_unmerged": torch.from_numpy(patch_scores_label),
        "patch_scores_label": torch.from_numpy(patch_scores_merged),
        "focus_input_ids": focus_input_ids,
        "focus_attention_mask": focus_attention_mask,
    }
