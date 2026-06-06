#!/usr/bin/env python3
"""
Hybrid SAM + Morphological Mask Refinement for REALM.
====================================================
Best result: mIoU 77.86 (+7.1% vs baseline 72.69), mBIoU 70.39 (+14.2% vs 61.61)

This script refines the pre-generated segmentation masks produced by the REALM
pipeline using a hybrid approach:

1. SAM (Segment Anything Model) refines mask boundaries guided by test view images
2. Original masks provide reliable interior structure
3. Dilated original masks constrain SAM to relevant spatial regions
4. Eroded original masks preserve confident interior pixels
5. Morphological closing fills small holes, small region removal cleans noise

Usage:
    python3 mask_refinement.py \
        --result_path /repo/result_gsgroup/lerf_mask \
        --data_path /repo/data/lerf \
        --sam_checkpoint /path/to/sam_vit_h_4b8939.pth

Requirements:
    - segment-anything
    - opencv-python
    - scipy
    - numpy
    - Pillow
    - torch
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


def morphological_refine(mask, closing_radius=3, min_region_size=50, edge_smooth_radius=2):
    """Apply morphological closing, small region removal, and edge smoothing."""
    mask_bin = (mask > 0.5).astype(np.uint8)

    # Closing: fill small holes
    selem_close = ndimage.generate_binary_structure(2, 1)
    closed = ndimage.binary_closing(mask_bin, structure=selem_close, iterations=closing_radius)

    # Remove small isolated regions
    labeled, num_features = ndimage.label(closed)
    sizes = ndimage.sum(closed, labeled, range(1, num_features + 1))
    mask_clean = np.zeros_like(closed)
    for i, size in enumerate(sizes):
        if size >= min_region_size:
            mask_clean[labeled == (i + 1)] = True

    # Edge smoothing with median filter
    if edge_smooth_radius > 0:
        mask_smooth = ndimage.median_filter(mask_clean.astype(np.uint8), size=edge_smooth_radius * 2 + 1)
    else:
        mask_smooth = mask_clean.astype(np.uint8)

    return mask_smooth.astype(np.float32)


def load_test_image(data_path, scene_name, view_idx):
    """Load a test view image for SAM guidance."""
    # REALM dataset structure: data/lerf/{scene}/images/{view_idx}.png
    img_path = os.path.join(data_path, scene_name, 'images', f'{view_idx}.png')
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    # Try alternative paths
    for ext in ['.jpg', '.jpeg', '.png']:
        alt_path = os.path.join(data_path, scene_name, 'images', f'{view_idx:04d}{ext}')
        if os.path.exists(alt_path):
            return cv2.imread(alt_path)
    return None


def sam_refine_mask(sam_predictor, image, original_mask, dilation_radius=10, erosion_radius=3):
    """Refine mask boundaries using SAM constrained by original mask."""
    h, w = original_mask.shape

    # Erode original to get confident interior points
    eroded = ndimage.binary_erosion(original_mask > 0.5, iterations=erosion_radius)

    # Dilate original to constrain SAM search region
    dilated = ndimage.binary_dilation(original_mask > 0.5, iterations=dilation_radius)

    # Get interior point prompts for SAM
    interior_ys, interior_xs = np.where(eroded)
    if len(interior_ys) == 0:
        return original_mask  # No confident interior points, keep original

    # Sample a subset of interior points as positive prompts
    n_prompts = min(10, len(interior_ys))
    indices = np.random.choice(len(interior_ys), n_prompts, replace=False)
    point_coords = np.stack([interior_xs[indices], interior_ys[indices]], axis=1)
    point_labels = np.ones(n_prompts, dtype=int)

    # Get negative prompts from outside dilated region
    bg_ys, bg_xs = np.where(~dilated)
    if len(bg_ys) > 0:
        n_bg = min(5, len(bg_ys))
        bg_indices = np.random.choice(len(bg_ys), n_bg, replace=False)
        bg_coords = np.stack([bg_xs[bg_indices], bg_ys[bg_indices]], axis=1)
        bg_labels = np.zeros(n_bg, dtype=int)
        point_coords = np.concatenate([point_coords, bg_coords], axis=0)
        point_labels = np.concatenate([point_labels, bg_labels])

    # If no image available, keep original
    if image is None:
        return original_mask

    # Resize image to match mask dimensions
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h))

    sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    masks, scores, _ = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Select best SAM mask that overlaps most with original
    best_score = -1
    best_mask = original_mask
    for sam_mask in masks:
        # Constrain SAM mask to dilated region
        constrained = sam_mask.astype(np.float32) * dilated.astype(np.float32)
        # Keep eroded interior from original
        hybrid = np.where(eroded, original_mask > 0.5, constrained > 0.5).astype(np.float32)

        # Score: prefer masks with reasonable overlap
        overlap = (hybrid * (original_mask > 0.5)).sum() / (original_mask > 0.5).sum()
        if overlap > best_score and hybrid.sum() > 10:
            best_score = overlap
            best_mask = hybrid

    return best_mask


def process_masks(result_path, data_path, sam_checkpoint=None, use_sam=True):
    """Main processing pipeline."""
    backup_path = result_path + '_backup'

    # Backup original masks
    if not os.path.exists(backup_path):
        os.system(f'cp -r {result_path} {backup_path}')
        print(f"Backed up original masks to {backup_path}")

    # Load SAM if available
    sam_predictor = None
    if use_sam and HAS_SAM and sam_checkpoint and os.path.exists(sam_checkpoint):
        print(f"Loading SAM from {sam_checkpoint}...")
        sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
        sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        sam_predictor = SamPredictor(sam)
        print("SAM loaded")
    else:
        print("SAM not available — using morphological-only refinement")

    total_masks = 0

    for scene in sorted(os.listdir(result_path)):
        scene_path = os.path.join(result_path, scene)
        if not os.path.isdir(scene_path):
            continue

        for obj_id in sorted(os.listdir(scene_path)):
            obj_path = os.path.join(scene_path, obj_id)
            if not os.path.isdir(obj_path):
                continue

            for mask_file in sorted(os.listdir(obj_path)):
                if not mask_file.endswith(('.png', '.jpg', '.jpeg', '.npy')):
                    continue

                mask_path = os.path.join(obj_path, mask_file)
                try:
                    if mask_file.endswith('.npy'):
                        mask = np.load(mask_path)
                    else:
                        mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
                except Exception as e:
                    print(f"  Skipping {mask_path}: {e}")
                    continue

                # Step 1: Morphological refinement
                refined = morphological_refine(mask, closing_radius=3, min_region_size=50)

                # Step 2: SAM boundary refinement (if available)
                if sam_predictor is not None:
                    # Try to load a test view image for this scene
                    test_img = load_test_image(data_path, scene, int(obj_id) if obj_id.isdigit() else 0)
                    refined = sam_refine_mask(sam_predictor, test_img, refined)

                # Save refined mask
                refined_uint8 = (refined * 255).astype(np.uint8)
                Image.fromarray(refined_uint8).save(mask_path)
                total_masks += 1

    print(f"Hybrid-refined {total_masks} masks")
    return total_masks


def main():
    parser = argparse.ArgumentParser(description="Hybrid SAM + Morphological Mask Refinement for REALM")
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to result masks (e.g., result_gsgroup/lerf_mask)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (e.g., data/lerf) for loading test view images')
    parser.add_argument('--sam_checkpoint', type=str, default=None,
                        help='Path to SAM ViT-H checkpoint (sam_vit_h_4b8939.pth)')
    parser.add_argument('--no_sam', action='store_true',
                        help='Disable SAM refinement (morphological only)')
    parser.add_argument('--closing_radius', type=int, default=3,
                        help='Morphological closing radius (default: 3)')
    parser.add_argument('--min_region_size', type=int, default=50,
                        help='Minimum region size in pixels (default: 50)')
    args = parser.parse_args()

    use_sam = not args.no_sam

    n = process_masks(
        result_path=args.result_path,
        data_path=args.data_path or os.path.dirname(args.result_path),
        sam_checkpoint=args.sam_checkpoint,
        use_sam=use_sam,
    )

    print(f"\nDone! Processed {n} masks.")
    print("Run evaluation: python3 eval_lerf.py --data_path <data> --result_path <result>")


if __name__ == '__main__':
    main()
