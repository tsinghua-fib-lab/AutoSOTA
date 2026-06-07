#!/usr/bin/env python
"""
Visualize processed MASS training data.

This utility samples slices from exported ``*_image.npy`` files, optional
``*_gt.npy`` files, and ``dataset.h5`` auto masks so users can quickly inspect
preprocessing outputs before launching training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import h5py
from pathlib import Path

def load_processed_data(data_dir, image_name):
    """Load processed image, optional GT, and auto masks for a sample."""
    data_dir = Path(data_dir)
    image_file = data_dir / f"{image_name}_image.npy"
    gt_file = data_dir / f"{image_name}_gt.npy"
    h5_file = data_dir / "dataset.h5"

    if not image_file.exists():
        return None, None, None

    image = np.load(image_file)
    gt = np.load(gt_file) if gt_file.exists() else None
    auto_masks = np.zeros((0, *image.shape), dtype=np.uint8)

    if h5_file.exists():
        with h5py.File(h5_file, "r") as h5f:
            if image_name in h5f and "auto_masks" in h5f[image_name]:
                auto_masks = h5f[image_name]["auto_masks"][:]

    return image, gt, auto_masks


def normalize_for_display(image, percentiles=(2, 98)):
    """Normalize image for display using percentile clipping."""
    p_low, p_high = np.percentile(image, percentiles)
    image_norm = np.clip(image, p_low, p_high)
    if p_high <= p_low:
        return np.zeros_like(image_norm)
    image_norm = (image_norm - p_low) / (p_high - p_low)
    return image_norm


def create_overlay(image_slice, mask_slice, alpha=0.4, mask_color='red'):
    """Create overlay of mask on image."""
    # Normalize image for display
    img_display = normalize_for_display(image_slice)
    
    img_rgb = np.stack([img_display, img_display, img_display], axis=-1)
    
    if mask_color == 'red':
        color = [1, 0, 0]
    elif mask_color == 'green':
        color = [0, 1, 0]
    elif mask_color == 'blue':
        color = [0, 0, 1]
    elif mask_color == 'yellow':
        color = [1, 1, 0]
    elif mask_color == 'cyan':
        color = [0, 1, 1]
    elif mask_color == 'magenta':
        color = [1, 0, 1]
    else:
        color = [1, 0, 0]  # default to red
    
    mask_binary = mask_slice > 0
    for i in range(3):
        img_rgb[:, :, i] = np.where(mask_binary, 
                                   (1 - alpha) * img_rgb[:, :, i] + alpha * color[i],
                                   img_rgb[:, :, i])
    
    return img_rgb


def visualize_sample(image, gt, auto_masks, image_name, slice_idx, auto_channels=None, save_dir=None):
    """Visualize a single slice with overlays."""
    if auto_channels is None:
        n_channels = min(3, auto_masks.shape[0])
        auto_channels = random.sample(range(auto_masks.shape[0]), n_channels) if auto_masks.shape[0] > 0 else []

    img_slice = image[slice_idx]

    has_gt = gt is not None
    n_plots = 1 + int(has_gt) + len(auto_channels)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Plot original image
    axes[0].imshow(normalize_for_display(img_slice), cmap='gray')
    axes[0].set_title(f'{image_name}\nSlice {slice_idx}\nOriginal Image')
    axes[0].axis('off')

    plot_idx = 1

    # Plot GT overlay if available
    if has_gt:
        gt_slice = gt[slice_idx]
        gt_overlay = create_overlay(img_slice, gt_slice, alpha=0.5, mask_color='green')
        axes[plot_idx].imshow(gt_overlay)
        axes[plot_idx].set_title(f'GT Labels Overlay\n({len(np.unique(gt_slice[gt_slice > 0]))} classes)')
        axes[plot_idx].axis('off')
        plot_idx += 1

    # Plot auto label overlays
    colors = ['red', 'blue', 'yellow', 'cyan', 'magenta']
    for i, channel_idx in enumerate(auto_channels):
        auto_slice = auto_masks[channel_idx, slice_idx]
        auto_overlay = create_overlay(img_slice, auto_slice, alpha=0.4, 
                                     mask_color=colors[i % len(colors)])
        axes[plot_idx + i].imshow(auto_overlay)
        axes[plot_idx + i].set_title(f'Auto Mask {channel_idx}\n({colors[i % len(colors)]})')
        axes[plot_idx + i].axis('off')

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{image_name}_slice{slice_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")

    plt.show()


def print_data_summary(image, gt, auto_masks, image_name):
    """Print summary statistics of the loaded data."""
    print(f"\n=== Data Summary for {image_name} ===")
    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Image mean: {image.mean():.3f}, std: {image.std():.3f}")

    if gt is not None:
        print(f"\nGT shape: {gt.shape}")
        gt_labels = np.unique(gt)
        print(f"GT labels: {gt_labels}")
        print(f"GT non-zero voxels: {np.sum(gt > 0)} ({np.sum(gt > 0) / gt.size * 100:.1f}%)")
    else:
        print("\nGT: not available")

    print(f"\nAuto masks shape: {auto_masks.shape}")
    if auto_masks.shape[0] > 0:
        print(f"Number of auto masks: {auto_masks.shape[0]}")
        non_empty_masks = np.sum([np.any(auto_masks[i] > 0) for i in range(auto_masks.shape[0])])
        print(f"Non-empty auto masks: {non_empty_masks}")
        avg_mask_size = np.mean([np.sum(auto_masks[i] > 0) for i in range(auto_masks.shape[0])])
        print(f"Average mask size: {avg_mask_size:.1f} voxels")
    else:
        print("No auto masks found")


def main():
    parser = argparse.ArgumentParser(description="Visualize processed training data")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Dataset directory containing *_image.npy files and dataset.h5")
    parser.add_argument("--save_dir", type=str, default='./visualization',
                       help="Directory to save visualization images (optional)")
    parser.add_argument("--n_samples", type=int, default=3,
                       help="Number of random samples to visualize")
    parser.add_argument("--n_slices", type=int, default=10,
                       help="Number of random slices per sample")
    parser.add_argument("--images", type=str, nargs='+', default=None,
                       help="Specific image names to visualize (without extension)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)

    image_files = sorted(data_dir.glob("*_image.npy"))

    if not image_files:
        print("No processed image files found!")
        return

    image_names = [f.name[:-len("_image.npy")] for f in image_files]

    print(f"Found {len(image_names)} processed images")
    
    if args.images:
        image_names = [name for name in image_names if name in args.images]
        print(f"Filtering to specified images: {args.images}")
        print(f"Found {len(image_names)} matching images")
    
    if not image_names:
        print("No matching images found!")
        return
    
    n_samples = min(args.n_samples, len(image_names))
    selected_images = random.sample(image_names, n_samples)
    
    print(f"\nVisualizing {n_samples} random samples...")
    
    for image_name in selected_images:
        print(f"\n{'='*50}")
        print(f"Loading {image_name}...")
        
        image, gt, auto_masks = load_processed_data(args.data_dir, image_name)

        if image is None:
            print(f"Failed to load data for {image_name}")
            continue

        print_data_summary(image, gt, auto_masks, image_name)

        # Select random slices to visualize
        n_slices = min(args.n_slices, image.shape[0])
        slice_indices = random.sample(range(image.shape[0]), n_slices)
        
        print(f"\nVisualizing slices: {slice_indices}")
        
        for slice_idx in slice_indices:
            if auto_masks.shape[0] > 0:
                n_auto_channels = min(3, auto_masks.shape[0])
                auto_channels = random.sample(range(auto_masks.shape[0]), n_auto_channels)
            else:
                auto_channels = []

            print(f"\nSlice {slice_idx}: Auto channels {auto_channels}")

            visualize_sample(image, gt, auto_masks, image_name, slice_idx, 
                           auto_channels, args.save_dir)

    print(f"\nVisualization complete!")


if __name__ == "__main__":
    main()
