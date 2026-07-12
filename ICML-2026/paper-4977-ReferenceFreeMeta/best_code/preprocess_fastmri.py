"""
Pre-process fastMRI singlecoil data into task/sample format for IPOD meta-learning.

Creates task directories with sample h5 files containing:
- forward_fft: fully-sampled k-space (cropped to target_resolution)
- mask: 1D Cartesian undersampling mask
- img_full: ground truth image (from RSS reconstruction or IFFT)
- img_zf: zero-filled reconstruction
- coordinates: 2D coordinate grid
- slice_idx: original slice index
"""

import h5py
import numpy as np
import os
import sys
import glob
from numpy import fft

def create_cartesian_mask(height, width, R=10, ACS_lines=24):
    """Create 1D Cartesian undersampling mask."""
    mask = np.zeros((1, height, width), dtype=np.float32)
    step = int(R)
    sampling_idx = list(range(0, height, step))

    # Add ACS lines at center
    center_start = height // 2 - ACS_lines // 2
    center_end = center_start + ACS_lines
    center_idx = list(range(center_start, center_end))

    sampling_idx = list(set(sampling_idx + center_idx))
    sampling_idx = [idx for idx in sampling_idx if 0 <= idx < height]
    sampling_idx.sort()

    mask[:, sampling_idx, :] = 1.0

    # Also add center region fully sampled
    actual_sampled = mask.sum()
    actual_AF = mask.size / actual_sampled

    return mask, actual_AF


def crop_center(img, crop_h, crop_w):
    """Crop the central region of an image."""
    h, w = img.shape[-2], img.shape[-1]
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[..., start_h:start_h+crop_h, start_w:start_w+crop_w]


def build_coordinate_train(L_PE, L_RO):
    """Generate 2D coordinate grid in [0,1] range."""
    x = np.linspace(0, 1, L_PE)
    y = np.linspace(0, 1, L_RO)
    x, y = np.meshgrid(x, y, indexing='ij')
    xy = np.stack([x, y], -1).reshape(-1, 2)
    xy = xy.reshape(L_PE, L_RO, 2)
    return xy


def normalize01(img):
    """Normalize image to [0, 1] range."""
    if len(img.shape) == 3:
        nimg = len(img)
    else:
        nimg = 1
        r, c = img.shape
        img = np.reshape(img, (nimg, r, c))
    img2 = np.empty(img.shape, dtype=img.dtype)
    for i in range(nimg):
        denominator = img[i].ptp()
        if denominator == 0:
            denominator = 1
        img2[i] = np.divide(img[i] - img[i].min(), denominator,
                           out=np.zeros_like(img[i]), where=denominator != 0)
    return np.squeeze(img2).astype(img.dtype)


def process_file(h5_path, output_dir, target_res=256, R_values=[4, 6, 10, 14],
                 ACS_lines=24, slices_per_file=None, prefix="task"):
    """
    Process a single fastMRI file and create task directories.

    For singlecoil data:
    - kspace shape: (nslices, freq_enc, phase_enc)
    - reconstruction_rss: (nslices, h, w) or (nslices, target_res, target_res)

    Each slice becomes one task with one sample.
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        kspace = f['kspace'][:]  # (nslices, freq_enc, phase_enc)

        # Get ground truth
        if 'reconstruction_rss' in f:
            gt_img = f['reconstruction_rss'][:]  # (nslices, h, w)
        elif 'reconstruction_esc' in f:
            gt_img = f['reconstruction_esc'][:]
        else:
            # Compute from fully-sampled k-space
            gt_img = np.zeros((kspace.shape[0], target_res, target_res), dtype=np.float32)
            for s in range(kspace.shape[0]):
                recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kspace[s])))
                gt_img[s] = crop_center(np.abs(recon), target_res, target_res)

        nslices = kspace.shape[0]
        if slices_per_file is not None:
            nslices = min(nslices, slices_per_file)

        print("Processing %d slices from %s" % (nslices, os.path.basename(h5_path)))
        print("  kspace shape: %s, gt_img shape: %s" % (str(kspace.shape), str(gt_img.shape)))

        task_counter = 0
        tasks_created = []

        for slice_idx in range(nslices):
            # Get k-space for this slice
            slice_ksp = kspace[slice_idx]  # (freq_enc, phase_enc)
            freq_enc, phase_enc = slice_ksp.shape

            # Crop k-space to get target resolution
            # We crop the k-space by taking central frequency region
            ksp_cropped = crop_center(slice_ksp, target_res, target_res)

            # Get ground truth image
            if gt_img.shape[-1] == target_res and gt_img.shape[-2] == target_res:
                gt_slice = gt_img[slice_idx]
            else:
                gt_slice = crop_center(gt_img[slice_idx], target_res, target_res)

            # Normalize using the zero-filled reconstruction as reference
            zf_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ksp_cropped)))
            norm_factor = np.max(np.abs(zf_recon))
            if norm_factor == 0:
                continue

            # Create different masks for different acceleration factors
            for R in R_values:
                # Create Cartesian 1D mask
                mask_1c, actual_AF = create_cartesian_mask(target_res, target_res, R=R, ACS_lines=ACS_lines)
                mask_1c = mask_1c[0]  # (h, w)
                # Actually, 1D Cartesian mask should be along phase encoding direction
                # In the paper, it's along one dimension
                mask = np.zeros((target_res, target_res), dtype=np.float32)
                sampling_idx = list(range(0, target_res, int(R)))
                center_start = target_res // 2 - ACS_lines // 2
                center_idx = list(range(center_start, center_start + ACS_lines))
                sampling_idx = list(set(sampling_idx + center_idx))
                sampling_idx = [idx for idx in sampling_idx if 0 <= idx < target_res]
                mask[:, sampling_idx] = 1.0  # sample along columns (phase encoding)

                # Create undersampled k-space
                forward_fft = ksp_cropped / norm_factor  # Normalized fully-sampled k-space
                forward_fft_und = forward_fft * mask  # Undersampled

                # For singlecoil, we don't have coil sensitivity maps
                # We create a dummy single-channel csmp
                csmp = np.ones((1, target_res, target_res), dtype=np.complex128)

                # Ground truth as complex (magnitude only for singlecoil RSS)
                gt_complex = gt_slice.astype(np.complex128) / norm_factor

                # Coordinates
                coordinates = build_coordinate_train(target_res, target_res)

                # Create task directory
                task_id_str = "%s_%04d" % (prefix, task_counter)
                task_dir = os.path.join(output_dir, "task_" + task_id_str)
                os.makedirs(task_dir, exist_ok=True)

                # Create sample file
                sample_path = os.path.join(task_dir, "sample_0000.h5")

                with h5py.File(sample_path, 'w') as hf_out:
                    hf_out.create_dataset('forward_fft', data=forward_fft.astype(np.complex64))
                    hf_out.create_dataset('forward_fft_und', data=forward_fft_und.astype(np.complex64))
                    hf_out.create_dataset('mask', data=mask.astype(np.float32))
                    hf_out.create_dataset('csmp', data=csmp.astype(np.complex64))
                    hf_out.create_dataset('img_full', data=gt_complex.astype(np.complex64))
                    hf_out.create_dataset('slice_idx', data=slice_idx)
                    hf_out.create_dataset('coordinates', data=coordinates.astype(np.float32))

                    # Metadata
                    hf_out.attrs['R'] = R
                    hf_out.attrs['ACS'] = ACS_lines
                    hf_out.attrs['Type'] = 'Cartesian_1D'
                    hf_out.attrs['actual_AF'] = actual_AF
                    hf_out.attrs['task_id'] = task_counter
                    hf_out.attrs['source_file'] = os.path.basename(h5_path)
                    hf_out.attrs['source_slice'] = slice_idx
                    hf_out.attrs['num_captions'] = 0

                tasks_created.append(task_id_str)
                task_counter += 1

        print("  Created %d tasks" % task_counter)
        return tasks_created


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/datasets/fastmri_dl')
    parser.add_argument('--output_dir', type=str, default='/datasets/fastmri_processed')
    parser.add_argument('--target_res', type=int, default=256)
    parser.add_argument('--R_values', type=int, nargs='+', default=[4, 6, 10, 14])
    parser.add_argument('--ACS_lines', type=int, default=24)
    parser.add_argument('--max_files', type=int, default=10)
    parser.add_argument('--slices_per_file', type=int, default=10)
    parser.add_argument('--is_eval', action='store_true', help='Process as evaluation data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all h5 files
    h5_files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
    if args.is_eval:
        # Use test files for evaluation
        h5_files = [f for f in h5_files if 'test' in f]
    else:
        # Use train files for training (exclude files with 'test')
        h5_files = [f for f in h5_files if 'test' not in f]

    h5_files = h5_files[:args.max_files]
    print("Processing %d files" % len(h5_files))

    all_tasks = []
    for i, h5_path in enumerate(h5_files):
        print("\n[%d/%d] %s" % (i+1, len(h5_files), os.path.basename(h5_path)))
        tasks = process_file(
            h5_path, args.output_dir,
            target_res=args.target_res,
            R_values=args.R_values,
            ACS_lines=args.ACS_lines,
            slices_per_file=args.slices_per_file,
            prefix="eval" if args.is_eval else "train"
        )
        all_tasks.extend(tasks)

    print("\nTotal tasks created: %d" % len(all_tasks))
    print("Output directory: %s" % args.output_dir)


if __name__ == '__main__':
    main()
