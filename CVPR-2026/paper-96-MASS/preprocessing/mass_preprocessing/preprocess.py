"""Raw NIfTI standardization before SAM2 mask generation.

This module reads medical images and optional labels, reorients them to a
consistent orientation, crops to the label/body/foreground region of interest,
and writes standardized NIfTI files for downstream SAM2 preprocessing steps.
"""

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_orientation_matrix(affine: np.ndarray) -> np.ndarray:
    """
    Project a non-orthonormal affine orientation onto the nearest valid rotation.
    """
    matrix = affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    spacing[spacing == 0] = 1.0
    orientation = matrix / spacing[np.newaxis, :]
    u, _, vt = np.linalg.svd(orientation)
    orthonormal = u @ vt

    fixed_affine = affine.copy()
    fixed_affine[:3, :3] = orthonormal * spacing[np.newaxis, :]
    return fixed_affine


def safe_read_image(image_file: Union[str, Path], verbose: bool = False) -> sitk.Image:
    """
    Read a NIfTI image with a nibabel fallback for files rejected by SimpleITK.
    """
    image_path = Path(image_file)
    try:
        return sitk.ReadImage(str(image_path))
    except RuntimeError as exc:
        message = f"Using nibabel fallback for {image_path}: {exc}"
        if verbose:
            logger.warning(message)
        else:
            logger.debug(message)

        nib_img = nib.load(str(image_path))
        data = np.asanyarray(nib_img.dataobj)
        if data.ndim != 3:
            raise

        fixed_affine = fix_orientation_matrix(nib_img.affine)
        data = np.transpose(data, (2, 1, 0))

        image = sitk.GetImageFromArray(data)
        spacing = np.linalg.norm(fixed_affine[:3, :3], axis=0)
        spacing[spacing == 0] = 1.0
        image.SetSpacing(spacing.tolist())
        image.SetOrigin(fixed_affine[:3, 3].tolist())
        direction = fixed_affine[:3, :3] / spacing[np.newaxis, :]
        image.SetDirection(direction.flatten().tolist())
        return image


def reorient_image_and_label(
    itk_img: sitk.Image,
    itk_lab: Optional[sitk.Image] = None,
    target_orientation: str = 'RAS'
) -> Union[sitk.Image, Tuple[sitk.Image, sitk.Image]]:
    """
    Reorient medical image and optionally its corresponding label mask to target orientation.
    
    Args:
        itk_img (sitk.Image): Input medical image
        itk_lab (Optional[sitk.Image]): Optional label mask corresponding to the image
        target_orientation (str): Target orientation code (default: 'RAS')
    
    Returns:
        Union[sitk.Image, Tuple[sitk.Image, sitk.Image]]: 
            - If itk_lab is None: returns only the reoriented image
            - If itk_lab is provided: returns tuple of (reoriented_image, reoriented_label)
    """
    reoriented_img = sitk.DICOMOrient(itk_img, target_orientation)
    
    # If no label provided, return only the image
    if itk_lab is None:
        return reoriented_img
    
    if itk_img.GetSize() != itk_lab.GetSize():
        raise ValueError(f"Image size {itk_img.GetSize()} doesn't match label size {itk_lab.GetSize()}")
    
    reoriented_lab = sitk.DICOMOrient(itk_lab, target_orientation)
    
    return reoriented_img, reoriented_lab



def crop_image_and_label(
    itk_img: sitk.Image,
    itk_lab: sitk.Image,
    margin: Union[int, List[int], Tuple[int, int, int]] = 10,
    margin_unit: str = "mm"
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Crop image and label based on the foreground region in the label mask.
    
    Args:
        itk_img (sitk.Image): Input medical image
        itk_lab (sitk.Image): Label mask with foreground regions
        margin (Union[int, List[int], Tuple[int, int, int]]): 
            Margin to add around the foreground bounding box.
            If int: same margin for all axes
            If list/tuple: margin for each axis [x, y, z]
        margin_unit (str): Unit of margin - "pixel" or "mm" (millimeters)
    
    Returns:
        Tuple[sitk.Image, sitk.Image]: Cropped (image, label)
    """
    cropped_img, cropped_lab = crop_images_by_mask(
        [itk_img, itk_lab],
        itk_lab,
        margin=margin,
        margin_unit=margin_unit,
    )
    return cropped_img, cropped_lab


def _normalize_margin(
    margin: Union[int, List[int], Tuple[int, int, int]],
    spacing: Tuple[float, float, float],
    margin_unit: str,
) -> List[int]:
    if isinstance(margin, int):
        margin = [margin, margin, margin]
    elif len(margin) != 3:
        raise ValueError("Margin must be an int or a list/tuple of 3 values")

    if margin_unit == "mm":
        margin = [int(np.ceil(margin[i] / spacing[i])) for i in range(3)]
    elif margin_unit != "pixel":
        raise ValueError(f"margin_unit must be 'pixel' or 'mm', got '{margin_unit}'")

    return [int(x) for x in margin]


def compute_foreground_roi(
    mask: sitk.Image,
    margin: Union[int, List[int], Tuple[int, int, int]] = 10,
    margin_unit: str = "mm",
) -> Tuple[List[int], List[int]]:
    """
    Compute a SimpleITK ROI from foreground voxels in a mask.
    """
    margin = _normalize_margin(margin, mask.GetSpacing(), margin_unit)
    label_array = sitk.GetArrayFromImage(mask)
    nonzero_indices = np.where(label_array > 0)

    if len(nonzero_indices[0]) == 0:
        raise ValueError("Label contains no foreground voxels")

    min_z, max_z = nonzero_indices[0].min(), nonzero_indices[0].max()
    min_y, max_y = nonzero_indices[1].min(), nonzero_indices[1].max()
    min_x, max_x = nonzero_indices[2].min(), nonzero_indices[2].max()

    lower_bound = [min_x, min_y, min_z]
    upper_bound = [max_x, max_y, max_z]

    image_size = mask.GetSize()
    crop_lower = []
    crop_upper = []

    for i in range(3):
        lower = max(0, lower_bound[i] - margin[i])
        crop_lower.append(lower)

        upper = min(image_size[i] - 1, upper_bound[i] + margin[i])
        crop_upper.append(upper)

    roi_index = [int(x) for x in crop_lower]
    roi_size = [int(crop_upper[i] - crop_lower[i] + 1) for i in range(3)]
    return roi_index, roi_size


def crop_images_by_mask(
    images: List[sitk.Image],
    mask: sitk.Image,
    margin: Union[int, List[int], Tuple[int, int, int]] = 10,
    margin_unit: str = "mm",
) -> Tuple[sitk.Image, ...]:
    """
    Crop one or more images with the same foreground ROI from a mask.
    """
    for image in images:
        if image.GetSize() != mask.GetSize():
            raise ValueError(f"Image size {image.GetSize()} doesn't match mask size {mask.GetSize()}")

    roi_index, roi_size = compute_foreground_roi(mask, margin=margin, margin_unit=margin_unit)
    return tuple(sitk.RegionOfInterest(image, size=roi_size, index=roi_index) for image in images)


def has_same_geometry(image: sitk.Image, reference: sitk.Image) -> bool:
    """
    Check whether two images share the grid needed for voxel-wise operations.
    """
    return (
        image.GetSize() == reference.GetSize()
        and np.allclose(image.GetSpacing(), reference.GetSpacing())
        and np.allclose(image.GetOrigin(), reference.GetOrigin())
        and np.allclose(image.GetDirection(), reference.GetDirection())
    )


def process_bcv_dataset(input_dir: str, output_dir: str):
    """
    Process all images and labels in the BCV dataset.
    
    Args:
        input_dir: Input directory containing 'img' and 'label' folders
        output_dir: Output directory for processed data
    """
    img_dir = os.path.join(input_dir, 'img')
    label_dir = os.path.join(input_dir, 'label')
    
    if not os.path.exists(img_dir):
        raise ValueError(f"Image directory not found: {img_dir}")
    if not os.path.exists(label_dir):
        raise ValueError(f"Label directory not found: {label_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
    
    if len(img_files) == 0:
        raise ValueError(f"No .nii.gz files found in {img_dir}")
    
    processed_count = 0
    skipped_count = 0
    
    for img_path in img_files:
        img_basename = os.path.basename(img_path)
        match = re.search(r'(\d+)', img_basename)
        
        if not match:
            print(f"Warning: Could not extract number from {img_basename}, skipping...")
            skipped_count += 1
            continue
        
        number = match.group(1)
        label_filename = f"label{number}.nii.gz"
        label_path = os.path.join(label_dir, label_filename)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {img_basename}, skipping...")
            skipped_count += 1
            continue
        
        try:
            img = sitk.ReadImage(img_path)
            lab = sitk.ReadImage(label_path)
            img.SetDirection((1,0,0, 0,-1,0, 0,0,1)) # BCV's image direction is wrong, so we fix it
            lab.SetDirection((1,0,0, 0,-1,0, 0,0,1)) 
            
            img, lab = reorient_image_and_label(img, lab, 'RAS')
            
            
            img_cropped, lab_cropped = crop_image_and_label(img, lab, margin=[30, 30, 8])
            
            img_name_without_ext = img_basename.replace('.nii.gz', '')
            output_img_path = os.path.join(output_dir, f"{img_name_without_ext}.nii.gz")
            output_label_path = os.path.join(output_dir, f"{img_name_without_ext}_gt.nii.gz")
            
            sitk.WriteImage(img_cropped, output_img_path)
            sitk.WriteImage(lab_cropped, output_label_path)
            
            print(f"Processed: {img_basename}")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_basename}: {str(e)}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Output saved to: {output_dir}")


def process_amos_dataset(input_dir: str, output_dir: str):
    """
    Process all images and labels in the AMOS dataset.
    
    Args:
        input_dir: Input directory containing imagesTr, labelsTr, imagesVa, labelsVa folders
        output_dir: Output directory for processed data (will create _ct and _mr subfolders)
    """
    images_tr_dir = os.path.join(input_dir, 'imagesTr')
    labels_tr_dir = os.path.join(input_dir, 'labelsTr')
    images_va_dir = os.path.join(input_dir, 'imagesVa')
    labels_va_dir = os.path.join(input_dir, 'labelsVa')
    
    for dir_path, dir_name in [(images_tr_dir, 'imagesTr'), (labels_tr_dir, 'labelsTr'),
                                (images_va_dir, 'imagesVa'), (labels_va_dir, 'labelsVa')]:
        if not os.path.exists(dir_path):
            raise ValueError(f"{dir_name} directory not found: {dir_path}")
    
    output_ct_dir = output_dir + '_ct'
    output_mr_dir = output_dir + '_mr'
    os.makedirs(output_ct_dir, exist_ok=True)
    os.makedirs(output_mr_dir, exist_ok=True)
    
    all_pairs = []
    
    train_images = glob.glob(os.path.join(images_tr_dir, 'amos_*.nii.gz'))
    for img_path in train_images:
        img_basename = os.path.basename(img_path)
        label_path = os.path.join(labels_tr_dir, img_basename)
        if os.path.exists(label_path):
            all_pairs.append((img_path, label_path))
    
    val_images = glob.glob(os.path.join(images_va_dir, 'amos_*.nii.gz'))
    for img_path in val_images:
        img_basename = os.path.basename(img_path)
        label_path = os.path.join(labels_va_dir, img_basename)
        if os.path.exists(label_path):
            all_pairs.append((img_path, label_path))
    
    if len(all_pairs) == 0:
        raise ValueError("No valid image-label pairs found")
    
    processed_ct = 0
    processed_mr = 0
    skipped_count = 0
    
    for img_path, label_path in all_pairs:
        img_basename = os.path.basename(img_path)
        match = re.search(r'amos_(\d+)', img_basename)
        
        if not match:
            print(f"Warning: Could not extract number from {img_basename}, skipping...")
            skipped_count += 1
            continue
        
        number = int(match.group(1))
        
        try:
            img = sitk.ReadImage(img_path)
            lab = sitk.ReadImage(label_path)
            
            img, lab = reorient_image_and_label(img, lab, 'RAS')
            
            img_cropped, lab_cropped = crop_image_and_label(img, lab, margin=[30, 30, 8])
            
            if number < 500:
                # CT image
                output_folder = output_ct_dir
                processed_ct += 1
            else:
                # MRI image
                output_folder = output_mr_dir
                processed_mr += 1
            
            output_img_path = os.path.join(output_folder, f"{number}.nii.gz")
            output_label_path = os.path.join(output_folder, f"{number}_gt.nii.gz")
            
            sitk.WriteImage(img_cropped, output_img_path)
            sitk.WriteImage(lab_cropped, output_label_path)
            
            print(f"Processed: {img_basename} -> {number}.nii.gz ({'CT' if number < 500 else 'MRI'})")
            
        except Exception as e:
            print(f"Error processing {img_basename}: {str(e)}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total CT processed: {processed_ct}")
    print(f"Total MRI processed: {processed_mr}")
    print(f"Total skipped: {skipped_count}")
    print(f"CT data saved to: {output_ct_dir}")
    print(f"MRI data saved to: {output_mr_dir}")


def process_kits_dataset(input_dir: str, output_dir: str):
    """
    Process all images and labels in the KiTS dataset.
    
    Args:
        input_dir: Input directory containing case_XXXXX folders
        output_dir: Output directory for processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    case_folders = sorted(glob.glob(os.path.join(input_dir, 'case_*')))
    case_folders = [f for f in case_folders if os.path.isdir(f)]
    
    if len(case_folders) == 0:
        raise ValueError(f"No case folders found in {input_dir}")
    
    processed_count = 0
    skipped_count = 0
    
    for case_folder in case_folders:
        case_name = os.path.basename(case_folder)
        match = re.search(r'case_(\d+)', case_name)
        
        if not match:
            print(f"Warning: Could not extract number from {case_name}, skipping...")
            skipped_count += 1
            continue
        
        case_number = int(match.group(1))
        
        img_path = os.path.join(case_folder, 'imaging.nii.gz')
        label_path = os.path.join(case_folder, 'segmentation.nii.gz')
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for {case_name}, skipping...")
            skipped_count += 1
            continue
            
        if not os.path.exists(label_path):
            print(f"Warning: Segmentation not found for {case_name}, skipping...")
            skipped_count += 1
            continue
        
        try:
            img = sitk.ReadImage(img_path)
            lab = sitk.ReadImage(label_path)
            
            img, lab = reorient_image_and_label(img, lab, 'RAS')
            
            img_cropped, lab_cropped = crop_image_and_label(img, lab, margin=[40, 40, 10])
            
            output_img_path = os.path.join(output_dir, f"{case_number}.nii.gz")
            output_label_path = os.path.join(output_dir, f"{case_number}_gt.nii.gz")
            
            sitk.WriteImage(img_cropped, output_img_path)
            sitk.WriteImage(lab_cropped, output_label_path)
            
            print(f"Processed: {case_name} -> {case_number}.nii.gz")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {case_name}: {str(e)}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Output saved to: {output_dir}")



def process_lits_dataset(input_dir: str, output_dir: str):
    """
    Process all images and labels in the LiTS dataset.
    
    Args:
        input_dir: Input directory containing volume-X.nii and segmentation-X.nii files
        output_dir: Output directory for processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    volume_files = sorted(glob.glob(os.path.join(input_dir, 'volume-*.nii')))
    
    if len(volume_files) == 0:
        raise ValueError(f"No volume-*.nii files found in {input_dir}")
    
    processed_count = 0
    skipped_count = 0
    
    for volume_path in volume_files:
        volume_basename = os.path.basename(volume_path)
        match = re.search(r'volume-(\d+)', volume_basename)
        
        if not match:
            print(f"Warning: Could not extract number from {volume_basename}, skipping...")
            skipped_count += 1
            continue
        
        number = int(match.group(1))
        
        segmentation_filename = f"segmentation-{number}.nii"
        segmentation_path = os.path.join(input_dir, segmentation_filename)
        
        if not os.path.exists(segmentation_path):
            print(f"Warning: Segmentation not found for volume-{number}.nii, skipping...")
            skipped_count += 1
            continue
        
        try:
            img = sitk.ReadImage(volume_path)
            lab = sitk.ReadImage(segmentation_path)

            if number in [28, 34]:
                img.SetSpacing((0.7, 0.7, 2.5)) # number 28 and 34 have wrong spacing
            
            img, lab = reorient_image_and_label(img, lab, 'RAS')
            
            img_cropped, lab_cropped = crop_image_and_label(img, lab, margin=[40, 40, 10])
            
            output_img_path = os.path.join(output_dir, f"{number}.nii.gz")
            output_label_path = os.path.join(output_dir, f"{number}_gt.nii.gz")
            
            sitk.WriteImage(img_cropped, output_img_path)
            sitk.WriteImage(lab_cropped, output_label_path)
            
            print(f"Processed: volume-{number}.nii -> {number}.nii.gz")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing volume-{number}.nii: {str(e)}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Output saved to: {output_dir}")


def process_structseg_head_oar_dataset(input_dir: str, output_dir: str):
    """
    Process all images and labels in the HaN_OAR dataset.
    
    Args:
        input_dir: Input directory containing numbered folders with data.nii.gz and label.nii.gz
        output_dir: Output directory for processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    numbered_folders = []
    for folder in all_folders:
        if folder.isdigit():
            numbered_folders.append(int(folder))
    
    if len(numbered_folders) == 0:
        raise ValueError(f"No numbered folders found in {input_dir}")
    
    numbered_folders.sort()
    
    processed_count = 0
    skipped_count = 0
    
    for folder_number in numbered_folders:
        folder_path = os.path.join(input_dir, str(folder_number))
        
        img_path = os.path.join(folder_path, 'data.nii.gz')
        label_path = os.path.join(folder_path, 'label.nii.gz')
        
        if not os.path.exists(img_path):
            print(f"Warning: data.nii.gz not found in folder {folder_number}, skipping...")
            skipped_count += 1
            continue
            
        if not os.path.exists(label_path):
            print(f"Warning: label.nii.gz not found in folder {folder_number}, skipping...")
            skipped_count += 1
            continue
        
        try:
            img = sitk.ReadImage(img_path)
            lab = sitk.ReadImage(label_path)
            
            img, lab = reorient_image_and_label(img, lab, 'RAS')
            
            # Using smaller margins since head and neck region is more compact
            img_cropped, lab_cropped = crop_image_and_label(img, lab, margin=[15, 15, 5])
            
            output_img_path = os.path.join(output_dir, f"{folder_number}.nii.gz")
            output_label_path = os.path.join(output_dir, f"{folder_number}_gt.nii.gz")
            
            sitk.WriteImage(img_cropped, output_img_path)
            sitk.WriteImage(lab_cropped, output_label_path)
            
            print(f"Processed: folder {folder_number} -> {folder_number}.nii.gz")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing folder {folder_number}: {str(e)}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Output saved to: {output_dir}")

def process_structseg_thoracic_oar_dataset(input_dir: str, output_dir: str):
    """
    Process all images and labels in the Thoracic_OAR dataset.
    
    Args:
        input_dir: Input directory containing numbered folders with data.nii.gz and label.nii.gz
        output_dir: Output directory for processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    numbered_folders = []
    for folder in all_folders:
        if folder.isdigit():
            numbered_folders.append(int(folder))
    
    if len(numbered_folders) == 0:
        raise ValueError(f"No numbered folders found in {input_dir}")
    
    numbered_folders.sort()
    
    processed_count = 0
    skipped_count = 0
    
    for folder_number in numbered_folders:
        folder_path = os.path.join(input_dir, str(folder_number))
        
        img_path = os.path.join(folder_path, 'data.nii.gz')
        label_path = os.path.join(folder_path, 'label.nii.gz')
        
        if not os.path.exists(img_path):
            print(f"Warning: data.nii.gz not found in folder {folder_number}, skipping...")
            skipped_count += 1
            continue
            
        if not os.path.exists(label_path):
            print(f"Warning: label.nii.gz not found in folder {folder_number}, skipping...")
            skipped_count += 1
            continue
        
        try:
            img = sitk.ReadImage(img_path)
            lab = sitk.ReadImage(label_path)
            
            img, lab = reorient_image_and_label(img, lab, 'RAS')
            
            # Using larger margins since thoracic region is more extensive than head/neck
            img_cropped, lab_cropped = crop_image_and_label(img, lab, margin=[30, 30, 10])
            
            output_img_path = os.path.join(output_dir, f"{folder_number}.nii.gz")
            output_label_path = os.path.join(output_dir, f"{folder_number}_gt.nii.gz")
            
            sitk.WriteImage(img_cropped, output_img_path)
            sitk.WriteImage(lab_cropped, output_label_path)
            
            print(f"Processed: folder {folder_number} -> {folder_number}.nii.gz (Thoracic)")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing folder {folder_number}: {str(e)}")
            skipped_count += 1
    
    print(f"\nThoracic OAR processing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Output saved to: {output_dir}")


def process_ctpelvic1k_dataset(input_dir: str, output_dir: str):
    """
    Process all images and labels in the CTPelvic1K dataset.
    
    Args:
        input_dir: Input directory containing CTPelvic1K_dataset6_data and ipcai2021_dataset6_Anonymized folders
        output_dir: Output directory for processed data
    """
    img_dir = os.path.join(input_dir, 'CTPelvic1K_dataset6_data')
    label_dir = os.path.join(input_dir, 'ipcai2021_dataset6_Anonymized')
    
    if not os.path.exists(img_dir):
        raise ValueError(f"Image directory not found: {img_dir}")
    if not os.path.exists(label_dir):
        raise ValueError(f"Label directory not found: {label_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted(glob.glob(os.path.join(img_dir, 'dataset6_CLINIC_*_data.nii.gz')))
    
    if len(img_files) == 0:
        raise ValueError(f"No dataset6_CLINIC_*_data.nii.gz files found in {img_dir}")
    
    processed_count = 0
    skipped_count = 0
    
    for img_path in img_files:
        img_basename = os.path.basename(img_path)
        match = re.search(r'dataset6_CLINIC_(\d+)_data', img_basename)
        
        if not match:
            print(f"Warning: Could not extract number from {img_basename}, skipping...")
            skipped_count += 1
            continue
        
        number_str = match.group(1)
        number = int(number_str)
        
        label_filename = f"dataset6_CLINIC_{number_str}_mask_4label.nii.gz"
        label_path = os.path.join(label_dir, label_filename)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {img_basename}, skipping...")
            skipped_count += 1
            continue
        
        try:
            img = sitk.ReadImage(img_path)
            lab = sitk.ReadImage(label_path)
            
            img, lab = reorient_image_and_label(img, lab, 'RAS')
            
            # Using moderate margins suitable for pelvic anatomy
            img_cropped, lab_cropped = crop_image_and_label(img, lab, margin=[40, 40, 10])
            
            output_img_path = os.path.join(output_dir, f"{number}.nii.gz")
            output_label_path = os.path.join(output_dir, f"{number}_gt.nii.gz")
            
            sitk.WriteImage(img_cropped, output_img_path)
            sitk.WriteImage(lab_cropped, output_label_path)
            
            print(f"Processed: {img_basename} -> {number}.nii.gz")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_basename}: {str(e)}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Output saved to: {output_dir}")

def process_mnm_dataset(input_dir: str, output_dir: str):
    """
    Process the MnM (Multi-Centre, Multi-Vendor & Multi-Disease) cardiac cine MRI dataset.
    - Reads 4D image/label pairs ({name}_sa.nii.gz, {name}_sa_gt.nii.gz) per subject.
    - Extracts the labeled time frames (typically ED & ES) to 3D volumes.
    - Reorients each 3D image/label to RAS.
    - Crops around the label foreground with a margin.
    - No resampling is performed.
    - Saves outputs as: {subject}_0(.nii.gz), {subject}_0_gt(.nii.gz), {subject}_1(.nii.gz), {subject}_1_gt(.nii.gz)

    Expected directory layout:
        input_dir/
            Training/Labeled/{subject}/
                {subject}_sa.nii.gz
                {subject}_sa_gt.nii.gz
            Validation/{subject}/
                {subject}_sa.nii.gz
                {subject}_sa_gt.nii.gz
            Testing/{subject}/
                {subject}_sa.nii.gz
                {subject}_sa_gt.nii.gz  # if available; otherwise case is skipped

    Args:
        input_dir (str): Root of the MnM dataset as described above.
        output_dir (str): Directory to place processed 3D frames.

    Notes:
        - Uses `reorient_image_and_label` and `crop_image_and_label` defined earlier in your script.
        - Margin is specified in millimeters to roughly mimic your prior CMR context
          ([in-plane, in-plane, through-plane] ~= [40 mm, 40 mm, 20 mm]).
    """

    crop_margin_mm = [100, 100, 20]  # generous in-plane, smaller through-plane
    phases = ['Training/Labeled', 'Validation', 'Testing']

    def _extract_3d_at_time(img4d: sitk.Image, t_index: int) -> sitk.Image:
        """Extract a 3D volume at time index t_index from a 4D image using SimpleITK Extract."""
        size = list(img4d.GetSize())  # (x, y, z, t)
        if len(size) != 4:
            raise ValueError(f"Expected 4D image for extraction, got dimension {len(size)}.")
        index = [0, 0, 0, 0]
        index[3] = int(t_index)
        size[3] = 0  # collapse the time dimension at t_index
        return sitk.Extract(img4d, size=size, index=index)

    os.makedirs(output_dir, exist_ok=True)

    subjects_processed = 0
    frames_saved = 0
    subjects_skipped = 0

    for phase in phases:
        phase_dir = os.path.join(input_dir, phase)
        if not os.path.isdir(phase_dir):
            print(f"Warning: phase directory not found: {phase_dir} (skipping phase)")
            continue

        subject_names = sorted(
            [d for d in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, d))]
        )

        for name in subject_names:
            subj_dir = os.path.join(phase_dir, name)
            img_path = os.path.join(subj_dir, f"{name}_sa.nii.gz")
            lab_path = os.path.join(subj_dir, f"{name}_sa_gt.nii.gz")

            if not os.path.exists(img_path):
                print(f"Warning: Image not found for {phase}/{name}, skipping...")
                subjects_skipped += 1
                continue
            if not os.path.exists(lab_path):
                print(f"Warning: Label not found for {phase}/{name}, skipping (no cropping possible)...")
                subjects_skipped += 1
                continue

            try:
                
                img4d = safe_read_image(img_path)
                lab4d = safe_read_image(lab_path)

                if img4d.GetDimension() != 4 or lab4d.GetDimension() != 4:
                    print(f"Warning: Expected 4D image/label for {phase}/{name}; got "
                          f"{img4d.GetDimension()}D/{lab4d.GetDimension()}D. Skipping...")
                    subjects_skipped += 1
                    continue

                lab_np = sitk.GetArrayFromImage(lab4d)  # shape: (t, z, y, x)
                labeled_frames = [t for t in range(lab_np.shape[0]) if lab_np[t].max() > 0]

                if len(labeled_frames) == 0:
                    print(f"Warning: No labeled frames found for {phase}/{name}, skipping...")
                    subjects_skipped += 1
                    continue

                # Use at most two frames (typically ED & ES). Preserve time order.
                if len(labeled_frames) > 2:
                    print(f"Note: Found {len(labeled_frames)} labeled frames for {phase}/{name}; "
                          f"keeping the first two in time order.")
                selected_frames = labeled_frames[:2]

                saved_this_subject = 0
                for count, t_idx in enumerate(selected_frames):
                    # 4D -> 3D extraction (preserves spacing/origin/direction for x,y,z)
                    img3d = _extract_3d_at_time(img4d, t_idx)
                    lab3d = _extract_3d_at_time(lab4d, t_idx)


                    # Crop around labeled foreground (margin in mm)
                    img_crop, lab_crop = crop_image_and_label(
                        img3d, lab3d, margin=crop_margin_mm, margin_unit="mm"
                    )

                    out_img = os.path.join(output_dir, f"{name}_{count}.nii.gz")
                    out_lab = os.path.join(output_dir, f"{name}_{count}_gt.nii.gz")
                    sitk.WriteImage(img_crop, out_img)
                    sitk.WriteImage(lab_crop, out_lab)

                    frames_saved += 1
                    saved_this_subject += 1

                subjects_processed += 1
                print(f"Processed: {phase}/{name} -> saved {saved_this_subject} frame(s) "
                      f"as {name}_0(.nii.gz){' and ' + name + '_1(.nii.gz)' if saved_this_subject > 1 else ''}")

                # Warn if only one labeled frame was present
                if len(selected_frames) == 1:
                    print(f"Warning: Only one labeled frame found for {phase}/{name}; saved _0 only.")

            except Exception as e:
                print(f"Error processing {phase}/{name}: {str(e)}")
                subjects_skipped += 1

    print("\nMnM processing complete!")
    print(f"Subjects processed: {subjects_processed}")
    print(f"Frames saved: {frames_saved}")
    print(f"Subjects skipped: {subjects_skipped}")
    print(f"Output saved to: {output_dir}")


def process_brats2018_dataset(input_dir: str, output_dir: str):
    """
    Process the BraTS 2018 dataset (HGG/LGG):
      - For each case, read 4 modalities (flair, t1, t2, t1ce) and seg.
      - Reorient all to RAS.
      - Compute a single crop ROI from the (reoriented) segmentation foreground.
      - Apply the same ROI to all modalities + seg (no resampling).
      - Save results into four modality-specific folders:
            brats18_flair, brats18_t1, brats18_t2, brats18_t1ce
        with filenames:
            {CaseID}.nii.gz          (image)
            {CaseID}_gt.nii.gz       (label)
    Args:
        input_dir (str): Root directory containing 'HGG' and 'LGG' subfolders.
        output_dir (str): Output root directory.
    Notes:
        - Uses your existing helpers: reorient_image_and_label, crop logic mimics crop_image_and_label,
          but ROI is computed once from the label and applied to all modalities for consistency.
        - No resampling is performed.
    """


    target_orientation = 'RAS'
    # A conservative margin for brain MR (mm). Adjust if you prefer tighter/looser crops.
    crop_margin_mm = [80, 80, 40]

    modalities = ["flair", "t1", "t2", "t1ce"]
    modality_to_out = {
        "flair": os.path.join(output_dir, "brats18_flair"),
        "t1":    os.path.join(output_dir, "brats18_t1"),
        "t2":    os.path.join(output_dir, "brats18_t2"),
        "t1ce":  os.path.join(output_dir, "brats18_t1ce"),
    }
    for p in modality_to_out.values():
        os.makedirs(p, exist_ok=True)

    def _compute_roi_from_label_mm(lab_img_ras: sitk.Image, margin_mm):
        """Return (index, size) for RegionOfInterest based on label foreground with mm margins."""
        lab_arr = sitk.GetArrayFromImage(lab_img_ras)  # z, y, x
        nz = np.where(lab_arr > 0)
        if len(nz[0]) == 0:
            raise ValueError("Label contains no foreground voxels")

        min_z, max_z = int(nz[0].min()), int(nz[0].max())
        min_y, max_y = int(nz[1].min()), int(nz[1].max())
        min_x, max_x = int(nz[2].min()), int(nz[2].max())

        lower = [min_x, min_y, min_z]
        upper = [max_x, max_y, max_z]

        spacing = lab_img_ras.GetSpacing()  # (sx, sy, sz) in mm
        margin_vox = [int(np.ceil(margin_mm[i] / spacing[i])) for i in range(3)]

        size = list(lab_img_ras.GetSize())  # (x, y, z)
        crop_lower = []
        crop_upper = []
        for i in range(3):
            lo = max(0, lower[i] - margin_vox[i])
            up = min(size[i] - 1, upper[i] + margin_vox[i])
            crop_lower.append(lo)
            crop_upper.append(up)

        crop_size = [int(crop_upper[i] - crop_lower[i] + 1) for i in range(3)]
        return crop_lower, crop_size

    def _save_pair(img: sitk.Image, lab: sitk.Image, out_img_path: str, out_lab_path: str):
        sitk.WriteImage(img, out_img_path)
        sitk.WriteImage(lab, out_lab_path)

    total_cases = 0
    skipped_cases = 0
    saved_counts = {m: 0 for m in modalities}

    for grade_folder in ["HGG", "LGG"]:
        grade_path = os.path.join(input_dir, grade_folder)
        if not os.path.isdir(grade_path):
            print(f"Warning: grade folder not found: {grade_path} (skipping)")
            continue

        case_ids = sorted([d for d in os.listdir(grade_path)
                           if os.path.isdir(os.path.join(grade_path, d))])

        for case_id in case_ids:
            case_dir = os.path.join(grade_path, case_id)

            img_paths = {m: os.path.join(case_dir, f"{case_id}_{m}.nii.gz") for m in modalities}
            seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")

            if not os.path.exists(seg_path):
                print(f"Warning: seg not found for {grade_folder}/{case_id}, skipping case.")
                skipped_cases += 1
                continue

            available_modalities = [m for m, p in img_paths.items() if os.path.exists(p)]
            if len(available_modalities) == 0:
                print(f"Warning: no modalities found for {grade_folder}/{case_id}, skipping case.")
                skipped_cases += 1
                continue

            try:
                # Reorient seg once, compute ROI, then apply the same ROI to each modality.
                first_mod = available_modalities[0]
                img_ref = sitk.ReadImage(img_paths[first_mod])
                lab = sitk.ReadImage(seg_path)

                # (Direction/origin/spacing alignment assumed across modalities in BraTS.)
                _, lab_ras = reorient_image_and_label(img_ref, lab, target_orientation)

                roi_index, roi_size = _compute_roi_from_label_mm(lab_ras, crop_margin_mm)

                saved_any = False
                for m in modalities:
                    img_path = img_paths[m]
                    if not os.path.exists(img_path):
                        print(f"Note: {m} missing for {grade_folder}/{case_id}, skipping this modality.")
                        continue

                    img = sitk.ReadImage(img_path)
                    img_ras, lab_ras_m = reorient_image_and_label(img, lab, target_orientation)

                    # We reoriented seg again alongside this modality to preserve identical orientation metadata.
                    # Crop with the same ROI for all modalities.
                    img_crop = sitk.RegionOfInterest(img_ras, size=roi_size, index=roi_index)
                    lab_crop = sitk.RegionOfInterest(lab_ras_m, size=roi_size, index=roi_index)

                    out_folder = modality_to_out[m]
                    out_img = os.path.join(out_folder, f"{case_id}.nii.gz")
                    out_lab = os.path.join(out_folder, f"{case_id}_gt.nii.gz")
                    _save_pair(img_crop, lab_crop, out_img, out_lab)

                    saved_counts[m] += 1
                    saved_any = True

                if saved_any:
                    total_cases += 1
                    print(f"Processed: {grade_folder}/{case_id}")
                else:
                    skipped_cases += 1
                    print(f"Warning: no modalities saved for {grade_folder}/{case_id} (all missing).")

            except Exception as e:
                skipped_cases += 1
                print(f"Error processing {grade_folder}/{case_id}: {str(e)}")

    print("\nBraTS 2018 processing complete!")
    print(f"Total cases processed: {total_cases}")
    print(f"Total cases skipped:   {skipped_cases}")
    for m in modalities:
        print(f"Saved in {m:5s}: {saved_counts[m]}")
    print(f"Outputs written to:")
    for m, p in modality_to_out.items():
        print(f"  - {m:5s}: {p}")




# ---------- Utilities specific to TotalSegmentator ----------

CT_TOTALSEG_CLASSES = [
    'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver', 'stomach',
    'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein', 'pancreas',
    'adrenal_gland_right', 'adrenal_gland_left', 'lung_upper_lobe_left',
    'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right',
    'lung_lower_lobe_right', 'kidney_cyst_left', 'kidney_cyst_right',
    'esophagus', 'trachea', 'thyroid_gland', 'small_bowel', 'duodenum',
    'colon', 'urinary_bladder', 'prostate', 'sacrum', 'vertebrae_L5',
    'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1',
    'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9',
    'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5',
    'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1',
    'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4',
    'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1', 'vertebrae_S1',
    'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5',
    'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10',
    'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2',
    'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6',
    'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10',
    'rib_right_11', 'rib_right_12', 'clavicula_left', 'clavicula_right',
    'scapula_left', 'scapula_right', 'sternum', 'costal_cartilages',
    'humerus_left', 'humerus_right', 'heart', 'pulmonary_vein',
    'brachiocephalic_trunk', 'subclavian_artery_right',
    'subclavian_artery_left', 'common_carotid_artery_right',
    'common_carotid_artery_left', 'brachiocephalic_vein_left',
    'brachiocephalic_vein_right', 'atrial_appendage_left',
    'superior_vena_cava', 'brain', 'skull', 'femur_left', 'femur_right',
    'hip_left', 'hip_right', 'spinal_cord', 'gluteus_maximus_left',
    'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right',
    'gluteus_minimus_left', 'gluteus_minimus_right', 'autochthon_left',
    'autochthon_right', 'iliopsoas_left', 'iliopsoas_right',
    'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left',
    'iliac_vena_right',
]

MRI_TOTALSEG_CLASSES = [
    'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver',
    'stomach', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left',
    'lung_left', 'lung_right', 'esophagus', 'small_bowel', 'duodenum',
    'colon', 'urinary_bladder', 'prostate', 'sacrum', 'vertebrae',
    'intervertebral_discs', 'spinal_cord', 'heart', 'aorta',
    'inferior_vena_cava', 'portal_vein_and_splenic_vein',
    'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left',
    'iliac_vena_right', 'humerus_left', 'humerus_right', 'scapula_left',
    'scapula_right', 'clavicula_left', 'clavicula_right', 'femur_left',
    'femur_right', 'hip_left', 'hip_right', 'gluteus_maximus_left',
    'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right',
    'gluteus_minimus_left', 'gluteus_minimus_right', 'autochthon_left',
    'autochthon_right', 'iliopsoas_left', 'iliopsoas_right', 'brain',
]

TOTALSEG_CLASS_MAPPINGS = {
    'CT': {name: idx + 1 for idx, name in enumerate(CT_TOTALSEG_CLASSES)},
    'MRI': {name: idx + 1 for idx, name in enumerate(MRI_TOTALSEG_CLASSES)},
}


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    for suffix in ('.nii.gz', '.nii'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def load_and_combine_totalseg_masks(
    segmentation_dir: Path,
    class_mapping: Dict[str, int],
    reference_image: sitk.Image,
    verbose: bool = False,
) -> sitk.Image:
    """
    Combine TotalSegmentator's per-class binary masks into one compact label map.
    """
    combined_array = np.zeros(reference_image.GetSize()[::-1], dtype=np.uint8)

    for mask_file in sorted(segmentation_dir.glob("*.nii.gz")):
        class_name = _strip_nii_suffix(mask_file)
        if class_name not in class_mapping:
            continue

        try:
            mask_img = safe_read_image(mask_file, verbose=verbose)
            if not has_same_geometry(mask_img, reference_image):
                mask_img = _resample_like(mask_img, reference_image, is_label=True)

            mask_array = sitk.GetArrayFromImage(mask_img)
            combined_array[mask_array > 0] = class_mapping[class_name]
        except Exception as exc:
            logger.warning(f"Failed to load TotalSegmentator mask {mask_file}: {exc}")

    combined_img = sitk.GetImageFromArray(combined_array)
    combined_img.CopyInformation(reference_image)
    return combined_img


def process_totalsegmentator_case(
    case_dir: Path,
    output_dir: Path,
    modality: str,
    class_mapping: Dict[str, int],
    crop_body: bool = True,
    verbose: bool = False,
    body_segmenter=None,
) -> bool:
    """
    Standardize one TotalSegmentator case into ``<case>.nii.gz`` and ``<case>_gt.nii.gz``.
    """
    case_name = case_dir.name
    image_filename = "ct.nii.gz" if modality == "CT" else "mri.nii.gz"
    image_path = case_dir / image_filename
    segmentation_dir = case_dir / "segmentations"

    if not image_path.exists():
        logger.warning(f"Image not found for {case_name}: {image_path}")
        return False
    if not segmentation_dir.exists():
        logger.warning(f"Segmentation directory not found for {case_name}: {segmentation_dir}")
        return False

    try:
        image = safe_read_image(image_path, verbose=verbose)
        label = load_and_combine_totalseg_masks(
            segmentation_dir,
            class_mapping,
            image,
            verbose=verbose,
        )

        if crop_body:
            if body_segmenter is None:
                body_segmenter = BodyMaskSegmenter(fast=True, num_threads=1)
            body_mask = body_segmenter.segment_body_from_path(str(image_path))
            if not has_same_geometry(body_mask, image):
                body_mask = _resample_like(body_mask, image, is_label=True)
            image, label = crop_images_by_mask([image, label], body_mask, margin=0, margin_unit="pixel")

        image, label = reorient_image_and_label(image, label, 'RAS')

        sitk.WriteImage(image, str(output_dir / f"{case_name}.nii.gz"))
        sitk.WriteImage(label, str(output_dir / f"{case_name}_gt.nii.gz"))
        return True
    except Exception as exc:
        logger.error(f"Error processing TotalSegmentator case {case_name}: {exc}")
        return False


def process_totalsegmentator_dataset(
    input_dir: str,
    output_dir: str,
    modality: str,
    crop_body: bool = True,
    cases: Optional[List[str]] = None,
    max_cases: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """
    Process the TotalSegmentator dataset from per-class masks to MASS NIfTI pairs.
    """
    modality = modality.upper()
    if modality not in TOTALSEG_CLASS_MAPPINGS:
        raise ValueError("TotalSegmentator modality must be 'CT' or 'MRI'")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if cases:
        case_dirs = [input_path / case for case in cases]
        missing = [case.name for case in case_dirs if not case.exists()]
        if missing:
            logger.warning(f"Skipping missing TotalSegmentator cases: {missing}")
        case_dirs = [case for case in case_dirs if case.exists()]
    else:
        case_dirs = sorted(case for case in input_path.iterdir() if case.is_dir() and case.name.startswith('s'))

    if max_cases is not None:
        case_dirs = case_dirs[:max_cases]
    if len(case_dirs) == 0:
        raise ValueError(f"No TotalSegmentator cases found in {input_path}")

    class_mapping = TOTALSEG_CLASS_MAPPINGS[modality]
    body_segmenter = BodyMaskSegmenter(fast=True, num_threads=1) if crop_body else None

    successful = 0
    failed = 0
    for case_dir in tqdm(case_dirs, desc="Processing TotalSegmentator cases"):
        if process_totalsegmentator_case(
            case_dir,
            output_path,
            modality,
            class_mapping,
            crop_body=crop_body,
            verbose=verbose,
            body_segmenter=body_segmenter,
        ):
            successful += 1
        else:
            failed += 1

    logger.info("TotalSegmentator processing complete")
    logger.info(f"Successful: {successful} cases")
    logger.info(f"Failed: {failed} cases")
    logger.info(f"Output saved to: {output_path}")


# ---------- Utilities specific to AutoPET ----------


def _which(exe: str) -> Optional[str]:
    from shutil import which
    return which(exe)

class BodyMaskSegmenter:
    """
    Wrap TotalSegmentator 'body' task so we can reuse one object and
    degrade gracefully across environments:

      1) Try official Python API:  from totalsegmentator.python_api import totalsegmentator
      2) Try v2 alias:             from totalsegmentatorv2.python_api import totalsegmentator
      3) Fallback to CLI:          TotalSegmentator -i <ct> -o <out> -ta body [--fast]

    Notes:
      - Python API keeps everything in-process (faster I/O, no shell).
      - CLI fallback cannot keep weights in RAM across cases.
    """
    def __init__(self, fast: bool = True, num_threads: int = 1):
        self.fast = bool(fast)
        self.num_threads = int(max(1, num_threads))
        self._api_callable = None
        self._api_name = None

        # 1) Official import path (documented in README)
        # https://github.com/wasserth/TotalSegmentator#python-api
        try:
            from totalsegmentator.python_api import totalsegmentator as _ts  # type: ignore
            self._api_callable = _ts
            self._api_name = "totalsegmentator.python_api"
        except Exception:
            # 2) Some installs expose a v2 namespace
            try:
                from totalsegmentatorv2.python_api import totalsegmentator as _ts  # type: ignore
                self._api_callable = _ts
                self._api_name = "totalsegmentatorv2.python_api"
            except Exception:
                self._api_callable = None
                self._api_name = None

        # Detect CLI if needed
        self._cli = _which("TotalSegmentator") or _which("totalsegmentator")  # second name just in case

        if (self._api_callable is None) and (self._cli is None):
            # Give a precise, actionable message (but do not crash constructor)
            print(
                "[BodyMaskSegmenter] Neither Python API nor CLI found. "
                "Install via `pip install TotalSegmentator` and ensure the console script "
                "`TotalSegmentator` is on PATH, or run inside the environment where it was installed. "
                "PyTorch>=2.0,<2.6 is required (Windows <2.4).",  # per README
                file=sys.stderr
            )

    def _run_python_api(self, ct_path: str, out_dir: str) -> None:
        """
        Run the Python API. Accept both file-path calling convention and
        handle API variants around task/fast keyword names.
        """
        assert self._api_callable is not None, "Python API not available"

        kwargs = dict(task="body")
        if self.fast:
            kwargs["fast"] = True

        try:
            self._api_callable(ct_path, out_dir, **kwargs)
            return
        except TypeError:
            # Retry without optional speed flags when the installed API is stricter.
            kwargs.pop("fast", None)
            self._api_callable(ct_path, out_dir, **kwargs)

    def _run_cli(self, ct_path: str, out_dir: str) -> None:
        """
        Call the CLI as documented in the README:
          TotalSegmentator -i ct.nii.gz -o seg -ta body [--fast]
        """
        if self._cli is None:
            raise RuntimeError(
                "TotalSegmentator CLI not found on PATH. Ensure `TotalSegmentator` is installed "
                "in this environment and the script directory is on PATH."
            )

        cmd = [self._cli, "-i", ct_path, "-o", out_dir, "-ta", "body"]
        if self.fast:
            cmd.append("--fast")

        # Thread env for a bit more control on CPU
        env = os.environ.copy()
        # ITK & OpenMP threads are common bottlenecks on CPU
        env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(self.num_threads))
        env.setdefault("OMP_NUM_THREADS", str(self.num_threads))

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"TotalSegmentator CLI failed (exit {res.returncode}).\n"
                f"Command: {' '.join(cmd)}\n--- STDOUT ---\n{res.stdout}\n--- STDERR ---\n{res.stderr}"
            )

    def segment_body_from_path(self, ct_path: str, tmp_root: Optional[str] = None) -> sitk.Image:
        """
        Run 'body' task on CT at ct_path and return a SimpleITK binary mask aligned to the CT.
        """
        tmp_dir = tempfile.mkdtemp(dir=tmp_root) if tmp_root else tempfile.mkdtemp()
        out_dir = os.path.join(tmp_dir, "ts_out")
        os.makedirs(out_dir, exist_ok=True)

        try:
            if self._api_callable is not None:
                self._run_python_api(ct_path, out_dir)
            else:
                self._run_cli(ct_path, out_dir)

            # Pick the "body" file from the subtask outputs
            # (the subtask produces body, body_trunc, body_extremities, skin)
            # https://github.com/wasserth/TotalSegmentator#subtasks
            hits = glob.glob(os.path.join(out_dir, "body.nii.gz"))
            if not hits:
                # be generous in case of slightly different naming
                hits = glob.glob(os.path.join(out_dir, "*body*.nii.gz"))
            if not hits:
                raise FileNotFoundError("TotalSegmentator did not produce a body mask file.")

            mask_img = safe_read_image(Path(hits[0]))
            arr = sitk.GetArrayFromImage(mask_img)
            arr = (arr > 0).astype("uint8")
            out = sitk.GetImageFromArray(arr)
            out.CopyInformation(mask_img)
            return out

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_study_date(folder_name: str) -> Optional[datetime]:
    """
    Extract a date from AutoPET study folder names like '09-22-2005-NA-PET-CT ...'.
    Returns a datetime or None if a date isn't found.
    """
    m = re.search(r'(\d{2})-(\d{2})-(\d{4})', folder_name)
    if not m:
        return None
    mm, dd, yyyy = m.groups()
    try:
        return datetime(int(yyyy), int(mm), int(dd))
    except Exception:
        return None

def _list_study_dirs(patient_dir: str) -> List[str]:
    """
    Return study directories that contain CTres.nii.gz, sorted by study date if parseable,
    otherwise by folder name.
    """
    candidates = []
    for d in os.listdir(patient_dir):
        p = os.path.join(patient_dir, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "CTres.nii.gz")):
            candidates.append(p)

    def _key(path):
        d = os.path.basename(path)
        dt = _parse_study_date(d)
        return (0, dt) if dt else (1, d)  # all with date come first, then name
    candidates.sort(key=_key)
    return candidates

def _resample_like(moving: sitk.Image, reference: sitk.Image, is_label: bool = True) -> sitk.Image:
    """
    Resample 'moving' to the grid of 'reference' (identity transform).
    """
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(reference)
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    res.SetTransform(sitk.Transform())
    return res.Execute(moving)

# ---------- Main pipeline ----------

def process_autopet_dataset(
    input_dir: str,
    output_dir: str,
    fast: bool = True,
    body_margin_mm: Tuple[int, int, int] = (5, 5, 5),
    num_threads: int = 1
) -> None:
    """
    Process the AutoPET dataset.

    Pipeline
    --------
    1) Read CTres.nii.gz, SUV.nii.gz, SEG.nii.gz
    2) Run TotalSegmentator 'body' on CTres to get a body mask
    3) Reorient all to RAS (using your reorient_image_and_label)
    4) Crop CT, SUV, SEG using the *body mask* (using your crop_image_and_label)
    5) Save:
        - CT + SEG   -> output_dir + "_ct"
        - SUV + SEG  -> output_dir + "_suv"
       Filenames: {patient_id}_{k}.nii.gz (image), {patient_id}_{k}_gt.nii.gz (label)
       where k is the study index (0,1,...) within each patient.

    Args
    ----
    input_dir : str
        Root directory containing patient folders 'PETCT_*'
    output_dir : str
        Output root. Two subfolders will be created: '<output_dir>_ct' and '<output_dir>_suv'
    fast : bool
        Pass-through to TotalSegmentator's 'fast' mode.
    body_margin_mm : (int,int,int)
        [x,y,z] margin (in mm) added around the body mask when cropping.
    num_threads : int
        Number of threads for TotalSegmentator (if supported by your version).

    Notes
    -----
    - Uses your helpers: safe_read_image, reorient_image_and_label, crop_image_and_label.
    - We DO NOT crop based on disease foreground; we crop using the TotalSegmentator body mask.
    - CTres and SUV in AutoPET should share grid/geometry. We still verify and resample the
      body mask to SUV's geometry if needed to guarantee identical crops.
    """
    out_ct_dir = output_dir + "_ct"
    out_suv_dir = output_dir + "_suv"
    os.makedirs(out_ct_dir, exist_ok=True)
    os.makedirs(out_suv_dir, exist_ok=True)

    segmenter = BodyMaskSegmenter(fast=fast, num_threads=num_threads)

    patient_dirs = sorted(
        [p for p in glob.glob(os.path.join(input_dir, "PETCT_*")) if os.path.isdir(p)]
    )
    if len(patient_dirs) == 0:
        raise ValueError(f"No 'PETCT_*' patient folders found under {input_dir}")

    processed = 0
    skipped = 0

    for patient_path in patient_dirs:
        patient_id = os.path.basename(patient_path)

        study_dirs = _list_study_dirs(patient_path)
        if len(study_dirs) == 0:
            print(f"Warning: no study folders with CTres for {patient_id}, skipping patient.")
            continue

        for study_idx, study_dir in enumerate(study_dirs):
            ctres_path = os.path.join(study_dir, "CTres.nii.gz")
            suv_path   = os.path.join(study_dir, "SUV.nii.gz")
            seg_path   = os.path.join(study_dir, "SEG.nii.gz")

            missing = [p for p in (ctres_path, suv_path, seg_path) if not os.path.exists(p)]
            if missing:
                print(f"Warning: missing files in {patient_id}/{os.path.basename(study_dir)} -> {missing}, skipping this study.")
                skipped += 1
                continue

            try:
                # ---- 1) Load images
                ct_itk  = safe_read_image(Path(ctres_path))
                suv_itk = safe_read_image(Path(suv_path))
                seg_itk = safe_read_image(Path(seg_path))

                # ---- 2) Body mask via TotalSegmentator (from file path for max compatibility)
                body_mask = segmenter.segment_body_from_path(ctres_path)

                # ---- 3) Reorient all to RAS
                # Keep body mask aligned with its source (CTres)
                ct_ras, body_ras = reorient_image_and_label(ct_itk, body_mask, 'RAS')
                # Keep SEG aligned with SUV (both are at PET grid); this ensures exact pairing
                suv_ras, seg_ras = reorient_image_and_label(suv_itk, seg_itk, 'RAS')

                # If body mask (from CTres) doesn't exactly match SUV geometry after reorient, resample it
                if (suv_ras.GetSize() != body_ras.GetSize() or
                    suv_ras.GetSpacing() != body_ras.GetSpacing() or
                    suv_ras.GetDirection() != body_ras.GetDirection()):
                    body_ras_for_suv = _resample_like(body_ras, suv_ras, is_label=True)
                else:
                    body_ras_for_suv = body_ras

                # ---- 4) Crop with body mask (same ROI across all)
                ct_crop, _ = crop_image_and_label(ct_ras, body_ras, margin=list(body_margin_mm), margin_unit="mm")

                suv_crop, _ = crop_image_and_label(suv_ras, body_ras_for_suv, margin=list(body_margin_mm), margin_unit="mm")

                seg_crop, _ = crop_image_and_label(seg_ras, body_ras_for_suv, margin=list(body_margin_mm), margin_unit="mm")

                # ---- 5) Save with {patient_id}_{k} convention
                base = f"{patient_id}_{study_idx}"

                sitk.WriteImage(ct_crop, os.path.join(out_ct_dir,  f"{base}.nii.gz"))
                sitk.WriteImage(seg_crop, os.path.join(out_ct_dir,  f"{base}_gt.nii.gz"))

                sitk.WriteImage(suv_crop, os.path.join(out_suv_dir, f"{base}.nii.gz"))
                sitk.WriteImage(seg_crop, os.path.join(out_suv_dir, f"{base}_gt.nii.gz"))

                processed += 1
                print(f"Processed: {patient_id} | study {study_idx} -> {base}.nii.gz")

            except Exception as e:
                print(f"Error processing {patient_id} | study {study_idx}: {e}")
                skipped += 1
                continue

    print("\nAutoPET processing complete!")
    print(f"Studies processed: {processed}")
    print(f"Studies skipped:   {skipped}")
    print(f"CT   saved to: {out_ct_dir}")
    print(f"SUV  saved to: {out_suv_dir}")


if __name__ == "__main__":
    DATASET_PROCESSORS = {
        'bcv':                  process_bcv_dataset,
        'amos':                 process_amos_dataset,
        'kits':                 process_kits_dataset,
        'lits':                 process_lits_dataset,
        'structseg_head_oar':   process_structseg_head_oar_dataset,
        'structseg_thoracic':   process_structseg_thoracic_oar_dataset,
        'ctpelvic1k':           process_ctpelvic1k_dataset,
        'mnm':                  process_mnm_dataset,
        'brats':                process_brats2018_dataset,
        'autopet':              process_autopet_dataset,
        'totalsegmentator':     process_totalsegmentator_dataset,
    }

    parser = argparse.ArgumentParser(
        description='Preprocess medical image datasets for MASS mask generation.'
    )
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_PROCESSORS.keys(),
                        help='Dataset to process. Choices: ' + ', '.join(DATASET_PROCESSORS.keys()))
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing the raw dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--modality', type=str, choices=['CT', 'MRI'],
                        help='Required for --dataset totalsegmentator')
    parser.add_argument('--no-crop', action='store_true',
                        help='For TotalSegmentator: skip body-region cropping')
    parser.add_argument('--cases', nargs='+',
                        help='For TotalSegmentator: process only these case folders')
    parser.add_argument('--max-cases', type=int,
                        help='For TotalSegmentator: process at most this many cases')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed orientation fallback warnings')

    args = parser.parse_args()
    if args.dataset == 'totalsegmentator':
        if args.modality is None:
            parser.error("--modality is required when --dataset totalsegmentator")
        process_totalsegmentator_dataset(
            args.input_dir,
            args.output_dir,
            modality=args.modality,
            crop_body=not args.no_crop,
            cases=args.cases,
            max_cases=args.max_cases,
            verbose=args.verbose,
        )
    else:
        DATASET_PROCESSORS[args.dataset](args.input_dir, args.output_dir)
    
