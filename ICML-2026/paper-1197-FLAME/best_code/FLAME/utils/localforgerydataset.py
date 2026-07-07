"""Dataset helpers for assembling localized forgery training samples."""

from __future__ import annotations

import io
import logging
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sam2.utils.transforms import SAM2Transforms
from torch.utils.data import Dataset

from .perturbations import apply_blur_to_image


logger = logging.getLogger(__name__)

class LocalForgeryDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        img_size: int = 256,
        global_img_size: int = 512,
        allow_multiple_targets: bool = False,
        is_training: bool = True,
        return_pil: bool = False,
        force_resize: bool = False,
        # AUTHENTIC IMAGE PARAMETERS
        authentic_ratio: float = 0.0,  # Ratio of authentic images to include (0.0 to 1.0)
        authentic_source_dir: str = None,  # Optional; authentic images are loaded from root/source
        # DATA SAMPLING PARAMETERS
        sample_ratio: float = 1.0,  # Ratio of samples to use from this dataset (0.0 to 1.0)
        # MASK CREATION PARAMETERS
        use_diff: bool = False,               # Whether to use difference-based masking
    ):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory containing 'source', 'target', and 'mask' folders.
            img_size (int): Size to which images and masks will be resized.
            allow_multiple_targets (bool): Whether to handle multiple targets per source.
            is_training (bool): If True, dataset is in training mode (random crops); if False, in evaluation mode (center crops).
            return_pil (bool): If True, returns PIL image for external processing (e.g., TruFor).
            force_resize (bool): If True, resize to img_size instead of cropping in training.
            authentic_ratio (float): Ratio of authentic images to include (0.0 to 1.0)
            authentic_source_dir (str): Optional; authentic images always use the dataset root/source folder
            sample_ratio (float): Ratio of samples to use from this dataset (0.0 to 1.0)
            use_diff (bool): Whether to use difference-based masking in create_combined_mask
        """
        self.root = root_dir
        self.img_size = img_size
        self.global_img_size = int(global_img_size)
        self.allow_multiple_targets = allow_multiple_targets
        self.is_training = is_training
        self.return_pil = return_pil
        self.dataset_name = os.path.basename(root_dir)
        self.force_resize = force_resize
        
        # AUTHENTIC IMAGE PARAMETERS
        self.authentic_ratio = authentic_ratio
        self.authentic_source_dir = authentic_source_dir
        
        # DATA SAMPLING PARAMETERS
        self.sample_ratio = sample_ratio
        
        # MASK CREATION PARAMETERS
        self.use_diff = use_diff

        # NOTE: Do not instantiate SAM2Transforms here. DataLoader with `spawn` must pickle the dataset,
        # and SAM2Transforms may hold TorchScript modules that are not picklable. We lazily create them
        # per-worker on first __getitem__ call.
        self._local_transforms = None
        self._global_transforms = None

        # Discover forgery samples
        self._discover_forgery_samples()
        
        # Discover authentic samples if enabled
        self.authentic_samples = []
        if self.authentic_ratio > 0.0:
            self._discover_authentic_samples()
        
        # Combine and shuffle samples
        self._combine_samples()

    def _compute_perturbation_params(self):
        """Compute specific perturbation parameters based on intensity (0-1.5)"""
        intensity = self.perturbation_intensity
        
        if self.perturbation_type == "gaussian_blur":
            # Map 0-1.5 to blur sigma 0-3.0
            self.blur_sigma = intensity
        elif self.perturbation_type == "jpeg_compression":
            # Map 0-1.5 to JPEG quality 95-10 (higher intensity = lower quality = more compression)
            # At intensity 0: quality=95 (minimal compression)
            # At intensity 1.5: quality=10 (heavy compression)
            self.jpeg_quality = max(10, int(95 - (intensity * 56.67)))
        elif self.perturbation_type == "gaussian_noise":
            # Map 0-1.5 to noise std 0-0.3
            self.noise_std = intensity * 0.2
        elif self.perturbation_type == "none":
            # No perturbation parameters needed
            pass
        elif self.perturbation_type == "gaussian_blur/gaussian_noise":
            # Combined perturbation: apply both blur and noise
            self.blur_sigma = intensity
            self.noise_std = intensity * 0.2
        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation_type}")

    def create_combined_mask(
        self,
        src_img_np,
        tgt_img_np,
        mask_rgba,
        threshold=5,
        allow_multiple_targets=False,
        use_diff=True,
    ):
        """
        Creates a combined mask using difference and non-literal masks.
        """
        array_is_rgba = mask_rgba.ndim == 3 and mask_rgba.shape[2] == 4
        if array_is_rgba and mask_rgba[:, :, 3].sum() == (
            mask_rgba.shape[0] * mask_rgba.shape[1] * 255
        ):
            non_literal_mask = (mask_rgba[:, :, 0] > 0).astype(np.uint8)
        elif mask_rgba.ndim == 3 and mask_rgba.shape[2] == 4:
            alpha = mask_rgba[:, :, 3]
            _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
            non_literal_mask = 1 - (binary // 255).astype(np.uint8)
        elif mask_rgba.ndim == 2:
            non_literal_mask = (mask_rgba // 255).astype(np.uint8)
        elif allow_multiple_targets and mask_rgba.ndim == 3 and mask_rgba.shape[2] == 4:
            non_literal_mask = (mask_rgba[:, :, 0] > 0).astype(np.uint8)
        elif (
            mask_rgba.ndim == 3
            and mask_rgba[:, :, 0].max() == 255
            and np.all(mask_rgba[:, :, 1] == mask_rgba[:, :, 0])
            and np.all(mask_rgba[:, :, 2] == mask_rgba[:, :, 0])
        ):
            non_literal_mask = (mask_rgba[:, :, 0] > 0).astype(np.uint8)
        else:
            non_literal_mask = (mask_rgba[:, :, 0] > 0).astype(np.uint8)

        if not use_diff:
            return non_literal_mask.astype(np.uint8)

        diff_mask = cv2.absdiff(tgt_img_np, src_img_np)
        if diff_mask.sum() > 0.0:
            if diff_mask.ndim == 3:
                diff_gray = cv2.cvtColor(diff_mask, cv2.COLOR_RGB2GRAY)
            else:
                diff_gray = diff_mask
            _, binary_diff = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
            binary_diff = (binary_diff > 0).astype(np.uint8)
        else:
            binary_diff = np.ones_like(tgt_img_np[:, :, 0], dtype=np.uint8)

        if binary_diff.shape != non_literal_mask.shape[:2]:
            binary_diff = cv2.resize(
                binary_diff,
                (non_literal_mask.shape[1], non_literal_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return np.logical_and(binary_diff, non_literal_mask).astype(np.uint8)

    def __len__(self):
        return len(self.all_samples)

    def _balanced_crop(self, src_img_pil, tgt_img_pil, mask_pil, is_training=True, return_coords: bool = False):
        """
        Balanced crop to a fixed square size.

        Training:
          - 50% probability: crop centered at forgery mask center (if mask exists)
          - 50% probability: random crop
          - For small images: reflect padding (cv2.BORDER_REFLECT_101) to enable cropping

        Eval:
          - Center crop (deterministic)
        
        Args:
            src_img_pil: Source image PIL object (can be None)
            tgt_img_pil: Target image PIL object
            mask_pil: Mask image PIL object
            is_training: Whether to use random cropping (True) or center cropping (False)
            
        Returns:
            tuple: (src_img_pil, tgt_img_pil, mask_pil) after cropping
                If return_coords=True, also returns a tensor norm_coords=[x1,y1,x2,y2]
                in original image normalized coordinates.
        """
        import random
        crop_size = int(self.img_size)

        # Ensure src/mask are aligned with target size
        orig_w, orig_h = tgt_img_pil.size
        if src_img_pil is not None and src_img_pil.size != (orig_w, orig_h):
            src_img_pil = src_img_pil.resize((orig_w, orig_h), Image.BILINEAR)
        if mask_pil.size != (orig_w, orig_h):
            mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)

        # Reflect pad to allow fixed-size crop
        pad_w = max(0, crop_size - orig_w)
        pad_h = max(0, crop_size - orig_h)
        left = right = top = bottom = 0
        if pad_w > 0 or pad_h > 0:
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top

            def _reflect_pad_pil(img: Image.Image) -> Image.Image:
                img_np = np.array(img)
                padded = cv2.copyMakeBorder(
                    img_np,
                    top,
                    bottom,
                    left,
                    right,
                    borderType=cv2.BORDER_REFLECT_101,
                )
                return Image.fromarray(padded)

            tgt_img_pil = _reflect_pad_pil(tgt_img_pil)
            mask_pil = _reflect_pad_pil(mask_pil)
            if src_img_pil is not None:
                src_img_pil = _reflect_pad_pil(src_img_pil)
            else:
                src_img_pil = None

        padded_w, padded_h = tgt_img_pil.size
        max_x = max(0, padded_w - crop_size)
        max_y = max(0, padded_h - crop_size)

        # Compute crop location
        if not is_training:
            crop_x = max_x // 2
            crop_y = max_y // 2
        else:
            use_mask_center = random.random() < 0.5
            crop_x = random.randint(0, max_x) if max_x > 0 else 0
            crop_y = random.randint(0, max_y) if max_y > 0 else 0

            if use_mask_center:
                mask_rgba = np.array(mask_pil)
                src_np = np.array(src_img_pil) if src_img_pil is not None else np.array(tgt_img_pil)
                tgt_np = np.array(tgt_img_pil)
                non_literal = self.create_combined_mask(
                    src_img_np=src_np,
                    tgt_img_np=tgt_np,
                    mask_rgba=mask_rgba,
                    allow_multiple_targets=self.allow_multiple_targets,
                    use_diff=False,
                )
                ys, xs = np.nonzero(non_literal)
                if xs.size > 0 and ys.size > 0:
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    crop_x = int(np.clip(cx - crop_size // 2, 0, max_x))
                    crop_y = int(np.clip(cy - crop_size // 2, 0, max_y))

        # Apply crop
        box = (crop_x, crop_y, crop_x + crop_size, crop_y + crop_size)
        tgt_img_pil = tgt_img_pil.crop(box)
        mask_pil = mask_pil.crop(box)
        if src_img_pil is not None:
            src_img_pil = src_img_pil.crop(box)
        else:
            src_img_pil = tgt_img_pil

        if not return_coords:
            return src_img_pil, tgt_img_pil, mask_pil

        # Map crop box from padded image coords back to original image coords and normalize.
        x1 = crop_x - left
        y1 = crop_y - top
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        x1c = float(max(0, x1))
        y1c = float(max(0, y1))
        x2c = float(min(orig_w, x2))
        y2c = float(min(orig_h, y2))

        norm_coords = torch.tensor(
            [
                x1c / float(max(orig_w, 1)),
                y1c / float(max(orig_h, 1)),
                x2c / float(max(orig_w, 1)),
                y2c / float(max(orig_h, 1)),
            ],
            dtype=torch.float32,
        )
        return src_img_pil, tgt_img_pil, mask_pil, norm_coords

    def _handle_image_sizing(
        self, src_img_pil, tgt_img_pil, mask_pil, is_training=True, return_coords: bool = False
    ):
        """
        Handle sizing for images:
          - training: balanced crop to fixed square (self.img_size) unless force_resize is enabled
          - eval: resize to fixed square (self.img_size)
        """
        if is_training and not self.force_resize:
            return self._balanced_crop(
                src_img_pil,
                tgt_img_pil,
                mask_pil,
                is_training=True,
                return_coords=return_coords,
            )

        # Inference/validation: resize to configured img_size, then align src/mask.
        desired_size = (int(self.img_size), int(self.img_size))
        if tgt_img_pil.size != desired_size:
            tgt_img_pil = tgt_img_pil.resize(desired_size, Image.BILINEAR)

        if src_img_pil is None:
            src_img_pil = tgt_img_pil
        elif src_img_pil.size != desired_size:
            src_img_pil = src_img_pil.resize(desired_size, Image.BILINEAR)

        if mask_pil.size != desired_size:
            mask_pil = mask_pil.resize(desired_size, Image.NEAREST)

        if return_coords:
            norm_coords = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            return src_img_pil, tgt_img_pil, mask_pil, norm_coords
        return src_img_pil, tgt_img_pil, mask_pil

    def _recursive_find_images(self, directory):
        """
        Recursively find all image files in a directory and its subdirectories.
        
        Args:
            directory (str): Directory to search for images
            
        Returns:
            dict: Mapping from base filename to list of full paths (for multiple versions of the same image)
        """
        image_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')
        image_map = {}
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    base_name = os.path.splitext(file)[0]
                    full_path = os.path.join(root, file)
                    # For BR-Gen dataset, multiple versions of the same image exist in different subdirectories
                    # So we store all versions in a list
                    if base_name not in image_map:
                        image_map[base_name] = []
                    image_map[base_name].append(full_path)
        
        return image_map

    def _discover_forgery_samples(self):
        """Discover forgery samples with support for nested directories"""
        if self.allow_multiple_targets:
            # Gather targets by base name
            target_dir = os.path.join(self.root, "target")
            self.target_groups = {}
            for target_file in sorted(os.listdir(target_dir)):
                base_name = "_".join(target_file.split("_")[:-1])
                self.target_groups.setdefault(base_name, []).append(target_file)

            # Flatten into (base_name, target_file) tuples
            self.forgery_samples = []
            for base_name, file_list in self.target_groups.items():
                for tgt in file_list:
                    self.forgery_samples.append({
                        'type': 'forgery',
                        'base_name': base_name,
                        'target_file': tgt
                    })
        else:
            # Single target per source with support for nested directories
            # For BR-Gen dataset, we need to handle nested directories
            # and map Forged → target, Mask → mask, RealImage → source
            
            # Get all image paths from nested directories
            forged_dir = os.path.join(self.root, "Forged")
            mask_dir = os.path.join(self.root, "Mask")
            realimage_dir = os.path.join(self.root, "RealImage")
            
            # Recursively find all images in each directory
            forged_images = self._recursive_find_images(forged_dir)
            mask_images = self._recursive_find_images(mask_dir)
            
            # For RealImage, we need to read from the text files and map to actual images
            realimage_map = {}
            
            # Read image lists from RealImage directory
            for root, _, files in os.walk(realimage_dir):
                for file in files:
                    if file.endswith('.txt'):
                        # Read the image list file
                        list_path = os.path.join(root, file)
                        with open(list_path, 'r') as f:
                            image_list = [line.strip() for line in f if line.strip()]
                        
                        # Get the dataset name from the file name
                        dataset_name = os.path.splitext(file)[0].split('_')[0]
                        # The actual images should be in the same directory as the list file
                        # but according to the BR-Gen structure, RealImage only contains list files
                        # and the actual images are stored elsewhere. However, for our purposes,
                        # we'll use the Forged images as both target and source for now
                        # since the RealImage list seems to be references to external datasets
            
            # Create samples by matching base names between Forged and Mask
            self.forgery_samples = []
            
            # Iterate through all mask images and find matching forged images
            # This way, each mask can be paired with multiple forged images from different generation methods
            for base_name, mask_paths in mask_images.items():
                if base_name in forged_images:
                    # Get all forged images for this base_name
                    forged_paths = forged_images[base_name]
                    # Get the first mask path (assuming all mask paths for the same base_name are identical)
                    mask_path = mask_paths[0]
                    
                    # Create a sample for each forged image
                    for forged_path in forged_paths:
                        self.forgery_samples.append({
                            'type': 'forgery',
                            'base_name': base_name,
                            'target_path': forged_path,
                            'mask_path': mask_path,
                            'source_path': forged_path,  # Use forged as source for now
                            'is_nested': True
                        })
            
            # If no nested samples found, fall back to original logic
            if not self.forgery_samples:
                # Original logic for flat directory structure
                target_dir = os.path.join(self.root, "target")
                self.img_names = sorted(os.listdir(target_dir))
                self.forgery_samples = []
                for img_name in self.img_names:
                    base_name = os.path.splitext(img_name)[0]
                    self.forgery_samples.append({
                        'type': 'forgery',
                        'base_name': base_name,
                        'target_file': img_name,
                        'is_nested': False
                    })
            
            logger.info(f"Discovered {len(self.forgery_samples)} forgery samples")

    def _discover_authentic_samples(self):
        """Discover authentic samples from specified directory"""
        import random
        
        # Always use source images under the dataset root for authentic samples.
        source_dir = os.path.join(self.root, "source")
        if self.authentic_source_dir:
            logger.debug(
                "Ignoring authentic_source_dir=%s; using %s instead",
                self.authentic_source_dir,
                source_dir,
            )
        
        if not os.path.exists(source_dir):
            logger.warning("Authentic source directory %s not found", source_dir)
            return
        
        # Get all image files
        authentic_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            authentic_files.extend([f for f in os.listdir(source_dir) if f.lower().endswith(ext)])
        
        authentic_files = sorted(authentic_files)
        
        if not authentic_files:
            logger.warning("No authentic images found in %s", source_dir)
            return
        
        # Calculate number of authentic samples needed
        num_forgery = len(self.forgery_samples)
        if self.authentic_ratio == 1.0:
            # If authentic_ratio is 1.0, use all available authentic images
            num_authentic_needed = float('inf')  # Will be limited to available authentic images
        else:
            num_authentic_needed = int(num_forgery * self.authentic_ratio / (1 - self.authentic_ratio))
        
        # Sample authentic images
        if len(authentic_files) >= num_authentic_needed:
            if self.is_training:
                # Random sampling for training
                sampled_files = random.sample(authentic_files, num_authentic_needed)
            else:
                # Deterministic sampling for testing
                step = len(authentic_files) // num_authentic_needed
                sampled_files = authentic_files[::step][:num_authentic_needed]
        else:
            # Use all available authentic images
            sampled_files = authentic_files
            logger.warning(
                "Only %s authentic images available (requested %s)",
                len(authentic_files),
                num_authentic_needed,
            )
        
        # Create authentic sample entries
        for auth_file in sampled_files:
            base_name = os.path.splitext(auth_file)[0]
            self.authentic_samples.append({
                'type': 'authentic',
                'base_name': base_name,
                'auth_file': auth_file,
                'source_dir': source_dir
            })
        
        logger.info(
            "Added %s authentic samples (ratio %.1f%%)",
            len(self.authentic_samples),
            self.authentic_ratio * 100.0,
        )

    def _combine_samples(self):
        """Combine forgery and authentic samples"""
        import random
        
        self.all_samples = self.forgery_samples + self.authentic_samples
        
        # Apply sample ratio if needed
        if self.sample_ratio < 1.0 and self.sample_ratio > 0.0:
            # Calculate number of samples to keep
            num_samples_to_keep = int(len(self.all_samples) * self.sample_ratio)
            if num_samples_to_keep == 0:
                num_samples_to_keep = 1  # Ensure at least one sample
            
            # Randomly select samples
            sampled_indices = random.sample(range(len(self.all_samples)), num_samples_to_keep)
            self.all_samples = [self.all_samples[i] for i in sampled_indices]
        
        if self.is_training:
            random.shuffle(self.all_samples)
        
        logger.info(
            "Dataset assembled | forgery=%s authentic=%s total=%s sample_ratio=%.2f final_samples=%s",
            len(self.forgery_samples),
            len(self.authentic_samples),
            len(self.forgery_samples) + len(self.authentic_samples),
            self.sample_ratio,
            len(self.all_samples),
        )

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset (authentic or forgery).
        """
        sample_info = self.all_samples[idx]
        sample_type = sample_info['type']

        local_transforms, global_transforms = self._get_transforms()
        
        if sample_type == 'authentic':
            return self._get_authentic_sample(sample_info, local_transforms, global_transforms)
        else:
            return self._get_forgery_sample(sample_info, local_transforms, global_transforms)

    def _get_transforms(self):
        if not self.is_training:
            return None, None
        if self._local_transforms is None:
            self._local_transforms = SAM2Transforms(resolution=self.img_size, mask_threshold=0.0)
        if self._global_transforms is None:
            self._global_transforms = SAM2Transforms(resolution=self.global_img_size, mask_threshold=0.0)
        return self._local_transforms, self._global_transforms
    


    def _get_authentic_sample(self, sample_info, local_transforms, global_transforms):
        """Get an authentic sample with all-zero mask"""
        base_name = sample_info['base_name']
        auth_file = sample_info['auth_file']
        source_dir = sample_info['source_dir']
        
        # Load authentic image
        auth_path = os.path.join(source_dir, auth_file)
        try:
            auth_img_pil = Image.open(auth_path).convert("RGB")
        except Exception:
            logger.exception("Could not load authentic image %s", auth_path)
            return {
                "orig": None, "streams": [], "mask": None, "source": None, "instruction": "",
                "orig_pil": None, "is_authentic": True, "sample_type": "authentic",
            }
        
        # For authentic images, source and target are the same
        src_img_pil = auth_img_pil.copy()
        tgt_img_pil = auth_img_pil.copy()
        
        # Create all-zero mask
        mask_pil = Image.new('L', auth_img_pil.size, 0)  # All zeros
        
        norm_coords = None
        if local_transforms is not None:
            # Training: produce local patch + coords, plus a global resized view.
            global_pil = tgt_img_pil.resize((self.global_img_size, self.global_img_size), Image.BILINEAR)
            global_out = global_transforms(global_pil)

            src_img, tgt_img, mask_img, norm_coords = self._handle_image_sizing(
                src_img_pil, tgt_img_pil, mask_pil, is_training=True, return_coords=True
            )
        else:
            # Inference/validation: keep full resolution tensors in [0,1]
            src_img, tgt_img, mask_img = self._handle_image_sizing(
                src_img_pil, tgt_img_pil, mask_pil, is_training=False
            )
            global_out = None

        # Store PIL image if requested
        orig_pil = tgt_img if self.return_pil else None

        # Get original tensor (always present).  During training ``orig_out`` is
        # SAM2/ImageNet-normalized for legacy compatibility, while
        # ``local_patch_raw`` keeps the exact resized/cropped RGB patch in
        # [0, 1] for the paper-style LAD operator.
        orig_tensor = TF.to_tensor(tgt_img)
        local_patch_raw = orig_tensor
        if local_transforms is not None:
            orig_out = local_transforms.transforms(orig_tensor).unsqueeze(0).squeeze(0)
            mask_tensor = torch.zeros(1, int(self.img_size), int(self.img_size), dtype=torch.float32)
        else:
            orig_out = orig_tensor
            mask_tensor = torch.zeros(1, orig_tensor.shape[-2], orig_tensor.shape[-1], dtype=torch.float32)

        src_tensor = TF.to_tensor(src_img)
        if local_transforms is not None:
            src_out = local_transforms.transforms(src_tensor).unsqueeze(0).squeeze(0)
        else:
            src_out = src_tensor
        
        # Ferret-SAM doesn't use perturbation streams
        streams_transformed = []
        
        result = {
            "orig": orig_out,                 # Training: normalized crop | Inference: raw full image
            "local_patch": orig_out if local_transforms is not None else None,
            "local_patch_raw": local_patch_raw,
            "global_image": global_out,
            "norm_coords": norm_coords,
            "streams": streams_transformed,    # List of ONLY perturbed/sharpened streams
            "mask": mask_tensor,
            "source": src_out,                # For visualization (unedited if available)
            "instruction": "authentic image",
            "dataset_name": self.dataset_name,
            "is_authentic": True,
            "sample_type": "authentic",
        }
        
        if self.return_pil:
            result["orig_pil"] = orig_pil
        
        return result
    
    def _get_forgery_sample(self, sample_info, local_transforms, global_transforms):
        """Get a forgery sample"""
        base_name = sample_info['base_name']
        
        try:
            # Check if this is a nested sample (BR-Gen dataset)
            is_nested = sample_info.get('is_nested', False)
            
            if is_nested:
                # Handle nested directory samples (BR-Gen dataset)
                # Directly use the full paths stored in sample_info
                tgt_path = sample_info['target_path']
                mask_path = sample_info['mask_path']
                src_path = sample_info['source_path']
                
                # Load target image
                if not os.path.exists(tgt_path):
                    raise FileNotFoundError(f"Target image not found: {tgt_path}")
                tgt_img_pil = Image.open(tgt_path).convert("RGB")
                tgt_img_pil_full = tgt_img_pil

                # Load mask image
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask image not found: {mask_path}")
                mask_pil = Image.open(mask_path)
                mask_pil_full = mask_pil

                # Load source image (if available)
                src_img_pil = None
                use_source = True
                try:
                    if os.path.exists(src_path):
                        src_img_pil = Image.open(src_path).convert("RGB")
                    else:
                        use_source = False
                except Exception:
                    logger.exception("Could not load source image %s, using target as source", src_path)
                    use_source = False
                src_img_pil_full = src_img_pil if src_img_pil is not None else tgt_img_pil_full
            else:
                # Original logic for flat directory structure
                if 'target_file' in sample_info:
                    target_file = sample_info['target_file']
                else:
                    # Legacy support
                    target_file = f"{base_name}.png"
                    if not os.path.exists(os.path.join(self.root, "target", target_file)):
                        target_file = f"{base_name}.jpg"

                # 1) Load source image (raw PIL) - handle case where source folder doesn't exist
                src_img_pil = None
                use_source = True
                try:
                    src_path_png = os.path.join(self.root, "source", f"{base_name}.png")
                    src_path_jpg = src_path_png.replace(".png", ".jpg")
                    
                    if os.path.exists(src_path_png):
                        src_img_pil = Image.open(src_path_png).convert("RGB")
                    elif os.path.exists(src_path_jpg):
                        src_img_pil = Image.open(src_path_jpg).convert("RGB")
                    else:
                        # No source image found - use target image as source (for datasets like sid_train)
                        # logger.warning("No source image found for %s, using target as source", base_name)
                        use_source = False
                except Exception:
                    logger.exception("Could not load source image %s, using target as source", src_path_png)
                    use_source = False

                # 2) Load target image (raw PIL)
                tgt_path = os.path.join(self.root, "target", target_file)
                if not os.path.exists(tgt_path):
                    raise FileNotFoundError(f"Target image not found: {tgt_path}")
                tgt_img_pil = Image.open(tgt_path).convert("RGB")
                tgt_img_pil_full = tgt_img_pil

                # 3) Load mask (raw PIL)
                mask_path = os.path.join(self.root, "mask", f"{base_name}.png")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask image not found: {mask_path}")
                mask_pil = Image.open(mask_path)
                mask_pil_full = mask_pil

                src_img_pil_full = src_img_pil if src_img_pil is not None else tgt_img_pil_full

            # 4) Handle sizing: training uses balanced crop; inference keeps original size
            norm_coords = None
            if local_transforms is not None:
                global_pil = tgt_img_pil_full.resize((self.global_img_size, self.global_img_size), Image.BILINEAR)
                global_out = global_transforms(global_pil)

                src_img, tgt_img, mask_img, norm_coords = self._handle_image_sizing(
                    src_img_pil, tgt_img_pil, mask_pil, is_training=True, return_coords=True
                )
            else:
                src_img, tgt_img, mask_img = self._handle_image_sizing(
                    src_img_pil, tgt_img_pil, mask_pil, is_training=False
                )
                global_out = None

            # Store the processed target PIL image for TruFor (only if requested)
            if self.return_pil:
                orig_pil = tgt_img_pil_full
            else:
                orig_pil = None

            # 6) Get original tensor (always present).  Keep the raw local
            # target patch separately so LAD is trained in the same [0, 1]
            # RGB space used by validation/inference and described in the
            # paper.  ``orig_out`` remains normalized for legacy callers.
            orig_tensor = TF.to_tensor(tgt_img)
            local_patch_raw = orig_tensor
            if local_transforms is not None:
                orig_out = local_transforms.transforms(orig_tensor).unsqueeze(0).squeeze(0)
            else:
                orig_out = orig_tensor

            src_tensor = TF.to_tensor(src_img)
            if local_transforms is not None:
                src_out = local_transforms.transforms(src_tensor).unsqueeze(0).squeeze(0)
            else:
                src_out = src_tensor

            # Ferret-SAM doesn't use perturbation streams
            streams_transformed = []

            # 8) Convert mask→binary, combine diff+nonliteral
            mask_rgba = np.array(mask_img)
            src_img_np = np.array(src_img)
            tgt_img_np = np.array(tgt_img)
            # For datasets without source images, disable diff-based masking
            binary_mask = self.create_combined_mask(
                src_img_np=src_img_np,
                tgt_img_np=tgt_img_np,
                mask_rgba=mask_rgba,
                allow_multiple_targets=self.allow_multiple_targets,
                use_diff=self.use_diff and use_source,  # Disable diff if no source
            )
            if local_transforms is not None:
                crop_size = int(self.img_size)
                if binary_mask.shape != (crop_size, crop_size):
                    binary_mask = cv2.resize(
                        binary_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST
                    )
            else:
                expected_h, expected_w = orig_tensor.shape[-2], orig_tensor.shape[-1]
                if binary_mask.shape != (expected_h, expected_w):
                    binary_mask = cv2.resize(
                        binary_mask, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST
                    )
            mask_tensor = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0)

            # 9) Load instruction text if available (skip if no source folder)
            instruction = ""
            if use_source:
                instr_path = os.path.join(self.root, "source", f"{base_name}.txt")
                if os.path.exists(instr_path):
                    with open(instr_path, "r") as f:
                        instruction = f.read().strip()

            # If the crop contains no forgery pixels, treat as authentic.
            is_authentic = bool(mask_tensor.sum().item() == 0)

            result = {
                "orig": orig_out,                 # Training: normalized crop | Inference: raw full image
                "local_patch": orig_out if local_transforms is not None else None,
                "local_patch_raw": local_patch_raw,
                "global_image": global_out,
                "norm_coords": norm_coords,
                "streams": streams_transformed,    # List of ONLY perturbed/sharpened streams
                "mask": mask_tensor,               # (1, H, W)
                "source": src_out,                # For visualization (unedited if available)
                "instruction": instruction,        # str
                "dataset_name": self.dataset_name, # str
                "is_authentic": is_authentic,
                "sample_type": "forgery",
            }
            
            # Only add PIL image if requested
            if self.return_pil:
                result["orig_pil"] = orig_pil

            return result
        except Exception as e:
            logger.exception("Could not load forgery sample %s, skipping", base_name)
            # Return a placeholder dictionary with None values to be filtered out later
            return {
                "orig": None, 
                "local_patch": None,
                "global_image": None,
                "norm_coords": None,
                "streams": [], 
                "mask": None, 
                "source": None,
                "instruction": "",
                "dataset_name": self.dataset_name,
                "is_authentic": False,
                "sample_type": "forgery",
                "orig_pil": None,
            }
