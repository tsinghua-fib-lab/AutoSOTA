import os
import os.path as osp
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
from tqdm import tqdm
import json


class OminiWorldDataset(Dataset):
    """
    Dataset for OminiWorld motion segmentation using VGGT
    Adapted from MotionSegmentationDataset for OminiWorld data structure
    """
    def __init__(self, data_dir, scene_list=None, split='train', transform=None, 
                 img_size=518, sequence_length=8, sample_stride=1,
                 enable_augmentation=True, crop_strategy='center', train_ratio=0.8):
        """
        Args:
            data_dir: Root directory (/data1/OminiWorld/annotations/OmniWorld-Game)
            scene_list: List of scene IDs to use (if None, uses all available scenes)
            split: 'train', 'val', or 'test'
            transform: Optional transforms for images
            img_size: Target image size for VGGT (default 518)
            sequence_length: Number of frames to sample from each video
            sample_stride: Stride for sampling frames (1 means consecutive frames)
            enable_augmentation: Whether to apply data augmentation
            crop_strategy: 'center', 'random', or 'smart' (aspect-ratio aware)
            train_ratio: Ratio for train/val split (remaining goes to val)
        """
        self.split = split
        self.data_dir = data_dir
        self.training = split == "train"
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.sample_stride = sample_stride
        self.enable_augmentation = enable_augmentation and self.training
        self.crop_strategy = crop_strategy
        
        # Default scene list if not provided
        if scene_list is None:
            self.scene_list = [
                '0caf78b8eeac', '9d2d94a36bc6', 'afafe7b21713', 'b0320ed0c3a2', 
                'b9a6b7c50e9a', 'bfb12b6cea73', 'c75817fcbd27', 
                'ca072bb30700',  'dbd3e34a840d'
            ]
        else:
            self.scene_list = scene_list
        
        # Set up transforms
        if transform is None:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = transform
        
        # Split scenes into train/val/test
        self.valid_scenes = self._get_valid_scenes()
        self.split_scenes = self._split_scenes(train_ratio)
        
        # Prepare final, validated lists
        self.scene_paths = []
        self.valid_samples = []
        
        for scene_id in tqdm(self.split_scenes, desc=f"Validating data for '{split}' split"):
            scene_path = osp.join(self.data_dir, scene_id)
            color_dir = osp.join(scene_path, "color")
            mask_dir = osp.join(scene_path, "gdino_mask")
            
            # Check if directories exist
            if not os.path.exists(color_dir) or not os.path.exists(mask_dir):
                print(f"Warning: Missing directories for {scene_id}, skipping.")
                continue
            
            # Get image and mask paths
            img_paths = self._get_image_paths(color_dir)
            if not img_paths:
                print(f"Warning: No images found in {color_dir}, skipping.")
                continue
            
            # Check if we have enough frames for sequence sampling
            valid_pair_count = self._count_valid_pairs(img_paths, mask_dir)
            if valid_pair_count < self.sequence_length:
                print(f"Warning: {scene_id} has only {valid_pair_count} valid image-mask pairs, need at least {self.sequence_length}. Skipping.")
                continue
            
            # If all checks pass, add this scene to our valid lists
            self.scene_paths.append(scene_path)
            self.valid_samples.append(scene_id)
        
        if len(self.valid_samples) == 0:
            raise RuntimeError(f"No valid scenes found for split '{self.split}' in directory {self.data_dir}. Please check your data.")
            
        print(f"Loaded {len(self.valid_samples)} valid scenes for '{split}' split")
        
        # Get reference image size
        sample_color_dir = osp.join(self.scene_paths[0], "color")
        sample_img_path = self._get_image_paths(sample_color_dir)[0]
        sample_img = Image.open(sample_img_path).convert('RGB')
        self.original_image_size = sample_img.size  # (W, H)
        print(f"Original image size: {self.original_image_size}")

    def _get_valid_scenes(self):
        """Get list of scenes that exist in the data directory"""
        valid_scenes = []
        for scene_id in self.scene_list:
            scene_path = osp.join(self.data_dir, scene_id)
            if os.path.exists(scene_path):
                valid_scenes.append(scene_id)
            else:
                print(f"Warning: Scene {scene_id} not found in {self.data_dir}")
        return valid_scenes

    def _split_scenes(self, train_ratio):
        """Split scenes into train/val/test sets"""
        # Sort for reproducible splits
        sorted_scenes = sorted(self.valid_scenes)
        total_scenes = len(sorted_scenes)
        
        if total_scenes == 0:
            return []
        
        # Calculate split indices
        train_end = int(total_scenes * train_ratio)
        val_end = train_end + int(total_scenes * 0.1)  # 10% for validation
        
        if self.split == 'train':
            return sorted_scenes[:train_end]
        elif self.split == 'val':
            return sorted_scenes[train_end:val_end] if val_end < total_scenes else sorted_scenes[train_end:]
        else:  # test
            return sorted_scenes[val_end:] if val_end < total_scenes else []

    def _get_image_paths(self, img_dir):
        """Get sorted list of image paths from directory (6-digit naming: 000000.png)"""
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob(osp.join(img_dir, ext)))
        return sorted(img_paths)

    def _count_valid_pairs(self, img_paths, mask_dir):
        """Count valid image-mask pairs"""
        valid_count = 0
        for img_path in img_paths:
            img_filename = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_filename)[0]
            
            # Check for corresponding mask (should be same naming: 000000.png)
            mask_path = osp.join(mask_dir, img_filename)  # Try same filename first
            if os.path.exists(mask_path):
                valid_count += 1
            else:
                # Try different extensions
                for mask_ext in ['.png', '.jpg', '.jpeg']:
                    potential_mask_path = osp.join(mask_dir, img_name_no_ext + mask_ext)
                    if os.path.exists(potential_mask_path):
                        valid_count += 1
                        break
        return valid_count

    def _apply_augmentation(self, images, masks):
        """
        Apply data augmentation to image-mask pairs
        Same as original implementation
        """
        import random
        
        if not self.enable_augmentation or len(images) != len(masks):
            return images, masks
        
        # Get original size from first image
        orig_w, orig_h = images[0].size
        
        # 1. Smart cropping strategy
        if self.crop_strategy == 'smart':
            # Calculate crop size maintaining aspect ratio close to square
            min_dim = min(orig_w, orig_h)
            crop_size = min_dim
            
            # Random crop position
            max_x = max(0, orig_w - crop_size)
            max_y = max(0, orig_h - crop_size)
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)
            
        elif self.crop_strategy == 'center':
            # Center crop to square
            crop_size = min(orig_w, orig_h)
            crop_x = (orig_w - crop_size) // 2
            crop_y = (orig_h - crop_size) // 2
            
        elif self.crop_strategy == 'random':
            # Random crop with size between 80% and 100% of min dimension
            crop_size = int(min(orig_w, orig_h) * random.uniform(0.8, 1.0))
            max_x = max(0, orig_w - crop_size)
            max_y = max(0, orig_h - crop_size)
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)
        else:
            # No cropping, just resize
            crop_x = crop_y = 0
            crop_size = min(orig_w, orig_h)
        
        # Apply crop to all images and masks
        cropped_images = []
        cropped_masks = []
        
        for img, mask in zip(images, masks):
            if self.crop_strategy != 'none':
                # Apply same crop to image and mask
                img_cropped = img.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
                mask_cropped = mask.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
            else:
                img_cropped = img
                mask_cropped = mask
                
            cropped_images.append(img_cropped)
            cropped_masks.append(mask_cropped)
        
        return cropped_images, cropped_masks

    def _smart_resize(self, images, masks, target_size):
        """
        Smart resize that maintains aspect ratio and quality
        Same as original implementation
        """
        resized_images = []
        resized_masks = []
        
        for img, mask in zip(images, masks):
            # Resize image with high quality
            img_resized = img.resize((target_size, target_size), Image.LANCZOS)
            # Resize mask with nearest neighbor to maintain binary values
            mask_resized = mask.resize((target_size, target_size), Image.NEAREST)
            
            resized_images.append(img_resized)
            resized_masks.append(mask_resized)
            
        return resized_images, resized_masks

    def _load_images_and_masks(self, img_paths, mask_paths, start_idx, end_idx):
        """
        Load and augment a sequence of images and masks
        Same as original implementation
        """
        # Load PIL images and masks
        pil_images = []
        pil_masks = []
        
        for i in range(start_idx, end_idx, self.sample_stride):
            if i >= len(img_paths):
                # Repeat last frame if needed
                img_path = img_paths[-1]
                mask_path = mask_paths[-1]
            else:
                img_path = img_paths[i]
                mask_path = mask_paths[i]
            
            # Load image and mask
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            pil_images.append(img)
            pil_masks.append(mask)
        
        # Apply augmentation if enabled
        if self.enable_augmentation:
            pil_images, pil_masks = self._apply_augmentation(pil_images, pil_masks)
        
        # Smart resize to target size
        pil_images, pil_masks = self._smart_resize(pil_images, pil_masks, self.img_size)
        
        # Convert to tensors
        images = []
        masks = []
        
        for img, mask in zip(pil_images, pil_masks):
            # Convert image to tensor and normalize to [0, 1]
            img_tensor = transforms.ToTensor()(img)
            # Apply normalization
            img_tensor = self.normalize(img_tensor)
            images.append(img_tensor)
            
            # Convert mask to tensor
            mask_tensor = transforms.ToTensor()(mask)
            mask_tensor = mask_tensor.squeeze(0)  # Remove channel dimension
            # Ensure binary mask
            mask_tensor = (mask_tensor > 0).float()
            masks.append(mask_tensor)
        
        # Stack into tensors [S, 3, H, W] and [S, H, W]
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        
        return images, masks

    def _get_frame_indices(self, total_frames):
        """Get start and end indices for frame sampling"""
        max_start_idx = max(0, total_frames - self.sequence_length * self.sample_stride)
        
        if self.training:
            # Random sampling during training
            start_idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
        else:
            # Fixed sampling during testing (center frames)
            start_idx = max_start_idx // 2
            
        end_idx = start_idx + self.sequence_length * self.sample_stride
        return start_idx, end_idx

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        # Get scene path and ID
        scene_path = self.scene_paths[idx]
        scene_id = self.valid_samples[idx]
        
        # Get color and mask directories
        color_dir = osp.join(scene_path, "color")
        mask_dir = osp.join(scene_path, "gdino_mask")
        
        # Get image paths
        img_paths = self._get_image_paths(color_dir)
        
        # Get corresponding mask paths based on image filenames
        mask_paths = []
        valid_img_paths = []
        
        for img_path in img_paths:
            img_filename = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_filename)[0]
            
            # Look for corresponding mask (try same filename first)
            mask_path = osp.join(mask_dir, img_filename)
            if os.path.exists(mask_path):
                valid_img_paths.append(img_path)
                mask_paths.append(mask_path)
            else:
                # Try different extensions
                found_mask = False
                for mask_ext in ['.png', '.jpg', '.jpeg']:
                    potential_mask_path = osp.join(mask_dir, img_name_no_ext + mask_ext)
                    if os.path.exists(potential_mask_path):
                        valid_img_paths.append(img_path)
                        mask_paths.append(potential_mask_path)
                        found_mask = True
                        break
                
                if not found_mask:
                    # Skip this image if no corresponding mask found
                    continue
        
        img_paths = valid_img_paths
        
        # Sample frame indices
        start_idx, end_idx = self._get_frame_indices(len(img_paths))
        
        # Load and augment images and masks together
        images, masks = self._load_images_and_masks(img_paths, mask_paths, start_idx, end_idx)
        
        # Create sample dictionary
        sample = {
            'images': images,           # [S, 3, H, W] - input video frames
            'motion_masks': masks,      # [S, H, W] - ground truth motion masks
            'sequence_name': scene_id,  # string - scene identifier
            'frame_indices': torch.tensor(list(range(start_idx, end_idx, self.sample_stride))),  # [S] - frame indices
        }
        
        # Add valid masks (all pixels are valid by default)
        valid_masks = torch.ones_like(masks)  # [S, H, W]
        sample['valid_masks'] = valid_masks
        
        return sample


# Example usage:
if __name__ == "__main__":
    # Example usage
    data_dir = "/data1/OminiWorld/annotations/OmniWorld-Game"
    
    # Create dataset for training
    train_dataset = OminiWorldDataset(
        data_dir=data_dir,
        split='train',
        img_size=518,
        sequence_length=8,
        sample_stride=1,
        enable_augmentation=True,
        crop_strategy='smart'
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create dataset for validation
    val_dataset = OminiWorldDataset(
        data_dir=data_dir,
        split='val',
        img_size=518,
        sequence_length=8,
        sample_stride=1,
        enable_augmentation=False,
        crop_strategy='center'
    )
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Images shape: {sample['images'].shape}")
    print(f"Motion masks shape: {sample['motion_masks'].shape}")
    print(f"Sequence name: {sample['sequence_name']}")