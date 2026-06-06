import os
import numpy as np
import torch
from torch.utils.data import Dataset

import os.path as osp
from glob import glob
import imageio as iio
from tqdm import tqdm
import logging
# from utils import masked_median_blur, SceneNormDict
from PIL import Image
import random
import torch.nn.functional as F
import h5py
from torchvision.io import read_image
# from .kubric import parse_tapir_track_info, load_target_tracks
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms


class MotionSegmentationDatasetGotmoving(Dataset):
    def __init__(self, data_dir, split, transform=None, 
                 img_size=518, sequence_length=8, sample_stride=1,
                 enable_augmentation=True, crop_strategy='center'):

        self.split = split
        self.data_dir = data_dir
        self.training = split == "train"
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.sample_stride = sample_stride
        self.enable_augmentation = enable_augmentation and self.training
        self.crop_strategy = crop_strategy

        if transform is None:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = transform

        # -----------------------------------------------------------
        # ① 根据 data_dir 类型决定：按 txt 读，还是递归扫描
        # -----------------------------------------------------------
        if os.path.isfile(self.data_dir) and self.data_dir.endswith(".txt"):
            print(f"Loading sequence roots from txt: {self.data_dir}")
            all_sequences = self._load_sequences_from_txt(self.data_dir)
        else:
            print(f"Scanning {data_dir} for sequences...")
            all_sequences = self._find_all_sequences(self.data_dir)

        self.img_dirs_list = []
        self.mask_dirs_list = []
        valid_samples = []
        
        # -----------------------------------------------------------
        # ② 验证每个序列中 frame 数是否足够（使用 .npz 作为 mask）
        # -----------------------------------------------------------
        for seq in tqdm(all_sequences, desc=f"Validating '{split}'"):
            img_dir = seq["img_dir"]
            mask_dir = seq["mask_dir"]
            seq_name = seq["name"]

            img_paths = self._get_image_paths(img_dir)
            if not img_paths:
                continue

            valid_pair_count = 0
            for img_path in img_paths:
                name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                # 只匹配 .npz 掩码
                mask_path = os.path.join(mask_dir, name_no_ext + ".npz")
                if os.path.exists(mask_path):
                    valid_pair_count += 1

            if valid_pair_count < self.sequence_length:
                continue

            self.img_dirs_list.append(img_dir)
            self.mask_dirs_list.append(mask_dir)
            valid_samples.append(seq_name)

        self.sample_list = valid_samples
        print(f"Found {len(self.sample_list)} valid sequences.")

        if len(self.sample_list) == 0:
            raise RuntimeError(
                f"No valid sequences found for split '{self.split}' "
                f"in directory / txt {self.data_dir}."
            )

        # -----------------------------------------------------------
        # ③ 读取一张图片作为参考尺寸
        # -----------------------------------------------------------
        sample_img_path = self._get_image_paths(self.img_dirs_list[0])[0]
        sample_img = Image.open(sample_img_path).convert('RGB')
        self.original_image_size = sample_img.size
        print(f"Original image size: {self.original_image_size}")

    # ---------------------------------------------------------------
    # 新增：按 txt 读取序列
    # txt 每行一个 video_root
    #   img_dir  = <video_root>
    #   mask_dir = <video_root>/masks
    # ---------------------------------------------------------------
    def _load_sequences_from_txt(self, list_path):
        seqs = []
        with open(list_path, "r") as f:
            roots = [line.strip() for line in f if line.strip()]

        for root in roots:
            img_dir = root
            mask_dir = os.path.join(root, "masks")

            if (not os.path.isdir(img_dir)) or (not os.path.isdir(mask_dir)):
                # 如果某一行路径写错 / 没有 masks 目录，就跳过
                continue

            img_paths = self._get_image_paths(img_dir)
            if not img_paths:
                continue

            # 先收集这个序列有哪些 .npz 掩码
            mask_files = sorted(glob(os.path.join(mask_dir, "*.npz")))
            mask_names = {
                os.path.splitext(os.path.basename(f))[0] for f in mask_files
            }

            # 统计有 mask 的帧
            valid_pairs = []
            for img_path in img_paths:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                if img_name in mask_names:
                    valid_pairs.append(img_path)

            if len(valid_pairs) >= self.sequence_length:
                seqs.append({
                    "img_dir": img_dir,
                    "mask_dir": mask_dir,
                    "name": root
                })

        print(f"[TXT] Collected {len(seqs)} sequences from {list_path}")
        return seqs

    # ---------------------------------------------------------------
    # 原有递归扫描逻辑：改为兼容 GOT 结构
    #   只找名为 'masks' 的目录
    #   img_dir = masks 的父目录
    #   mask 后缀改为 .npz
    # ---------------------------------------------------------------
    def _find_all_sequences(self, root):
        seqs = []
        # 找所有 'masks' 文件夹
        mask_dirs = glob(os.path.join(root, "**", "masks"), recursive=True)

        for mask_dir in mask_dirs:
            parent = os.path.dirname(mask_dir)  # video_root
            img_dir = parent  # 图片直接在 video_root 下面

            if not os.path.isdir(img_dir):
                continue

            img_files = self._get_image_paths(img_dir)
            mask_files = sorted(glob(os.path.join(mask_dir, "*.npz")))

            if not img_files or not mask_files:
                continue

            mask_names = {
                os.path.splitext(os.path.basename(f))[0] for f in mask_files
            }

            valid_pairs = []
            for img_path in img_files:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                if img_name in mask_names:
                    valid_pairs.append(img_path)

            if len(valid_pairs) >= self.sequence_length:
                seqs.append({
                    "img_dir": img_dir,
                    "mask_dir": mask_dir,
                    "name": os.path.relpath(parent, root)
                })
        
        return seqs


    def _apply_augmentation(self, images, masks):
        """
        Apply data augmentation to image-mask pairs
        
        Args:
            images: List of PIL images
            masks: List of PIL masks
            
        Returns:
            Tuple of (augmented_images, augmented_masks)
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
        
        
        # 3. Color augmentation (only for images, not masks)
        # if random.random() < 0.3:  # 30% chance
        #     # Random brightness, contrast, saturation
        #     brightness_factor = random.uniform(0.8, 1.2)
        #     contrast_factor = random.uniform(0.8, 1.2)
        #     saturation_factor = random.uniform(0.8, 1.2)
            
        #     color_transform = transforms.ColorJitter(
        #         brightness=(brightness_factor, brightness_factor),
        #         contrast=(contrast_factor, contrast_factor),
        #         saturation=(saturation_factor, saturation_factor)
        #     )
            
        #     cropped_images = [color_transform(img) for img in cropped_images]
        
        return cropped_images, cropped_masks

    def _smart_resize(self, images, masks, target_size):
        """
        Smart resize that maintains aspect ratio and quality
        
        Args:
            images: List of PIL images
            masks: List of PIL masks  
            target_size: Target size (int) for output
            
        Returns:
            Tuple of (resized_images, resized_masks)
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
    
    def _get_image_paths(self, img_dir):
        """Get sorted list of image paths from directory"""
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob(osp.join(img_dir, ext)))
        return sorted(img_paths)

    def _get_mask_paths(self, mask_dir):
        """Get sorted list of mask paths from directory"""
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        mask_paths = []
        for ext in extensions:
            mask_paths.extend(glob(osp.join(mask_dir, ext)))
        return sorted(mask_paths)

    def _load_images_and_masks(self, img_paths, mask_paths, start_idx, end_idx):
        """
        Load and augment a sequence of images and masks
        
        Args:
            img_paths: List of image paths
            mask_paths: List of mask paths
            start_idx: Start index for sampling
            end_idx: End index for sampling
            
        Returns:
            Tuple of (images_tensor, masks_tensor)
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
        return len(self.sample_list)

    def __getitem__(self, idx):
        try:
            # Get paths
            img_dir = self.img_dirs_list[idx]
            mask_dir = self.mask_dirs_list[idx]
            seq_name = self.sample_list[idx]
            
            # Get image paths first
            img_paths = self._get_image_paths(img_dir)
            
            # Get corresponding mask paths based on image filenames
            mask_paths = []
            for img_path in img_paths:
                # Extract filename from image path
                img_filename = os.path.basename(img_path)
                # Create corresponding mask path
                mask_path = os.path.join(mask_dir, img_filename)
                
                # Check if mask file exists (handle different extensions if needed)
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                else:
                    # Try different extensions for mask files
                    img_name_no_ext = os.path.splitext(img_filename)[0]
                    for mask_ext in ['.png', '.jpg', '.jpeg']:
                        potential_mask_path = os.path.join(mask_dir, img_name_no_ext + mask_ext)
                        if os.path.exists(potential_mask_path):
                            mask_paths.append(potential_mask_path)
                            break
                    else:
                        # If no mask found, skip this image
                        continue
            
            # Now img_paths and mask_paths should have matching filenames
            # But we need to ensure they have the same length
            if len(mask_paths) != len(img_paths):
                # Filter img_paths to only include those with corresponding masks
                valid_img_paths = []
                valid_mask_paths = []
                
                for img_path in img_paths:
                    img_filename = os.path.basename(img_path)
                    img_name_no_ext = os.path.splitext(img_filename)[0]
                    
                    # Look for corresponding mask
                    mask_path = None
                    for mask_ext in ['.png', '.jpg', '.jpeg']:
                        potential_mask_path = os.path.join(mask_dir, img_name_no_ext + mask_ext)
                        if os.path.exists(potential_mask_path):
                            mask_path = potential_mask_path
                            break
                    
                    if mask_path is not None:
                        valid_img_paths.append(img_path)
                        valid_mask_paths.append(mask_path)
                
                img_paths = valid_img_paths
                mask_paths = valid_mask_paths
            
            # Sample frame indices
            start_idx, end_idx = self._get_frame_indices(len(img_paths))
            
            # Load and augment images and masks together
            images, masks = self._load_images_and_masks(img_paths, mask_paths, start_idx, end_idx)
            # print(mask_paths)

            # Create sample dictionary
            sample = {
                'images': images,           # [S, 3, H, W] - input video frames
                'motion_masks': masks,      # [S, H, W] - ground truth motion masks
                'sequence_name': seq_name,  # string - sequence identifier
                'frame_indices': torch.tensor(list(range(start_idx, end_idx, self.sample_stride))),  # [S] - frame indices
            }
            
            # Add valid masks (all pixels are valid by default)
            # You can modify this if you have invalid regions to ignore
            valid_masks = torch.ones_like(masks)  # [S, H, W]
            sample['valid_masks'] = valid_masks
            
            return sample
        except Exception as e:
            # 打印警告信息
            print(f"[WARN] Skipping broken sample at idx={idx}, seq={self.sample_list[idx]}: {e}")
            return None


def create_motion_seg_dataloader(cfg, split='train'):
    """
    Factory function to create motion segmentation dataloader with augmentation support
    """
    # Dataset parameters
    dataset_params = {
        'data_dir': cfg.data_dir,
        'split': split,
        'img_size': getattr(cfg, 'img_size', 518),
        'sequence_length': getattr(cfg, 'sequence_length', 8),
        'sample_stride': getattr(cfg, 'sample_stride', 1),
        'enable_augmentation': getattr(cfg, 'enable_augmentation', split == 'train'),
        'crop_strategy': getattr(cfg, 'crop_strategy', 'smart' if split == 'train' else 'center'),
    }
    
    # Create dataset
    dataset = MotionSegmentationDataset(**dataset_params)
    
    # Dataloader parameters
    dataloader_params = {
        'batch_size': getattr(cfg, 'batch_size', 2),
        'shuffle': split == 'train',
        'num_workers': getattr(cfg, 'num_workers', 4),
        'pin_memory': True,
        'drop_last': split == 'train',
    }
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, **dataloader_params)
    
    return dataloader


# Example configuration class for motion segmentation
class MotionSegConfig:
    def __init__(self):
        # Data settings
        self.data_dir = "/data2/hxk/Recon/SegAnyMo/data/HOI4D"
        self.img_size = 518
        self.sequence_length = 8  # Number of frames per sample
        self.sample_stride = 1    # Frame sampling stride
        
        # Training settings
        self.batch_size = 2       # Small batch size due to video sequences
        self.num_workers = 4
        self.learning_rate = 1e-4
        self.num_epochs = 100
        
        # Model settings  
        self.vggt_model_path = "/data2/hxk/model.pt"
        
        # Logging
        self.log_dir = "logs/motion_segmentation"
        self.print_freq = 50


def test_motion_seg_dataset():
    """
    Test function to verify dataset loading
    """
    cfg = MotionSegConfig()
    cfg.data_dir = "/data2/hxk/Recon/SegAnyMo/data/HOI4D"  # Update this path
    
    # try:
    # Create dataset
    dataset = MotionSegmentationDataset(
        data_dir=cfg.data_dir,
        split='train',
        img_size=518,
        sequence_length=4,
        sample_stride=1
    )
    
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    # Test loading one sample
    if len(dataset) > 0:
        sample = dataset[1594]
        print(sample)
        print(f"Sample keys: {sample.keys()}")
        print(f"Images shape: {sample['images'].shape}")
        print(f"Motion masks shape: {sample['motion_masks'].shape}")
        print(f"Sequence name: {sample['sequence_name']}")
        print(f"Images range: [{sample['images'].min():.3f}, {sample['images'].max():.3f}]")
        print(f"Masks range: [{sample['motion_masks'].min():.3f}, {sample['motion_masks'].max():.3f}]")
        print("Dataset test passed!")
    else:
        print("Warning: Dataset is empty")
            
    # except Exception as e:
    # print(f"Dataset test failed: {e}")


if __name__ == "__main__":
    test_motion_seg_dataset()