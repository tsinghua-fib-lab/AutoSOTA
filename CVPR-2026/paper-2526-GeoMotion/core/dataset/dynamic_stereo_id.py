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
# from .kubric import parse_tapir_track_info, load_target_tracks
import h5py
from torchvision.io import read_image
from torchvision import transforms

class Stereo_dataset(Dataset):
    def __init__(self, data_dir, split, transform=None,
                 factor=1, depth_type="depths",
                 track_method="gt_point_tracks", load_dino=True, dino_later=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = osp.join(data_dir,split)
        self.training = split == "train"
        self.factor = factor
        self.depth_type = depth_type
        self.track_method = track_method
        self.load_dino = load_dino
        self.dino_later = dino_later
        
        self.sample_list = [item for item in sorted(os.listdir(self.data_dir))
                    if os.path.isdir(os.path.join(self.data_dir, item))]

        img_dirs_list, mask_dirs_list, track_dirs_list = [], [], []    

        for seq_name in tqdm(self.sample_list, desc="loading data"):
            # Load image list.
            img_dir = osp.join(self.data_dir,seq_name, "images")
            img_dirs_list.append(img_dir)
            mask_dirs_list.append(img_dir.replace("images", "dynamic_masks"))
            track_dirs_list.append(img_dir.replace("images", f'{self.track_method}_random'))
            
        self.img_dirs_list = img_dirs_list
        self.mask_dirs_list = mask_dirs_list
        self.track_dirs_list = track_dirs_list
        
        # get image size, should under 1024x1024
        img_dir = self.img_dirs_list[0]
        img_paths = sorted(glob(osp.join(img_dir, "*.png"))) + sorted(
            glob(osp.join(img_dir, "*.jpg")) + sorted(glob(osp.join(img_dir, "*.jpeg")))
        )
        img_paths = img_paths[::4]
        frame_names = [osp.splitext(osp.basename(p))[0] for p in img_paths]
        num_frames = len(img_paths)

        img = torch.from_numpy(np.array([iio.imread(img_paths[0])])).squeeze().permute(2, 0, 1) # [1, 512, 512, 3]
        self.image_size = [img.shape[1],img.shape[2]]
        self.step = 8
        self.q_ts = list(range(0, num_frames, self.step))
        # tracks_2d = load_target_tracks(frame_names[self.q_ts[0]], self.track_dirs_list[0], num_frames, frame_names)
        # num = tracks_2d.shape[0]
        # max_num = num * len(self.q_ts)
        num=0
        max_num = num
        random_num = [256, 512, 768, 1024, 1536, 2048]
        random_num = [x for x in random_num if x <= max_num and x >= num]
        if num not in random_num:
            random_num.append(num)
        self.random_num = sorted(random_num)
        
        print("data list loading done !")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """
        Be sure to note that visible mask cannot have a row with all False values.
        """

        # Load images dir
        img_dir = self.img_dirs_list[idx]

        # Load dynamic masks. need normilize [0,1]
        mask_dir = self.mask_dirs_list[idx]
        mask_paths = sorted(glob(osp.join(mask_dir, "*.png"))) + sorted(
            glob(osp.join(mask_dir, "*.jpg")) + sorted(glob(osp.join(mask_dir, "*.jpeg")))
        )
        # random choose image sets
        random_start = random.choice([0, 1, 2, 3])
        mask_paths = mask_paths[random_start::4]
        mask_list = []
        for path in mask_paths: # 15s
            mask_tensor = read_image(path) / 255.0
            mask_tensor = mask_tensor.squeeze(0)
            # mask = np.array(Image.open(path).convert('L')) / 255
            # mask_tensor = torch.from_numpy(mask)
            mask_list.append(mask_tensor)
        dynamic_masks = torch.stack(mask_list, dim=0)
        
        # Load depths. [24,256,256]
        depth_dir = img_dir.replace("images", self.depth_type)
        depth_paths = sorted(glob(osp.join(depth_dir, "*.png"))) + sorted(
            glob(osp.join(depth_dir, "*.jpg")) + sorted(glob(osp.join(depth_dir, "*.jpeg")))
        )
        depth_paths = depth_paths[random_start::4]
        depth_list = []
        for path in depth_paths: # 13s
            depth_img = np.array(Image.open(path).convert('L') )
            depth_image_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
            depth_tensor = torch.from_numpy(depth_image_normalized)
            depth_list.append(depth_tensor)
        depths = torch.stack(depth_list, dim=0)
                        
        # Load 2D tracks prompt. [2, N, L]
        frame_names = [osp.splitext(osp.basename(p))[0] for p in depth_paths]
        num_frames = len(frame_names)
        
        if self.track_method == 'bootstapir':
            tracks_dir = self.track_dirs_list[idx]
            total_num = random.choice(self.random_num)
            q_t = random.choice(self.q_ts)
            # tracks_2d = load_target_tracks(frame_names[q_t], tracks_dir, num_frames, frame_names)
            # track_2d, occs, dists = (
            #     tracks_2d[..., :2],
            #     tracks_2d[..., 2],
            #     tracks_2d[..., 3],
            # )
            # visibles, invisibles, confidences, visib_value, confi_value = parse_tapir_track_info(occs, dists)

            # randomly pick pts to get a new track
            # num_sample = int(total_num * ratio)
            pts_num = track_2d.shape[0]
            indices = torch.randperm(pts_num)
            selected_indices = indices[:total_num]
            
            track_2d = track_2d[selected_indices]
            visibles = visibles[selected_indices]
            confidences = confidences[selected_indices]
            visib_value, confi_value = visib_value[selected_indices], confi_value[selected_indices]
            
            # Check if all rows in visibles are False = if all rows in ~visible are true
            rows_all_false = torch.all((~visibles), dim=1)
            track_2d = track_2d[~rows_all_false,:,:]
            visibles = visibles[~rows_all_false,:]
            confidences = confidences[~rows_all_false,:]
            visib_value, confi_value = visib_value[~rows_all_false,:], confi_value[~rows_all_false,:]
            
            cols_all_false = torch.all(~visibles.permute(1,0), dim=1)
            
            for t in range(cols_all_false.size(0)):
                if cols_all_false[t]:
                    max_v, max_conf_index = torch.max(confidences[:, t], dim=0)
                    print(f'get invisible time and confi: {t} and {max_v}')
                    
                    visibles[max_conf_index, t] = True
            
            track_2d = track_2d.permute(2,0,1)
            visible_mask = visibles
                
        elif self.track_method == 'gt_point_tracks':
            track_dir = self.track_dirs_list[idx]
            track_path = osp.join(track_dir,"gt_tracks.npy")
            track_2d = torch.from_numpy(np.load(track_path))
            track_2d = track_2d.permute(2,0,1)
        elif self.track_method == 'cotracker':
            track_dir = self.track_dirs_list[idx]
            track_path = osp.join(track_dir,"pred_tracks.npy")
            track_2d = torch.from_numpy(np.load(track_path))
            track_2d = track_2d.permute(2,1,0)
            
            visible_path = osp.join(track_dir,"pred_visibility.npy")
            visible_mask = torch.from_numpy(np.load(visible_path)).permute(1,0)

        # Load DINO feature
        if self.load_dino:
            dino_dir = img_dir.replace("images", "dinos")
            dino_paths = sorted(glob(osp.join(dino_dir, "*.h5")))
            dino_paths = dino_paths[random_start::4]
            # with h5py.File(dino_paths[q_t], 'r') as hf:
            #     features = hf['dinos'][:]
            features = np.load(dino_paths[q_t], mmap_mode="r").squeeze()
            dino_qt = torch.from_numpy(features).unsqueeze(0)
                            
        sample = {'depths': depths, 
                'track_2d': track_2d,  'dynamic_mask': dynamic_masks,
                'video_dir': img_dir, 'visible_mask': visible_mask,
                'confidences': confidences, 'q_t': q_t, 
                'visib_value': visib_value, 'confi_value': confi_value,
                }
        sample['case_name'] = f'{osp.basename(osp.dirname(img_dir))}'
        if self.load_dino:
            sample['dinos'] = dino_qt
        return sample
    

class StereoMotionSegmentationDataset(Dataset):
    """
    Dataset for motion segmentation using Stereo dataset structure
    Adapted from Stereo_dataset to work with MotionSegmentationDataset interface
    """
    def __init__(self, data_dir, split, transform=None, 
                 img_size=518, sequence_length=8, sample_stride=1,
                 enable_augmentation=True, crop_strategy='center',
                 factor=1, depth_type="depths", track_method="gt_point_tracks",
                 load_dino=False, dino_later=False):
        """
        Args:
            data_dir: Root directory containing split folders (train/test/val)
            split: 'train' or 'test' or 'val'
            transform: Optional transforms for images
            img_size: Target image size for VGGT (default 518)
            sequence_length: Number of frames to sample from each video
            sample_stride: Stride for sampling frames (1 means consecutive frames)
            enable_augmentation: Whether to apply data augmentation
            crop_strategy: 'center', 'random', or 'smart' (aspect-ratio aware)
            factor: Factor for image scaling (from original Stereo_dataset)
            depth_type: Type of depth data to load
            track_method: Method for track loading
            load_dino: Whether to load DINO features
            dino_later: Whether to load DINO features later
        """
        self.split = split
        self.data_dir = osp.join(data_dir, split)
        self.training = split == "train"
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.sample_stride = sample_stride
        self.enable_augmentation = enable_augmentation and self.training
        self.crop_strategy = crop_strategy
        
        # Stereo dataset specific parameters
        self.factor = factor
        self.depth_type = depth_type
        self.track_method = track_method
        self.load_dino = load_dino
        self.dino_later = dino_later
        
        # Set up transforms
        if transform is None:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = transform
        
        # Load sample list from data directory
        self.sample_list = [item for item in sorted(os.listdir(self.data_dir))
                           if os.path.isdir(os.path.join(self.data_dir, item))]
        
        # Prepare directories
        img_dirs_list, mask_dirs_list, depth_dirs_list, track_dirs_list = [], [], [], []
        valid_samples = []
        
        for seq_name in tqdm(self.sample_list, desc=f"Loading stereo data for '{split}' split"):
            img_dir = osp.join(self.data_dir, seq_name, "images")
            mask_dir = osp.join(self.data_dir, seq_name, "dynamic_masks")
            depth_dir = osp.join(self.data_dir, seq_name, self.depth_type)
            track_dir = osp.join(self.data_dir, seq_name, f'{self.track_method}_random')
            
            # Check if required directories exist
            if not os.path.exists(img_dir):
                print(f"Warning: image directory not found for {seq_name}, skipping.")
                continue
            if not os.path.exists(mask_dir):
                print(f"Warning: mask directory not found for {seq_name}, skipping.")
                continue
            
            # Get image paths and check if we have enough frames
            img_paths = self._get_image_paths(img_dir)
            if len(img_paths) < self.sequence_length:
                print(f"Warning: {seq_name} has only {len(img_paths)} images, need at least {self.sequence_length}. Skipping.")
                continue
            
            # Check if masks exist for images
            mask_paths = self._get_image_paths(mask_dir)
            if len(mask_paths) < self.sequence_length:
                print(f"Warning: {seq_name} has only {len(mask_paths)} masks, need at least {self.sequence_length}. Skipping.")
                continue
            
            # Add to valid lists
            img_dirs_list.append(img_dir)
            mask_dirs_list.append(mask_dir)
            depth_dirs_list.append(depth_dir)
            track_dirs_list.append(track_dir)
            valid_samples.append(seq_name)
        
        self.sample_list = valid_samples
        self.img_dirs_list = img_dirs_list
        self.mask_dirs_list = mask_dirs_list
        self.depth_dirs_list = depth_dirs_list
        self.track_dirs_list = track_dirs_list
        
        if len(self.sample_list) == 0:
            raise RuntimeError(f"No valid sequences found for split '{self.split}' in directory {self.data_dir}")
        
        print(f"Loaded {len(self.sample_list)} valid sequences for '{split}' split")
        
        # Get reference image size
        if len(self.img_dirs_list) > 0:
            sample_img_path = self._get_image_paths(self.img_dirs_list[0])[0]
            img = np.array(iio.imread(sample_img_path))
            if len(img.shape) == 3:
                self.original_image_size = (img.shape[1], img.shape[0])  # (W, H)
            else:
                self.original_image_size = (img.shape[0], img.shape[1])  # (W, H)
            print(f"Original image size: {self.original_image_size}")
        
        # Set up frame sampling parameters (from original Stereo_dataset)
        self.step = 4  # Sample every 4th frame (as in original)
    
    def _get_image_paths(self, img_dir):
        """Get sorted list of image paths from directory"""
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob(osp.join(img_dir, ext)))
        return sorted(img_paths)
    
    def _apply_augmentation(self, images, masks):
        """
        Apply data augmentation to image-mask pairs
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
        
        # Color augmentation (only for images, not masks)
        if random.random() < 0.3:  # 30% chance
            # Random brightness, contrast, saturation
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            saturation_factor = random.uniform(0.8, 1.2)
            
            color_transform = transforms.ColorJitter(
                brightness=(brightness_factor, brightness_factor),
                contrast=(contrast_factor, contrast_factor),
                saturation=(saturation_factor, saturation_factor)
            )
            
            cropped_images = [color_transform(img) for img in cropped_images]
        
        return cropped_images, cropped_masks
    
    def _smart_resize(self, images, masks, target_size):
        """
        Smart resize that maintains aspect ratio and quality
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
    
    def _get_frame_indices(self, total_frames):
        """Get start and end indices for frame sampling using stereo dataset strategy"""
        # Sample every step-th frame (like original stereo dataset)
        available_indices = list(range(0, total_frames, self.step))
        
        if len(available_indices) < self.sequence_length:
            # If not enough frames with step sampling, use consecutive frames
            max_start_idx = max(0, total_frames - self.sequence_length * self.sample_stride)
            if self.training:
                start_idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
            else:
                start_idx = max_start_idx // 2
            end_idx = start_idx + self.sequence_length * self.sample_stride
            return start_idx, end_idx, False  # False means no step sampling
        
        # Use step sampling
        max_start_seq = len(available_indices) - self.sequence_length
        if self.training:
            start_seq_idx = random.randint(0, max_start_seq) if max_start_seq > 0 else 0
        else:
            start_seq_idx = max_start_seq // 2
        
        # Get actual frame indices
        selected_indices = available_indices[start_seq_idx:start_seq_idx + self.sequence_length]
        return min(selected_indices), max(selected_indices) + 1, True  # True means step sampling used
    
    def _load_images_and_masks_stereo_style(self, img_dir, mask_dir, random_start=None):
        """
        Load images and masks using stereo dataset style (with random start and step sampling)
        """
        # Get all paths
        img_paths = self._get_image_paths(img_dir)
        mask_paths = self._get_image_paths(mask_dir)
        id_paths = self._get_image_paths(mask_dir.replace('/dynamic_masks', '/instance_id_maps'))
        
        # Apply random start like in original stereo dataset
        if random_start is None:
            random_start = random.choice([0, 1, 2, 3]) if self.training else 0
        
        # Sample paths with step
        img_paths = img_paths[random_start::self.step]
        mask_paths = mask_paths[random_start::self.step]
        
        # Ensure we have enough frames
        min_frames = min(len(img_paths), len(mask_paths))
        if min_frames < self.sequence_length:
            # Fall back to consecutive sampling
            all_img_paths = self._get_image_paths(img_dir)
            all_mask_paths = self._get_image_paths(mask_dir)
            max_start = max(0, min(len(all_img_paths), len(all_mask_paths)) - self.sequence_length)
            start_idx = random.randint(0, max_start) if self.training and max_start > 0 else 0
            img_paths = all_img_paths[start_idx:start_idx + self.sequence_length]
            mask_paths = all_mask_paths[start_idx:start_idx + self.sequence_length]
        else:
            # Sample sequence_length frames
            if self.training:
                start_idx = random.randint(0, max(0, min_frames - self.sequence_length))
            else:
                start_idx = max(0, min_frames - self.sequence_length) // 2
            img_paths = img_paths[start_idx:start_idx + self.sequence_length]
            mask_paths = mask_paths[start_idx:start_idx + self.sequence_length]
        
        # Load PIL images and masks
        pil_images = []
        pil_masks = []
        
        for img_path, mask_path in zip(img_paths, mask_paths):
            # Load image
            img = Image.open(img_path).convert('RGB')
            pil_images.append(img)
            
            # Load mask and normalize to [0,1] like in stereo dataset
            mask = Image.open(mask_path).convert('L')
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
            # Convert image to tensor and normalize
            img_tensor = transforms.ToTensor()(img)
            img_tensor = self.normalize(img_tensor)
            images.append(img_tensor)
            
            # Convert mask to tensor and normalize to [0,1] like stereo dataset
            mask_array = np.array(mask)
            mask_normalized = mask_array / 255.0
            mask_tensor = torch.from_numpy(mask_normalized).float()
            # Ensure binary mask
            import pdb;pdb.set_trace()
            mask_tensor = (mask_tensor > 0).float()
            masks.append(mask_tensor)
        
        # Stack into tensors [S, 3, H, W] and [S, H, W]
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        
        return images, masks
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        # Get directories
        img_dir = self.img_dirs_list[idx]
        mask_dir = self.mask_dirs_list[idx]
        seq_name = self.sample_list[idx]
        
        # Load images and masks using stereo style
        images, masks = self._load_images_and_masks_stereo_style(img_dir, mask_dir)
        
        # Create frame indices
        frame_indices = torch.tensor(list(range(images.shape[0])))
        
        # Create sample dictionary (compatible with MotionSegmentationDataset interface)
        sample = {
            'images': images,           # [S, 3, H, W] - input video frames
            'motion_masks': masks,      # [S, H, W] - ground truth motion masks
            'sequence_name': seq_name,  # string - sequence identifier
            'frame_indices': frame_indices,  # [S] - frame indices
        }
        
        # Add valid masks (all pixels are valid by default)
        valid_masks = torch.ones_like(masks)  # [S, H, W]
        sample['valid_masks'] = valid_masks
        
        return sample


def create_stereo_motion_dataset(data_dir, split, **kwargs):
    """
    Factory function to create StereoMotionSegmentationDataset
    
    Args:
        data_dir: Root directory containing train/test/val folders
        split: 'train', 'test', or 'val'
        **kwargs: Additional arguments for dataset configuration
    
    Returns:
        StereoMotionSegmentationDataset instance
    """
    return StereoMotionSegmentationDataset(data_dir, split, **kwargs)

def test_motion_seg_dataset():
    """
    Test function to verify dataset loading
    """
    data_dir = "/data0/hexiankang/code/SegAnyMo/data/dynamic_replica_data"  # Update this path
    
    # try:
    # Create dataset
    dataset = StereoMotionSegmentationDataset(
        data_dir=data_dir,
        split='train',
        img_size=518,
        sequence_length=8,
        sample_stride=12
    )
    
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    # Test loading one sample
    if len(dataset) > 0:
        sample = dataset[100]
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