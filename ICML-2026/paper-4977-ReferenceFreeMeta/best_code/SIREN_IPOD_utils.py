
import os
import time
import copy
import glob
import random
import gc
import threading
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import fft

# Custom module imports
from utils import build_coordinate_train, MYTVLoss
from model_siren import siren_model

# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe division handling division by zero"""
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def normalize01(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    if len(img.shape) == 3:
        nimg = len(img)
    else:
        nimg = 1
        r, c = img.shape
        img = np.reshape(img, (nimg, r, c))
    
    img2 = np.empty(img.shape, dtype=img.dtype)
    for i in range(nimg):
        img2[i] = div0(img[i] - img[i].min(), img[i].ptp())
    
    return np.squeeze(img2).astype(img.dtype)


def calculate_psnr(pred: np.ndarray, target: np.ndarray, 
                   data_range: Optional[float] = None) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    pred_abs = normalize01(np.abs(pred))
    target_abs = normalize01(np.abs(target))
    
    if data_range is None:
        data_range = np.max(target_abs)
    
    mse = np.mean((pred_abs - target_abs) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(data_range / np.sqrt(mse))


def calculate_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM)"""
    pred_abs = normalize01(np.abs(pred))
    target_abs = normalize01(np.abs(target))
    
    pred_norm = pred_abs / np.max(pred_abs)
    target_norm = target_abs / np.max(target_abs)
    
    # SSIM constants
    K1, K2, L = 0.01, 0.03, 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # Statistics
    mu_x = np.mean(pred_norm)
    mu_y = np.mean(target_norm)
    sigma_x2 = np.var(pred_norm)
    sigma_y2 = np.var(target_norm)
    sigma_xy = np.mean((pred_norm - mu_x) * (target_norm - mu_y))
    
    # SSIM calculation
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    
    return numerator / denominator


# ============================================================================
# Dataset Class
# ============================================================================

class OptimizedMRIMetaDataset(Dataset):
    """
    Optimized MRI Meta-Learning Dataset
    Supports intelligent caching, preloading, and multi-threaded data loading
    """
    
    def __init__(
        self,
        data_dir: str,
        tasks: Optional[List[str]] = None,
        load_captions: bool = True,
        cache_size: int = 50,
        preload_samples: bool = True,
        num_workers: int = 4,
        memory_limit_gb: float = 8
    ):
        """
        Args:
            data_dir: Root directory of data
            tasks: List of specific task folder names
            load_captions: Whether to load caption data
            cache_size: Number of tasks to cache
            preload_samples: Whether to preload samples into memory
            num_workers: Number of data loading threads
            memory_limit_gb: Memory usage limit in GB
        """
        self.data_dir = data_dir
        self.load_captions = load_captions
        self.cache_size = cache_size
        self.preload_samples = preload_samples
        self.num_workers = num_workers
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        
        # Cache management
        self.sample_cache: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()
        self.cache_access_count: Dict[str, int] = {}
        self.current_memory_usage = 0
        
        # Scan task directories
        self._scan_tasks(tasks)
        
        # Preload hot samples
        if self.preload_samples:
            self._preload_hot_samples()
    
    def _scan_tasks(self, tasks: Optional[List[str]]) -> None:
        """Scan and validate task directories"""
        if tasks is None:
            self.task_dirs = [
                d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('task_')
            ]
        else:
            self.task_dirs = tasks
        
        self.task_metadata: Dict[str, Dict] = {}
        valid_tasks = []
        
        print("Scanning task directories...")
        for task in tqdm(self.task_dirs, desc="Loading task metadata"):
            task_path = os.path.join(self.data_dir, task)
            sample_files = sorted(glob.glob(os.path.join(task_path, "sample_*.h5")))
            
            if len(sample_files) == 0:
                continue
            
            valid_tasks.append(task)
            
            # Read task parameters
            try:
                with h5py.File(sample_files[0], 'r') as hf:
                    task_params = {key: hf.attrs[key] for key in hf.attrs.keys()}
                    has_captions = 'num_captions' in hf.attrs
                    num_captions = hf.attrs['num_captions'] if has_captions else 0
            except Exception as e:
                print(f"Error reading {sample_files[0]}: {e}")
                task_params = {}
                has_captions = False
                num_captions = 0
            
            self.task_metadata[task] = {
                'task_id': task,
                'task_params': task_params,
                'file_paths': sample_files,
                'has_captions': has_captions,
                'num_captions': num_captions
            }
        
        self.task_dirs = valid_tasks
        total_samples = sum(len(self.task_metadata[t]['file_paths']) for t in valid_tasks)
        
        print(f"Found {len(self.task_dirs)} valid tasks with {total_samples} total samples")
    
    def _estimate_sample_size(self, file_path: str) -> int:
        """Estimate memory footprint of a single sample"""
        try:
            with h5py.File(file_path, 'r') as hf:
                total_size = 0
                for key in ['forward_fft_und', 'mask', 'csmp', 'forward_fft', 'img_full']:
                    if key in hf:
                        total_size += hf[key].size * 8  # Assume complex64/float64
                return total_size
        except:
            return 50 * 1024 * 1024  # Default 50MB
    
    def _preload_hot_samples(self) -> None:
        """Preload frequently used samples into memory"""
        print("Starting sample preloading...")
        
        preload_files = []
        actual_tasks_count = min(self.cache_size, len(self.task_dirs))
        
        # Adjust preload strategy based on memory size
        if self.memory_limit_bytes >= 150 * 1024 * 1024 * 1024:  # 150GB+
            samples_per_task = "all"
            print(f"Large memory mode: preloading all samples from first {actual_tasks_count} tasks")
            
            for i, task_id in enumerate(self.task_dirs[:self.cache_size]):
                task_files = self.task_metadata[task_id]['file_paths']
                preload_files.extend([(task_id, f) for f in task_files])
                
                if i < 5:
                    print(f"  Task {task_id}: {len(task_files)} samples (all)")
                elif i == 5:
                    print(f"  ... {actual_tasks_count-5} more tasks")
                    
        elif self.memory_limit_bytes >= 50 * 1024 * 1024 * 1024:  # 50GB+
            samples_per_task = 30
            print(f"Medium memory mode: {samples_per_task} samples per task from first {actual_tasks_count} tasks")
            
            for i, task_id in enumerate(self.task_dirs[:self.cache_size]):
                task_files = self.task_metadata[task_id]['file_paths']
                num_samples_to_load = min(samples_per_task, len(task_files))
                selected_files = random.sample(task_files, num_samples_to_load)
                preload_files.extend([(task_id, f) for f in selected_files])
                
                if i < 5:
                    print(f"  Task {task_id}: {num_samples_to_load}/{len(task_files)} samples")
        else:
            samples_per_task = 3
            print(f"Standard memory mode: {samples_per_task} samples per task from first {actual_tasks_count} tasks")
            
            for i, task_id in enumerate(self.task_dirs[:self.cache_size]):
                task_files = self.task_metadata[task_id]['file_paths']
                num_samples_to_load = min(samples_per_task, len(task_files))
                selected_files = random.sample(task_files, num_samples_to_load)
                preload_files.extend([(task_id, f) for f in selected_files])
                
                if i < 5:
                    print(f"  Task {task_id}: {num_samples_to_load}/{len(task_files)} samples")
        
        actual_total = len(preload_files)
        estimated_memory = actual_total * 50  # Estimate 50MB per sample
        
        print(f"Preload statistics:")
        print(f"  Target tasks: {actual_tasks_count}")
        print(f"  Preload mode: {samples_per_task}")
        print(f"  Total samples: {actual_total}")
        print(f"  Estimated memory: {estimated_memory:.1f}MB")
        print(f"  Memory limit: {self.memory_limit_bytes/1024/1024:.1f}MB")
        
        # Parallel loading with thread pool
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {
                executor.submit(self._load_sample_to_cache, task_id, file_path): (task_id, file_path)
                for task_id, file_path in preload_files
            }
            
            for future in tqdm(as_completed(future_to_file), 
                             total=len(future_to_file), 
                             desc="Preloading progress"):
                task_id, file_path = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Preload failed {file_path}: {e}")
        
        print(f"Preload complete, cache size: {len(self.sample_cache)} samples")
        print(f"Memory usage: {self.current_memory_usage / 1024 / 1024:.1f} MB")
    
    def _load_sample_to_cache(self, task_id: str, file_path: str) -> Optional[Dict]:
        """Load sample to cache"""
        cache_key = f"{task_id}_{os.path.basename(file_path)}"
        
        # Check if already cached
        with self.cache_lock:
            if cache_key in self.sample_cache:
                return self.sample_cache[cache_key]
        
        # Load sample
        sample_data = self._load_sample_from_disk(file_path)
        if sample_data is None:
            return None
        
        # Estimate size
        sample_size = self._estimate_memory_usage(sample_data)
        
        # Check memory limit
        with self.cache_lock:
            # Evict least used if memory insufficient
            while (self.current_memory_usage + sample_size > self.memory_limit_bytes and 
                   len(self.sample_cache) > 0):
                self._evict_least_used()
            
            # Add to cache
            self.sample_cache[cache_key] = sample_data
            self.cache_access_count[cache_key] = 1
            self.current_memory_usage += sample_size
        
        return sample_data
    
    def _estimate_memory_usage(self, sample_data: Dict) -> int:
        """Estimate sample memory usage"""
        total_size = 0
        for key, value in sample_data.items():
            if isinstance(value, np.ndarray):
                total_size += value.nbytes
            elif isinstance(value, torch.Tensor):
                total_size += value.element_size() * value.nelement()
            elif isinstance(value, (list, str)):
                total_size += 1024  # Estimate
        return total_size
    
    def _evict_least_used(self) -> None:
        """Remove least recently used cache item"""
        if not self.sample_cache:
            return
        
        # Find item with lowest access count
        least_used_key = min(self.cache_access_count.keys(), 
                           key=lambda k: self.cache_access_count[k])
        
        # Estimate freed memory
        sample_data = self.sample_cache[least_used_key]
        freed_size = self._estimate_memory_usage(sample_data)
        
        # Delete cache
        del self.sample_cache[least_used_key]
        del self.cache_access_count[least_used_key]
        self.current_memory_usage -= freed_size
    
    def __len__(self) -> int:
        return len(self.task_dirs)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get task metadata"""
        task_id = self.task_dirs[idx]
        return self.task_metadata[task_id]
    
    def get_samples(
        self,
        task_idx: int,
        num_samples: Optional[int] = None,
        indices: Optional[List[int]] = None,
        caption_mode: str = 'specific'
    ) -> Dict:
        """
        Efficiently get task samples, prioritizing cache retrieval
        
        Args:
            task_idx: Task index
            num_samples: Number of samples to retrieve
            indices: Specific sample indices
            caption_mode: Caption selection mode ('specific', 'random', 'all', or int)
            
        Returns:
            Dictionary containing task_id, task_params, samples, and has_captions
        """
        task_id = self.task_dirs[task_idx]
        file_paths = self.task_metadata[task_id]['file_paths']
        
        # Determine which file indices to load
        if indices is not None:
            selected_indices = indices
        elif num_samples is not None:
            if num_samples >= len(file_paths):
                selected_indices = list(range(len(file_paths)))
            else:
                selected_indices = random.sample(range(len(file_paths)), num_samples)
        else:
            selected_indices = list(range(len(file_paths)))
        
        # Batch load samples
        samples = []
        cache_hits = 0
        cache_misses = 0
        
        # Separate cached and uncached files
        cached_samples = []
        uncached_indices = []
        
        for idx in selected_indices:
            file_path = file_paths[idx]
            cache_key = f"{task_id}_{os.path.basename(file_path)}"
            
            with self.cache_lock:
                if cache_key in self.sample_cache:
                    # Cache hit
                    sample_data = copy.deepcopy(self.sample_cache[cache_key])
                    # Re-select caption (because it's random)
                    self._update_caption_selection(sample_data, caption_mode)
                    cached_samples.append((idx, sample_data))
                    cache_hits += 1
                    # Update access count
                    self.cache_access_count[cache_key] += 1
                else:
                    # Cache miss
                    uncached_indices.append(idx)
                    cache_misses += 1
        
        # Parallel load uncached samples
        if uncached_indices:
            uncached_files = [file_paths[idx] for idx in uncached_indices]
            
            if len(uncached_files) == 1:
                # Single file direct load
                sample_data = self._load_sample_from_disk(uncached_files[0])
                if sample_data:
                    self._update_caption_selection(sample_data, caption_mode)
                    cached_samples.append((uncached_indices[0], sample_data))
                    # Async add to cache
                    self._async_cache_sample(task_id, uncached_files[0], sample_data)
            else:
                # Multiple files parallel load
                with ThreadPoolExecutor(max_workers=min(self.num_workers, len(uncached_files))) as executor:
                    future_to_idx = {
                        executor.submit(self._load_sample_from_disk, file_path): (idx, file_path)
                        for idx, file_path in zip(uncached_indices, uncached_files)
                    }
                    
                    for future in as_completed(future_to_idx):
                        idx, file_path = future_to_idx[future]
                        try:
                            sample_data = future.result()
                            if sample_data:
                                self._update_caption_selection(sample_data, caption_mode)
                                cached_samples.append((idx, sample_data))
                                # Async add to cache
                                self._async_cache_sample(task_id, file_path, sample_data)
                        except Exception as e:
                            print(f"Load failed {file_path}: {e}")
        
        # Sort by original order and extract samples
        cached_samples.sort(key=lambda x: x[0])
        samples = [sample for _, sample in cached_samples]
        
        # Print cache statistics (reduce output frequency)
        if cache_hits + cache_misses > 0 and random.random() < 0.1:  # 10% probability
            hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            print(f"Cache hit rate: {hit_rate:.1f}% ({cache_hits}/{cache_hits + cache_misses})")
        
        return {
            'task_id': task_id,
            'task_params': self.task_metadata[task_id]['task_params'],
            'samples': samples,
            'has_captions': self.task_metadata[task_id]['has_captions']
        }
    
    def _async_cache_sample(self, task_id: str, file_path: str, sample_data: Dict) -> None:
        """Asynchronously add sample to cache"""
        def cache_worker():
            cache_key = f"{task_id}_{os.path.basename(file_path)}"
            sample_size = self._estimate_memory_usage(sample_data)
            
            with self.cache_lock:
                # Check memory limit
                while (self.current_memory_usage + sample_size > self.memory_limit_bytes and 
                       len(self.sample_cache) > 0):
                    self._evict_least_used()
                
                # Add to cache
                if cache_key not in self.sample_cache:
                    self.sample_cache[cache_key] = copy.deepcopy(sample_data)
                    self.cache_access_count[cache_key] = 1
                    self.current_memory_usage += sample_size
        
        # Execute cache operation in background thread
        threading.Thread(target=cache_worker, daemon=True).start()
    
    def _update_caption_selection(self, sample_data: Dict, caption_mode: str) -> None:
        """Update sample's caption selection"""
        if not self.load_captions or not sample_data.get('has_captions'):
            return
        
        captions = sample_data.get('captions', [])
        if not captions:
            return
        
        if caption_mode == 'random':
            selected_caption = random.choice(captions)
        elif caption_mode == 'all':
            selected_caption = captions
        elif isinstance(caption_mode, int) and 0 <= caption_mode < len(captions):
            selected_caption = captions[caption_mode]
        elif caption_mode == 'specific':
            spe_index = random.choice([0, 2, 4])
            selected_caption = captions[spe_index] if spe_index < len(captions) else captions[0]
        else:
            selected_caption = captions[0]
        
        sample_data['selected_caption'] = selected_caption
    
    def _load_sample_from_disk(self, file_path: str) -> Optional[Dict]:
        """Load single sample from disk"""
        try:
            with h5py.File(file_path, 'r') as hf:
                # Extract basic MRI data
                forward_fft_und = hf['forward_fft_und'][:]
                mask = hf['mask'][:]
                csmp = hf['csmp'][:]
                forward_fft = hf['forward_fft'][:]
                img_full = hf['img_full'][:]
                slice_idx = hf['slice_idx'][()]
                
                # Load caption data
                captions = None
                if self.load_captions and 'num_captions' in hf.attrs:
                    num_captions = hf.attrs['num_captions']
                    captions = []
                    
                    for i in range(num_captions):
                        caption_key = f'caption_{i}'
                        if caption_key in hf:
                            caption = hf[caption_key][()]
                            if isinstance(caption, bytes):
                                caption = caption.decode('utf-8')
                            captions.append(caption)
                
                # Data preprocessing (done in one pass)
                zf_img = np.sum(
                    fft.fftshift(fft.ifft2(fft.fftshift(forward_fft_und.transpose(1,2,0), axes=(0,1)), axes=(0,1)), axes=(0,1)) * 
                    np.conj(csmp.transpose(1,2,0)), 
                    axis=2
                )
                norm_factor = np.max(np.abs(zf_img))
                gt_ksp = forward_fft / norm_factor
                gt_img = img_full / norm_factor
                
                # Dimension transpose
                mask_transposed = mask.transpose(1, 2, 0)
                gt_ksp_transposed = gt_ksp.transpose(1, 2, 0)
                csmp_transposed = csmp.transpose(1, 2, 0)
                
                # Generate coordinates
                nRow, nCol = gt_img.shape
                coordinates = build_coordinate_train(L_RO=nRow, L_PE=nCol)
                
                # Build return data
                sample_data = {
                    'forward_fft_und': forward_fft_und,
                    'mask': mask,
                    'mask_transposed': mask_transposed,
                    'csmp': csmp,
                    'csmp_transposed': csmp_transposed,
                    'forward_fft': forward_fft,
                    'gt_ksp': gt_ksp,
                    'gt_ksp_transposed': gt_ksp_transposed,
                    'gt_img': gt_img,
                    'coordinates': coordinates,
                    'norm_factor': norm_factor,
                    'slice_idx': slice_idx,
                    'captions': captions,
                    'selected_caption': captions[0] if captions else None,
                    'has_captions': captions is not None
                }
                
                return sample_data
        
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            return {
                'cache_size': len(self.sample_cache),
                'memory_usage_mb': self.current_memory_usage / 1024 / 1024,
                'memory_limit_mb': self.memory_limit_bytes / 1024 / 1024,
                'hit_rates': dict(self.cache_access_count)
            }
    
    def clear_cache(self) -> None:
        """Clear cache"""
        with self.cache_lock:
            self.sample_cache.clear()
            self.cache_access_count.clear()
            self.current_memory_usage = 0
        gc.collect()
        print("Cache cleared")
    
    def warm_up_cache(self, task_indices: Optional[List[int]] = None, 
                      samples_per_task: int = 2) -> None:
        """Warm up cache for specified tasks"""
        if task_indices is None:
            task_indices = list(range(min(10, len(self.task_dirs))))
        
        print(f"Warming up cache for {len(task_indices)} tasks...")
        
        for task_idx in tqdm(task_indices, desc="Warming cache"):
            try:
                task_data = self.get_samples(task_idx, num_samples=samples_per_task)
                # Data already loaded and cached by get_samples
            except Exception as e:
                print(f"Failed to warm up task {task_idx}: {e}")
        
        stats = self.get_cache_stats()
        print(f"Warm up complete, cache: {stats['cache_size']} samples, memory: {stats['memory_usage_mb']:.1f}MB")


# ============================================================================
# Model Definition
# ============================================================================

class SirenModel(nn.Module):
    """SIREN-based implicit neural representation model for MRI reconstruction"""
    
    def __init__(self, w0: float = 30):
        """
        Args:
            w0: Omega_0 parameter for SIREN
        """
        super(SirenModel, self).__init__()
        
        self.w0 = w0
        
        # Magnitude and phase branches
        self.model_mag = siren_model(
            num_layers=8, 
            input_dim=2, 
            hidden_dim=256, 
            out_dim=1, 
            w0=self.w0
        )
        self.model_phi = siren_model(
            num_layers=8, 
            input_dim=2, 
            hidden_dim=256, 
            out_dim=1, 
            w0=self.w0
        )
    
    def forward(self, coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation
        
        Args:
            coordinates: Coordinate input [N, 2] where N is number of pixels
            
        Returns:
            pre_intensity_mag: Predicted magnitude [N, 1]
            pre_intensity_phi: Predicted phase [N, 1]
        """
        pre_intensity_mag = self.model_mag(coordinates)
        pre_intensity_phi = self.model_phi(coordinates)
        
        return pre_intensity_mag.float(), pre_intensity_phi.float()
    
    def get_complex_image(
        self, 
        coordinates: torch.Tensor, 
        shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Get complex image
        
        Args:
            coordinates: Coordinate input [N, 2]
            shape: Output image shape (H, W)
            
        Returns:
            Complex image
        """
        H, W = shape
        pre_intensity_mag, pre_intensity_phi = self.forward(coordinates)
        
        # Reshape to image shape
        pre_intensity_mag = pre_intensity_mag.view(H, W, 1)
        pre_intensity_phi = pre_intensity_phi.view(H, W, 1)
        
        # Convert to complex
        pre_intensity = torch.complex(pre_intensity_mag, pre_intensity_phi)
        
        return pre_intensity


# ============================================================================
# Reptile Trainer
# ============================================================================

class ReptileTrainer:
    """Reptile meta-learning trainer for MRI reconstruction"""
    
    def __init__(
        self,
        encoding_config: Dict,
        network_config: Dict,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
        samples_per_task: int = 4,
        device: str = 'cuda:0',
        load_index: Optional[int] = None
    ):
        """
        Args:
            encoding_config: TCNN encoding configuration
            network_config: TCNN network configuration
            inner_lr: Inner loop learning rate
            meta_lr: Meta learning rate (Reptile "step size")
            inner_steps: Number of inner loop steps
            samples_per_task: Number of samples per task
            device: Computing device
            load_index: Epoch index to load checkpoint from
        """
        self.encoding_config = encoding_config
        self.network_config = network_config
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.samples_per_task = samples_per_task
        self.device = device
        self.start = 0
        
        # Create model
        self.model = SirenModel().to(device)
        
        # Load checkpoint if specified
        if load_index:
            checkpoint = torch.load(
                f'./checkpoints/model_epoch_{load_index}.pth',
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start = load_index
            print(f'Loaded parameters from epoch {load_index}')
        
        # Create meta optimizer
        self.meta_optimizer = torch.optim.Adam([
            {'params': self.model.model_mag.parameters(), 'lr': meta_lr},
            {'params': self.model.model_phi.parameters(), 'lr': meta_lr},
        ])
        
        # Create learning rate scheduler
        self.scheduler = lr_scheduler.StepLR(self.meta_optimizer, step_size=500, gamma=0.5)
        
        # Loss functions
        self.criterion = nn.L1Loss()
        self.TVconstrain = MYTVLoss()
    
    def inner_loop_adaptation(
        self,
        task_samples: List[Dict],
        visualize_path: str = './progress'
    ) -> Tuple[SirenModel, float]:
        """
        Inner loop adaptation for task - create model copy and update on task samples
        
        Args:
            task_samples: List of task samples
            visualize_path: Path to save visualization
            
        Returns:
            adapted_model: Adapted model
            final_loss: Final loss value
        """
        # Create model copy
        adapted_model = SirenModel().to(self.device)
        
        # Copy parameters
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Create inner loop optimizer
        inner_optimizer = torch.optim.Adam([
            {'params': adapted_model.model_mag.parameters(), 'lr': self.inner_lr},
            {'params': adapted_model.model_phi.parameters(), 'lr': self.inner_lr},
        ])
        
        # Training mode
        adapted_model.train()
        
        # Inner loop training
        total_loss = 0.0
        sample_losses = []
        
        for step_ind in range(self.inner_steps):
            # Randomly select a sample
            sample = random.choice(task_samples)
            
            # Extract data and move to device
            mask = torch.tensor(sample['mask_transposed']).to(self.device)
            csmp = torch.tensor(sample['csmp_transposed']).to(self.device).to(torch.complex64)
            gt_ksp = torch.tensor(sample['gt_ksp_transposed']).to(self.device).to(torch.complex64)
            coordinates = torch.tensor(sample['coordinates']).to(self.device).float()
            text_caption = sample['selected_caption']
            if isinstance(text_caption, str):
                text_caption = [text_caption]
            
            if step_ind == 0:
                print(f"Used Caption: {text_caption}")
            
            # Get image dimensions
            nRow, nCol = sample['gt_img'].shape
            
            # Forward propagation
            pre_intensity_mag, pre_intensity_phi = adapted_model.forward(coordinates.view(-1, 2))
            pre_intensity = torch.complex(
                pre_intensity_mag.view(nRow, nCol, 1),
                pre_intensity_phi.view(nRow, nCol, 1)
            )
            
            # Calculate multi-channel image (multiply with csmp)
            pre_intensity_multi = pre_intensity * csmp
            
            # Calculate k-space representation
            fft_pre_intensity = torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.fftshift(pre_intensity_multi, dim=(0, 1)),
                    dim=(0, 1)
                ),
                dim=(0, 1)
            )
            
            # Calculate loss (only at sampled locations)
            mae_ksp_loss = self.criterion(
                torch.view_as_real(fft_pre_intensity[mask==1]).float(),
                torch.view_as_real(gt_ksp[mask==1]).float()
            )
            TV_loss = self.TVconstrain(pre_intensity_mag.view(nRow, nCol, 1)) + \
                      self.TVconstrain(pre_intensity_phi.view(nRow, nCol, 1))
            loss = mae_ksp_loss + 2 * TV_loss
            
            sample_losses.append(loss.item())
            
            # Backpropagation and parameter update
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
            
            # Periodic visualization
            if (step_ind + 1) % 50 == 0:
                plt.figure(figsize=(12, 4))
                
                # Show current reconstruction
                plt.subplot(1, 2, 1)
                plt.imshow(torch.abs(pre_intensity[:,:,0].detach().cpu()), cmap='gray')
                plt.title(f"Step {step_ind+1}")
                plt.colorbar()
                
                # Show loss curve
                plt.subplot(1, 2, 2)
                plt.plot(sample_losses)
                plt.title(f"Loss Curve (Current: {mae_ksp_loss.item():.6f})")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{visualize_path}/step_{step_ind+1}.png", dpi=100)
                plt.close()
            
            total_loss += mae_ksp_loss.item()
        
        final_loss = loss.item()
        print(f"Inner Loop Adaptation Complete, Final Loss: {final_loss:.6f}")
        
        return adapted_model, final_loss
    
    def adaptive_reptile_update(
        self,
        adapted_models: List,
        task_losses: List[float]
    ) -> List[float]:
        """
        Adaptive Reptile update with task-weighted meta learning rate
        
        Args:
            adapted_models: List of adapted model instances or parameter dicts
            task_losses: List of task losses
            
        Returns:
            normalized_weights: List of normalized task weights
        """
        # Calculate inverse losses as weight basis (lower loss = higher weight)
        inverse_losses = [1.0 / (loss + 1e-8) for loss in task_losses]
        
        # Normalize weights so they sum to 1
        total_inverse = sum(inverse_losses)
        normalized_weights = [inv_loss / total_inverse for inv_loss in inverse_losses]
        
        print(f"Task weight distribution: {[f'{w:.3f}' for w in normalized_weights]}")
        
        with torch.no_grad():
            # Define submodules to update
            update_modules = [
                'model_mag',  # MLP network - magnitude
                'model_phi',  # MLP network - phase
            ]
            
            updated_params_count = 0
            total_params_count = 0
            
            # Check adapted_models type
            if adapted_models and isinstance(adapted_models[0], dict):
                # If parameter dict format, use original method
                print("Detected parameter dict format, updating by parameter name")
                return self._update_from_param_dicts(adapted_models, normalized_weights)
            
            # Update each module
            for module_name in update_modules:
                try:
                    # Get meta model's module
                    meta_module = getattr(self.model, module_name)
                    module_updated_params = 0
                    
                    # Weighted update for each task's adapted model
                    for adapted_model, weight in zip(adapted_models, normalized_weights):
                        # Get corresponding module from current task's adapted model
                        adapted_module = getattr(adapted_model, module_name)
                        
                        # Calculate weighted update step size
                        weighted_meta_lr = self.meta_lr * weight
                        
                        # Update each parameter in the module
                        for p_meta, p_adapted in zip(meta_module.parameters(), adapted_module.parameters()):
                            # Reptile update: θ = θ + ε * weight * (φ - θ)
                            update = weighted_meta_lr * (p_adapted.data - p_meta.data)
                            p_meta.data.add_(update)
                            
                            if weight == normalized_weights[0]:  # Only count on first task
                                module_updated_params += p_meta.numel()
                    
                    updated_params_count += module_updated_params
                    total_params_count += sum(p.numel() for p in meta_module.parameters())
                    
                    print(f"Updated module {module_name}: {module_updated_params:,} parameters")
                    
                except AttributeError as e:
                    print(f"Module {module_name} does not exist: {e}")
            
            print(f"Total updated parameters: {updated_params_count:,}/{total_params_count:,}")
        
        return normalized_weights
    
    def _update_from_param_dicts(
        self,
        adapted_model_params_list: List[Dict],
        normalized_weights: List[float]
    ) -> List[float]:
        """
        Update from parameter dict list with detailed statistics
        
        Args:
            adapted_model_params_list: List of parameter dicts
            normalized_weights: Normalized weights
            
        Returns:
            normalized_weights: Normalized weights (returned for consistency)
        """
        # Get original model parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Define parameter name patterns to update and their descriptions
        update_patterns = {
            'model_mag': 'MLP Network - Magnitude Branch',
            'model_phi': 'MLP Network - Phase Branch',
        }
        
        # Initialize statistics
        update_stats = {desc: {'params': 0, 'layers': 0} for desc in update_patterns.values()}
        total_updated = 0
        
        # Weighted update for each task
        for task_idx, (adapted_model_params, weight) in enumerate(zip(adapted_model_params_list, normalized_weights)):
            task_updated = 0
            
            for name, param in self.model.named_parameters():
                # Check if parameter name matches update pattern
                for pattern, description in update_patterns.items():
                    if pattern in name and name in adapted_model_params:
                        # Calculate weighted update step size
                        weighted_meta_lr = self.meta_lr * weight
                        
                        # Apply weighted Reptile update
                        update = weighted_meta_lr * (adapted_model_params[name] - original_params[name])
                        param.data.add_(update)
                        
                        # Statistics (only count on first task to avoid double counting)
                        if task_idx == 0:
                            update_stats[description]['params'] += param.numel()
                            update_stats[description]['layers'] += 1
                            total_updated += param.numel()
                        
                        task_updated += param.numel()
                        break  # Exit after finding matching pattern
            
            print(f"  Task {task_idx+1} (weight: {weight:.3f}): updated {task_updated:,} parameters")
        
        # Print detailed update statistics
        print(f"\nReptile Update Statistics:")
        print(f"{'='*50}")
        
        for description, stats in update_stats.items():
            if stats['params'] > 0:
                print(f"{description}:")
                print(f"   Parameters: {stats['params']:,}")
                print(f"   Layers: {stats['layers']}")
                print(f"   Average per layer: {stats['params']//max(stats['layers'],1):,} parameters")
        
        print(f"{'='*50}")
        print(f"Total updated parameters: {total_updated:,}")
        print(f"Meta learning rate: {self.meta_lr}")
        
        return normalized_weights
    
    def evaluate_model(
        self,
        model: SirenModel,
        test_samples: List[Dict]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model performance on test samples
        
        Args:
            model: Model to evaluate
            test_samples: List of test samples
            
        Returns:
            avg_loss: Average loss
            metrics: Evaluation metrics dict
        """
        model.eval()
        total_loss = 0
        all_psnrs = []
        all_ssims = []
        
        with torch.no_grad():
            for sample in test_samples:
                # Extract data and move to device
                mask = torch.tensor(sample['mask_transposed']).to(self.device)
                csmp = torch.tensor(sample['csmp_transposed']).to(self.device).to(torch.complex64)
                gt_ksp = torch.tensor(sample['gt_ksp_transposed']).to(self.device).to(torch.complex64)
                gt_img = torch.tensor(sample['gt_img']).to(self.device).to(torch.complex64)
                coordinates = torch.tensor(sample['coordinates']).to(self.device).float()
                text_caption = sample['selected_caption']
                if isinstance(text_caption, str):
                    text_caption = [text_caption]
                print(f"Used Caption: {text_caption}")
                
                # Get image dimensions
                nRow, nCol = sample['gt_img'].shape
                
                # Forward propagation
                pre_intensity_mag, pre_intensity_phi = model.forward(coordinates.view(-1, 2))
                pre_intensity = torch.complex(
                    pre_intensity_mag.view(nRow, nCol, 1),
                    pre_intensity_phi.view(nRow, nCol, 1)
                )
                
                # Calculate multi-channel image
                pre_intensity_multi = pre_intensity * csmp
                
                # Calculate k-space representation
                fft_pre_intensity = torch.fft.fftshift(
                    torch.fft.fft2(
                        torch.fft.fftshift(pre_intensity_multi, dim=(0, 1)),
                        dim=(0, 1)
                    ),
                    dim=(0, 1)
                )
                
                # Calculate loss
                mae_ksp_loss = self.criterion(
                    torch.view_as_real(fft_pre_intensity).float(),
                    torch.view_as_real(gt_ksp).float()
                )
                total_loss += mae_ksp_loss.item()
                
                # Calculate image quality metrics
                pred_img = pre_intensity.squeeze().cpu().numpy()
                target_img = gt_img.cpu().numpy()
                
                psnr = calculate_psnr(pred_img, target_img)
                ssim = calculate_ssim(pred_img, target_img)
                
                all_psnrs.append(psnr)
                all_ssims.append(ssim)
        
        # Calculate average metrics
        avg_loss = total_loss / len(test_samples)
        avg_psnr = np.mean(all_psnrs)
        avg_ssim = np.mean(all_ssims)
        
        return avg_loss, {
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
    
    def _plot_training_curves(self, stats: Dict, save_dir: str) -> None:
        """Plot training curves"""
        plt.figure(figsize=(12, 10))
        
        # Loss curves
        plt.subplot(3, 1, 1)
        plt.plot(stats['meta_losses'], 'b-', label='Meta Loss')
        if len(stats['eval_losses']) > 0:
            # Create x-axis points corresponding to evaluation epochs
            eval_epochs = np.linspace(0, len(stats['meta_losses'])-1, len(stats['eval_losses']))
            plt.plot(eval_epochs, stats['eval_losses'], 'r-', label='Eval Loss')
        plt.legend()
        plt.title('Training and Evaluation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # PSNR curves
        plt.subplot(3, 1, 2)
        if len(stats['eval_psnrs']) > 0:
            plt.plot(stats['eval_psnrs'], 'g-')
            plt.title('Evaluation PSNR')
            plt.xlabel('Evaluation')
            plt.ylabel('PSNR (dB)')
            plt.grid(True)
        
        # SSIM curves
        plt.subplot(3, 1, 3)
        if len(stats['eval_ssims']) > 0:
            plt.plot(stats['eval_ssims'], 'm-')
            plt.title('Evaluation SSIM')
            plt.xlabel('Evaluation')
            plt.ylabel('SSIM')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()
    
    def train(
        self,
        dataset: OptimizedMRIMetaDataset,
        epochs: int,
        save_dir: str = './reptile_checkpoints',
        eval_interval: int = 5,
        save_visuals: bool = True,
        tasks_per_epoch: int = 1,
        samples_per_task: int = 1
    ) -> Dict:
        """
        Execute Reptile-based meta-learning training
        
        Args:
            dataset: Dataset instance
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            eval_interval: Evaluation interval
            save_visuals: Whether to save visualizations
            tasks_per_epoch: Number of tasks per epoch
            samples_per_task: Number of samples per task
            
        Returns:
            training_stats: Training statistics dict
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Warm up cache for hot tasks
        print("Warming up data cache...")
        hot_tasks = list(range(min(30, len(dataset))))
        dataset.warm_up_cache(hot_tasks, samples_per_task=2)
        
        # Training statistics
        training_stats = {
            'meta_losses': [],
            'eval_losses': [],
            'eval_psnrs': [],
            'eval_ssims': [],
            'data_loading_times': [],
            'training_times': []
        }
        
        best_eval_psnr = 0
        total_tasks = len(dataset)
        
        print(f"Starting Reptile training with {tasks_per_epoch} tasks per epoch")
        print(f"Using {samples_per_task} samples per task")
        
        for epoch in range(self.start, epochs):
            epoch_start_time = time.time()
            data_loading_time = 0
            training_time = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Meta learning rate: {self.meta_lr:.6f}")
            
            # Training phase
            self.model.train()
            
            # Randomly select tasks
            task_indices = random.sample(range(total_tasks), min(tasks_per_epoch, total_tasks))
            print(f"Selected {len(task_indices)} tasks for this epoch")
            
            # Initialize meta loss
            epoch_meta_loss = 0.0
            batch_task_losses = []
            batch_adapted_models = []
            
            # Process each selected task
            for batch_idx, task_idx in enumerate(task_indices):
                # Data loading time statistics
                data_load_start = time.time()
                
                task_metadata = dataset[task_idx]
                task_id = task_metadata['task_id']
                
                task_data = dataset.get_samples(task_idx, num_samples=samples_per_task)
                task_samples = task_data['samples']
                
                data_loading_time += time.time() - data_load_start
                
                print(f"Task {batch_idx+1}/{len(task_indices)}: {task_id}")
                print(f"  Selected {len(task_samples)} samples")
                
                # Pure training time statistics
                train_start = time.time()
                
                # Inner loop adaptation
                adapted_model, task_loss = self.inner_loop_adaptation(task_samples)
                
                batch_task_losses.append(task_loss)
                batch_adapted_models.append({
                    name: param.clone() 
                    for name, param in adapted_model.named_parameters()
                })
                
                # Record task loss
                epoch_meta_loss += task_loss
                
                training_time += time.time() - train_start
                
                print(f"  Task Loss: {task_loss:.6f}")
            
            # Reptile update time also counts as training time
            update_start = time.time()
            weights = self.adaptive_reptile_update(batch_adapted_models, batch_task_losses)
            training_time += time.time() - update_start
            
            for i, weight in enumerate(weights):
                print(f"  Task {i+1} weight: {weight:.4f} (loss: {batch_task_losses[i]:.6f})")
            
            # Calculate average meta loss
            epoch_meta_loss = epoch_meta_loss / len(task_indices) if task_indices else 0
            training_stats['meta_losses'].append(epoch_meta_loss)
            training_stats['data_loading_times'].append(data_loading_time)
            training_stats['training_times'].append(training_time)
            
            # Print performance statistics
            epoch_time = time.time() - epoch_start_time
            gpu_utilization = (training_time / epoch_time) * 100 if epoch_time > 0 else 0
            
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            print(f"Data loading: {data_loading_time:.2f}s, Training: {training_time:.2f}s")
            print(f"GPU Utilization: {gpu_utilization:.1f}%")
            print(f"Meta Loss: {epoch_meta_loss:.6f}")
            
            # Periodic cache cleanup
            if (epoch + 1) % (eval_interval * 2) == 0:
                print("Cleaning and rewarming cache...")
                dataset.clear_cache()
                dataset.warm_up_cache(hot_tasks, samples_per_task=1)
            
            # Periodic evaluation and saving
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                print("\nEvaluation:")
                eval_meta_loss = 0
                eval_psnrs = []
                eval_ssims = []
                
                # Randomly select tasks for evaluation
                eval_tasks = random.sample(range(len(dataset)), min(5, len(dataset)))
                
                for eval_idx, task_idx in enumerate(eval_tasks):
                    task_metadata = dataset[task_idx]
                    task_id = task_metadata['task_id']
                    
                    # Use more samples for more stable evaluation
                    eval_samples_count = min(3, len(dataset.get_samples(task_idx, num_samples=3)['samples']))
                    
                    task_data = dataset.get_samples(task_idx, num_samples=eval_samples_count)
                    loaded_samples = task_data['samples']
                    support_set = loaded_samples
                    
                    print(f"Evaluating task {eval_idx+1}/{len(eval_tasks)}: {task_id}")
                    print(f"  Using {len(support_set)} adaptation samples")
                    
                    # Inner loop adaptation
                    for eval_ind in range(eval_samples_count):
                        adapted_model, _ = self.inner_loop_adaptation(support_set[eval_ind:eval_ind+1])
                        
                        # Evaluate on test set
                        eval_loss, metrics = self.evaluate_model(adapted_model, support_set[eval_ind:eval_ind+1])
                        
                        # Collect metrics
                        eval_meta_loss += eval_loss
                        eval_psnrs.append(metrics['psnr'])
                        eval_ssims.append(metrics['ssim'])
                    
                    print(f"  Eval Loss: {eval_loss:.6f}, PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}")
                
                # Calculate average evaluation metrics
                avg_eval_loss = eval_meta_loss / len(eval_tasks) if eval_tasks else 0
                avg_eval_psnr = np.mean(eval_psnrs) if eval_psnrs else 0
                avg_eval_ssim = np.mean(eval_ssims) if eval_ssims else 0
                
                # Save evaluation metrics
                training_stats['eval_losses'].append(avg_eval_loss)
                training_stats['eval_psnrs'].append(avg_eval_psnr)
                training_stats['eval_ssims'].append(avg_eval_ssim)
                
                print(f"Average Eval - Loss: {avg_eval_loss:.6f}, PSNR: {avg_eval_psnr:.2f}dB, SSIM: {avg_eval_ssim:.4f}")
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.meta_optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'meta_loss': epoch_meta_loss,
                    'eval_psnr': avg_eval_psnr,
                    'eval_ssim': avg_eval_ssim,
                    'encoding_config': self.encoding_config,
                    'network_config': self.network_config
                }
                
                # Save latest model
                torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
                
                # Save separately if best model
                if avg_eval_psnr > best_eval_psnr:
                    best_eval_psnr = avg_eval_psnr
                    torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                    print(f"Saved new best model with PSNR: {best_eval_psnr:.2f}dB")
                
                # Plot training curves
                self._plot_training_curves(training_stats, save_dir)
            
            # Print cache statistics
            if (epoch + 1) % 10 == 0:
                cache_stats = dataset.get_cache_stats()
                print(f"Cache: {cache_stats['cache_size']} samples, "
                      f"{cache_stats['memory_usage_mb']:.1f}MB / {cache_stats['memory_limit_mb']:.1f}MB")
        
        return training_stats
