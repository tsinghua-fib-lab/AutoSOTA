# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Train the trajectory-based motion segmentation network.
"""
import os
import shutil
import argparse
import random
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from core.utils.utils import load_config_file, save_model,save_motion_seg_model, cls_iou, cal_roc, get_feat
from core.dataset.kubric import Kubric_dataset, find_traj_label
from core.dataset.dynamic_stereo import Stereo_dataset, StereoMotionSegmentationDataset
from core.dataset.waymo import Waymo_dataset
from core.dataset.hoi4d import HOI_dataset, MotionSegConfig, MotionSegmentationDataset
from core.dataset.got10k_aug import MotionSegmentationDatasetGOT
from core.dataset.ytvos18m import MotionSegmentationDatasetYTVOS
from core.dataset.ominiworld import OminiWorldDataset
from core.dataset.dynamicverse_aug import MotionSegmentationDatasetDynamic
from core.dataset.gotmoving_aug import MotionSegmentationDatasetGotmoving
from core.dataset.base import ProbabilisticDataset
from core.network.traj_oa_depth import traj_oa_depth
from core.network.transfomer import traj_seg
from core.network import loss_func
from core.utils.visualize import Visualizer,read_imgs_from_path
from core.dataset.data_utils import normalize_point_traj_torch
from glob import glob
import numpy as np
from PIL import Image
import json
from itertools import zip_longest
# from vggt.models.vggt_seg import MotionSegmentationVGGT, CombinedLoss
# from pi3.models.pi3 import create_pi3_motion_segmentation_model, CombinedLoss
# from pi3.models.pi3_gate import create_pi3_motion_segmentation_model, CombinedLoss
from pi3.models.pi3_conf_flow_cam_lowfeature_wo_mean import create_pi3_motion_segmentation_model, CombinedLoss
from pi3.utils.geometry import process_video_with_improved_sliding_window
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from motion_seg_inference import compute_optical_flow_magnitude_batch, normalize_flow_magnitudes,initialize_raft_model

# 添加新的导入
import logging
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import socket

# --- DDP Imports: 新增的导入 ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# --- End DDP Imports ---


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.autograd.set_detect_anomaly(True)


class ProbabilisticDataset(Dataset):
    def __init__(self, datasets, probabilities):
        """
        More flexible version that accepts any number of datasets
        
        Args:
            datasets: List of dataset objects
            probabilities: List of probabilities (must sum to 1)
        """
        if len(datasets) != len(probabilities):
            raise ValueError("Number of datasets must match number of probabilities")
        
        # if abs(sum(probabilities) - 1.0) > 1e-6:
        #     raise ValueError("Probabilities must sum to 1")
        
        self.datasets = datasets
        self.probabilities = probabilities
        self.cumulative_probs = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
        
    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)
    
    def __getitem__(self, idx):
        rand_value = random.random()
        
        # Find which dataset to sample from
        for i, cum_prob in enumerate(self.cumulative_probs):
            if rand_value <= cum_prob:
                selected_dataset = self.datasets[i]
                return selected_dataset[idx % len(selected_dataset)]
        
        # Fallback (shouldn't happen with proper probabilities)
        return self.datasets[-1][idx % len(self.datasets[-1])]


def find_free_port():
    """找到一个空闲的端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

# --- DDP Helper Functions: 新增的辅助函数 ---
def setup_distributed():
    """初始化分布式训练环境"""
    if not dist.is_available() or not torch.cuda.is_available():
        return

    # torchrun 会自动设置这些环境变量
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 如果 world_size 为 1，则无需初始化
    if world_size == 1:
        return

    # 为当前进程设置对应的 GPU
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",  # NCCL 是 NVIDIA GPU 推荐的后端
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    print(f"Distributed training enabled: rank {rank}, world_size {world_size}")

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
# --- End DDP Helper Functions ---


def setup_logger(log_dir, rank=0):
    """设置logger"""
    if rank != 0:
        return None
    
    # 创建logger
    logger = logging.getLogger('motion_segmentation')
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件handler
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_visualization(images, predictions, ground_truth, save_path, sample_idx=0, max_samples=4):
    """
    Save visualization of motion segmentation results
    
    Args:
        images: [B, S, 3, H, W] input images
        predictions: [B, S, H, W] predicted motion masks
        ground_truth: [B, S, H, W] ground truth motion masks
        save_path: directory to save visualizations
        sample_idx: which sample in batch to visualize
        max_samples: maximum number of samples to save
    """
    
    batch_size = min(images.shape[0], max_samples)
    seq_len = images.shape[1]
    
    def denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # img_tensor: [C,H,W] torch.Tensor
        mean = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
        return img_tensor * std + mean
    
    for b in range(batch_size):
        # Create a figure with subplots
        fig, axes = plt.subplots(3, seq_len, figsize=(seq_len * 4, 12))
        if seq_len == 1:
            axes = axes.reshape(3, 1)
        
        for t in range(seq_len):
            # Original image
            # img = images[b, t].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            # img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
            img = denormalize(images[b, t].cpu())  # [C,H,W]
            img = img.numpy().transpose(1, 2, 0)  # [H,W,3]
            img = np.clip(img, 0, 1)
            axes[0, t].imshow(img)
            axes[0, t].set_title(f'Frame {t}')
            axes[0, t].axis('off')
            
            # Ground truth mask
            gt_mask = ground_truth[b, t].cpu().numpy()
            im_gt = axes[1, t].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            axes[1, t].set_title(f'GT Mask {t}')
            axes[1, t].axis('off')
            
            # Prediction mask
            pred_mask = torch.sigmoid(predictions[b, t]).cpu().numpy() if not isinstance(predictions, np.ndarray) else predictions[b, t]
            im_pred = axes[2, t].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            axes[2, t].set_title(f'Pred Mask {t}')
            axes[2, t].axis('off')
        
        # Add row labels
        axes[0, 0].set_ylabel('Original', rotation=90, size='large')
        axes[1, 0].set_ylabel('Ground Truth', rotation=90, size='large')
        axes[2, 0].set_ylabel('Prediction', rotation=90, size='large')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create overlay visualization
    for b in range(batch_size):
        fig, axes = plt.subplots(2, seq_len, figsize=(seq_len * 4, 8))
        if seq_len == 1:
            axes = axes.reshape(2, 1)
            
        for t in range(seq_len):
            img = images[b, t].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            
            # GT overlay
            gt_mask = ground_truth[b, t].cpu().numpy()
            axes[0, t].imshow(img)
            axes[0, t].imshow(gt_mask, alpha=0.5, cmap='Reds')
            axes[0, t].set_title(f'GT Overlay {t}')
            axes[0, t].axis('off')
            
            # Prediction overlay
            pred_mask = torch.sigmoid(predictions[b, t]).cpu().numpy() if not isinstance(predictions, np.ndarray) else predictions[b, t]
            axes[1, t].imshow(img)
            axes[1, t].imshow(pred_mask, alpha=0.5, cmap='Blues')
            axes[1, t].set_title(f'Pred Overlay {t}')
            axes[1, t].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(save_path) , f'overlay_{b:03d}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def setup_motion_seg_dataset(cfg):
    """
    Setup datasets and dataloaders for motion segmentation training
    """
    # Define transforms for VGGT (no need for complex transforms, handled in dataset)
    train_transform = None  # MotionSegmentationDataset handles transforms internally
    test_transform = None
    
    # Initialize dataset containers
    train_datasets = []
    
    # Create training datasets
    for i, dataset_name in enumerate(cfg.train_dataset):
        data_dir = cfg.train_root[i]
        
        if dataset_name == 'hoi4d':
            # from motion_segmentation_dataset import MotionSegmentationDataset
            hoi_train_dataset = MotionSegmentationDataset(
                data_dir=data_dir,
                split='train',
                transform=train_transform,
                img_size=cfg.img_size,
                sequence_length=cfg.sequence_length,
                sample_stride=1,
            )
            train_datasets.append(hoi_train_dataset)
        
        elif dataset_name == 'omniworld':
            omni_train_dataset = OminiWorldDataset(
                data_dir=data_dir,
                split='train',
                img_size=cfg.img_size,
                sequence_length=cfg.sequence_length,
                sample_stride=8,
                enable_augmentation=True
            )
            train_datasets.append(omni_train_dataset)
            
        elif dataset_name == 'dynamic_stereo':
            stereo_train_dataset = StereoMotionSegmentationDataset(
                data_dir, 
                transform=train_transform, 
                split='train', 
                img_size=cfg.img_size, 
                sequence_length=cfg.sequence_length, 
                sample_stride=12
            )
            train_datasets.append(stereo_train_dataset)

        elif dataset_name == 'got10k':
            got_train_dataset = MotionSegmentationDatasetGOT(
                data_dir, 
                transform=train_transform, 
                split='train', 
                img_size=cfg.img_size, 
                sequence_length=cfg.sequence_length, 
                sample_stride=4
            )
            train_datasets.append(got_train_dataset)

        elif dataset_name == 'dynamicverse':
            dynamicverse_train_dataset = MotionSegmentationDatasetDynamic(
                data_dir,
                transform=train_transform, 
                split='train', 
                img_size=cfg.img_size, 
                sequence_length=cfg.sequence_length, 
                sample_stride=1
            )
            train_datasets.append(dynamicverse_train_dataset)

        elif dataset_name == 'gotmoving':
            got_moving_dataset = MotionSegmentationDatasetGotmoving(
                # data_dir="/data1/got_train_video_roots_with_masks.txt",
                data_dir=data_dir,
                transform=train_transform, 
                split='train', 
                img_size=cfg.img_size, 
                sequence_length=cfg.sequence_length, 
                sample_stride=2
            )
            train_datasets.append(got_moving_dataset)

        elif dataset_name == 'ytvos18m':
            got_train_dataset = MotionSegmentationDatasetYTVOS(
                data_dir, 
                transform=train_transform, 
                split='train', 
                img_size=cfg.img_size, 
                sequence_length=cfg.sequence_length, 
                sample_stride=1
            )
            train_datasets.append(got_train_dataset)
        else:
            raise NotImplementedError(f"Train dataset {dataset_name} not supported for motion segmentation")
    
    # Combine datasets using different strategies
    if len(train_datasets) == 1:
        combined_dataset = train_datasets[0]
    else:
        # Check if probabilistic sampling is requested
        if hasattr(cfg, 'dataset_sampling_strategy') and cfg.dataset_sampling_strategy == 'probabilistic':
            # Use probabilistic sampling
            if hasattr(cfg, 'dataset_probabilities'):
                probabilities = cfg.dataset_probabilities
            else:
                # Equal probabilities by default
                probabilities = [1.0 / len(train_datasets)] * len(train_datasets)
            
            combined_dataset = ProbabilisticDataset(train_datasets, probabilities)
            print(f"Combined {len(train_datasets)} training datasets using probabilistic sampling")
            print(f"Dataset probabilities: {probabilities}")
        else:
            # Use concatenation (original behavior)
            combined_dataset = ConcatDataset(train_datasets)
            print(f"Combined {len(train_datasets)} training datasets with total {len(combined_dataset)} samples")
    
    # --- DDP Change: 不在这里创建 DataLoader ---
    # train_loader 将在 main 函数中创建，因为它需要 DistributedSampler
    
    # Create test dataloaders (测试时通常不需要分布式)
    test_loaders = {}
    
    for i, dataset_name in enumerate(cfg.test_dataset):
        data_dir = cfg.test_root[i]
        
        if dataset_name == 'test':
            test_dataset = MotionSegmentationDataset(
                data_dir=data_dir,
                split='test',  # or 'val'
                transform=test_transform,
                img_size=cfg.img_size,
                sequence_length=cfg.sequence_length,
                sample_stride=2 #cfg.get('sample_stride', 4)
            )
            
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Use batch_size=1 for testing
            shuffle=False,
            num_workers=min(cfg.num_workers, 4),  # Fewer workers for test
            pin_memory=True,
            drop_last=False
        )
        
        test_loaders[dataset_name] = test_loader
    
    # 返回 dataset 对象，而不是 loader
    return combined_dataset, test_loaders

def setup_motion_seg_model(cfg, device):
    """Setup motion segmentation model"""

    # Prefer config paths and fall back to environment variables for reproducibility.
    pi3_model_path = getattr(cfg, "vggt_model_path", None) or os.environ.get("PI3_MODEL_PATH", None)
    raft_model_path = getattr(cfg, "raft_model_path", None) or os.environ.get("RAFT_MODEL_PATH", None)

    # Create model with motion cues
    model = create_pi3_motion_segmentation_model(
        pi3_model_path=pi3_model_path,
    )
    initialize_raft_model(device=device, raft_model_path=raft_model_path)
    return model

def visualize_conf_on_images(images, conf, idx=0, save_path=None):
    """
    可视化 conf heatmap 叠加在图像上
    Args:
        images: Tensor [B, N, C, H, W]
        conf: Tensor [N, H, W, 1]
        idx: 选择 batch 内的第 idx 个样本
        save_path: 若指定则保存图片
    """
    # 转为 CPU numpy
    img = images[idx].detach().cpu().permute(0, 2, 3, 1).numpy()  # [N, H, W, 3]
    conf_map = conf.detach().cpu().squeeze(-1).numpy()             # [N, H, W]
    
    num_frames = img.shape[0]
    plt.figure(figsize=(12, num_frames * 3))
    
    for i in range(num_frames):
        frame = img[i]
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)  # 归一化到 [0,1]
        conf_i = conf_map[i]
        conf_i = (conf_i - conf_i.min()) / (conf_i.max() - conf_i.min() + 1e-8)

        plt.subplot(num_frames, 2, 2*i+1)
        plt.imshow(frame)
        plt.title(f"Frame {i} Image")
        plt.axis('off')

        plt.subplot(num_frames, 2, 2*i+2)
        plt.imshow(frame)
        plt.imshow(conf_i, cmap='jet', alpha=0.5)  # 半透明叠加
        plt.title(f"Frame {i} + Conf Heatmap")
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

# --- DDP Change: 为函数添加 device 参数 ---
def train_motion_seg_epoch(cfg, model, optimizer, train_loader, epoch, test_loaders, device, logger):
    """
    Training loop for motion segmentation using VGGT features
    """
    model.train()
    
    # Initialize loss function
    criterion = CombinedLoss()
    running_loss = 0.0
    running_metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # --- DDP Change: 仅在主进程中显示进度条或打印信息 ---
    is_main_process = dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized()
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # 创建可视化保存目录
    if is_main_process:
        vis_dir = os.path.join(cfg.log_dir, 'visualizations', f'epoch_{epoch:03d}')
        os.makedirs(vis_dir, exist_ok=True)
    
    for idx, sample in enumerate(train_loader): 
        if sample is None:
            continue
        # --- DDP Change: 将数据移动到正确的 device ---
        images = sample["images"].float().to(device)          # [B, N, C, H, W]
        motion_gt = sample["motion_masks"].float().to(device) # [B, N, H, W]

        # flow
        with torch.no_grad(): 
            # 标准双向光流
            flow_magnitudes = compute_optical_flow_magnitude_batch(
                images, device, 
                bidirectional=True,
                fusion_method='max'
            )  # [B, N, H, W]

            # 归一化光流幅度
            # flows = normalize_flow_magnitudes(
            #     flow_magnitudes, 
            #     method='percentile',
            #     percentile=95,
            #     min_val=0.0, 
            #     max_val=1.0
            # )

        # print(images.shape, motion_gt.shape)
        # import pdb; pdb.set_trace()
        valid_masks = sample.get("valid_masks", None)
        if valid_masks is not None:
            valid_masks = valid_masks.float().to(device)
        
        optimizer.zero_grad()

        # ================= Pi3 推理 =================
        # print("Running Pi3 inference on training batch...")
        video_tensor = images  # 用训练 batch
        ddp_model = model.module if hasattr(model, "module") else model
        # from pi3.models.visualization import visualize_hidden_features

        prediction = ddp_model(images, flow_magnitudes)
        motion_pred = prediction['motion_mask']  # [B, N, H, W]

        # ================= Loss 计算 =================
        loss, loss_components = criterion(motion_pred, motion_gt)
        loss.backward()
        
        if hasattr(cfg, 'gradient_clip') and cfg.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=cfg.gradient_clip)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        with torch.no_grad():
            batch_metrics = compute_metrics(motion_pred.detach(), motion_gt)
            # import pdb; pdb.set_trace()
            for key in running_metrics:
                running_metrics[key] += batch_metrics[key]
        
        # --- DDP Change: 仅在主进程中执行日志记录和模型保存 ---
        if is_main_process:
            if idx % cfg.print_freq == 0:
                
                # 计算当前的平均指标
                current_avg_iou = running_metrics['iou'] / (idx + 1)
                current_avg_f1 = running_metrics['f1'] / (idx + 1)
                
                # 记录训练信息
                log_msg = (f"Epoch {epoch:3d}/{cfg.max_epochs} | "
                          f"Iter {idx:4d}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"BCE: {loss_components['bce_loss']:.4f} | "
                          f"Dice: {loss_components['dice_loss']:.4f} | "
                          f"IoU: {current_avg_iou:.4f} | "
                          f"F1: {current_avg_f1:.4f} | "
                          f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                          )
                
                if logger:
                    logger.info(log_msg)
                else:
                    print(log_msg)
                
            
            if idx % cfg.save_freq == 0 and idx > 0:
                test_results = {}
                for test_name, test_loader in test_loaders.items():
                    # --- DDP Change: 传递 device 参数 ---
                    test_metrics = test_motion_seg_epoch(cfg, ddp_model, test_loader, device, logger, vis_dir, epoch)
                    test_results[test_name] = test_metrics
                
                # --- DDP Change: 保存模型时，使用 model.module 访问原始模型 ---
                save_motion_seg_model(model.module if hasattr(model, 'module') else model, cfg.log_dir, epoch, test_results)
                
                model.train() # 切换回训练模式
    
    # Epoch结束统计
    avg_loss = running_loss / len(train_loader)
    avg_metrics = {k: v / len(train_loader) for k, v in running_metrics.items()}
    
    if is_main_process:
        
        epoch_msg = (f"Epoch {epoch:3d} Summary | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Avg IoU: {avg_metrics['iou']:.4f} | "
                    f"Avg F1: {avg_metrics['f1']:.4f} | "
                    )
        
        if logger:
            logger.info("="*80)
            logger.info(epoch_msg)
            logger.info("="*80)
        else:
            print("="*80)
            print(epoch_msg)
            print("="*80)
    
    return avg_loss


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """
    Compute comprehensive metrics for motion segmentation
    """
    pred_binary = (pred > threshold).float()
    
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-6)
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

# --- DDP Change: 为函数添加 device 参数 ---
@torch.no_grad()
def test_motion_seg_epoch(cfg, model, test_loader, device, logger=None, vis_dir=None, epoch=0):
    """
    Evaluation function for motion segmentation
    """
    model.eval()
    
    total_metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    total_samples = 0
    
    # 创建测试可视化目录
    if vis_dir:
        test_vis_dir = os.path.join(vis_dir, 'test_samples')
        os.makedirs(test_vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            # --- DDP Change: 将数据移动到正确的 device ---
            images = sample["images"].float().to(device)
            motion_gt = sample["motion_masks"].float().to(device)
            valid_masks = sample.get("valid_masks", None)
            if valid_masks is not None:
                valid_masks = valid_masks.float().to(device)

            # flow
            flow_magnitudes = compute_optical_flow_magnitude_batch(
                images, device, 
                bidirectional=True,
                fusion_method='max'
            )  # [B, N, H, W]

            prediction = model(images, flow_magnitudes)
            motion_pred = prediction['motion_mask']  # [B, N, H, W]
  
            
            batch_metrics = compute_metrics(motion_pred, motion_gt)
            
            batch_size = images.shape[0]
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key] * batch_size
            total_samples += batch_size
            
            # 保存测试可视化 (只保存前几个样本)
            if vis_dir:
                vis_path = os.path.join(test_vis_dir, f'test_sample_{idx:03d}.png')
                save_visualization(
                    images, motion_pred, motion_gt, 
                    vis_path, sample_idx=idx, max_samples=1
                )
    
    avg_metrics = {k: v / total_samples for k, v in total_metrics.items()}
    
    # 仅在主进程打印测试结果
    is_main_process = dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized()
    if is_main_process:
        test_msg = (f"Test Results | "
                   f"IoU: {avg_metrics['iou']:.4f} | "
                   f"Precision: {avg_metrics['precision']:.4f} | "
                   f"Recall: {avg_metrics['recall']:.4f} | "
                   f"F1: {avg_metrics['f1']:.4f}")
        
        if logger:
            logger.info(test_msg)
        else:
            print(test_msg)
    
    return avg_metrics

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

def main(cfg):
    """
    Main training function for motion segmentation
    """
    # --- DDP Change: 1. 初始化分布式环境 ---
    setup_distributed()
    # 获取当前进程的 local_rank，用于指定 GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = (rank == 0)

    # --- DDP Change: 2. 日志和文件夹创建仅在主进程中进行 ---
    logger = None
    if is_main_process:
        if not os.path.exists(cfg.log_dir):
            os.makedirs(cfg.log_dir)
        # shutil.copy(args.config_file, cfg.log_dir)
        # shutil.copy(
        #     args.config_file,
        #     os.path.join(cfg.log_dir, os.path.basename(args.config_file))
        # )
        
        # 设置logger
        logger = setup_logger(cfg.log_dir, rank)
        
        # 记录配置信息
        logger.info("="*80)
        logger.info("Training Configuration")
        logger.info("="*80)
        for key, value in vars(cfg).items():
            logger.info(f"{key}: {value}")
        logger.info("="*80)

    # --- DDP Change: 3. 设置 DataLoader 使用 DistributedSampler ---
    train_dataset, test_loaders = setup_motion_seg_dataset(cfg)
    
    # DistributedSampler 会为每个进程分配不同的数据子集
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    # DataLoader 不再需要 shuffle=True，因为 sampler 会处理
    # 注意：batch_size 现在是 PER-GPU 的 batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=collate_skip_none,
        shuffle=(train_sampler is None)  # 单GPU时才shuffle
    )


    if is_main_process:
        logger.info(f'Data loader ready... Training samples: {len(train_loader.dataset)}')
        for test_name, test_loader in test_loaders.items():
            logger.info(f'Test {test_name}: {len(test_loader.dataset)} samples')
    
    # --- DDP Change: 4. 将模型移动到指定的 GPU ---
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    model = setup_motion_seg_model(cfg, device).to(device)
    
    # --- DDP Change: 5. 使用 DDP 包装模型 ---
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process:
        # 访问原始模型需要使用 model.module
        original_model = model.module if world_size > 1 else model
        total_params = sum(p.numel() for p in original_model.parameters())
        trainable_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # 优化器设置 (访问原始模型参数)
    original_model = model.module if world_size > 1 else model
    if cfg.freeze_backbone:
        # Only optimize motion-related parameters when backbone is frozen

        motion_params = list(original_model.conf_decoder.parameters()) + \
                list(original_model.conf_head.parameters()) + \
                list(original_model.motion_aware_decoder.parameters())
        
        optimizer = torch.optim.AdamW(
            motion_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        # Optimize all parameters when backbone is not frozen
        optimizer = torch.optim.AdamW(
            original_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999)
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.min_lr
    )
        
    # 从 checkpoint 恢复
    start_epoch = 0
    best_iou = 0.0
    if hasattr(cfg, 'resume_path') and cfg.resume_path:
        # 所有进程都需要加载模型权重以保持同步
        try:
            # 加载到 CPU 以避免 GPU 内存冲突
            checkpoint = torch.load(cfg.resume_path, map_location='cpu')
            
            # 加载 state_dict
            original_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 只有主进程加载优化器、scheduler等状态
            if is_main_process:
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'best_iou' in checkpoint:
                    best_iou = checkpoint['best_iou']
                logger.info(f'Loaded model from {cfg.resume_path}')
                logger.info(f'Resuming from epoch {start_epoch}, best IoU: {best_iou:.4f}')

        except Exception as e:
            if is_main_process:
                logger.error(f'Failed to load checkpoint: {e}')
                logger.info('Training from scratch')
    else:
        if is_main_process:
            logger.info('Training from scratch')
    
    # Training loop
    for epoch in range(start_epoch, cfg.max_epochs):
        # --- DDP Change: 6. 设置 sampler 的 epoch ---
        # 这对于分布式训练中正确的随机打乱至关重要
        if train_sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        
        if is_main_process:
            logger.info(f"\nStarting Epoch {epoch}/{cfg.max_epochs}")
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # --- DDP Change: 传递 device 参数 ---
        train_loss = train_motion_seg_epoch(cfg, model, optimizer, train_loader, epoch, test_loaders, device, logger)
        
        scheduler.step()
        
        # --- DDP Change: 7. 评估和保存只在主进程中进行 ---
        if is_main_process:
            test_results = {}
            for test_name, test_loader in test_loaders.items():
                test_metrics = test_motion_seg_epoch(cfg, original_model, test_loader, device, logger, 
                                                   os.path.join(cfg.log_dir, 'visualizations', f'epoch_{epoch:03d}'), 
                                                   epoch)
                test_results[test_name] = test_metrics
            
            # 保存时使用 model.module
            save_motion_seg_model(model.module if world_size > 1 else model, cfg.log_dir, epoch, test_results)
            
            epoch_iou = max([results.get('iou', 0.0) for results in test_results.values()]) if test_results else 0.0
            
            # 更新最佳IoU
            if epoch_iou > best_iou:
                best_iou = epoch_iou
                # 保存最佳模型
                best_model_path = os.path.join(cfg.log_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': (model.module if world_size > 1 else model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_iou': best_iou,
                    'test_results': test_results
                }, best_model_path)
                logger.info(f"New best model saved with IoU: {best_iou:.4f}")
            
            # 保存训练状态到CSV文件
            save_training_log_csv(cfg.log_dir, epoch, train_loss, test_results, optimizer.param_groups[0]['lr'])
            
            logger.info(f"Epoch {epoch} completed - Train Loss: {train_loss:.4f}, Best Test IoU: {epoch_iou:.4f}, Best Overall IoU: {best_iou:.4f}")

    if is_main_process:
        logger.info("Training completed!")
        logger.info(f"Best IoU achieved: {best_iou:.4f}")

    # --- DDP Change: 8. 清理进程组 ---
    cleanup_distributed()


def save_training_log_csv(log_dir, epoch, train_loss, test_results, lr):
    """保存训练日志到CSV文件"""
    import csv
    import os
    
    csv_path = os.path.join(log_dir, 'training_log.csv')
    
    # 准备数据行
    row_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'learning_rate': lr
    }
    
    # 添加测试结果
    for test_name, metrics in test_results.items():
        for metric_name, metric_value in metrics.items():
            row_data[f'{test_name}_{metric_name}'] = metric_value
    
    # 写入CSV
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)


def plot_training_curves(log_dir):
    """绘制训练曲线"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    csv_path = os.path.join(log_dir, 'training_log.csv')
    
    if not os.path.exists(csv_path):
        return
    
    try:
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失
        axes[0, 0].plot(df['epoch'], df['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 学习率
        axes[0, 1].plot(df['epoch'], df['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # 测试IoU
        iou_columns = [col for col in df.columns if 'iou' in col.lower()]
        for col in iou_columns:
            axes[1, 0].plot(df['epoch'], df[col], label=col)
        axes[1, 0].set_title('Test IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 测试F1
        f1_columns = [col for col in df.columns if 'f1' in col.lower()]
        for col in f1_columns:
            axes[1, 1].plot(df['epoch'], df[col], label=col)
        axes[1, 1].set_title('Test F1')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting training curves: {e}")


# debug_bceloss 函数保持不变，因为它是一个独立的调试工具

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trajectory-based motion segmentation network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config_file', metavar='DIR',help='path to config file')
    args = parser.parse_args()
    cfg = load_config_file(args.config_file)
    
    main(cfg)
