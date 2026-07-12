# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: utils.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/12/9
# -----------------------------------------
import torch
import numpy as np
import numpy.typing as npt
import nibabel as nib
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.stats import multivariate_normal
from torch import fft
import datetime
import time
from sklearn.manifold import TSNE
import seaborn as sns
from typing import List, Tuple, Union, Optional
from sigpy.mri import poisson

Tensor = torch.Tensor
Array = npt.NDArray



def create_undersampling_mask(width, height, R=4, ACS_lines=24, mask_type='Cartesian', channels=1):
    """
    创建指定类型的欠采样掩码
    
    参数:
    width, height: 图像尺寸
    R: 欠采样率
    ACS_lines: 自动校准信号线数量
    mask_type: 'Cartesian'或'Gaussian'
    channels: 通道数
    
    返回:
    mask: 欠采样掩码
    mask_type: 掩码类型描述
    """
    mask = np.zeros((width, height, channels), dtype=np.float32)
    
    if mask_type == 'Cartesian':
        # 笛卡尔采样 - 等间隔采样，现在在height方向上采样（列方向）
        step = int(R)
        # 采样线的索引
        sampling_idx = list(range(0, height, step))
        
        # 添加中心区域ACS线
        center_start = height // 2 - ACS_lines // 2
        center_end = center_start + ACS_lines
        center_idx = list(range(center_start, center_end))
        
        # 合并采样线索引并确保在有效范围内
        sampling_idx = list(set(sampling_idx + center_idx))
        sampling_idx = [idx for idx in sampling_idx if 0 <= idx < height]
        
        # 设置掩码值 - 注意这里用[:, sampling_idx, :] 而不是 [sampling_idx, :, :]
        mask[:, sampling_idx, :] = 1.0
        mask_info = f"Cartesian_R{R}_ACS{ACS_lines}"
        
    elif mask_type == 'Gaussian':
        # 高斯采样 - 中心密，边缘稀疏
        mask_random = np.zeros((width, height), dtype=np.float32)
        
        # 定义中心点
        center_x, center_y = width // 2, height // 2
        
        # 添加中心区域ACS - 确保完全采样
        center_width = min(width, ACS_lines)
        center_height = min(height, ACS_lines)
        
        center_start_w = width // 2 - center_width // 2
        center_end_w = center_start_w + center_width
        center_start_h = height // 2 - center_height // 2
        center_end_h = center_start_h + center_height
        
        # 将中心区域设为1
        mask_random[center_start_w:center_end_w, center_start_h:center_end_h] = 1.0
        
        # 计算每个点到中心的距离
        xx, yy = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        x_dists = xx - center_x
        y_dists = yy - center_y
        dists = np.sqrt(x_dists**2 + y_dists**2)
        
        # 归一化距离
        max_dist = np.sqrt(width**2 + height**2) / 2
        norm_dists = dists / max_dist
        
        # 创建概率映射 - 非线性变化，中心概率高，边缘概率低
        prob_map = (1.0 - norm_dists)**2  # 平方使概率变化更加非线性
        
        # 调整中心区域的概率为0（因为已经全部采样）
        prob_map[center_start_w:center_end_w, center_start_h:center_end_h] = 0
        
        # 调整概率以达到目标采样率
        # 计算已采样点数和目标点数
        sampled_points = np.sum(mask_random)
        target_points = width * height / R
        remaining_points = target_points - sampled_points
        
        # 调整概率映射以获得所需的剩余点数
        if remaining_points > 0 and np.sum(prob_map) > 0:
            prob_map = prob_map / np.sum(prob_map) * remaining_points
            
            # 生成随机数并与概率映射比较
            random_map = np.random.rand(width, height)
            new_samples = (random_map < prob_map).astype(np.float32)
            
            # 合并新采样点
            mask_random = np.logical_or(mask_random, new_samples).astype(np.float32)
        
        # 扩展到所有通道
        for c in range(channels):
            mask[:, :, c] = mask_random
            
        mask_info = f"VarDensity_R{R}_ACS{ACS_lines}"
    
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")
    
    return mask, mask_info

def get_multi_mask(img, size, type='gaussian2d', acc_factor=8, center_fraction=0.04):
    """
    生成三维MRI采样掩码（去掉batch_size维度）
    
    参数:
    img: 输入图像张量，用于确定掩码的形状和设备类型
    size: 图像尺寸（假设为正方形）
    type: 采样掩码类型
    acc_factor: 加速因子（下采样倍数）
    center_fraction: 中心区域保留比例
    
    返回:
    三维mask张量，形状与img相同（去掉batch维度）
    """
    coil_num = img.shape[0]
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
    
    if type == 'gaussian2d':
        mask = torch.zeros_like(img)
        spread = 0.1
        cov_factor = 3 * size * (1 / 128) 
        mean = [size // 2, size // 2] 
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        
        samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
        
    elif type == 'uniformrandom2d':
        mask = torch.zeros_like(img)
        mask_vec = torch.zeros([4, size * size])
        samples = np.random.choice(size * size, int(Nsamp))
        mask_vec[:, samples] = 1
        mask_b = mask_vec.view(size, size)
        mask[...] = mask_b
        
    elif type == 'gaussian1d':
        mask = torch.zeros_like(img)
        mean = size // 2
        std = size * (20.0 / 128)
        Nsamp_center = int(size * center_fraction)
        
        samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[..., int_samples] = 1
        c_from = size // 2 - Nsamp_center // 2
        mask[..., c_from:c_from + Nsamp_center] = 1
        
    elif type == 'uniform1d':
        mask = torch.zeros_like(img)
        Nsamp_center = int(size * center_fraction)
        samples = np.random.choice(size, int(Nsamp))
        mask[..., samples] = 1
        # ACS region
        c_from = size // 2 - Nsamp_center // 2
        mask[..., c_from:c_from + Nsamp_center] = 1
        
    elif type == 'poisson':
        mask = poisson((size, size), accel=acc_factor)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).repeat(coil_num, 1, 1)  
        
    else:
        raise NotImplementedError(f'Mask type {type} is currently not supported.')

    return mask

def save_reconstruction_images(normRec, normOrg, iteration, save_dir,method,vmax=0.7):
        """单独保存重建图像和error map"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 计算error map
        error_map = np.abs(normOrg - normRec)
        
        # 单独保存重建图像
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(normRec, cmap='gray', vmin=0, vmax=vmax)
        ax.axis('off')
    
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        save_path_rec = os.path.join(save_dir, method + f'reconstructed_iter_{iteration}.png')
        plt.savefig(save_path_rec, dpi=300, bbox_inches='tight', pad_inches=0,
                    facecolor='black', edgecolor='black')
        plt.close()
        
        
        
        # 单独保存error map
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(error_map, cmap='magma', vmin=0, vmax=0.1)
        ax.axis('off')
    
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        save_path_error = os.path.join(save_dir, method + f'error_map_iter_{iteration}.png')
        plt.savefig(save_path_error, dpi=300, bbox_inches='tight', pad_inches=0,
                    facecolor='black', edgecolor='black')
        plt.close()
        
        print(f"已保存重建图像到: {save_path_rec}")
        print(f"已保存error map到: {save_path_error}")
      
def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

class MYTVLoss(nn.Module):
    def __init__(self):
        super(MYTVLoss, self).__init__()

    def forward(self, x):
        L_PE, L_RO = x.shape[0], x.shape[1]
        tv_loss = (torch.sum(torch.abs(x[1:, :, :] - x[:L_PE-1, :, :]))+torch.sum(torch.abs(x[:,1:, :] - x[:,:L_RO-1, :])) )/ ((L_PE-1)*(L_RO-1))
        return tv_loss
 
def norm_np(x):
    return np.abs(x)/ np.max(np.abs(x))

def build_coordinate_train(L_PE, L_RO):
    x = np.linspace(0, 1, L_PE)              #*********
    y = np.linspace(0, 1, L_RO)               #*********
    x, y = np.meshgrid(x, y, indexing='ij')  # (L, L), (L, L)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (L*L, 2)
    xy = xy.reshape(L_PE, L_RO, 2)
    return xy

def flip_img(img, axes):
    """
    Flip the image along the given axes.
    :param img:     (..., H, W, C)      torch.float32
    :param axes:    (N)                 int
    :return:        (..., H, W, C)      torch.float32
    """
    for axis in axes:
        img = np.flip(img, [axis])
    return img                                                                                                                                   

def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==3:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    for i in range(nimg):
        img2[i]=div0(img[i]-img[i].min(),img[i].ptp())
        #img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)

def div0( a, b ):
    """ This function handles division by zero """
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c