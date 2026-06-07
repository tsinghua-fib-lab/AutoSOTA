import os
import time
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import cv2
from scipy.ndimage import distance_transform_edt
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.structures import Volumes
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from volume_illusion.model import (
    VolumeModel,
    create_dual_view_cameras, create_voxel_optimizer, apply_density_constraints
)
from volume_illusion.renderer import create_volume_renderer
from volume_illusion.visualization import generate_rotating_volume, image_grid
from volume_illusion.component_pruning import select_components_by_projection


def load_supervision_image(image_path, device='cuda', target_size=None):
    if isinstance(image_path, str):
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            alpha = np.array(img)[:, :, 3] / 255.0
            alpha = (alpha > 0.5).astype(np.float32)
        else:
            img_gray = img.convert('L')
            alpha = np.array(img_gray) / 255.0
            alpha = (alpha > 0.5).astype(np.float32)
    else:
        alpha = image_path
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.cpu().numpy()
        if len(alpha.shape) == 3:
            if alpha.shape[2] == 4:
                alpha = alpha[:, :, 3]
            else:
                alpha = np.mean(alpha, axis=2)
        alpha = (alpha > 0.5).astype(np.float32)
    if target_size is not None:
        if isinstance(alpha, torch.Tensor):
            alpha_np = alpha.detach().cpu().numpy()
        else:
            alpha_np = alpha
        img_alpha = Image.fromarray((alpha_np * 255).astype(np.uint8))
        img_alpha = img_alpha.resize((target_size[1], target_size[0]), Image.NEAREST)
        alpha = np.asarray(img_alpha).astype(np.float32) / 255.0
        alpha = (alpha > 0.5).astype(np.float32)
    alpha_tensor = torch.from_numpy(alpha).float().to(device)
    return alpha_tensor

def load_rgb_image(image_path, device='cuda', target_size=None):
    img = Image.open(image_path).convert('RGB')
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    rgb = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
    return rgb

def render_for_silhouette(volume_model: 'VolumeModel', cameras):
    return volume_model(cameras)


def render_for_rgb(volume_model: 'VolumeModel', cameras):
    req = volume_model.log_densities.requires_grad

    volume_model.log_densities.requires_grad_(False)
    out = volume_model(cameras)
    volume_model.log_densities.requires_grad_(req)
    return out

def compute_edge_weights(mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
    dist_fg = distance_transform_edt(mask_np)
    dist_bg = distance_transform_edt(1 - mask_np)
    w = np.exp(-dist_fg / sigma) * mask_np + np.exp(-dist_bg / sigma) * (1 - mask_np)
    return torch.from_numpy(w).float()


# PWA suppresses background noise by emphasizing distant pixels.
def compute_bg_far_weight(mask: torch.Tensor, power: float = 1.5, w_max: float = 3.0) -> torch.Tensor:
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask_np = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)

    dist_bg = distance_transform_edt(1 - mask_np)
    maxv = float(dist_bg.max())
    if maxv <= 0:
        w = np.ones_like(dist_bg, dtype=np.float32)
    else:
        dist_n = dist_bg / maxv
        w = 1.0 + (w_max - 1.0) * np.power(dist_n, power)
    return torch.from_numpy(w.astype(np.float32))

def projection_edge_loss(volume_model,
                         target_alpha1: torch.Tensor,
                         target_alpha2: torch.Tensor,
                         weight1: torch.Tensor,
                         weight2: torch.Tensor,
                         camera1, camera2,
                         target_rgb1: torch.Tensor | None = None,
                         target_rgb2: torch.Tensor | None = None,
                         keep_ratio: float = 0.2,
                         lambda_iou: float = 0.2,
                         lambda_bce: float = 1.0,
                         lambda_outside: float = 1.0,
                         lambda_fill: float = 0.3,
                         lambda_rgb: float = 0.0):
    pred_rgba1 = volume_model(camera1)
    pred_rgba2 = volume_model(camera2)

    pred_alpha1 = pred_rgba1[..., 3].clamp(0.0, 1.0)
    pred_alpha2 = pred_rgba2[..., 3].clamp(0.0, 1.0)

    if keep_ratio < 1.0:
        drop_mask1 = (torch.rand_like(weight1) < keep_ratio).float()
        drop_mask2 = (torch.rand_like(weight2) < keep_ratio).float()
        eff_weight1 = weight1 * drop_mask1
        eff_weight2 = weight2 * drop_mask2
    else:
        eff_weight1 = weight1
        eff_weight2 = weight2

    bce1 = F.binary_cross_entropy(pred_alpha1.squeeze(0), target_alpha1.squeeze(0), weight=eff_weight1, reduction='sum')
    bce2 = F.binary_cross_entropy(pred_alpha2.squeeze(0), target_alpha2.squeeze(0), weight=eff_weight2, reduction='sum')
    denom1 = eff_weight1.sum().clamp_min(1e-6)
    denom2 = eff_weight2.sum().clamp_min(1e-6)
    bce_loss = bce1 / denom1 + bce2 / denom2

    def _soft_iou(pred, target, eps=1e-6):
        inter = (pred * target).sum((-2, -1))
        union = pred.sum((-2, -1)) + target.sum((-2, -1)) - inter
        return 1.0 - ((inter + eps) / (union + eps)).mean()

    iou_loss = _soft_iou(pred_alpha1, target_alpha1) + _soft_iou(pred_alpha2, target_alpha2)

    outside_loss = ((pred_alpha1 * (1 - target_alpha1)).mean() +
                    (pred_alpha2 * (1 - target_alpha2)).mean())

    def _fill_loss(pred, target):
        fg = target > 0.5
        if fg.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return ((pred - 1.0) ** 2 * fg.float()).sum() / fg.sum()

    fill_loss = _fill_loss(pred_alpha1, target_alpha1) + _fill_loss(pred_alpha2, target_alpha2)

    rgb_loss = torch.tensor(0.0, device=pred_alpha1.device)
    if target_rgb1 is not None and target_rgb2 is not None and lambda_rgb > 0.0:
        pr1 = pred_rgba1[0, ..., :3].permute(2, 0, 1)
        pr2 = pred_rgba2[0, ..., :3].permute(2, 0, 1)

        fg1 = (target_alpha1 > 0.5).float().unsqueeze(0)
        fg2 = (target_alpha2 > 0.5).float().unsqueeze(0)
        denom_fg1 = fg1.sum().clamp_min(1e-6)
        denom_fg2 = fg2.sum().clamp_min(1e-6)

        l1_1 = (torch.abs(pr1 - target_rgb1) * fg1).sum() / denom_fg1
        l1_2 = (torch.abs(pr2 - target_rgb2) * fg2).sum() / denom_fg2
        rgb_loss = l1_1 + l1_2

    total = (lambda_bce * bce_loss +
             lambda_iou * iou_loss +
             lambda_outside * outside_loss +
             lambda_fill * fill_loss +
             lambda_rgb * rgb_loss)

    return total, {
        'bce_loss': bce_loss.detach(),
        'iou_loss': iou_loss.detach(),
        'outside_loss': outside_loss.detach(),
        'fill_loss': fill_loss.detach(),
        'rgb_loss': rgb_loss.detach(),
    }

def axial_area_variance_loss(volume_model: 'VolumeModel', threshold: float = 0.2):
    densities = volume_model.get_densities()[0, 0]
    active = (densities > threshold).float()
    slice_area = active.sum((0, 1))
    if slice_area.sum() == 0:
        return torch.tensor(0.0, device=densities.device)
    norm = slice_area.mean()
    var = ((slice_area - norm) ** 2).mean()
    return var / (norm ** 2 + 1e-6)


def binary_voxel_train(
    supervision_image1=None,  
    supervision_image2=None,  
    volume_size=128,
    volume_extent_world=3.0,
    render_scale: float = 2.0,
    render_size: int | tuple[int, int] | None = None,
    n_pts_per_ray: int = 100,
    n_iter=800,
    lr=0.05,
    device='cuda',
    output_dir='results',
    gumbel_temperature=2.0,
    temperature_decay=0.99,
    constraint_strength=0.1,
    azim1: float = 0.0,
    azim2: float = 180.0,
    elev1: float = 0.0,
    elev2: float = 0.0,
    orthographic: bool = True,
    interior_min_density: float = 0.6,
    interior_weight: float = 0.1,
    interior_kernel: int = 5,
    decouple_training: bool = True,
    shape_ratio: float = 0.7,
    freeze_density_mapping: bool = True,
    disable_pruning_after_boundary: bool = True
):

    print("\n=== Starting binary voxel optimization ===")

    if volume_size not in (128, 256):
        raise ValueError("volume_size must be 128 or 256")
    print(f"Using fixed world extent {volume_extent_world} (no scale coupling)")

    if render_size is None:
        render_size = int(volume_size * render_scale)

    custom_res = isinstance(render_size, (tuple, list))
    if custom_res:
        render_h, render_w = int(render_size[0]), int(render_size[1])
    else:
        render_h = render_w = int(render_size)

    print(f"Volume grid: {volume_size}^3")
    print(f"Render resolution: {render_h}x{render_w} (scale={render_scale})")
    print(f"Iterations: {n_iter}")
    print(f"Learning rate: {lr}")
    print(f"Gumbel temperature: {gumbel_temperature}")
    print(f"External supervision enabled: {supervision_image1 is not None and supervision_image2 is not None}")
    proj_type = 'Orthographic' if orthographic else 'Perspective'
    print(f"Camera setup: (azim1={azim1}°, elev1={elev1}°) vs (azim2={azim2}°, elev2={elev2}°) | {proj_type}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"binary_voxel_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "models"), exist_ok=True)
    
    base_vs = volume_size
    volume_size_list = [base_vs, base_vs, base_vs]
    print(f"Voxel grid shape: {volume_size_list} (D,H,W)")

    renderer = create_volume_renderer((render_h, render_w), volume_extent_world,
                                      n_pts_per_ray=n_pts_per_ray,
                                      device=device, orthographic=orthographic)
    
    if base_vs == 256:
        preset = {
            'valid_cube_ratio': 0.85,
            'init_cube_ratio': 0.80,
            'pos_logit': 1.5,
            'neg_logit': -1.0,
            'outside_max': 3.5,
            'outside_rise_end': 0.80,
            'raydrop_start_keep': 1.0,
            'prune_start_ratio': 0.60,
            'thin_neighbors': 3,
            'thin_density_th': 0.15,
            'final_prune_th': 0.40,
        }
    else:
        preset = {
            'valid_cube_ratio': 0.85,
            'init_cube_ratio': 0.85,
            'pos_logit': 1.5,
            'neg_logit': -1.0,
            'outside_max': 4.0,
            'outside_rise_end': 0.80,
            'raydrop_start_keep': 1.0,
            'prune_start_ratio': 0.60,
            'thin_neighbors': 3,
            'thin_density_th': 0.15,
            'final_prune_th': 0.40,
        }
    print(f"Preset: {preset}")
    volume_model = VolumeModel(
        renderer=renderer,
        volume_size=volume_size_list,
        voxel_size=volume_extent_world / base_vs,
        gumbel_temperature=gumbel_temperature,
        hard_gumbel=False,
        init_cube_ratio=preset['init_cube_ratio'],
        valid_cube_ratio=preset['valid_cube_ratio'],
        init_logit_pos=preset['pos_logit'],
        init_logit_neg=preset['neg_logit']
    ).to(device)

    interior_kernel_eff = interior_kernel
    if base_vs == 256:
        interior_kernel_eff = int(5 * (base_vs / 128))
        if interior_kernel_eff % 2 == 0:
            interior_kernel_eff += 1
    interior_min_density_eff = interior_min_density
    if base_vs == 256:
        interior_min_density_eff = min(0.5, interior_min_density)
    
    camera_view1, camera_view2 = create_dual_view_cameras(device=device,
                                                         azim1=azim1, azim2=azim2,
                                                         elev1=elev1, elev2=elev2,
                                                         orthographic=orthographic)
    
    if supervision_image1 is not None and supervision_image2 is not None:
        print("Loading supervision images...")
        target_alpha1 = load_supervision_image(supervision_image1, device, (render_h, render_w))
        target_alpha2 = load_supervision_image(supervision_image2, device, (render_h, render_w))
        target_rgb1 = load_rgb_image(supervision_image1, device, (render_h, render_w))
        target_rgb2 = load_rgb_image(supervision_image2, device, (render_h, render_w))
    else:
        print("Synthesizing supervision from the initial model...")
        with torch.no_grad():
            rendered1 = volume_model(camera_view1)
            rendered2 = volume_model(camera_view2)
            target_alpha1 = rendered1[0, ..., 3]
            target_alpha2 = rendered2[0, ..., 3]
            target_rgb1 = rendered1[0, ..., :3].permute(2, 0, 1)
            target_rgb2 = rendered2[0, ..., :3].permute(2, 0, 1)
            target_alpha1 = (target_alpha1 > 0.5).float()
            target_alpha2 = (target_alpha2 > 0.5).float()
    
    print(f"Target mask shape (view1): {target_alpha1.shape}")
    print(f"Target mask shape (view2): {target_alpha2.shape}")

    edge_weight1 = compute_edge_weights(target_alpha1, sigma=2.0).to(device)
    edge_weight2 = compute_edge_weights(target_alpha2, sigma=2.0).to(device)
    
    bg_far_w1 = compute_bg_far_weight(target_alpha1).to(device)
    bg_far_w2 = compute_bg_far_weight(target_alpha2).to(device)
    
    
    print("\nSaving initial renders...")
    with torch.no_grad():
        rendered1 = volume_model(camera_view1)
        rendered2 = volume_model(camera_view2)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(rendered1[0, ..., :3].cpu().numpy())
        axes[0, 0].set_title('Initial State - View1 (0°)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(rendered2[0, ..., :3].cpu().numpy())
        axes[0, 1].set_title('Initial State - View2 (180°)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(rendered1[0, ..., 3].cpu().numpy(), cmap='gray')
        axes[1, 0].set_title('Initial Alpha Mask - View1')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(rendered2[0, ..., 3].cpu().numpy(), cmap='gray')
        axes[1, 1].set_title('Initial Alpha Mask - View2')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "initial_state.png"))
        plt.close()
    
    
    # Shape-Color Decoupled (SCD) schedule freezes colors until geometry converges.
    if decouple_training:
        volume_model.log_colors1.requires_grad_(False)
        volume_model.log_colors2.requires_grad_(False)
        optimizer = torch.optim.Adam([volume_model.log_densities], lr=lr)
    else:
        optimizer = create_voxel_optimizer(volume_model, lr=lr)
    

    
    optimizer_frozen = False
    print("\n=== Entering training loop ===")

    prune_start = int(n_iter * preset['prune_start_ratio'])

   
    shape_iters = int(n_iter * shape_ratio)
    mapping_frozen = False 

    
    for iteration in range(n_iter):
        
        if iteration == round(n_iter * 0.75):
            print('Reducing learning rate...')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
      
        if not (decouple_training and mapping_frozen and iteration >= shape_iters and freeze_density_mapping):
            warm_ratio   = 0.15
            shape_ratio_sched  = 0.40
            sparse_ratio = 0.80

            if iteration < n_iter * warm_ratio:  
                t = iteration / (n_iter * warm_ratio)
                volume_model.inner_temperature = 0.3 + 0.2 * t           # 0.3→0.5
                volume_model.outer_scale      = 0.15 + 0.35 * t          # 0.15→0.5
                volume_model.density_bias     = 0.0
            elif iteration < n_iter * shape_ratio_sched:
                t = (iteration - n_iter * warm_ratio) / (n_iter * (shape_ratio_sched - warm_ratio))
                volume_model.inner_temperature = 0.5 + 1.0 * t           # 0.5→1.5
                volume_model.outer_scale      = 0.5 + 0.5 * t           # 0.5→1.0
                volume_model.density_bias     = 0.0
            elif iteration < n_iter * sparse_ratio:
                t = (iteration - n_iter * shape_ratio_sched) / (n_iter * (sparse_ratio - shape_ratio_sched))
                volume_model.inner_temperature = 1.5 + 1.5 * t           # 1.5→3.0
                volume_model.outer_scale      = 1.0 - 0.4 * t            # 1.0→0.6
                volume_model.density_bias     = 1.0 * t                  # 0→1.0
            else:
                t = (iteration - n_iter * sparse_ratio) / (n_iter * (1 - sparse_ratio))
                volume_model.inner_temperature = 3.0 + 2.0 * t           # 3.0→5.0
                volume_model.outer_scale      = 0.6 - 0.3 * t            # 0.6→0.3
                volume_model.density_bias     = 1.0                      
        
        optimizer.zero_grad()

       
        densities = volume_model.get_densities()

        
        with torch.no_grad():
            active_voxel_ratio = float((densities > 0.25).float().mean().item())

        
        outside_base = 1.0
        outside_max  = preset['outside_max']
        progress_ratio = iteration / n_iter
        if progress_ratio < preset['outside_rise_end']:
            outside_w = outside_base
        else:
            t_out = (progress_ratio - preset['outside_rise_end']) / max(1e-6, (1 - preset['outside_rise_end']))
            t_out = max(0.0, min(1.0, t_out))
            outside_w = outside_base + (outside_max - outside_base) * t_out


        if decouple_training and iteration == shape_iters:
            print("==> Switching to color optimization (density frozen)...")
            volume_model.log_densities.requires_grad_(False)
            volume_model.log_colors1.requires_grad_(True)
            volume_model.log_colors2.requires_grad_(True)
            if freeze_density_mapping:
                mapping_frozen = True
            
            optimizer = torch.optim.Adam([
                volume_model.log_colors1,
                volume_model.log_colors2
            ], lr=lr * 2.0)

        if base_vs == 256:
            keep_ratio = min(1.0, preset['raydrop_start_keep'] + (1.0 - preset['raydrop_start_keep']) * progress_ratio)
        else:
            keep_ratio = 1.0


        total_loss = torch.tensor(0.0, device=densities.device)
        
        bce_loss = torch.tensor(0.0, device=densities.device)
        iou_loss = torch.tensor(0.0, device=densities.device)
        rgb_loss_total = torch.tensor(0.0, device=densities.device)
        constraint_loss = torch.tensor(0.0, device=densities.device)
        interior_loss = torch.tensor(0.0, device=densities.device)
        area_var_loss = torch.tensor(0.0, device=densities.device)

        in_shape_phase = (iteration < shape_iters) or (not decouple_training)
        in_color_phase = (iteration >= shape_iters) and decouple_training

        if in_shape_phase:
            
            pred_sil_1 = render_for_silhouette(volume_model, camera_view1)
            pred_sil_2 = render_for_silhouette(volume_model, camera_view2)

            pred_alpha1 = pred_sil_1[..., 3].clamp(0.0, 1.0)
            pred_alpha2 = pred_sil_2[..., 3].clamp(0.0, 1.0)

           
            w1 = (edge_weight1 * bg_far_w1) * keep_ratio + 1e-8
            w2 = (edge_weight2 * bg_far_w2) * keep_ratio + 1e-8
            
            bce1 = F.binary_cross_entropy(pred_alpha1.squeeze(0), target_alpha1, weight=w1, reduction='sum')
            bce2 = F.binary_cross_entropy(pred_alpha2.squeeze(0), target_alpha2, weight=w2, reduction='sum')
            
            bce_loss = bce1 / w1.sum().clamp_min(1e-6) + bce2 / w2.sum().clamp_min(1e-6)

            def _soft_iou(p, t):
                inter = (p * t).sum()
                union = p.sum() + t.sum() - inter
                return 1.0 - (inter + 1e-6) / (union + 1e-6)

            iou_loss = _soft_iou(pred_alpha1, target_alpha1) + _soft_iou(pred_alpha2, target_alpha2)

            outside1 = (pred_alpha1 * (1 - target_alpha1) * bg_far_w1.unsqueeze(0)).sum() / bg_far_w1.sum().clamp_min(1e-6)
            outside2 = (pred_alpha2 * (1 - target_alpha2) * bg_far_w2.unsqueeze(0)).sum() / bg_far_w2.sum().clamp_min(1e-6)
            outside_loss = (outside1 + outside2) / 2.0

            def _fill(p, t):
                fg = (t > 0.5).float()
                if fg.sum() == 0:
                    return torch.tensor(0.0, device=p.device)
                return ((p - 1.0) ** 2 * fg).sum() / fg.sum()

            fill_loss = _fill(pred_alpha1, target_alpha1) + _fill(pred_alpha2, target_alpha2)

            sil_loss = (1.0 * bce_loss + 0.2 * iou_loss + outside_w * outside_loss + 0.3 * fill_loss)
            total_loss = total_loss + sil_loss

        if in_color_phase:
           
            pred_rgb_1 = render_for_rgb(volume_model, camera_view1)
            pred_rgb_2 = render_for_rgb(volume_model, camera_view2)

            rgb1 = pred_rgb_1[..., :3].permute(0, 3, 1, 2)
            rgb2 = pred_rgb_2[..., :3].permute(0, 3, 1, 2)

            
            fg1 = (target_alpha1 > 0.5).float().unsqueeze(0)
            fg2 = (target_alpha2 > 0.5).float().unsqueeze(0)
            l1_rgb = ((torch.abs(rgb1 - target_rgb1) * fg1).sum() / fg1.sum().clamp_min(1e-6) +
                      (torch.abs(rgb2 - target_rgb2) * fg2).sum() / fg2.sum().clamp_min(1e-6))

            
            wrong1 = (torch.abs(rgb1 - target_rgb2) * fg1).mean()
            wrong2 = (torch.abs(rgb2 - target_rgb1) * fg2).mean()
            mutual_ex = ((torch.abs(rgb1 - target_rgb1) * fg1).mean() + (torch.abs(rgb2 - target_rgb2) * fg2).mean())
            cross_loss = 1.0 * mutual_ex - 0.5 * (wrong1 + wrong2)

            
            def _color_tv(c):
                dx = c[:, :, 1:, :, :] - c[:, :, :-1, :, :]
                dy = c[:, :, :, 1:, :] - c[:, :, :, :-1, :]
                dz = c[:, :, :, :, 1:] - c[:, :, :, :, :-1]
                return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()) / 3.0

            colors1_vol = torch.sigmoid(volume_model.log_colors1).unsqueeze(0)
            colors2_vol = torch.sigmoid(volume_model.log_colors2).unsqueeze(0)
            tv_color = _color_tv(colors1_vol) + _color_tv(colors2_vol)

            rgb_loss_total = (l1_rgb + cross_loss + 0.1 * tv_color)
            total_loss = total_loss + rgb_loss_total

        
        loss_details = {
            'bce_loss': bce_loss.detach(),
            'iou_loss': iou_loss.detach(),
            'rgb_loss': rgb_loss_total.detach(),
        }

      
        
        
        if in_color_phase and progress_ratio > 0.3:
            pred_rgba1_full = volume_model(camera_view1)
            pred_rgba2_full = volume_model(camera_view2)

            pred_rgb1_full = pred_rgba1_full[..., :3].permute(0, 3, 1, 2)
            pred_rgb2_full = pred_rgba2_full[..., :3].permute(0, 3, 1, 2)

            fg1_mask = (target_alpha1 > 0.5).float().unsqueeze(0)
            fg2_mask = (target_alpha2 > 0.5).float().unsqueeze(0)

            mut1 = (torch.abs(pred_rgb1_full - target_rgb1) * fg1_mask).mean()
            mut2 = (torch.abs(pred_rgb2_full - target_rgb2) * fg2_mask).mean()
            mutual_ex = mut1 + mut2

            wrong1 = (torch.abs(pred_rgb1_full - target_rgb2) * fg1_mask).mean()
            wrong2 = (torch.abs(pred_rgb2_full - target_rgb1) * fg2_mask).mean()
            wrong_ex = wrong1 + wrong2

            lam_pos = 1.0
            lam_neg = 0.5
            cross_loss_extra = lam_pos * mutual_ex - lam_neg * wrong_ex
            total_loss += cross_loss_extra

       
        # Multi-pass smoothing mitigates voxel ringing before PAC pruning.
        lambda_smooth_init = 0.1
        decay_rate = 0.99
        lambda_smooth = lambda_smooth_init * (decay_rate ** iteration)

        if in_shape_phase and lambda_smooth > 1e-4:
            neighbor_avg = F.avg_pool3d(densities, kernel_size=3, stride=1, padding=1)
            lap_loss = torch.abs(densities - neighbor_avg).mean()
            total_loss += lambda_smooth * lap_loss * active_voxel_ratio

            lambda_tv_init = 0.05
            lambda_tv = lambda_tv_init * (decay_rate ** iteration)
            if lambda_tv > 1e-5:
                dx = densities[:, :, 1:, :, :] - densities[:, :, :-1, :, :]
                dy = densities[:, :, :, 1:, :] - densities[:, :, :, :-1, :]
                dz = densities[:, :, :, :, 1:] - densities[:, :, :, :, :-1]
                tv_loss = (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()) / 3.0
                total_loss += lambda_tv * tv_loss * active_voxel_ratio

        ramp_start = int(n_iter * 0.4)
        if in_shape_phase and iteration < ramp_start:
            constraint_strength_now = 0.0
        else:
            progress = (iteration - ramp_start) / max(1, (n_iter - ramp_start))
            constraint_strength_now = constraint_strength * progress

        
        if in_shape_phase:
            # IVP keeps interior voxels from collapsing during sparsification.
            with torch.no_grad():
                occ = (densities > 0.4).float()
                pooled = F.avg_pool3d(occ, kernel_size=interior_kernel_eff, stride=1, padding=interior_kernel_eff // 2)
                interior_mask = (pooled >= 1.0).float()
            outside_mask = 1.0 - interior_mask

            sparsity_loss = (densities * outside_mask).mean() * constraint_strength_now * 2.0
            binary_loss_reg = (4 * densities * (1 - densities) * outside_mask).mean() * constraint_strength_now * 2.0
            constraint_loss = sparsity_loss + binary_loss_reg
            total_loss += constraint_loss * active_voxel_ratio

       
        if in_shape_phase:
            if interior_mask.sum() > 0:
                interior_densities = densities * interior_mask
                min_deficit = F.relu(interior_min_density_eff - interior_densities) ** 2
                interior_loss = min_deficit.sum() / (interior_mask.sum() + 1e-6)
                total_loss += interior_weight * interior_loss
            else:
                interior_loss = torch.tensor(0.0, device=densities.device)

       
        if in_shape_phase:
            area_var_loss = axial_area_variance_loss(volume_model, threshold=0.2)
            total_loss += area_var_loss * active_voxel_ratio * (128.0 / base_vs)

        
        total_loss.backward()
        optimizer.step()

        if iteration >= prune_start and iteration % (60 if base_vs == 256 else 100) == 0 and (not decouple_training or iteration < shape_iters or not disable_pruning_after_boundary):
            
            opt_info = volume_model.get_optimization_info()
            active_ratio = opt_info["active_voxels"] / max(1, opt_info["total_voxels"])

            if active_ratio >= 0.01: 
                base_min = 0.25 if base_vs == 256 else 0.15
                dynamic_thresh = max(base_min, 0.3 * volume_model.outer_scale)  # 随 outer_scale 调整

                
                # PAC: retain only components whose dual-view projections align with the targets.
                keep_mask = select_components_by_projection(
                    volume_model,
                    camera_view1,
                    camera_view2,
                    target_alpha1,
                    target_alpha2,
                    density_threshold=max(dynamic_thresh, 0.20),
                    bg_far_w1=bg_far_w1,
                    bg_far_w2=bg_far_w2,
                    k_max=4,
                    score_thresh=0.1,
                    min_ratio=0.001
                )

                with torch.no_grad():
                    neg_val = float(volume_model.init_logit_neg)
                    volume_model.log_densities.data[0][~keep_mask] = neg_val

                
                volume_model.prune_thin_connections(
                    density_threshold=max(preset['thin_density_th'] * 0.9, dynamic_thresh * 0.9),
                    min_neighbors=max(4, preset['thin_neighbors'])
                )
            else:
                
                pass
        
        
        if iteration % 20 == 0:
            opt_info = volume_model.get_optimization_info()

            
            with torch.no_grad():
                grad = volume_model.log_densities.grad
                grad_mean = grad.mean().item() if grad is not None else 0.0
                grad_std  = grad.std().item()  if grad is not None else 0.0
                grad_max  = grad.max().item()  if grad is not None else 0.0
                grad_min  = grad.min().item()  if grad is not None else 0.0

            print(f'Iter {iteration:04d}: total={total_loss.item():.4f}, '
                  f'BCE={loss_details["bce_loss"].item():.4f}, '
                  f'IoU={loss_details["iou_loss"].item():.4f}, '
                  f'RGB={loss_details["rgb_loss"].item():.4f}, '
                  f'constraint={constraint_loss.item():.4f}, interior={interior_loss.item():.4f}, '
                  f'areaVar={area_var_loss.item():.4f}, '
                  f'active={opt_info["active_voxels"]}/{opt_info["total_voxels"]}, '
                  f'T1={volume_model.inner_temperature:.2f}, scale={volume_model.outer_scale:.2f}, '
                  f'Grad(mu={grad_mean:.5f}, sigma={grad_std:.5f}, max={grad_max:.5f}, min={grad_min:.5f})')
        
        
        if iteration % 100 == 0 or iteration == n_iter - 1:
            with torch.no_grad():
                rendered1 = volume_model(camera_view1)
                rendered2 = volume_model(camera_view2)
                
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                
                axes[0, 0].imshow(rendered1[0, ..., :3].cpu().numpy())
                axes[0, 0].set_title(f'Iter{iteration} - Render View1')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(rendered2[0, ..., :3].cpu().numpy())
                axes[0, 1].set_title(f'Iter{iteration} - Render View2')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(target_alpha1.cpu().numpy(), cmap='gray')
                axes[0, 2].set_title('Target Mask1')
                axes[0, 2].axis('off')
                
                axes[0, 3].imshow(target_alpha2.cpu().numpy(), cmap='gray')
                axes[0, 3].set_title('Target Mask2')
                axes[0, 3].axis('off')
                
                
                pred_alpha1 = rendered1[0, ..., 3].cpu().numpy()
                pred_alpha2 = rendered2[0, ..., 3].cpu().numpy()
                
                axes[1, 0].imshow(pred_alpha1, cmap='gray')
                axes[1, 0].set_title('Predicted Mask1')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(pred_alpha2, cmap='gray')
                axes[1, 1].set_title('Predicted Mask2')
                axes[1, 1].axis('off')
                
                
                axes[1, 2].imshow((pred_alpha1 > 0.5).astype(float), cmap='gray')
                axes[1, 2].set_title('Binary Prediction1')
                axes[1, 2].axis('off')
                
                axes[1, 3].imshow((pred_alpha2 > 0.5).astype(float), cmap='gray')
                axes[1, 3].set_title('Binary Prediction2')
                axes[1, 3].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "images", f"training_{iteration:04d}.png"))
                plt.close(fig)
                
                
                if (not decouple_training) or (iteration < shape_iters):
                    densities_vis = volume_model.get_densities()
                    mid_w = volume_size_list[2] // 2
                    middle_slice = densities_vis[0, 0, :, :, mid_w].cpu().numpy()
                    
                    plt.figure(figsize=(8, 6))
                    plt.imshow(middle_slice, cmap='hot')
                    plt.colorbar()
                    plt.title(f'Iteration {iteration} - Density Middle Slice')
                    plt.savefig(os.path.join(output_path, "images", f"density_{iteration:04d}.png"))
                    plt.close()
    
    
    with torch.no_grad():
        print("\nRunning final PAC filtering...")
        keep_mask_final = select_components_by_projection(
            volume_model,
            camera_view1,
            camera_view2,
            target_alpha1,
            target_alpha2,
            density_threshold=preset['final_prune_th'],
            bg_far_w1=bg_far_w1,
            bg_far_w2=bg_far_w2,
            k_max=4,
            score_thresh=0.1,
            min_ratio=0.001
        )
        neg_val = float(volume_model.init_logit_neg)
        volume_model.log_densities.data[0][~keep_mask_final] = neg_val

    
    final_model_path = os.path.join(output_path, "models", f"binary_voxel_model_{timestamp}.pt")
    torch.save({
        'state_dict': volume_model.state_dict(),
        'volume_size': volume_size_list,
        'voxel_size': volume_model._voxel_size,
        'gumbel_temperature': volume_model.gumbel_temperature,
        'inner_temperature': volume_model.inner_temperature,
        'outer_scale': volume_model.outer_scale,
        'density_bias': volume_model.density_bias,
        'timestamp': timestamp
    }, final_model_path)
    
    print("\n=== Training complete ===")
    print(f"Model saved to: {final_model_path}")
    print(f"Artifacts saved to: {output_path}")
    
    
    final_info = volume_model.get_optimization_info()
    print(f"Final active voxels: {final_info['active_voxels']}/{final_info['total_voxels']} "
          f"({final_info['active_voxels']/final_info['total_voxels']*100:.1f}%)")
    print(f"Final density range: [{final_info['density_min']:.3f}, {final_info['density_max']:.3f}]")
    
    return volume_model, output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="二值体素优化训练")
    
    parser.add_argument('--supervision_image1', type=str, default=None,
                        help='第一个视角的监督图像路径')
    parser.add_argument('--supervision_image2', type=str, default=None,
                        help='第二个视角的监督图像路径')
    parser.add_argument('--volume_size', type=int, default=128,
                        help='体积大小')
    parser.add_argument('--n_iter', type=int, default=800,
                        help='训练迭代次数')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--gpu', type=int, help='指定使用的GPU ID')
    parser.add_argument('--gumbel_temperature', type=float, default=2.0,
                        help='Gumbel Softmax初始温度')
    parser.add_argument('--temperature_decay', type=float, default=0.99,
                        help='温度衰减率')
    parser.add_argument('--constraint_strength', type=float, default=0.1,
                        help='密度约束强度')
    # 新增：渲染分辨率缩放系数与每光线采样点数
    parser.add_argument('--render_scale', type=float, default=2.0,
                        help='渲染分辨率倍率 (render_size = volume_size * render_scale)')
    # 非正方形渲染分辨率（若指定，高于 render_scale）
    parser.add_argument('--render_width', type=int, default=None,
                        help='渲染图像宽度 (像素)，与 --render_height 同时使用')
    parser.add_argument('--render_height', type=int, default=None,
                        help='渲染图像高度 (像素)，与 --render_width 同时使用')
    parser.add_argument('--pts_per_ray', type=int, default=150,
                        help='每条光线的采样点数 (n_pts_per_ray)')
    parser.add_argument('--azim1', type=float, default=0.0,
                        help='第一个视角的方位角 (度)')
    parser.add_argument('--azim2', type=float, default=180.0,
                        help='第二个视角的方位角 (度)')
    # 新增：俯仰角
    parser.add_argument('--elev1', type=float, default=0.0,
                        help='第一个视角的俯仰角 (度)')
    parser.add_argument('--elev2', type=float, default=0.0,
                        help='第二个视角的俯仰角 (度)')
    # 默认使用正交投影；若想使用透视投影，传入 --no_orthographic
    parser.add_argument('--no_orthographic', action='store_false', dest='orthographic',
                        help='使用透视投影 (默认正交)')
    parser.set_defaults(orthographic=True)
    # 解耦训练相关开关
    parser.add_argument('--shape_ratio', type=float, default=0.7,
                        help='形状阶段比例（0-1），默认0.6')
    # 默认启用解耦训练；提供关闭开关
    parser.add_argument('--no_decouple_training', action='store_false', dest='decouple_training',
                        help='关闭解耦训练（默认开启）')
    parser.set_defaults(decouple_training=True)
    # 冻结密度映射（保持 α 恒定），默认开启，可通过 --no_freeze_density_mapping 关闭
    parser.add_argument('--no_freeze_density_mapping', action='store_false', dest='freeze_density_mapping',
                        help='在颜色阶段不冻结密度映射')
    parser.set_defaults(freeze_density_mapping=True)
    # 边界后禁用剪枝（默认禁用），可通过 --enable_pruning_after_boundary 启用
    parser.add_argument('--enable_pruning_after_boundary', action='store_false', dest='disable_pruning_after_boundary',
                        help='在颜色阶段允许继续剪枝（默认禁用）')
    parser.set_defaults(disable_pruning_after_boundary=True)
    
    args = parser.parse_args()
    
    # 处理GPU选择
    if args.gpu is not None:
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        args.device = device
    
    # 加载监督图像（如果提供）
    supervision_image1 = None
    supervision_image2 = None
    
    if args.supervision_image1 and args.supervision_image2:
        print(f"加载监督图像...")
        print(f"图像1: {args.supervision_image1}")
        print(f"图像2: {args.supervision_image2}")
        
        # 确保文件存在
        if not os.path.exists(args.supervision_image1):
            print(f"错误：找不到监督图像1: {args.supervision_image1}")
            return
        if not os.path.exists(args.supervision_image2):
            print(f"错误：找不到监督图像2: {args.supervision_image2}")
            return
        
        
        supervision_image1 = args.supervision_image1
        supervision_image2 = args.supervision_image2
        
        print(f"监督图像路径已设置")
    
    
    render_size_arg = None
    if args.render_width is not None and args.render_height is not None:
        render_size_arg = (args.render_height, args.render_width)

    
    model, output_path = binary_voxel_train(
        supervision_image1=supervision_image1,
        supervision_image2=supervision_image2,
        volume_size=args.volume_size,
        n_iter=args.n_iter,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        gumbel_temperature=args.gumbel_temperature,
        temperature_decay=args.temperature_decay,
        constraint_strength=args.constraint_strength,
        azim1=args.azim1,
        azim2=args.azim2,
        elev1=args.elev1,
        elev2=args.elev2,
        orthographic=args.orthographic,
        render_scale=args.render_scale,
        render_size=render_size_arg,
        n_pts_per_ray=args.pts_per_ray,
        
        decouple_training=args.decouple_training,
        shape_ratio=args.shape_ratio,
        freeze_density_mapping=args.freeze_density_mapping,
        disable_pruning_after_boundary=args.disable_pruning_after_boundary
    )
    
    print("训练完成！")
    print(f"模型和结果保存在: {output_path}")


if __name__ == "__main__":
    main() 
