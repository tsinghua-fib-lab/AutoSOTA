

import torch
import torch.nn.functional as F
from pytorch3d.structures import Volumes
import numpy as np


class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size=[64] * 3, voxel_size=0.1, 
                 gumbel_temperature=1.0, hard_gumbel=False,
                 init_cube_ratio: float | tuple[float, float, float] = 0.85,
                 valid_cube_ratio: float = 0.95,
                 init_logit_pos: float = 1.5,
                 init_logit_neg: float = -1.0):

        super().__init__()
        
        self.volume_size = volume_size
        self.gumbel_temperature = gumbel_temperature
        self.hard_gumbel = hard_gumbel
        self._voxel_size = voxel_size
        self._renderer = renderer


        self.init_cube_ratio = init_cube_ratio  
        self.valid_cube_ratio = valid_cube_ratio  

        
        self.init_logit_pos = init_logit_pos
        self.init_logit_neg = init_logit_neg


        self.inner_temperature: float = 1.0  # T1
        self.outer_scale: float = 1.0        # T2
        self.density_bias: float = 0.0       # b
 
        self.density_scale: float = 0.9

        print(f"Initializing voxel grid: {volume_size}")


        self.register_buffer(
            'spatial_mask',
            self._create_spatial_mask(volume_size, self.valid_cube_ratio)
        )


        self.log_densities = self._init_center_cube_logits(volume_size,
                                                         cube_ratio=self.init_cube_ratio,
                                                         pos_logit=self.init_logit_pos,
                                                         neg_logit=self.init_logit_neg)

        init_val = 0.3
        logit_val = float(np.log(init_val / (1.0 - init_val)))  # approx -0.847
        self.log_colors1 = torch.nn.Parameter(torch.full((3, *volume_size), logit_val))
        self.log_colors2 = torch.nn.Parameter(torch.full((3, *volume_size), logit_val))


        self.register_buffer('axis_dir', torch.tensor([1.0, 0.0, 0.0]))
   
        self.blend_k: float = 25.0
        
        print(f"Initialization complete: grid {volume_size}, Gumbel temperature {gumbel_temperature}")
    
    def _init_center_cube_logits(self, volume_size,
                                 cube_ratio: float | tuple[float, float, float] = 0.6,
                                 pos_logit: float = 1.5,
                                 neg_logit: float = -1.0):

        D, H, W = volume_size

    
        if isinstance(cube_ratio, (tuple, list)) and len(cube_ratio) == 3:
            ratio_d, ratio_h, ratio_w = cube_ratio
        else:
            ratio_d = ratio_h = ratio_w = float(cube_ratio)

        
        logits = torch.full((1, D, H, W), neg_logit)  

       
        size_d = int(D * ratio_d)
        size_h = int(H * ratio_h)
        size_w = int(W * ratio_w)

        half_d = size_d // 2
        half_h = size_h // 2
        half_w = size_w // 2

        center_d, center_h, center_w = D // 2, H // 2, W // 2

        start_d = max(0, center_d - half_d)
        end_d = min(D, center_d + half_d)
        start_h = max(0, center_h - half_h)
        end_h = min(H, center_h + half_h)
        start_w = max(0, center_w - half_w)
        end_w = min(W, center_w + half_w)
        
        logits[0, start_d:end_d, start_h:end_h, start_w:end_w] = pos_logit  # bias toward dense interior
        
      
        center_voxels = (end_d - start_d) * (end_h - start_h) * (end_w - start_w)
        total_voxels = D * H * W
        print(f"Central high-density region: {center_voxels}/{total_voxels} voxels ({center_voxels/total_voxels*100:.1f}%)")
        
        return torch.nn.Parameter(logits)


    def prune_thin_connections(self, density_threshold: float = 0.15, min_neighbors: int = 3):

        with torch.no_grad():

            densities_raw = torch.sigmoid(self.inner_temperature * self.log_densities - self.density_bias)
            densities_raw = torch.clamp(densities_raw * self.outer_scale, 0.0, 1.0)

            binary = (densities_raw[0] > density_threshold).float()  # (D, H, W)
            if binary.sum() == 0:
                return  


            kernel = torch.ones((1, 1, 3, 3, 3), device=binary.device)
            neighbor_cnt = F.conv3d(binary.unsqueeze(0).unsqueeze(0), kernel, padding=1)
            neighbor_cnt = neighbor_cnt.squeeze(0).squeeze(0) - binary  


            thin_mask = (binary == 1) & (neighbor_cnt < min_neighbors)

            if thin_mask.sum() == 0:
                return

            neg_val = float(self.init_logit_neg)
            self.log_densities.data[0][thin_mask] = neg_val
        return

    def _create_spatial_mask(self, volume_size, cube_ratio: float):

        D, H, W = volume_size
        mask = torch.zeros((1, D, H, W))

        eff_D = int(D * cube_ratio)
        eff_H = int(H * cube_ratio)
        eff_W = int(W * cube_ratio)

        start_d = (D - eff_D) // 2
        end_d = start_d + eff_D
        start_h = (H - eff_H) // 2
        end_h = start_h + eff_H
        start_w = (W - eff_W) // 2
        end_w = start_w + eff_W

        mask[:, start_d:end_d, start_h:end_h, start_w:end_w] = 1.0
        print(f"Spatial mask ready: allowed voxels {mask.sum().item():.0f}/{mask.numel()} ({mask.mean().item()*100:.1f}%)")
        return mask
    
    def apply_gumbel_softmax(self, logits):

        binary_logits = torch.stack([torch.zeros_like(logits), logits], dim=-1)

        binary_probs = F.gumbel_softmax(
            binary_logits, 
            tau=self.gumbel_temperature, 
            hard=self.hard_gumbel, 
            dim=-1
        )
        

        densities = binary_probs[..., 1]
        
        return densities
    
    def get_densities(self):

        logits = self.log_densities  # (1, D, H, W)

        densities_raw = torch.sigmoid(self.inner_temperature * logits - self.density_bias)
        densities_raw = torch.clamp(densities_raw * self.outer_scale * self.density_scale, 0.0, 1.0)


        densities_raw = densities_raw * self.spatial_mask

        densities = densities_raw.permute(0, 2, 3, 1).unsqueeze(1)

        return densities
    
    def get_colors(self):

        raise NotImplementedError("Use _colors_from_logits and the forward-side blending.")

    def _colors_from_logits(self, logits: torch.Tensor):
        """Convert logits into a (1,3,H,W,D) color tensor."""
        raw = torch.sigmoid(logits) * self.spatial_mask
        return raw.permute(0, 2, 3, 1).unsqueeze(0)
    
    def forward(self, cameras):
        """Render the current volume for the provided cameras."""
        batch_size = cameras.R.shape[0]
        

        densities = self.get_densities()

        # Two independent color volumes
        c1 = self._colors_from_logits(self.log_colors1)
        c2 = self._colors_from_logits(self.log_colors2)

        if batch_size > 1:
            densities = densities.expand(batch_size, -1, -1, -1, -1)
            c1 = c1.expand(batch_size, -1, -1, -1, -1)
            c2 = c2.expand(batch_size, -1, -1, -1, -1)


        centers = cameras.get_camera_center()               # (B,3)
        view_dir = torch.nn.functional.normalize(-centers, dim=-1)
        axis = self.axis_dir.to(view_dir.device)            # (+X)

        dot = torch.matmul(view_dir, axis)                  # (B,)
        w = (dot >= 0).float().view(batch_size, 1, 1, 1, 1)  # hard switch, no blending

        colors = w * c1 + (1 - w) * c2
        
        
        volumes = Volumes(
            densities=densities,
            features=colors,
            voxel_size=self._voxel_size,
        )
        

        return self._renderer(cameras=cameras, volumes=volumes)[0]
    
    def get_optimization_info(self):

        with torch.no_grad():

            densities_raw = torch.sigmoid(self.inner_temperature * self.log_densities - self.density_bias)
            densities_raw = torch.clamp(densities_raw * self.outer_scale, 0.0, 1.0)
            
            logits = self.log_densities
            
            return {
                "density_mean": densities_raw.mean().item(),
                "density_std": densities_raw.std().item(),
                "density_min": densities_raw.min().item(),
                "density_max": densities_raw.max().item(),
                "active_voxels": (densities_raw > 0.25).sum().item(),
                "total_voxels": densities_raw.numel(),
                "logits_mean": logits.mean().item(),
                "logits_std": logits.std().item(),
                "inner_temperature": self.inner_temperature,
                "outer_scale": self.outer_scale,
                "density_bias": self.density_bias
            }


def binary_iou_loss(pred_mask, target_mask, smooth=1e-6):


    pred_binary = (pred_mask > 0.5).float()
    target_binary = (target_mask > 0.5).float()

    intersection = (pred_binary * target_binary).sum(dim=(-2, -1))
    union = pred_binary.sum(dim=(-2, -1)) + target_binary.sum(dim=(-2, -1)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    

    return 1.0 - iou.mean()


def projection_silhouette_loss(volume_model, target_alpha_view1, target_alpha_view2, 
                              camera_view1, camera_view2, iou_weight=1.0):

    pred_rgba_view1 = volume_model(camera_view1)  # [1, H, W, 4]
    pred_rgba_view2 = volume_model(camera_view2)  # [1, H, W, 4]
    

    pred_alpha1 = pred_rgba_view1[..., 3]  # [1, H, W]
    pred_alpha2 = pred_rgba_view2[..., 3]  # [1, H, W]
    

    if target_alpha_view1.dim() == 2:
        target_alpha_view1 = target_alpha_view1.unsqueeze(0)  # [1, H, W]
    if target_alpha_view2.dim() == 2:
        target_alpha_view2 = target_alpha_view2.unsqueeze(0)  # [1, H, W]

    iou_loss1 = binary_iou_loss(pred_alpha1, target_alpha_view1)
    iou_loss2 = binary_iou_loss(pred_alpha2, target_alpha_view2)
    total_iou_loss = iou_loss1 + iou_loss2
    

    total_loss = iou_weight * total_iou_loss
    
    return total_loss, {
        'iou_loss': total_iou_loss,
        'view1_iou_loss': iou_loss1,
        'view2_iou_loss': iou_loss2,
    }


def create_dual_view_cameras(device='cuda', distance=1.5, fov=60.0,
                             azim1: float = 0.0, azim2: float = 180.0,
                             elev1: float = 0.0, elev2: float = 0.0,
                             orthographic: bool = False):

    from pytorch3d.renderer import FoVPerspectiveCameras, OrthographicCameras, look_at_view_transform
    

    R1, T1 = look_at_view_transform(
        dist=distance,
        elev=elev1,
        azim=azim1,
        device=device
    )
    
    if orthographic:
        camera_view1 = OrthographicCameras(
            R=R1, T=T1,
            device=device
        )
    else:
        camera_view1 = FoVPerspectiveCameras(
            R=R1, T=T1, fov=fov,
            znear=0.1, zfar=10.0,
            device=device
        )
    

    R2, T2 = look_at_view_transform(
        dist=distance,
        elev=elev2,
        azim=azim2,
        device=device
    )
    
    if orthographic:
        camera_view2 = OrthographicCameras(
            R=R2, T=T2,
            device=device
        )
    else:
        camera_view2 = FoVPerspectiveCameras(
            R=R2, T=T2, fov=fov,
            znear=0.1, zfar=10.0,
            device=device
        )
    
    return camera_view1, camera_view2


def create_voxel_optimizer(volume_model, lr=0.01, weight_decay=0.0):

    import torch.optim as optim
    
    # Optimize both density and color fields jointly
    optimizer = optim.Adam([
        volume_model.log_densities,
        volume_model.log_colors1,
        volume_model.log_colors2
    ], lr=lr, weight_decay=weight_decay)
    
    return optimizer


def apply_density_constraints(volume_model, constraint_strength=0.1):

    densities = volume_model.get_densities()
    
    sparsity_loss = densities.mean() * constraint_strength

    binary_loss = torch.mean(4 * densities * (1 - densities)) * constraint_strength
    
    return sparsity_loss + binary_loss 