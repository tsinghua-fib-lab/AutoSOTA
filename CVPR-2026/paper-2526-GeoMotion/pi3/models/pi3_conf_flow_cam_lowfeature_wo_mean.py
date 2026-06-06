import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
import os

import torch.nn.functional as F
from typing import Optional, Dict, Any

# import pdb;pdb.set_trace()
from .dinov2.layers import Mlp
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from ..utils.geometry import homogenize_points, depth_edge


class MotionAwareDecoder(nn.Module):
    """融合原始光流、相机位姿和几何特征用于运动分割
    
    核心思想：
    - low_hidden: 低层几何特征 [B*N, hw, 2*C] (第5层+第15层)
    - hidden: 高层特征 [B*N, hw, 2*C] (最后两层)
    - 先将4*C降维到2048，再和光流、相机特征融合
    """
    
    def __init__(self, hidden_dim=1024, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # ===== 特征降维层 =====
        # 将 low_hidden + hidden (4*C) 降维到 2048
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2048),
            nn.ReLU(inplace=True),
        )
        
        # ===== 光流编码器 - 卷积版本 =====
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # ===== 融合投影层 =====
        # feature_proj(2048) + flow(128) + camera(512) = 2688 -> 2048
        self.fusion_proj = nn.Linear(2048 + 128 + 512, 2048)
        
    def forward(self, low_hidden, hidden, optical_flow, camera_hidden, 
                patch_start_idx, H, W, B, N):
        """
        Args:
            low_hidden: [B*N, hw, 2*C] 第5层+第15层
            hidden: [B*N, hw, 2*C] 最后两层
            optical_flow: [B, N, H, W]
            camera_hidden: [B*N, hw, 512]
        
        Returns:
            enhanced_hidden: [B*N, hw, 2048]
        """
        # 1. 提取patch部分（跳过特殊token）
        low_hidden_patch = low_hidden[:, patch_start_idx:]  # [B*N, num_patches, 2*C]
        hidden_patch = hidden[:, patch_start_idx:]  # [B*N, num_patches, 2*C]
        camera_patch = camera_hidden[:, patch_start_idx:]  # [B*N, num_patches, 512]
        
        # 2. 拼接low和high特征，然后降维到2048
        combined_hidden = torch.cat([low_hidden_patch, hidden_patch], dim=-1)  # [B*N, num_patches, 4*C]
        projected_hidden = self.feature_proj(combined_hidden)  # [B*N, num_patches, 2048]
        
        # 3. 用卷积提取光流特征
        flow_feat = self.flow_encoder(
            optical_flow.view(B*N, 1, H, W)
        )  # [B*N, 128, H, W]
        
        # 4. 下采样到patch分辨率
        flow_feat_down = F.interpolate(
            flow_feat,
            size=(H // self.patch_size, W // self.patch_size),
            mode='bilinear',
            align_corners=False
        )  # [B*N, 128, patch_h, patch_w]
        
        num_patches = flow_feat_down.shape[2] * flow_feat_down.shape[3]
        flow_patches = flow_feat_down.permute(0, 2, 3, 1).reshape(B*N, num_patches, 128)
        
        # 5. 融合所有特征（camera直接用512维）
        combined = torch.cat([
            projected_hidden,  # [B*N, num_patches, 2048]
            flow_patches,      # [B*N, num_patches, 128]
            camera_patch       # [B*N, num_patches, 512]
        ], dim=-1)  # [B*N, num_patches, 2688]
        
        # 6. 融合投影到2048
        fused_patch = self.fusion_proj(combined)  # [B*N, num_patches, 2048]
        
        # 7. 保留特殊token（从hidden中取，也需要降维）
        special_tokens = torch.cat([
            low_hidden[:, :patch_start_idx],
            hidden[:, :patch_start_idx]
        ], dim=-1)  # [B*N, patch_start_idx, 4*C]
        special_tokens_proj = self.feature_proj(special_tokens)  # [B*N, patch_start_idx, 2048]
        
        enhanced_hidden = torch.cat([
            special_tokens_proj,  # [B*N, patch_start_idx, 2048]
            fused_patch           # [B*N, num_patches, 2048]
        ], dim=1)  # [B*N, hw, 2048]
        
        return enhanced_hidden

class Pi3MotionSeg(nn.Module, PyTorchModelHubMixin):
    """
    Pi3 model adapted for motion segmentation with pose and point cloud motion cues
    """
    def __init__(
        self,
        pos_type='rope100',
        decoder_size='large',
        pi3_model_path: Optional[str] = None,
        freeze_backbone: bool = True,
        motion_feature_dim: int = 64,
    ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):  # eg rope100 
            if RoPE2D is None: 
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
            
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        self.motion_aware_decoder = MotionAwareDecoder(
            hidden_dim=2*self.dec_embed_dim,  # 2048 for large
            patch_size=self.patch_size
        )

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # ----------------------
        #   Motion Feature Extractor
        # ----------------------
        # self.motion_decoder = deepcopy(self.point_decoder)
        # self.motion_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        
        # Load pretrained Pi3 model if provided
        if pi3_model_path:
            self._load_pretrained_pi3(pi3_model_path)
        
        # Freeze backbone if specified
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _load_pretrained_pi3(self, model_path: str):
        """Load pretrained Pi3 model weights"""
        from safetensors.torch import load_file
        state_dict = load_file(model_path)

        filtered_state_dict = {k: v for k, v in state_dict.items() 
                                if not k.startswith('mask_') and not k.startswith('motion_')}
        self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pretrained Pi3 model from {model_path}")

    def _freeze_backbone(self):
        """Freeze the backbone components"""
        components_to_freeze = [
            'encoder', 'decoder', 'register_token', 'position_getter',
            'point_decoder', 'point_head', 'camera_decoder', 'camera_head'
        ]
        
        for name, param in self.named_parameters():
            for component in components_to_freeze:
                if name.startswith(component):
                    param.requires_grad = False
                    break

    def decode(self, hidden, N, H, W, imgs):
        """Decoder function adapted from Pi3"""
        BN, hw, _ = hidden.shape
        B = BN // N

        # 用于保存最后两层的输出
        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]  # 包含register tokens

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        # 选择第5层和第15层
        selected_layers = [5]
        selected_layers_2 = [15]
        selected_feats = []
        selected_feats_2 = []

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            # 收集第5层的输出
            if i in selected_layers:
                if i % 2 == 1:  # 奇数层
                    hidden_temp = hidden.reshape(B, N, hw, -1)
                    feat = hidden_temp.reshape(B*N, hw, -1)
                else:  # 偶数层
                    feat = hidden
                selected_feats.append(feat)
            
            # 收集第15层的输出
            if i in selected_layers_2:
                if i % 2 == 1:  # 奇数层
                    hidden_temp = hidden.reshape(B, N, hw, -1)
                    feat = hidden_temp.reshape(B*N, hw, -1)
                else:  # 偶数层
                    feat = hidden
                selected_feats_2.append(feat)
            
            # 收集最后两层的输出
            if i >= len(self.decoder) - 2:
                final_output.append(hidden.reshape(B*N, hw, -1))

        # 获取第5层和第15层的特征
        fused_feat_5 = torch.stack(selected_feats, dim=0).mean(dim=0) if len(selected_feats) > 0 else None
        fused_feat_15 = torch.stack(selected_feats_2, dim=0).mean(dim=0) if len(selected_feats_2) > 0 else None
        
        # 将第5层和第15层的特征concatenate
        if fused_feat_5 is not None and fused_feat_15 is not None:
            cat_output = torch.cat([fused_feat_5, fused_feat_15], dim=-1)  # [B*N, hw, 2*C]
        else:
            # 如果某层不存在，fallback到最后一层
            cat_output = final_output[-1]
        
        # 最后两层的输出
        last_two_layers = torch.cat(final_output, dim=-1)  # [B*N, hw, 2*C] - 拼接倒数第2层和倒数第1层
        
        return cat_output, last_two_layers, pos.reshape(B*N, hw, -1)

    def forward(self, imgs, flows=None):
        """
        Forward pass for motion segmentation
        
        Args:
            imgs: [B, N, 3, H, W] or [N, 3, H, W]
            flows: [B, N, H, W] 原始光流幅度（不归一化）
            
        Returns:
            Dict with motion_mask, points, conf, camera_poses
        """
        if len(imgs.shape) == 4:
            imgs = imgs.unsqueeze(0)
            if flows is not None and len(flows.shape) == 3:
                flows = flows.unsqueeze(0)
                
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        imgs_reshaped = imgs.reshape(B*N, _, H, W)
        
        if self.freeze_backbone:
            with torch.no_grad():
                hidden = self.encoder(imgs_reshaped, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]
                low_hidden, hidden, pos = self.decode(hidden, N, H, W,imgs)
        else:
            hidden = self.encoder(imgs_reshaped, is_training=True)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]
            low_hidden, hidden, pos = self.decode(hidden, N, H, W)

        # import pdb; pdb.set_trace()
        # 获取camera_hidden用于光流-相机融合
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        # ===== 核心：融合光流和相机信息到hidden中 =====
        if flows is not None and hasattr(self, 'motion_aware_decoder'):
            hidden_with_motion = self.motion_aware_decoder(
                low_hidden=low_hidden,
                hidden=hidden,
                optical_flow=flows,
                camera_hidden=camera_hidden,
                patch_start_idx=self.patch_start_idx,
                H=H, W=W,
                B=B, N=N
            )
        else:
            hidden_with_motion = hidden
        # import pdb; pdb.set_trace()
        # 用融合后的hidden传入decoders
        # visualize_imgs_and_hidden(imgs=imgs, hidden=hidden_with_motion[:, self.patch_start_idx:],layer_id=9, save_dir="debug/debug_vis_motorbike_8", num_frames=15, title_prefix="Encoder Output")
        # import pdb; pdb.set_trace()
        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden_with_motion, xpos=pos)
        # visualize_imgs_and_hidden(imgs=imgs, hidden=camera_hidden[:, self.patch_start_idx:],layer_id=9, save_dir="./debug_vis_cam", num_frames=15, title_prefix="Encoder Output")

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

            motion_mask = torch.sigmoid(conf).squeeze(-1)

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            motion_mask=motion_mask,
        )
    
def visualize_flows(flows, imgs=None, save_dir="./debug_flows"):
    """
    可视化 flow 序列（支持单独 flow 或与原图对比）

    Args:
        flows (torch.Tensor): [B, N, H, W] 形状的张量（单通道 flow magnitude）
        imgs (torch.Tensor, optional): [B, N, 3, H, W] 原始图像张量（可选）
        save_dir (str): 保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    flows = flows.detach().cpu()
    B, N, H, W = flows.shape

    has_imgs = imgs is not None
    if has_imgs:
        imgs = imgs.detach().cpu()

    for i in range(N):
        plt.figure(figsize=(10, 5) if has_imgs else (6, 6))

        if has_imgs:
            img_i = imgs[0, i].permute(1, 2, 0).numpy()
            flow_i = flows[0, i].numpy()

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img_i)
            ax[0].set_title(f"Image {i}")
            ax[0].axis("off")

            im = ax[1].imshow(flow_i, cmap="turbo")
            ax[1].set_title(f"Flow {i}")
            ax[1].axis("off")

            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            save_path = os.path.join(save_dir, f"flow_compare_{i:02d}.png")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        else:
            flow_i = flows[0, i].numpy()
            plt.imshow(flow_i, cmap="turbo")
            plt.colorbar()
            plt.title(f"Flow Frame {i}")
            plt.axis("off")

            save_path = os.path.join(save_dir, f"flow_{i:02d}.png")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()

    print(f"✅ Saved {N} flow visualizations to {save_dir}/")

def visualize_imgs_and_hidden(
    imgs: torch.Tensor,
    hidden: torch.Tensor,
    layer_id: int,
    save_dir: str = "./debug_vis",
    num_frames: int = 5,
    title_prefix: str = "",
    cmap: str = "viridis"
) -> None:
    """
    可视化输入图像和对应的hidden特征
    
    Args:
        imgs: 输入图像张量，shape [batch, num_frames, 3, H, W] 或 [num_frames, 3, H, W]
        hidden: 特征张量，shape [batch*num_frames, num_patches, feat_dim] 或 [num_frames, num_patches, feat_dim]
        layer_id: 层号，用于文件名和标题
        save_dir: 保存目录
        num_frames: 要可视化的帧数
        title_prefix: 标题前缀（如"Encoder"、"Layer 15"等）
        cmap: 热力图颜色映射
    
    Returns:
        None (直接保存文件)
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ===== 数据预处理 =====
    imgs_vis = imgs.detach().cpu()
    hidden_vis = hidden.detach().cpu()
    
    # 处理imgs维度
    if len(imgs_vis.shape) == 5:  # [B, N, 3, H, W]
        imgs_vis = imgs_vis[0]  # 取第一个batch [N, 3, H, W]
    elif len(imgs_vis.shape) != 4:  # [N, 3, H, W]
        raise ValueError(f"imgs shape应该是[B, N, 3, H, W]或[N, 3, H, W]，但得到{imgs_vis.shape}")
    
    N = imgs_vis.shape[0]
    H_img, W_img = imgs_vis.shape[2], imgs_vis.shape[3]
    
    # 处理hidden维度
    if len(hidden_vis.shape) == 3:  # [B*N, hw, C]
        num_patches, feat_dim = hidden_vis.shape[1], hidden_vis.shape[2]
    else:
        raise ValueError(f"hidden shape应该是[B*N, hw, C]，但得到{hidden_vis.shape}")
    
    # ===== 特征统计 =====
    feat_norm = hidden_vis.norm(dim=-1)  # [B*N, hw]
    feat_mean = hidden_vis.mean(dim=-1)  # [B*N, hw]
    feat_var = hidden_vis.var(dim=-1)    # [B*N, hw]
    feat_max = hidden_vis.max(dim=-1)[0]  # [B*N, hw]
    
    print(f"\n{'='*60}")
    print(f"[VIS] {title_prefix} Layer {layer_id}")
    print(f"{'='*60}")
    print(f"Images shape:   {imgs_vis.shape}")
    print(f"Hidden shape:   {hidden_vis.shape} (frames={N}, patches={num_patches}, dims={feat_dim})")
    print(f"Feature Norm    - min: {feat_norm.min():.4f}, max: {feat_norm.max():.4f}, mean: {feat_norm.mean():.4f}")
    print(f"Feature Mean    - min: {feat_mean.min():.4f}, max: {feat_mean.max():.4f}, mean: {feat_mean.mean():.4f}")
    print(f"Feature Var     - min: {feat_var.min():.4f}, max: {feat_var.max():.4f}, mean: {feat_var.mean():.4f}")
    print(f"Feature Max     - min: {feat_max.min():.4f}, max: {feat_max.max():.4f}, mean: {feat_max.mean():.4f}")
    
    # ===== 推断空间维度 =====
    side_len = int(np.sqrt(num_patches))
    if side_len * side_len != num_patches:
        print(f"[WARNING] num_patches={num_patches} 不是完全平方数，使用近似 side_len={side_len}")
    
    # ===== 逐帧可视化 =====
    num_vis_frames = min(num_frames, N)
    
    for n in range(num_vis_frames):
        frame_hidden = hidden_vis[n]  # [hw, C]
        frame_img = imgs_vis[n]  # [3, H, W]
        
        # 转换图像格式
        img_np = frame_img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # ===== 特征热力图处理 =====
        
        # 1. Norm热力图
        feat_norm_frame = frame_hidden.norm(dim=-1)  # [hw]
        heatmap_norm = feat_norm_frame.view(side_len, side_len).numpy()
        heatmap_norm_tensor = torch.from_numpy(heatmap_norm).unsqueeze(0).unsqueeze(0).float()
        heatmap_norm_up = F.interpolate(
            heatmap_norm_tensor, 
            size=(H_img, W_img), 
            mode='bilinear', 
            align_corners=False
        ).squeeze().numpy()
        
        # 2. Mean热力图
        feat_mean_frame = frame_hidden.mean(dim=-1)  # [hw]
        heatmap_mean = feat_mean_frame.view(side_len, side_len).numpy()
        heatmap_mean_tensor = torch.from_numpy(heatmap_mean).unsqueeze(0).unsqueeze(0).float()
        heatmap_mean_up = F.interpolate(
            heatmap_mean_tensor, 
            size=(H_img, W_img), 
            mode='bilinear', 
            align_corners=False
        ).squeeze().numpy()
        
        # 3. Max热力图
        feat_max_frame = frame_hidden.max(dim=-1)[0]  # [hw]
        heatmap_max = feat_max_frame.view(side_len, side_len).numpy()
        heatmap_max_tensor = torch.from_numpy(heatmap_max).unsqueeze(0).unsqueeze(0).float()
        heatmap_max_up = F.interpolate(
            heatmap_max_tensor, 
            size=(H_img, W_img), 
            mode='bilinear', 
            align_corners=False
        ).squeeze().numpy()
        
        # 4. RGB特征可视化（前3个通道）
        if feat_dim >= 3:
            feat_rgb = frame_hidden[:, :3].view(side_len, side_len, 3).numpy()
            feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min() + 1e-8)
            feat_rgb_tensor = torch.from_numpy(feat_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
            feat_rgb_up = F.interpolate(
                feat_rgb_tensor, 
                size=(H_img, W_img), 
                mode='bilinear', 
                align_corners=False
            ).squeeze().permute(1, 2, 0).numpy()
        else:
            feat_rgb_up = None
        
        # ===== 绘制 =====
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f"{title_prefix} - Layer {layer_id} - Frame {n}", fontsize=16, fontweight='bold')
        
        # 原始图像
        axs[0, 0].imshow(img_np)
        axs[0, 0].set_title("Input Image", fontsize=12, fontweight='bold')
        axs[0, 0].axis('off')
        
        # Norm热力图
        im1 = axs[0, 1].imshow(heatmap_norm_up, cmap=cmap)
        axs[0, 1].set_title(f"Feature Norm\n(min:{feat_norm[n].min():.2f}, max:{feat_norm[n].max():.2f})", fontsize=12, fontweight='bold')
        axs[0, 1].axis('off')
        plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
        
        # Mean热力图
        im2 = axs[0, 2].imshow(heatmap_mean_up, cmap='coolwarm')
        axs[0, 2].set_title(f"Feature Mean\n(min:{feat_mean[n].min():.2f}, max:{feat_mean[n].max():.2f})", fontsize=12, fontweight='bold')
        axs[0, 2].axis('off')
        plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)
        
        # Max热力图
        im3 = axs[1, 0].imshow(heatmap_max_up, cmap='plasma')
        axs[1, 0].set_title(f"Feature Max\n(min:{feat_max[n].min():.2f}, max:{feat_max[n].max():.2f})", fontsize=12, fontweight='bold')
        axs[1, 0].axis('off')
        plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)
        
        # RGB特征
        if feat_rgb_up is not None:
            axs[1, 1].imshow(feat_rgb_up, clim=(0, 1))
            axs[1, 1].set_title("Feature RGB\n(Ch 0, 1, 2)", fontsize=12, fontweight='bold')
        else:
            axs[1, 1].text(0.5, 0.5, "Feat Dim < 3", ha='center', va='center')
            axs[1, 1].set_title("Feature RGB", fontsize=12, fontweight='bold')
        axs[1, 1].axis('off')
        
        # 特征统计文本
        stats_text = (
            f"Frame: {n}/{N}\n"
            f"Patches: {num_patches} ({side_len}x{side_len})\n"
            f"Feat Dim: {feat_dim}\n\n"
            f"Norm   - μ:{feat_norm[n].mean():.3f}\n"
            f"Mean   - μ:{feat_mean[n].mean():.3f}\n"
            f"Var    - μ:{feat_var[n].mean():.3f}\n"
            f"Max    - μ:{feat_max[n].mean():.3f}"
        )
        axs[1, 2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center')
        axs[1, 2].axis('off')
        
        # 保存
        save_path = os.path.join(save_dir, f"layer{layer_id:02d}_frame{n}.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[VIS] Saved {save_path}")
    
    print(f"{'='*60}\n")

class CombinedLoss(nn.Module):
    """
    Combined loss function for motion segmentation with motion consistency
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, 
                 focal_weight: float = 0.0, motion_consistency_weight: float = 0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.motion_consistency_weight = motion_consistency_weight
        self.bce_loss = nn.BCELoss(reduction='mean')
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
        """Dice loss for handling class imbalance"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
        """Focal loss for handling hard examples"""
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def motion_consistency_loss(self, pred_mask: torch.Tensor, world_points: torch.Tensor, 
                              world_points_conf: torch.Tensor, camera_poses: torch.Tensor):
        """
        Encourage consistency between predicted mask and actual 3D motion
        """
        B, N, H, W = pred_mask.shape
        
        if N <= 1:
            return torch.tensor(0.0, device=pred_mask.device)
        
        # Compute 3D point motion magnitude
        point_motion = torch.zeros_like(world_points[..., 0])
        point_motion[:, 1:] = torch.norm(
            world_points[:, 1:] - world_points[:, :-1], dim=-1
        )
        
        # Compute camera motion magnitude (broadcast to image size)
        camera_motion = torch.zeros(B, N, device=pred_mask.device)
        camera_motion[:, 1:] = torch.norm(
            camera_poses[:, 1:, :3, 3] - camera_poses[:, :-1, :3, 3], dim=-1
        )
        camera_motion = camera_motion.view(B, N, 1, 1).expand(-1, -1, H, W)
        
        # Object motion should be high when point motion is high but camera motion is low
        # and vice versa: when camera motion is high, object motion should be relatively lower
        expected_object_motion = point_motion / (camera_motion + 1e-6)
        expected_object_motion = torch.clamp(expected_object_motion, 0, 1)
        
        # Mask with confidence
        valid_mask = world_points_conf > 0.5
        expected_object_motion = expected_object_motion * valid_mask
        
        # L2 loss between predicted mask and expected motion
        consistency_loss = F.mse_loss(
            pred_mask * valid_mask,
            expected_object_motion,
            reduction='mean'
        )
        
        return consistency_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                world_points: Optional[torch.Tensor] = None,
                world_points_conf: Optional[torch.Tensor] = None,
                camera_poses: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        """
        Args:
            pred: Predicted motion masks [B, N, H, W]
            target: Ground truth motion masks [B, N, H, W]
            world_points: [B, N, H, W, 3] for motion consistency
            world_points_conf: [B, N, H, W] confidence
            camera_poses: [B, N, 4, 4] for motion consistency
            mask: Valid pixel mask [B, N, H, W] (optional)
        """
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        total_loss = 0
        loss_dict = {}
        
        if self.bce_weight > 0:
            bce = self.bce_loss(pred, target)
            total_loss += self.bce_weight * bce
            loss_dict['bce_loss'] = bce.item()
        
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice
            loss_dict['dice_loss'] = dice.item()
        
        if self.focal_weight > 0:
            focal = self.focal_loss(pred, target)
            total_loss += self.focal_weight * focal
            loss_dict['focal_loss'] = focal.item()
        
        # Motion consistency loss (unsupervised)
        if (self.motion_consistency_weight > 0 and 
            world_points is not None and world_points_conf is not None and camera_poses is not None):
            motion_consistency = self.motion_consistency_loss(
                pred, world_points, world_points_conf, camera_poses
            )
            total_loss += self.motion_consistency_weight * motion_consistency
            loss_dict['motion_consistency_loss'] = motion_consistency.item()
        
        return total_loss, loss_dict


# Factory function
def create_pi3_motion_segmentation_model(
    pi3_model_path: Optional[str] = None,
    pos_type: str = 'rope100',
    decoder_size: str = 'large',
    freeze_backbone: bool = True,
    motion_feature_dim: int = 64
):
    """
    Factory function to create Pi3-based motion segmentation model with motion cues
    """
    model = Pi3MotionSeg(
        pos_type=pos_type,
        decoder_size=decoder_size,
        pi3_model_path=pi3_model_path,
        freeze_backbone=freeze_backbone,
        motion_feature_dim=motion_feature_dim
    )
    
    return model


# Evaluation metrics
def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Compute IoU for motion segmentation"""
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    iou = intersection / (union + 1e-6)
    return iou.item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Compute comprehensive metrics for motion segmentation"""
    pred_binary = (pred > threshold).float()
    
    # IoU
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    # Precision, Recall, F1
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def analyze_motion_consistency(outputs: Dict[str, torch.Tensor]):
    """
    Analyze the consistency between predicted masks and 3D motion cues
    """
    motion_mask = outputs['motion_mask']  # [B, N, H, W]
    world_points = outputs['world_points']  # [B, N, H, W, 3]
    world_points_conf = outputs['world_points_conf']  # [B, N, H, W]
    camera_poses = outputs['camera_poses']  # [B, N, 4, 4]
    
    B, N, H, W = motion_mask.shape
    
    if N <= 1:
        return {}
    
    # Compute 3D point motion
    point_motion = torch.zeros_like(world_points[..., 0])
    point_motion[:, 1:] = torch.norm(
        world_points[:, 1:] - world_points[:, :-1], dim=-1
    )
    
    # Compute camera motion
    camera_motion = torch.zeros(B, N)
    camera_motion[:, 1:] = torch.norm(
        camera_poses[:, 1:, :3, 3] - camera_poses[:, :-1, :3, 3], dim=-1
    )
    
    # Valid regions
    valid_mask = world_points_conf > 0.5
    
    # Correlation between predicted mask and point motion
    mask_flat = motion_mask[valid_mask]
    point_motion_flat = point_motion[valid_mask]
    
    if len(mask_flat) > 0:
        correlation = torch.corrcoef(torch.stack([mask_flat, point_motion_flat]))[0, 1]
    else:
        correlation = torch.tensor(0.0)
    
    return {
        'mask_point_motion_correlation': correlation.item(),
        'avg_camera_motion': camera_motion.mean().item(),
        'avg_point_motion': point_motion[valid_mask].mean().item() if valid_mask.sum() > 0 else 0.0,
        'avg_predicted_motion': motion_mask.mean().item(),
    }


# Example usage and training script template
def example_training_step(model, images, ground_truth_masks, optimizer, loss_fn):
    """
    Example training step showing how to use the enhanced model
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(images)
    motion_mask = outputs['motion_mask']
    
    # Compute loss with motion consistency
    loss, loss_dict = loss_fn(
        pred=motion_mask,
        target=ground_truth_masks,
        world_points=outputs.get('world_points'),
        world_points_conf=outputs.get('world_points_conf'),
        camera_poses=outputs.get('camera_poses')
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Analysis
    consistency_metrics = analyze_motion_consistency(outputs)
    
    return {
        **loss_dict,
        **consistency_metrics,
        'total_loss': loss.item()
    }

