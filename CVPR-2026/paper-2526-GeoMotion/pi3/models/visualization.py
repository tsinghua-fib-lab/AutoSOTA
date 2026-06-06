import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
import os
from typing import Optional, Tuple, List
import seaborn as sns

class HiddenFeatureVisualizer:
    """Hidden特征可视化工具"""
    
    def __init__(self, patch_size: int = 14, patch_start_idx: int = 0):
        self.patch_size = patch_size
        self.patch_start_idx = patch_start_idx
        
    def extract_patch_features(self, hidden: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        从hidden中提取patch特征
        Args:
            hidden: [BN, num_patches, hidden_dim] 
            H, W: 原图尺寸
        Returns:
            patch_features: [BN, patch_h, patch_w, hidden_dim]
        """
        BN, num_patches, hidden_dim = hidden.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # 去除register tokens，只保留patch tokens
        patch_tokens = hidden[:, self.patch_start_idx:]  # [BN, patch_h*patch_w, hidden_dim]
        
        # 重塑为spatial grid
        patch_features = patch_tokens.reshape(BN, patch_h, patch_w, hidden_dim)
        
        return patch_features
    
    def visualize_feature_pca(self, 
                             hidden: torch.Tensor, 
                             video_tensor: torch.Tensor,
                             save_dir: str = "hidden_vis",
                             num_components: int = 3,
                             frame_indices: Optional[List[int]] = None) -> None:
        """
        使用PCA可视化hidden特征
        Args:
            hidden: [BN, num_patches, hidden_dim]
            video_tensor: [B, N, 3, H, W] 原始视频
            save_dir: 保存目录
            num_components: PCA组件数量
            frame_indices: 要可视化的帧索引，None表示可视化所有帧
        """
        os.makedirs(save_dir, exist_ok=True)
        
        B, N, _, H, W = video_tensor.shape
        BN = hidden.shape[0]
        assert BN == B * N, f"BN {BN} != B*N {B*N}"
        
        # 提取patch特征
        patch_features = self.extract_patch_features(hidden, H, W)  # [BN, patch_h, patch_w, hidden_dim]
        
        # PCA降维
        features_flat = patch_features.reshape(-1, patch_features.shape[-1])  # [BN*patch_h*patch_w, hidden_dim]
        
        # 随机采样以节省内存
        if features_flat.shape[0] > 100000:
            indices = torch.randperm(features_flat.shape[0])[:100000]
            sample_features = features_flat[indices].cpu().numpy()
        else:
            sample_features = features_flat.cpu().numpy()
            
        pca = PCA(n_components=num_components)
        pca_features = pca.fit_transform(sample_features)
        
        # 将PCA应用到所有特征
        all_pca_features = pca.transform(features_flat.cpu().numpy())
        all_pca_features = all_pca_features.reshape(BN, patch_features.shape[1], patch_features.shape[2], num_components)
        
        # 归一化到[0, 1]
        for i in range(num_components):
            comp = all_pca_features[..., i]
            comp_min, comp_max = comp.min(), comp.max()
            all_pca_features[..., i] = (comp - comp_min) / (comp_max - comp_min + 1e-8)
        
        # 选择要可视化的帧
        if frame_indices is None:
            frame_indices = list(range(min(N, 8)))  # 默认显示前8帧
        
        # 可视化每个batch的每一帧
        for b in range(B):
            for frame_idx in frame_indices:
                if frame_idx >= N:
                    continue
                    
                bn_idx = b * N + frame_idx
                
                # 原始图像
                orig_img = video_tensor[b, frame_idx].permute(1, 2, 0).cpu().numpy()
                orig_img = (orig_img * 0.229 + 0.485).clip(0, 1)  # 反标准化
                
                # PCA特征图
                pca_feat = all_pca_features[bn_idx]  # [patch_h, patch_w, num_components]
                
                # 上采样到原图尺寸
                pca_feat_upsampled = F.interpolate(
                    torch.from_numpy(pca_feat).permute(2, 0, 1).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0).numpy()
                
                # 创建可视化
                fig, axes = plt.subplots(2, num_components + 1, figsize=(4*(num_components+1), 8))
                
                # 第一行：原图和各个PCA组件
                axes[0, 0].imshow(orig_img)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                
                for i in range(num_components):
                    axes[0, i+1].imshow(pca_feat_upsampled[..., i], cmap='viridis')
                    axes[0, i+1].set_title(f'PCA Component {i+1}')
                    axes[0, i+1].axis('off')
                
                # 第二行：RGB合成和特征统计
                if num_components >= 3:
                    rgb_feat = pca_feat_upsampled[..., :3]
                    axes[1, 0].imshow(rgb_feat)
                    axes[1, 0].set_title('PCA RGB Composite')
                    axes[1, 0].axis('off')
                    
                    # 特征分布统计
                    for i in range(min(num_components, 3)):
                        axes[1, i+1].hist(pca_feat_upsampled[..., i].flatten(), bins=50, alpha=0.7)
                        axes[1, i+1].set_title(f'Component {i+1} Distribution')
                        axes[1, i+1].set_xlabel('Feature Value')
                        axes[1, i+1].set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'pca_vis_batch{b}_frame{frame_idx}.png'), dpi=150, bbox_inches='tight')
                plt.close()
                
        print(f"PCA可视化保存到: {save_dir}")
        print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
    
    def visualize_attention_maps(self, 
                                hidden: torch.Tensor,
                                video_tensor: torch.Tensor,
                                save_dir: str = "attention_vis",
                                num_heads: int = 8) -> None:
        """
        可视化注意力图（通过特征相似性）
        Args:
            hidden: [BN, num_patches, hidden_dim]
            video_tensor: [B, N, 3, H, W]
            save_dir: 保存目录
            num_heads: 分割头数
        """
        os.makedirs(save_dir, exist_ok=True)
        
        B, N, _, H, W = video_tensor.shape
        BN = hidden.shape[0]
        
        # 提取patch特征
        patch_features = self.extract_patch_features(hidden, H, W)  # [BN, patch_h, patch_w, hidden_dim]
        patch_h, patch_w, hidden_dim = patch_features.shape[1:]
        
        # 将特征分割为多个头
        head_dim = hidden_dim // num_heads
        patch_features = patch_features.reshape(BN, patch_h, patch_w, num_heads, head_dim)
        
        for b in range(min(B, 2)):  # 只可视化前2个batch
            for frame_idx in range(min(N, 4)):  # 只可视化前4帧
                bn_idx = b * N + frame_idx
                
                # 原始图像
                orig_img = video_tensor[b, frame_idx].permute(1, 2, 0).cpu().numpy()
                orig_img = (orig_img * 0.229 + 0.485).clip(0, 1)
                
                fig, axes = plt.subplots(2, num_heads//2, figsize=(16, 8))
                axes = axes.flatten()
                
                for head in range(num_heads):
                    head_features = patch_features[bn_idx, :, :, head, :]  # [patch_h, patch_w, head_dim]
                    
                    # 计算中心patch与所有patch的相似性
                    center_h, center_w = patch_h // 2, patch_w // 2
                    center_feat = head_features[center_h, center_w]  # [head_dim]
                    
                    # 计算相似性
                    similarity = F.cosine_similarity(
                        head_features.reshape(-1, head_dim),
                        center_feat.unsqueeze(0),
                        dim=1
                    ).reshape(patch_h, patch_w)
                    
                    # 上采样到原图尺寸
                    similarity_upsampled = F.interpolate(
                        similarity.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()
                    
                    # 创建attention overlay
                    overlay = plt.cm.jet(similarity_upsampled)[:, :, :3]
                    blended = 0.6 * orig_img + 0.4 * overlay
                    
                    axes[head].imshow(blended)
                    axes[head].set_title(f'Head {head+1} Attention')
                    axes[head].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'attention_batch{b}_frame{frame_idx}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
        print(f"注意力图保存到: {save_dir}")
    
    def visualize_feature_similarity(self,
                                   hidden: torch.Tensor,
                                   video_tensor: torch.Tensor,
                                   save_dir: str = "similarity_vis") -> None:
        """
        可视化特征相似性
        Args:
            hidden: [BN, num_patches, hidden_dim]
            video_tensor: [B, N, 3, H, W]
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        B, N, _, H, W = video_tensor.shape
        BN = hidden.shape[0]
        
        # 提取patch特征
        patch_features = self.extract_patch_features(hidden, H, W)  # [BN, patch_h, patch_w, hidden_dim]
        patch_h, patch_w = patch_features.shape[1:3]
        
        # 计算相似性矩阵
        for b in range(min(B, 1)):  # 只可视化第一个batch
            batch_features = patch_features[b*N:(b+1)*N]  # [N, patch_h, patch_w, hidden_dim]
            
            # 重塑为[N, num_patches, hidden_dim]
            batch_features_flat = batch_features.reshape(N, -1, patch_features.shape[-1])
            
            # 计算帧间相似性
            similarity_matrix = torch.zeros(N, N, patch_h * patch_w)
            
            for i in range(N):
                for j in range(N):
                    sim = F.cosine_similarity(
                        batch_features_flat[i],  # [num_patches, hidden_dim]
                        batch_features_flat[j],  # [num_patches, hidden_dim]
                        dim=1
                    )  # [num_patches]
                    similarity_matrix[i, j] = sim
            
            # 可视化平均相似性
            avg_similarity = similarity_matrix.mean(dim=2).numpy()  # [N, N]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_similarity, annot=True, cmap='viridis', 
                       xticklabels=[f'Frame {i}' for i in range(N)],
                       yticklabels=[f'Frame {i}' for i in range(N)])
            plt.title(f'Frame-wise Feature Similarity (Batch {b})')
            plt.xlabel('Target Frame')
            plt.ylabel('Source Frame')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'similarity_matrix_batch{b}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 可视化每个patch位置的时序相似性
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # 选择几个有代表性的patch位置
            positions = [
                (patch_h//4, patch_w//4),      # 左上
                (patch_h//4, 3*patch_w//4),    # 右上
                (patch_h//2, patch_w//2),      # 中心
                (3*patch_h//4, patch_w//4),    # 左下
                (3*patch_h//4, 3*patch_w//4),  # 右下
                (patch_h//2, patch_w//4),      # 左中
            ]
            
            for idx, (ph, pw) in enumerate(positions):
                patch_idx = ph * patch_w + pw
                patch_similarity = similarity_matrix[:, :, patch_idx].numpy()
                
                im = axes[idx].imshow(patch_similarity, cmap='viridis', vmin=0, vmax=1)
                axes[idx].set_title(f'Patch ({ph}, {pw}) Similarity')
                axes[idx].set_xlabel('Frame')
                axes[idx].set_ylabel('Frame')
                plt.colorbar(im, ax=axes[idx])
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'patch_similarity_batch{b}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
                
        print(f"相似性可视化保存到: {save_dir}")
    
    def visualize_tsne(self,
                      hidden: torch.Tensor,
                      video_tensor: torch.Tensor,
                      save_dir: str = "tsne_vis",
                      perplexity: int = 30) -> None:
        """
        使用t-SNE可视化特征聚类
        Args:
            hidden: [BN, num_patches, hidden_dim]
            video_tensor: [B, N, 3, H, W]
            save_dir: 保存目录
            perplexity: t-SNE perplexity参数
        """
        os.makedirs(save_dir, exist_ok=True)
        
        B, N, _, H, W = video_tensor.shape
        
        # 提取patch特征
        patch_features = self.extract_patch_features(hidden, H, W)
        features_flat = patch_features.reshape(-1, patch_features.shape[-1]).cpu().numpy()
        
        # 随机采样以节省计算时间
        if features_flat.shape[0] > 10000:
            indices = np.random.choice(features_flat.shape[0], 10000, replace=False)
            sample_features = features_flat[indices]
            
            # 创建标签（batch和frame信息）
            batch_labels = []
            frame_labels = []
            for idx in indices:
                bn_idx = idx // (patch_features.shape[1] * patch_features.shape[2])
                b = bn_idx // N
                n = bn_idx % N
                batch_labels.append(b)
                frame_labels.append(n)
        else:
            sample_features = features_flat
            batch_labels = []
            frame_labels = []
            for bn in range(hidden.shape[0]):
                b = bn // N
                n = bn % N
                for _ in range(patch_features.shape[1] * patch_features.shape[2]):
                    batch_labels.append(b)
                    frame_labels.append(n)
        
        # 计算t-SNE
        print("计算t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_features = tsne.fit_transform(sample_features)
        
        # 按batch着色
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                            c=batch_labels, cmap='tab10', alpha=0.6, s=1)
        plt.title('t-SNE: Colored by Batch')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)
        
        # 按frame着色
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                            c=frame_labels, cmap='viridis', alpha=0.6, s=1)
        plt.title('t-SNE: Colored by Frame')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tsne_clustering.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE可视化保存到: {save_dir}")


# 使用示例
def visualize_hidden_features(hidden, video_tensor, motion_prior_weights=None):
    """
    完整的hidden特征可视化流程
    Args:
        hidden
        video_tensor: [B, N, 3, H, W] 视频tensor
        motion_prior_weights: [B, N, H, W] 运动先验（可选）
    """
    visualizer = HiddenFeatureVisualizer()
    
    # 获取hidden特征
    # hidden, pos, (B, N, H, W) = model.forward_backbone(video_tensor)
    
    print(f"Hidden shape: {hidden.shape}")
    print(f"Video shape: {video_tensor.shape}")
    
    # 1. PCA可视化
    print("生成PCA可视化...")
    visualizer.visualize_feature_pca(hidden, video_tensor, save_dir="hidden_pca_vis")
    
    # 2. 注意力图可视化
    print("生成注意力图可视化...")
    visualizer.visualize_attention_maps(hidden, video_tensor, save_dir="hidden_attention_vis")
    
    # 3. 特征相似性可视化
    print("生成特征相似性可视化...")
    visualizer.visualize_feature_similarity(hidden, video_tensor, save_dir="hidden_similarity_vis")
    
    # 4. t-SNE可视化（可选，计算时间较长）
    print("生成t-SNE可视化...")
    visualizer.visualize_tsne(hidden, video_tensor, save_dir="hidden_tsne_vis")
    
    print("所有可视化完成！")
    
    return hidden