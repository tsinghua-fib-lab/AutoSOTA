"""Task-encoding layers for in-context segmentation.

Given reference image features and binary reference masks, these modules produce
multi-scale visual prior tokens that define the segmentation task for MASS/Iris.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from ..components.attention_blocks import TaskQueryAttentionBlock
from ..components.pixel_ops import PixelShuffle3d, PixelUnshuffle3d
from ..components.positional import LearnablePositionalEncoding3D


class TaskEncodingLayer_SubPixel(nn.Module):
    """
    Task encoding layer used by MASS/Iris to produce visual prior tokens from
    reference image features and binary reference masks.
    """

    def __init__(
        self,
        feat_dim: int,
        prior_dim: int,
        block_num: int = 2,
        task_query_num: int = 10,
        scale: int = 16,
    ):
        super().__init__()

        self.task_query = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(task_query_num, prior_dim))
        )

        self.attn_layers = nn.ModuleList(
            [
                TaskQueryAttentionBlock(
                    feat_dim,
                    heads=feat_dim // 32,
                    dim_head=32,
                    attn_drop=0,
                    proj_drop=0,
                )
                for _ in range(block_num)
            ]
        )

        self.mask_dim = 1
        expand_scale = (scale ** 3) / feat_dim
        expanded_ch = int(feat_dim * expand_scale)
        shuffled_ch = int(expanded_ch / (scale**3))

        self.norm_foreground = nn.LayerNorm(feat_dim)
        self.norm_query = nn.LayerNorm(prior_dim)
        self.norm_combined = nn.LayerNorm(feat_dim)
        self.norm_post = nn.LayerNorm(feat_dim)

        self.expand = nn.Sequential(
            nn.InstanceNorm3d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(feat_dim, expanded_ch, kernel_size=1, padding=0),
        )
        self.pixel_shuffle = PixelShuffle3d(scale)

        self.merge = nn.Sequential(
            nn.InstanceNorm3d(shuffled_ch + self.mask_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(shuffled_ch + self.mask_dim, shuffled_ch, kernel_size=3, padding=1),
        )

        self.pixel_unshuffle = PixelUnshuffle3d(scale)
        self.squeeze = nn.Sequential(
            nn.InstanceNorm3d(expanded_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(expanded_ch, feat_dim, kernel_size=1, padding=0),
        )

        self.pos_embed = LearnablePositionalEncoding3D(shuffled_ch, 128, 128, 128)

    def forward(
        self,
        ref_feat: torch.Tensor,
        ref_lab: torch.Tensor,
        output_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            ref_feat: Reference feature map with shape [B, C, D, H, W].
            ref_lab: Reference masks with shape [B, K, D', H', W'].
            output_debug: Whether to return a few token norms for diagnostics.

        Returns:
            Visual prior tokens with shape [B, K, M + 1, C].
        """
        te_details = [] if output_debug else None
        ref_lab = ref_lab.float()

        upsampled_ref_feat = F.interpolate(
            ref_feat,
            size=ref_lab.shape[-3:],
            mode="trilinear",
            align_corners=True,
        )

        batch_size, dim, _, _, _ = upsampled_ref_feat.shape
        _, num_classes, _, _, _ = ref_lab.shape

        # Average reference features inside each binary mask; this gives one
        # foreground summary token per requested class.
        foreground_visual_prior_token = torch.matmul(
            upsampled_ref_feat.view(batch_size, dim, -1),
            ref_lab.view(batch_size, num_classes, -1)
            .permute(0, 2, 1)
            .to(upsampled_ref_feat.dtype),
        )

        den = (
            ref_lab.sum(dim=(2, 3, 4), keepdim=True)
            .view(batch_size, num_classes, 1)
            .permute(0, 2, 1)
            .clamp(min=1)
        )
        foreground_visual_prior_token /= den

        if output_debug:
            te_details.append(torch.norm(foreground_visual_prior_token[0, :, 0]))

        origin_x = ref_feat
        x = self.pixel_shuffle(self.expand(ref_feat))

        # Process each class mask as its own task while sharing the feature map.
        x_expanded = x.unsqueeze(1).expand(-1, num_classes, -1, -1, -1, -1)
        x_expanded = x_expanded.reshape(batch_size * num_classes, *x.shape[1:])

        ref_lab_reshaped = ref_lab.reshape(batch_size * num_classes, 1, *ref_lab.shape[2:])

        merged_x = torch.cat([x_expanded, ref_lab_reshaped], dim=1)
        merged_x = self.merge(merged_x)
        merged_x = self.pos_embed(merged_x)
        merged_x = self.pixel_unshuffle(merged_x)

        origin_x_expanded = origin_x.unsqueeze(1).expand(
            -1, num_classes, -1, -1, -1, -1
        )
        origin_x_expanded = origin_x_expanded.reshape(
            batch_size * num_classes, *origin_x.shape[1:]
        )

        merged_x = self.squeeze(merged_x) + origin_x_expanded
        merged_x = merged_x.view(batch_size * num_classes, dim, -1)
        merged_x = merged_x.permute(0, 2, 1).contiguous()

        task_query = self.task_query.unsqueeze(0).expand(
            batch_size * num_classes, -1, -1
        )
        foreground_token = foreground_visual_prior_token.permute(0, 2, 1).reshape(
            batch_size * num_classes, 1, dim
        )

        if output_debug:
            te_details.append(torch.norm(self.task_query[0, :]))

        task_query = self.norm_query(task_query)
        foreground_token = self.norm_foreground(foreground_token)

        # Learned task queries attend to mask-conditioned reference features;
        # the foreground token anchors them to the selected anatomy/object.
        combined_query = torch.cat([task_query, foreground_token], dim=1)
        combined_query = self.norm_combined(combined_query)

        for layer in self.attn_layers:
            combined_query = layer(merged_x, combined_query)

        combined_query = self.norm_post(combined_query)

        if output_debug:
            te_details.append(torch.norm(combined_query[0, 0, :]))

        combined_query = combined_query.view(
            batch_size, num_classes, *combined_query.shape[1:]
        )

        if output_debug:
            return combined_query, te_details
        return combined_query
