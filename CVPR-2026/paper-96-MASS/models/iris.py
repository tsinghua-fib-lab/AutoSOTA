"""MASS/Iris in-context segmentation model.

The Iris model encodes a reference image and binary reference mask into visual
prior tokens, fuses those priors with target image features, and predicts one
or more segmentation classes without task-specific finetuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any

from .modules.encoder import UNet_Encoder
from .modules.decoder import UNet_Decoder
from .modules.task_encoding import TaskEncodingLayer_SubPixel
from .modules.fusion import HierarchyPriorClassifier
from .utils import get_block, get_norm

from utils.registry import register_model

@register_model("iris")
class Iris(nn.Module):
    """
    Iris model: In-context Reference Image guided Segmentation.
    
    This model enables flexible adaptation to novel segmentation tasks through
    reference examples without fine-tuning.
    
    Args:
        in_ch: Input channel dimension
        base_ch: Base channel dimension, used when channels is not provided
        channels: Explicit channel dimensions for each stage [ch0, ch1, ch2, ch3, ch4]
        scale: Downsampling scale for each level
        kernel_size: Kernel size for each level
        block: Block type ('BasicBlock', 'Bottleneck', etc.)
        num_block: Number of blocks per level
        pool: Whether to use pooling for downsampling
        norm: Normalization type ('bn', 'in', 'ln')
        tn: Number of task priors
        num_prior_stage: Number of prior fusion stages
        ema_moment: EMA momentum for buffer updates
    """
    def __init__(
        self, 
        in_ch: int, 
        base_ch: Optional[int] = None,
        channels: Optional[List[int]] = None,
        scale: List[int] = [2,2,2,2], 
        kernel_size: List[int] = [3,3,3,3,3], 
        block: str = 'BasicBlock', 
        num_block: List[int] = [2,2,2,2],
        pool: bool = True, 
        norm: str = 'in', 
        tn: int = 72,
        num_prior_stage: int = 3,
        ema_moment: float = 0.99,
    ):
        super().__init__()
        self.ema_moment = ema_moment
        
        if channels is None:
            if base_ch is None:
                raise ValueError("Either 'channels' or 'base_ch' must be provided")
            channels = [base_ch, 2*base_ch, 4*base_ch, 8*base_ch, 16*base_ch]
        else:
            if len(channels) != 5:
                raise ValueError(f"'channels' must have 5 values, got {len(channels)}")

        self.encoder = UNet_Encoder(
            in_ch=in_ch, 
            base_ch=base_ch,
            block=block, 
            scale=scale, 
            num_block=num_block, 
            pool=pool, 
            norm=norm, 
            kernel_size=kernel_size,
            channels=channels
        )
        
        self.decoder = UNet_Decoder(
            in_ch=in_ch, 
            base_ch=base_ch,
            block=block, 
            scale=scale, 
            num_block=num_block, 
            norm=norm, 
            kernel_size=kernel_size, 
            num_prior_stage=num_prior_stage,
            channels=channels
        )
        
        self.num_prior_stage = num_prior_stage
        
        self.task_embedding = nn.ModuleList()
        for i in range(num_prior_stage):
            # Stage mapping: i=0 -> ch[4] (deepest), i=1 -> ch[3], i=2 -> ch[2]
            dim = channels[4-i]

            # Buffer priors are used by optional label-indexed evaluation paths.
            # Each prior stores 10 learned query tokens plus one foreground token.
            self.register_buffer(
                f'task_prior_{i}', 
                nn.init.xavier_uniform_(torch.zeros(tn, 10+1, dim))
            )
            
            self.task_embedding.append(
                TaskEncodingLayer_SubPixel(
                    dim, 
                    dim, 
                    block_num=2, 
                    task_query_num=10, 
                    scale=2**(4-i)
                )
            )

        self.out = HierarchyPriorClassifier(sum(channels[4-i] for i in range(num_prior_stage)), channels[0])

    def forward(
        self, 
        tgt_img: torch.Tensor, 
        ref_img: Optional[torch.Tensor] = None, 
        ref_lab: Optional[torch.Tensor] = None, 
        tgt_idx: Optional[torch.Tensor] = None, 
        mask_type: Optional[torch.Tensor] = None, 
        update_buffer: bool = False, 
        output_prior: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], List]]:
        """
        Forward pass through the Iris model.
        
        Args:
            tgt_img: Target image to segment
            ref_img: Reference image (optional)
            ref_lab: Reference label (optional)
            tgt_idx: Target class indices (optional)
            mask_type: Mask type (optional)
            update_buffer: Whether to update the task prior buffer
            output_prior: Whether to output the prior tokens
            
        Returns:
            Segmentation output tensor, and optionally prior tokens and debug info
        """
        tgt_feat_list = self.encoder(tgt_img)
        
        if ref_img is not None and ref_lab is not None:
            # In-context mode: encode the reference pair into visual priors.
            ref_feat_list = self.encoder(ref_img)

            if output_prior:
                prior_list, TE_details = self.encode_visual_prior(
                    ref_feat_list[:self.num_prior_stage], 
                    ref_lab, 
                    output_debug=True
                )
            else:
                prior_list = self.encode_visual_prior(
                    ref_feat_list[:self.num_prior_stage], 
                    ref_lab
                )
        else:
            # During buffer-based evaluation/finetuning, priors are indexed
            # directly instead of encoded from reference images.
            prior_list = self.get_visual_prior_from_buffer(tgt_idx)
        
        if update_buffer:
            self.ema_update_prior_token(prior_list, tgt_idx)
        
        output, posterior_list = self.decoder(tgt_feat_list, prior_list)
        output = self.out(output, posterior_list)
        
        if output_prior:
            return output, prior_list, TE_details
        else:
            return output

    def forward_with_encoded_prior(
        self, 
        tgt_img: torch.Tensor, 
        encoded_prior_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass with pre-encoded priors.
        
        Args:
            tgt_img: Target image to segment
            encoded_prior_list: List of encoded prior tokens
            
        Returns:
            Segmentation output tensor
        """
        tgt_feat_list = self.encoder(tgt_img)
        output, posterior_list = self.decoder(tgt_feat_list, encoded_prior_list)
        output = self.out(output, posterior_list)
        
        return output
    
    def get_visual_prior_from_buffer(
        self, 
        tgt_idx: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get prior tokens from the buffer.
        
        Args:
            tgt_idx: Target class indices
            
        Returns:
            List of prior tokens
        """

        prior_list = []
        
        for i in range(self.num_prior_stage):
           
            current_prior = getattr(self, f'task_prior_{i}')

            # tgt_idx can contain several class ids; indexing preserves that
            # class dimension for multi-class prediction.
            prior_list.append(current_prior[tgt_idx])

        return prior_list

    def ema_update_prior_token(
        self, 
        prior_list: List[torch.Tensor], 
        tgt_idx: torch.Tensor
    ) -> None:
        """
        Update prior tokens in buffer using EMA.
        
        Args:
            prior_list: List of prior tokens
            tgt_idx: Target class indices
        """
        for i, prior in enumerate(prior_list):
            current_prior = getattr(self, f'task_prior_{i}')
            for j, tgt in enumerate(tgt_idx):
                # Negative ids are padding slots and should not update a prior.
                valid_idx = tgt >= 0
                valid_tgt = tgt[valid_idx]
                
                current_prior[valid_tgt] = (
                    current_prior[valid_tgt] * self.ema_moment +
                    (1 - self.ema_moment) * prior[j, valid_idx, :].detach()
                )

            self.register_buffer(f'task_prior_{i}', current_prior)
    
    def encode_image_feature(
        self, 
        image: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Encode image features.
        
        Args:
            image: Input image
            
        Returns:
            List of feature maps
        """
        feat_list = self.encoder(image)
        return feat_list[:self.num_prior_stage]


    def encode_visual_prior(
        self, 
        ref_feat_list: List[torch.Tensor], 
        ref_lab: torch.Tensor, 
        output_debug: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List]]:
        """
        Encode visual priors from reference features and labels.
        
        Args:
            ref_feat_list: List of reference feature maps
            ref_lab: Reference label
            output_debug: Whether to output debug information
            
        Returns:
            List of visual prior tokens, and optionally debug info
        """
        visual_prior_token_list = []

        for i, ref_feat in enumerate(ref_feat_list):
            # Each decoder scale receives its own visual prior tokens.
            if output_debug:
                visual_prior_token, TE_details = self.task_embedding[i](
                    ref_feat, 
                    ref_lab, 
                    output_debug=True
                )
            else:
                visual_prior_token = self.task_embedding[i](ref_feat, ref_lab)

            visual_prior_token_list.append(visual_prior_token)

        if output_debug:
            return visual_prior_token_list, TE_details
        else:
            return visual_prior_token_list



    def encode_dense_feature(
        self,
        img: torch.Tensor,
        ref_lab: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode dense features with original spatial resolution.
        Uses provided mask as reference, or an all-ones mask if not provided.
        
        Args:
            img: Input image [B, C, D, H, W]
            ref_lab: Optional reference mask [B, 1, D, H, W]. If None, uses all-ones mask.
            
        Returns:
            Dense feature map (last decoder layer before classifier)
        """
        feat_list = self.encoder(img)
        
        if ref_lab is None:
            B, C, *spatial_dims = img.shape
            ref_lab = torch.ones(B, 1, *spatial_dims, device=img.device)
        else:
            ref_lab = ref_lab.contiguous()
        
        ref_feat_list = feat_list[:self.num_prior_stage]
        prior_list = self.encode_visual_prior(ref_feat_list, ref_lab)
        
        dense_feature, posterior_list = self.decoder(feat_list, prior_list)
        
        return dense_feature
