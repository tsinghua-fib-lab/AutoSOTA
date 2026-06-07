# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList, BaseModule
from torch import Tensor

from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .detr_layers import DetrTransformerEncoderLayer
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, get_text_sine_pos_embed, coordinate_to_encoding
from typing import List
import math
from .grounding_dino_layers import GroundingDinoTransformerDecoderLayer

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None
from mmdet.registry import MODELS


class PetDINOTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,                       # used for text/visual prompt to interact with image features (cross attention)
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                # text prompt
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,             # used for image feature to interact with text prompt (cross attention)
                pos_text: Tensor = None,                        # originally None
                text_self_attention_masks: Tensor = None,       # used for text self attention
                position_ids: Tensor = None,                    # used to compute pos_text
                # prompt_type
                prompt_type: str = 'Text'):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output_for_text = query.clone()                                             # torch.Size([2, 14078, 256])
        output_for_visual = query                                                   # torch.Size([2, 14078, 256])
        reference_points = self.get_encoder_reference_points(                       # torch.Size([2, 14078, 4, 2])
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers and text_attention_mask is not None:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(                         # torch.Size([2, 224, 256])
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # text prompt and image feature mutual cross attention
            if self.fusion_layers and text_attention_mask is not None:
                output_for_text, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output_for_text,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            # text prompt self attention
            if self.text_layers and text_attention_mask is not None:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                # NOTE. text_self_attention_masks must not have an all-False row, otherwise memory_text will contain NaN values
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            # image feature self attention
            if prompt_type == 'Text':
                output_for_text = layer(
                    query=output_for_text,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                output = output_for_text + 0 * output_for_visual
            elif prompt_type == 'Visual':
                output_for_visual = layer(
                    query=output_for_visual,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                output = 0 * output_for_text + output_for_visual
            else:
                raise Exception(f"prompt_type should be Text or Visual, but got {self.prompt_type}")

        return output, memory_text


@MODELS.register_module()
class VisualPromptEncoder(BaseModule):

    def __init__(self,
                 num_layers: int,
                 visual_layer_cfg: ConfigType,
                 pos_visual_mode_id: int = 0,
                 num_cp: int = -1,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.visual_layer_cfg = visual_layer_cfg
        self.pos_visual_mode_id = pos_visual_mode_id
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()
    
    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        
        self.visual_layers = ModuleList([
            PetDINOVisualPromptEncoderLayer(**self.visual_layer_cfg)
            for _ in range(self.num_layers)
        ])

        self.embed_dims = self.visual_layers[0].embed_dims
        self.ref_point_head = MLP(input_dim=self.embed_dims * 2, hidden_dim=self.embed_dims, output_dim=self.embed_dims, num_layers=2)
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.visual_layers[i] = checkpoint_wrapper(self.visual_layers[i])
    
    def forward(self,
                # prompt image feat
                prompt_query: Tensor = None,
                prompt_query_pos: Tensor = None,
                prompt_key_padding_mask: Tensor = None,                       
                prompt_spatial_shapes: Tensor = None,
                prompt_level_start_index: Tensor = None,
                prompt_valid_ratios: Tensor = None,
                # visual prompt
                memory_visual: Tensor = None,
                visual_attention_mask: Tensor = None,           # used for image feature to interact with visual prompt (cross attention)
                pos_visual: Tensor = None,                      # originally None
                visual_self_attention_masks: Tensor = None,     # used for self_attention
                visual_position_ids: Tensor = None,             # similar to text, compute pos_visual via position_ids for visual prompt self attn
                normalized_coordinate: Tensor = None,           # used to compute B in the paper
                visual_prompt_idx: Tensor = None,               # used to pick global prompt features after all attention, use_global_box=True
                visual_prompts_batch_idx: Tensor = None,        # used to re-combine after cross attention, prompt_across_batch=True
                # prompt_type
                use_coordinate_to_encoding: bool = False):
        """Forward function of Transformer encoder.

        Args:
            
        """
        prompt_output = prompt_query                                        # for extract visual embedding from visual prompt image
        if self.visual_layers and visual_self_attention_masks is not None:
            # NOTE. Prepare for generating pos_visual_prompt, which is used as positional encoding in MSDeformAttn
            # NOTE. Refer to pre_decoder in grounding_dino.py
            if use_coordinate_to_encoding:
                bs, _, _ = normalized_coordinate.shape
                visual_prompt_reference_points = normalized_coordinate
            else:
                bs, _, _ = normalized_coordinate.shape
                coords_unact = normalized_coordinate
                visual_prompt_reference_points = coords_unact
                visual_prompt_reference_points = visual_prompt_reference_points.sigmoid()
            # NOTE. Refer to forward in class DinoTransformerDecoder declared in dino_layers.py
            # NOTE. Important: convert pre-padding normalized_coordinate to post-padding normalized_coordinate
            if visual_prompt_reference_points.shape[-1] == 4:
                visual_prompt_reference_points_input = \
                    visual_prompt_reference_points[:, :, None] * torch.cat(
                        [prompt_valid_ratios, prompt_valid_ratios], -1)[:, None]                        # torch.Size([bs, K, 4, 4])
            else:
                assert visual_prompt_reference_points.shape[-1] == 2
                visual_prompt_reference_points_input = \
                    visual_prompt_reference_points[:, :, None] * prompt_valid_ratios[:, None]
            # NOTE. get pos_visual_prompt, equivalent to B in the paper;
            query_sine_embed = coordinate_to_encoding(
                visual_prompt_reference_points_input[:, :, 0, :])
            pos_visual_prompt = self.ref_point_head(query_sine_embed)

            # NOTE. positional encoding for self attention
            if self.pos_visual_mode_id == 0:
                # 0. directly use pos_visual_prompt
                pos_visual = pos_visual_prompt
            elif self.pos_visual_mode_id == 1:
                # 1. positional encoding is None
                pos_visual = None
            elif self.pos_visual_mode_id == 2:
                # 2. text-like positional encoding, e.g. [0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
                pos_visual = get_text_sine_pos_embed(
                                visual_position_ids[..., None],
                                num_pos_feats=256,
                                exchange_xy=False)
            elif self.pos_visual_mode_id == 3:
                # 3. only highlight global box positional encoding, e.g. [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
                new_visual_position_ids = torch.zeros_like(visual_position_ids)
                assert visual_prompt_idx is not None
                for i in range(bs):
                    valid_idx_for_visual_prompt_idx = visual_prompt_idx[i]!=-1  # NOTE. pad_value is -1
                    new_visual_position_ids[i][visual_prompt_idx[i][valid_idx_for_visual_prompt_idx]] = 1
                pos_visual = get_text_sine_pos_embed(
                                new_visual_position_ids[..., None],
                                num_pos_feats=256,
                                exchange_xy=False)
            else:
                raise Exception('pos_visual_mode_id should be 0, 1, 2, or 3.')
        
        # get memory_visual by MSDeformAttn and selfAttn
        if self.visual_layers and visual_self_attention_masks is not None:
            for layer_id in range(len(self.visual_layers)):
                # NOTE. query_pos in dino_layers.py is also obtained from visual_prompt_reference_points_input (dino_layers.py line85)
                visual_num_heads = self.visual_layers[layer_id].self_attn_cfg.num_heads
                memory_visual = self.visual_layers[layer_id](
                    query=memory_visual,                                                        # torch.Size([2, 10, 256])
                    query_pos=pos_visual_prompt,                                                # torch.Size([2, 10, 256])      # NOTE. #TODO. experiment with/without this, or whether Value part should have attention.
                    query_pos_for_self_attn=pos_visual,                                         # torch.Size([2, 10, 256])
                    value=prompt_output,                                                        # torch.Size([2, 11736, 256])
                    key_padding_mask=prompt_key_padding_mask,                                   # torch.Size([2, 11736]), related to image padding; if bs=1, key_padding_mask is None
                    self_attn_mask=~visual_self_attention_masks.repeat(visual_num_heads, 1, 1),     # note we use ~ for mask here. # should repeat or not ? (should)
                    # kwargs
                    spatial_shapes=prompt_spatial_shapes,
                    level_start_index=prompt_level_start_index,
                    valid_ratios=prompt_valid_ratios,
                    reference_points=visual_prompt_reference_points_input,
                    # cross_attn_mask=None,                                             # NOTE. no need for image features to extract from memory_visual, so no mask needed
                    batch_idx=visual_prompts_batch_idx,                                 # NOTE. important, for prompt_across_batch==True, re-arrange query by batch_idx
                )
        
        # NOTE. if use global box
        if visual_prompt_idx is not None:
            # pick features from memory_visual according to visual_prompt_idx
            new_memory_visual_list = []
            for i in range(bs):
                valid_idx_for_visual_prompt_idx = visual_prompt_idx[i]!=-1  # NOTE. pad_value is -1
                new_memory_visual_list.append(memory_visual[i][visual_prompt_idx[i][valid_idx_for_visual_prompt_idx]])
            # lengths differ after picking, pad to the same length
            max_len = max([item.shape[0] for item in new_memory_visual_list])
            new_memory_visual = torch.rand(bs, max_len, memory_visual.shape[-1], device=memory_visual.device)
            # new_memory_visual = torch.zeros(*(bs, max_len, memory_visual.shape[-1]), device=memory_visual.device)
            for i in range(bs):
                new_memory_visual[i, :new_memory_visual_list[i].shape[0], :] = new_memory_visual_list[i]
                # need_pad_num = max_len - new_memory_visual_list[i].shape[0]
                # need_repeat_num = math.ceil(need_pad_num/new_memory_visual_list[i].shape[0])
                # new_memory_visual[i, new_memory_visual_list[i].shape[0]:, :] = new_memory_visual_list[i].repeat(need_repeat_num, 1)[:need_pad_num]
            memory_visual = new_memory_visual

        return memory_visual


class PetDINOVisualPromptEncoderLayer(BaseModule):
    # NOTE. ref class DeformableDetrTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
    
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,                          # torch.Size([2, 9, 256])           memory_visual 
                key: Tensor = None,                     # None                              None
                value: Tensor = None,                   # torch.Size([2, 11645, 256])       output_
                query_pos: Tensor = None,               # torch.Size([2, 9, 256])           pos_visual_prompt
                query_pos_for_self_attn: Tensor = None, # torch.Size([2, 9, 256])           pos_visual
                key_pos: Tensor = None,                 # None                              None
                self_attn_mask: Tensor = None,          # torch.Size([8, 9, 9])             ~visual_self_attention_masks.repeat(visual_num_heads, 1, 1)
                cross_attn_mask: Tensor = None,         # None                              None
                key_padding_mask: Tensor = None,        # torch.Size([2, 11645])            key_padding_mask
                batch_idx: Tensor = None,               # 
                **kwargs) -> Tensor:                    # 'spatial_shapes', 'level_start_index', 'valid_ratios', 'reference_points'
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.cross_attn(                # deformable cross attention, so key and key_pos are both None
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,          # None
            key_padding_mask=key_padding_mask,  # None
            **kwargs)
        query = self.norms[0](query)

        # NOTE. re-arrange query by batch_idx
        if batch_idx is not None:
            arranged_query = torch.zeros(*(query.shape), device=query.device)
            for i in range(query.shape[0]):
                batch_idx[i][batch_idx[i]==-1] = i
                for j in range(query.shape[0]):
                    arranged_query[i][batch_idx[i]==j] = query[j][batch_idx[i]==j]
            query = arranged_query

        # NOTE. apply self attention for aggregation
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos_for_self_attn,
            key_pos=query_pos_for_self_attn,
            attn_mask=self_attn_mask,
            **kwargs                            # **kwargs was not added before, but actually adding it here has no effect since it's MultiheadAttention
            )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


class PetDINOTransformerDecoder(DinoTransformerDecoder):
    """Mostly identical to GroundingDinoTransformerDecoder, only replacing self.layers with PetDINOTransformerDecoderLayer.
    """

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            PetDINOTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)


class PetDINOTransformerDecoderLayer(GroundingDinoTransformerDecoderLayer):
    """Based on GroundingDinoTransformerDecoderLayer, determines whether to perform cross attention based on prompt_type.
    """

    def __init__(self,
                 with_cross_attn_visual=True,
                 remove_query_pos=False,
                 lasf=False,
                 cross_attn_visual_cfg=dict(),
                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        self.with_cross_attn_visual = with_cross_attn_visual
        self.remove_query_pos = remove_query_pos
        self.lasf = lasf    # ref OV-DINO.
        if self.lasf:
            assert self.with_cross_attn_visual, "Before turning on Lasf, you need to first turn on with_cross_attn_visual."
        # cross_attn_visual is independent of cross_attn_text
        self.cross_attn_visual_cfg = cross_attn_visual_cfg
        if self.cross_attn_visual_cfg and 'batch_first' not in self.cross_attn_visual_cfg:
            self.cross_attn_visual_cfg['batch_first'] = True
        super().__init__(**kwargs)
        if self.cross_attn_visual_cfg:
            self.cross_attn_visual = MultiheadAttention(**self.cross_attn_visual_cfg)
            self.norm_extra = build_norm_layer(self.norm_cfg, self.embed_dims)[1]

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,      # dn_mask
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                prompt_type: str = 'Text',
                memory_so: Tensor = None,
                **kwargs) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # kwargs: spatial_shapes, level_start_index, valid_ratios and reference_points
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,        # torch.Size([4, 1082, 256]), pos embedding of reference_points of these queries
            key_pos=query_pos,
            attn_mask=self_attn_mask,   # dn_mask: torch.Size([1082, 1082])
            **kwargs)                   # kwargs has no effect here since it's MultiheadAttention
        query = self.norms[0](query)
        # cross attention between query and text
        if prompt_type == 'Text':
            query = self.cross_attn_text(
                query=query,                # torch.Size([1, 900, 256])
                query_pos=query_pos,
                key=memory_text,            # torch.Size([1, 3, 256])
                value=memory_text,
                key_padding_mask=text_attention_mask)   # tensor([[False, False, False]], device='cuda:0')
            query = self.norms[1](query)
        elif prompt_type == 'Visual':
            if self.with_cross_attn_visual:
                if self.lasf:
                    K = memory_so
                    V = memory_so
                    attention_mask = None   # key_padding_mask=None, all positions are valid by default
                else:
                    K = memory_text
                    V = memory_text
                    attention_mask = text_attention_mask
                if self.remove_query_pos:
                    query = self.cross_attn_text(
                        query=query,                # torch.Size([1, 900, 256])
                        query_pos=None,             # NOTE. query_pos info may resonate with pos in memory_text, causing info leakage
                        key=K,                      # torch.Size([1, 3, 256])
                        value=V,
                        key_padding_mask=attention_mask)   # tensor([[False, False, False]], device='cuda:0')
                else:
                    query = self.cross_attn_text(
                        query=query,                # torch.Size([1, 900, 256])
                        query_pos=query_pos,        # NOTE. query_pos info may resonate with pos in memory_text, causing info leakage
                        key=K,                      # torch.Size([1, 3, 256])
                        value=V,
                        key_padding_mask=attention_mask)   # tensor([[False, False, False]], device='cuda:0')
                query = self.norms[1](query)
            else:
                query_ = query.clone()
                query_ = self.cross_attn_text(
                    query=query_,                # torch.Size([1, 900, 256])
                    query_pos=query_pos,
                    key=memory_text,            # torch.Size([1, 3, 256])
                    value=memory_text,
                    key_padding_mask=text_attention_mask)   # tensor([[False, False, False]], device='cuda:0')
                query_ = self.norms[1](query_)
                query = query + 0 * query_
        else:
            raise Exception(f"prompt_type should be Text or Visual, but got {self.prompt_type}")
        # cross attention between query and image
        query = self.cross_attn(                # deformable cross attention, key and key_pos are both None; think about how deformable cross attention works
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,          # None
            key_padding_mask=key_padding_mask,  # None
            **kwargs)
        query = self.norms[2](query)
        if self.cross_attn_visual_cfg:
            if prompt_type == 'Visual':
                query = self.cross_attn_visual(
                    query=query,                # torch.Size([1, 900, 256])
                    query_pos=query_pos,
                    key=memory_text,            # torch.Size([1, 3, 256])
                    value=memory_text,
                    key_padding_mask=text_attention_mask)   # tensor([[False, False, False]], device='cuda:0')
                query = self.norm_extra(query)
            elif prompt_type == 'Text':
                query_ = query.clone()
                query_ = self.cross_attn_visual(
                    query=query_,                # torch.Size([1, 900, 256])
                    query_pos=query_pos,
                    key=memory_text,            # torch.Size([1, 3, 256])
                    value=memory_text,
                    key_padding_mask=text_attention_mask)   # tensor([[False, False, False]], device='cuda:0')
                query_ = self.norm_extra(query_)
                query = query + 0 * query_
            else:
                raise Exception(f"prompt_type should be Text or Visual, but got {self.prompt_type}")
        query = self.ffn(query)
        query = self.norms[3](query)

        return query


@MODELS.register_module()
class VisualPromptEnhancer(BaseModule):

    def __init__(self,
                 num_layers: int,
                 visual_layer_cfg: ConfigType,
                 num_cp: int = -1,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.visual_layer_cfg = visual_layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()
    
    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.visual_layers = ModuleList([
            PetDINOVisualPromptEnhancerLayer(**self.visual_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.visual_layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.visual_layers[i] = checkpoint_wrapper(self.visual_layers[i])
    
    def forward(self,
                # visual prompt
                memory_visual: Tensor = None,
                visual_self_attention_masks: Tensor = None,     # used for self_attention
                ):
        """Forward function of VisualPromptEnhancer.
            
        """
        for layer_id in range(len(self.visual_layers)):
            # NOTE. query_pos in dino_layers.py is also obtained from visual_prompt_reference_points_input (dino_layers.py line85)
            visual_num_heads = self.visual_layers[layer_id].self_attn_cfg.num_heads
            memory_visual = self.visual_layers[layer_id](
                query=memory_visual,                                                            # torch.Size([2, 10, 256])
                query_pos=None,                                                                 # None
                self_attn_mask=~visual_self_attention_masks.repeat(visual_num_heads, 1, 1),     # note we use ~ for mask here. # should repeat or not ? (should)
            )
        return memory_visual


class PetDINOVisualPromptEnhancerLayer(BaseModule):
    # NOTE. ref class DeformableDetrTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),    # NOTE. when to use LN vs BN?
                 init_cfg: OptConfigType = None,
                 ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
    
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,                          # torch.Size([2, 9, 256])           memory_visual 
                query_pos: Tensor = None,               # torch.Size([2, 9, 256])           pos_visual_prompt
                self_attn_mask: Tensor = None,          # torch.Size([8, 9, 9])             ~visual_self_attention_masks.repeat(visual_num_heads, 1, 1)
                **kwargs) -> Tensor:                    # 'spatial_shapes', 'level_start_index', 'valid_ratios', 'reference_points'
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # NOTE. apply self attention for aggregation
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,                # None
            key_pos=query_pos,                  # None
            attn_mask=self_attn_mask,
            **kwargs                            # **kwargs was not added before, but actually adding it here has no effect since it's MultiheadAttention
            )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query
