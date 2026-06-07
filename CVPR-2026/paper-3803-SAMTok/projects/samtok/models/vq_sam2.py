import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print("use npu success!")
except:
    print("npu not enabled!")
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from xtuner.registry import BUILDER
# from xtuner.model.utils import guess_load_checkpoint

import os.path as osp
from typing import List, Optional

def guess_load_checkpoint(pth_model):
    if osp.isfile(pth_model):
        state_dict = torch.load(pth_model, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    elif osp.isdir(pth_model):
        try:
            from xtuner.utils.zero_to_any_dtype import (
                get_state_dict_from_zero_checkpoint,
            )
        except ImportError:
            raise ImportError(
                "The provided PTH model appears to be a DeepSpeed checkpoint. "
                "However, DeepSpeed library is not detected in current "
                "environment. This suggests that DeepSpeed may not be "
                "installed or is incorrectly configured. Please verify your "
                "setup."
            )
        state_dict = get_state_dict_from_zero_checkpoint(
            osp.dirname(pth_model), osp.basename(pth_model)
        )
    else:
        raise FileNotFoundError(f"Cannot find {pth_model}")
    return state_dict

class VQ_SAM2Model(BaseModel):
    def __init__(
        self,
        hf_model,
        sam2_pretrained_weights=None,
        pretrained_pth=None,
        freeze_sam2_decoder=False,
        freeze_codebook=False,
        box_input=False,
    ):
        super(VQ_SAM2Model, self).__init__()
        vq_sam2_config = hf_model['config']
        sam2_config = BUILDER.build(vq_sam2_config['sam2_config'])
        vq_sam2_config.update({'sam2_config': sam2_config})
        vq_sam2_config = BUILDER.build(vq_sam2_config)
        hf_model.update({'config': vq_sam2_config})
        
        self.hf_model = BUILDER.build(hf_model)

        # if sam2_pretrained_weights is not None:
        #     self.hf_model.model.load_pretrained_weights(sam2_pretrained_weights, strict=False)
        
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_pth}")
        
        self.hf_model.model.requires_grad_(False)
        if not freeze_sam2_decoder:
            self.hf_model.model.sam2_model.sam_mask_decoder.requires_grad_(True)
            self.hf_model.model.sam2_model.sam_mask_decoder.pred_obj_score_head.requires_grad_(False)
            self.hf_model.model.sam2_model.sam_mask_decoder.iou_prediction_head.requires_grad_(False)
        if not freeze_codebook:
            self.hf_model.model.sam2_model.sam_mask_encoder.requires_grad_(True)

        self.box_input = box_input
        self.freeze_codebook = freeze_codebook
    
    def state_dict(self, *args, **kwargs):
        from collections import OrderedDict

        to_return = OrderedDict()

        for n, p in self.hf_model.state_dict().items():
            to_return.update({n: p})

        return to_return

    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(n)
        # exit(0)
        pixel_values = data.pop('pixel_values', None)
        gt_masks = data.pop('masks', None)
        gt_boxes = data.pop('boxes', None) if self.box_input else None

        output = self.hf_model(pixel_values, gt_masks, gt_boxes, freeze_codebook=self.freeze_codebook)

        return {
            'loss_recon': output['loss_recon'],
            'loss_quant': output['loss_quant'],
        }