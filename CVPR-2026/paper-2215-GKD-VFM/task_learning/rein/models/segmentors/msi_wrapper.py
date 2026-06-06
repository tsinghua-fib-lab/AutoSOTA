import torch
import torch.nn.functional as F
from typing import List
from torch import Tensor

from mmseg.registry import MODELS
from rein.models.segmentors.frozen_encoder_decoder import FrozenBackboneEncoderDecoder

@MODELS.register_module()
class MultiScaleModel(FrozenBackboneEncoderDecoder):
    """Multi-scale inference with efficient 3-scale averaging."""
    
    def __init__(self, **kwargs):
        self.ms_scales = kwargs.pop('ms_scales', [0.5, 1.0, 1.5])
        super().__init__(**kwargs)
    
    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """3-scale inference: run at [0.5, 1.0, 1.5] scales, average logits."""
        assert len(inputs) == 1, "Multi-scale requires batch_size=1"
        
        img = inputs[0]  # [C, H, W] after pipeline preprocessing
        img_meta = batch_img_metas[0]
        ori_shape = img_meta['ori_shape']
        
        all_logits = []
        
        for scale in self.ms_scales:
            h = int(round(ori_shape[0] * scale))
            w = int(round(ori_shape[1] * scale))
            
            if scale == 1.0:
                s_img = img
            else:
                s_img = F.interpolate(img.unsqueeze(0), size=(h, w),
                                      mode='bilinear', align_corners=False).squeeze(0)
            
            s_meta = dict(img_meta)
            s_meta['img_shape'] = (h, w)
            
            logit = super().inference(s_img.unsqueeze(0), [s_meta])
            
            if logit.shape[-2:] != ori_shape:
                logit = F.interpolate(logit, size=ori_shape,
                                      mode='bilinear', align_corners=False)
            
            all_logits.append(logit)
        
        return torch.stack(all_logits).mean(dim=0)
