# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
import torch

from .utils import convert_to_numpy


class SubjectAnnotator:
    def __init__(self, cfg, device=None):
        self.mode = cfg.get('MODE', "salientmasktrack")
        self.use_aug = cfg.get('USE_AUG', False)
        self.use_crop = cfg.get('USE_CROP', False)
        self.roi_only = cfg.get('ROI_ONLY', False)
        self.return_mask = cfg.get('RETURN_MASK', True)

        from .inpainting import InpaintingAnnotator
        self.inp_anno = InpaintingAnnotator(cfg['INPAINTING'], device=device)
        if self.use_aug:
            from .maskaug import MaskAugAnnotator
            self.maskaug_anno = MaskAugAnnotator(cfg={})
        assert self.mode in ["plain", "salient", "mask", "bbox", "salientmasktrack", "salientbboxtrack", "masktrack",
                             "bboxtrack", "label", "caption", "all"]

    def forward(self, image=None, mode=None, return_mask=None, mask_cfg=None, mask=None, bbox=None, label=None, caption=None):
        return_mask = return_mask if return_mask is not None else self.return_mask

        if mode == "plain":
            return {"image": image, "mask": None} if return_mask else image

        inp_res = self.inp_anno.forward(image,  mask=mask, bbox=bbox, label=label, caption=caption, mode=mode, return_mask=True, return_source=True)
        src_image = inp_res['src_image']
        mask = inp_res['mask']

        if self.use_aug and mask_cfg is not None:
            mask = self.maskaug_anno.forward(mask, mask_cfg)

        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        if (binary_mask is None or binary_mask.size == 0 or cv2.countNonZero(binary_mask) == 0):
            x, y, w, h = 0, 0, binary_mask.shape[1], binary_mask.shape[0]
        else:
            x, y, w, h = cv2.boundingRect(binary_mask)

        ret_mask = mask.copy()
        ret_image = src_image.copy()

        if self.roi_only:
            ret_image[mask == 0] = 255

        if self.use_crop:
            ret_image = ret_image[y:y + h, x:x + w]
            ret_mask = ret_mask[y:y + h, x:x + w]

        if return_mask:
            return {"image": ret_image, "mask": ret_mask}
        else:
            return ret_image


