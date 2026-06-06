# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import math
import random
from abc import ABCMeta

import numpy as np
import torch
from PIL import Image, ImageDraw
from .utils import convert_to_numpy, convert_to_pil, single_rle_to_mask, get_mask_box, read_video_one_frame

class InpaintingAnnotator:
    def __init__(self, cfg, device=None):
        self.use_aug = cfg.get('USE_AUG', True)
        self.return_mask = cfg.get('RETURN_MASK', True)
        self.return_source = cfg.get('RETURN_SOURCE', True)
        self.mask_color = cfg.get('MASK_COLOR', 128)
        self.mode = cfg.get('MODE', "mask")
        assert self.mode in ["salient", "mask", "bbox", "salientmasktrack", "salientbboxtrack", "maskpointtrack", "maskbboxtrack", "masktrack", "bboxtrack", "label", "caption", "all"]
        if self.mode in ["salient", "salienttrack"]:
            from .salient import SalientAnnotator
            self.salient_model = SalientAnnotator(cfg['SALIENT'], device=device)
        if self.mode in ['masktrack', 'bboxtrack', 'salienttrack']:
            from .sam2 import SAM2ImageAnnotator
            self.sam2_model = SAM2ImageAnnotator(cfg['SAM2'], device=device)
        if self.mode in ['label', 'caption']:
            from .gdino import GDINOAnnotator
            from .sam2 import SAM2ImageAnnotator
            self.gdino_model = GDINOAnnotator(cfg['GDINO'], device=device)
            self.sam2_model = SAM2ImageAnnotator(cfg['SAM2'], device=device)
        if self.mode in ['all']:
            from .salient import SalientAnnotator
            from .gdino import GDINOAnnotator
            from .sam2 import SAM2ImageAnnotator
            self.salient_model = SalientAnnotator(cfg['SALIENT'], device=device)
            self.gdino_model = None #GDINOAnnotator(cfg['GDINO'], device=device)
            self.sam2_model = SAM2ImageAnnotator(cfg['SAM2'], device=device)
        if self.use_aug:
            from .maskaug import MaskAugAnnotator
            self.maskaug_anno = MaskAugAnnotator(cfg={})

    def apply_plain_mask(self, image, mask, mask_color):
        bool_mask = mask > 0
        out_image = image.copy()
        out_image[bool_mask] = mask_color
        out_mask = np.where(bool_mask, 255, 0).astype(np.uint8)
        return out_image, out_mask
        
    def apply_seg_mask(self, image, mask, mask_color, mask_cfg=None):
        out_mask = (mask * 255).astype('uint8')
        if self.use_aug and mask_cfg is not None:
            out_mask = self.maskaug_anno.forward(out_mask, mask_cfg)
        bool_mask = out_mask > 0
        out_image = image.copy()
        out_image[bool_mask] = mask_color
        return out_image, out_mask
        
    def forward(self, image=None, mask=None, bbox=None, label=None, caption=None, mode=None, return_mask=None, return_source=None, mask_color=None, mask_cfg=None):
        mode = mode if mode is not None else self.mode
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = return_source if return_source is not None else self.return_source
        mask_color = mask_color if mask_color is not None else self.mask_color

        image = convert_to_numpy(image)
        out_image, out_mask = None, None
        if mode in ['salient']:
            mask = self.salient_model.forward(image)
            out_image, out_mask = self.apply_plain_mask(image, mask, mask_color)
        elif mode in ['mask']:
            mask_h, mask_w = mask.shape[:2]
            h, w = image.shape[:2]
            if (mask_h ==h) and (mask_w == w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            out_image, out_mask = self.apply_plain_mask(image, mask, mask_color)
        elif mode in ['bbox']:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w, x2)), int(min(h, y2))
            out_image = image.copy()
            out_image[y1:y2, x1:x2] = mask_color
            out_mask = np.zeros((h, w), dtype=np.uint8)
            out_mask[y1:y2, x1:x2] = 255
        elif mode in ['salientmasktrack']:
            mask = self.salient_model.forward(image)
            resize_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            out_mask = self.sam2_model.forward(image=image, mask=resize_mask, task_type='mask', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image, out_mask, mask_color, mask_cfg)
        elif mode in ['salientbboxtrack']:
            mask = self.salient_model.forward(image)
            bbox = get_mask_box(np.array(mask), threshold=1)
            out_mask = self.sam2_model.forward(image=image, input_box=bbox, task_type='input_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image, out_mask, mask_color, mask_cfg)
        elif mode in ['maskpointtrack']:
            out_mask = self.sam2_model.forward(image=image, mask=mask, task_type='mask_point', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image, out_mask, mask_color, mask_cfg)
        elif mode in ['maskbboxtrack']:
            out_mask = self.sam2_model.forward(image=image, mask=mask, task_type='mask_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image, out_mask, mask_color, mask_cfg)
        elif mode in ['masktrack']:
            resize_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            out_mask = self.sam2_model.forward(image=image, mask=resize_mask, task_type='mask', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image, out_mask, mask_color, mask_cfg)
        elif mode in ['bboxtrack']:
            out_mask = self.sam2_model.forward(image=image, input_box=bbox, task_type='input_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image, out_mask, mask_color, mask_cfg)
        elif mode in ['label']:
            gdino_res = self.gdino_model.forward(image, classes=label)
            if 'boxes' in gdino_res and len(gdino_res['boxes']) > 0:
                bboxes = gdino_res['boxes'][0]
            else:
                raise ValueError(f"Unable to find the corresponding boxes of label: {label}")
            out_mask = self.sam2_model.forward(image=image, input_box=bboxes, task_type='input_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image, out_mask, mask_color, mask_cfg)
        elif mode in ['caption']:
            gdino_res = self.gdino_model.forward(image, caption=caption)
            if 'boxes' in gdino_res and len(gdino_res['boxes']) > 0:
                bboxes = gdino_res['boxes'][0]
            else:
                raise ValueError(f"Unable to find the corresponding boxes of caption: {caption}")
            out_mask = self.sam2_model.forward(image=image, input_box=bboxes, task_type='input_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image, out_mask, mask_color, mask_cfg)

        ret_data = {"image": out_image}
        if return_mask:
            ret_data["mask"] = out_mask
        if return_source:
            ret_data["src_image"] = image
        return ret_data




class InpaintingVideoAnnotator:
    def __init__(self, cfg, device=None):
        self.use_aug = cfg.get('USE_AUG', True)
        self.return_frame = cfg.get('RETURN_FRAME', True)
        self.return_mask = cfg.get('RETURN_MASK', True)
        self.return_source = cfg.get('RETURN_SOURCE', True)
        self.mask_color = cfg.get('MASK_COLOR', 128)
        self.mode = cfg.get('MODE', "mask")
        assert self.mode in ["salient", "mask", "bbox", "salientmasktrack", "salientbboxtrack", "maskpointtrack", "maskbboxtrack", "masktrack", "bboxtrack", "label", "caption", "all"]
        if self.mode in ["salient", "salienttrack"]:
            from .salient import SalientAnnotator
            self.salient_model = SalientAnnotator(cfg['SALIENT'], device=device)
        if self.mode in ['masktrack', 'bboxtrack', 'salienttrack']:
            from .sam2 import SAM2VideoAnnotator
            self.sam2_model = SAM2VideoAnnotator(cfg['SAM2'], device=device)
        if self.mode in ['label', 'caption']:
            from .gdino import GDINOAnnotator
            from .sam2 import SAM2VideoAnnotator
            self.gdino_model = GDINOAnnotator(cfg['GDINO'], device=device)
            self.sam2_model = SAM2VideoAnnotator(cfg['SAM2'], device=device)
        if self.mode in ['all']:
            from .salient import SalientAnnotator
            from .gdino import GDINOAnnotator
            from .sam2 import SAM2VideoAnnotator
            self.salient_model = SalientAnnotator(cfg['SALIENT'], device=device)
            self.gdino_model = GDINOAnnotator(cfg['GDINO'], device=device)
            self.sam2_model = SAM2VideoAnnotator(cfg['SAM2'], device=device)
        if self.use_aug:
            from .maskaug import MaskAugAnnotator
            self.maskaug_anno = MaskAugAnnotator(cfg={})
    
    def apply_plain_mask(self, frames, mask, mask_color, return_frame=True):
        out_frames = []
        num_frames = len(frames)
        bool_mask = mask > 0
        out_masks = [np.where(bool_mask, 255, 0).astype(np.uint8)] * num_frames
        if not return_frame:
            return None, out_masks
        for i in range(num_frames):
            masked_frame = frames[i].copy()
            masked_frame[bool_mask] = mask_color
            out_frames.append(masked_frame)
        return out_frames, out_masks

    def apply_seg_mask(self, mask_data, frames, mask_color, mask_cfg=None, return_frame=True):
        out_frames = []
        out_masks = [(single_rle_to_mask(val[0]["mask"]) * 255).astype('uint8') for key, val in mask_data['annotations'].items()]
        if not return_frame:
            return None, out_masks
        num_frames = min(len(out_masks), len(frames))
        for i in range(num_frames):
            sub_mask = out_masks[i]
            if self.use_aug and mask_cfg is not None:
                sub_mask = self.maskaug_anno.forward(sub_mask, mask_cfg)
                out_masks[i] = sub_mask
            bool_mask = sub_mask > 0
            masked_frame = frames[i].copy()
            masked_frame[bool_mask] = mask_color
            out_frames.append(masked_frame)
        out_masks = out_masks[:num_frames]
        return out_frames, out_masks

    def forward(self, frames=None, video=None, mask=None, bbox=None, label=None, caption=None, mode=None, return_frame=None, return_mask=None, return_source=None, mask_color=None, mask_cfg=None):
        mode = mode if mode is not None else self.mode
        return_frame = return_frame if return_frame is not None else self.return_frame
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = return_source if return_source is not None else self.return_source
        mask_color = mask_color if mask_color is not None else self.mask_color

        out_frames, out_masks = [], []
        if mode in ['salient']:
            first_frame = frames[0] if frames is not None else read_video_one_frame(video)
            mask = self.salient_model.forward(first_frame)
            out_frames, out_masks = self.apply_plain_mask(frames, mask, mask_color, return_frame)
        elif mode in ['mask']:
            first_frame = frames[0] if frames is not None else read_video_one_frame(video)
            mask_h, mask_w = mask.shape[:2]
            h, w = first_frame.shape[:2]
            if (mask_h ==h) and (mask_w == w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            out_frames, out_masks = self.apply_plain_mask(frames, mask, mask_color, return_frame)
        elif mode in ['bbox']:
            first_frame = frames[0] if frames is not None else read_video_one_frame(video)
            num_frames = len(frames)
            x1, y1, x2, y2 = bbox
            h, w = first_frame.shape[:2]
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w, x2)), int(min(h, y2))
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            out_masks = [mask] * num_frames
            if not return_frame:
                out_frames = None
            else:
                for i in range(num_frames):
                    masked_frame = frames[i].copy()
                    masked_frame[y1:y2, x1:x2] = mask_color
                    out_frames.append(masked_frame)
        elif mode in ['salientmasktrack']:
            first_frame = frames[0] if frames is not None else read_video_one_frame(video)
            salient_mask = self.salient_model.forward(first_frame)
            mask_data = self.sam2_model.forward(video=video, mask=salient_mask, task_type='mask')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
        elif mode in ['salientbboxtrack']:
            first_frame = frames[0] if frames is not None else read_video_one_frame(video)
            salient_mask = self.salient_model.forward(first_frame)
            bbox = get_mask_box(np.array(salient_mask), threshold=1)
            mask_data = self.sam2_model.forward(video=video, input_box=bbox, task_type='input_box')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
        elif mode in ['maskpointtrack']:
            mask_data = self.sam2_model.forward(video=video, mask=mask, task_type='mask_point')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
        elif mode in ['maskbboxtrack']:
            mask_data = self.sam2_model.forward(video=video, mask=mask, task_type='mask_box')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
        elif mode in ['masktrack']:
            mask_data = self.sam2_model.forward(video=video, mask=mask, task_type='mask')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
        elif mode in ['bboxtrack']:
            mask_data = self.sam2_model.forward(video=video, input_box=bbox, task_type='input_box')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
        elif mode in ['label']:
            first_frame = frames[0] if frames is not None else read_video_one_frame(video)
            gdino_res = self.gdino_model.forward(first_frame, classes=label)
            if 'boxes' in gdino_res and len(gdino_res['boxes']) > 0:
                bboxes = gdino_res['boxes'][0]
            else:
                raise ValueError(f"Unable to find the corresponding boxes of label: {label}")
            mask_data = self.sam2_model.forward(video=video, input_box=bboxes, task_type='input_box')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
        elif mode in ['caption']:
            first_frame = frames[0] if frames is not None else read_video_one_frame(video)
            gdino_res = self.gdino_model.forward(first_frame, caption=caption)
            if 'boxes' in gdino_res and len(gdino_res['boxes']) > 0:
                bboxes = gdino_res['boxes'][0]
            else:
                raise ValueError(f"Unable to find the corresponding boxes of caption: {caption}")
            mask_data = self.sam2_model.forward(video=video, input_box=bboxes, task_type='input_box')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)

        ret_data = {}
        if return_frame:
            ret_data["frames"] = out_frames
        if return_mask:
            ret_data["masks"] = out_masks
        return ret_data



