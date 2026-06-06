# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

from .utils import convert_to_numpy


class LayoutBboxAnnotator:
    def __init__(self, cfg, device=None):
        self.bg_color = cfg.get('BG_COLOR', [255, 255, 255])
        self.box_color = cfg.get('BOX_COLOR', [0, 0, 0])
        self.frame_size = cfg.get('FRAME_SIZE', [720, 1280])  # [H, W]
        self.num_frames = cfg.get('NUM_FRAMES', 81)
        ram_tag_color_path = cfg.get('RAM_TAG_COLOR_PATH', None)
        self.color_dict = {'default': tuple(self.box_color)}
        if ram_tag_color_path is not None:
            lines = [id_name_color.strip().split('#;#') for id_name_color in open(ram_tag_color_path).readlines()]
            self.color_dict.update({id_name_color[1]: tuple(eval(id_name_color[2])) for id_name_color in lines})

    def forward(self, bbox, frame_size=None, num_frames=None, label=None, color=None):
        frame_size = frame_size if frame_size is not None else self.frame_size
        num_frames = num_frames if num_frames is not None else self.num_frames
        assert len(bbox) == 2, 'bbox should be a list of two elements (start_bbox & end_bbox)'
        # frame_size = [H, W]
        # bbox = [x1, y1, x2, y2]
        label = label[0] if label is not None and isinstance(label, list) else label
        if label is not None and label in self.color_dict:
            box_color = self.color_dict[label]
        elif color is not None:
            box_color = color
        else:
            box_color = self.color_dict['default']
        start_bbox, end_bbox = bbox
        start_bbox = [start_bbox[0], start_bbox[1], start_bbox[2] - start_bbox[0], start_bbox[3] - start_bbox[1]]
        start_bbox = np.array(start_bbox, dtype=np.float32)
        end_bbox = [end_bbox[0], end_bbox[1], end_bbox[2] - end_bbox[0], end_bbox[3] - end_bbox[1]]
        end_bbox = np.array(end_bbox, dtype=np.float32)
        bbox_increment = (end_bbox - start_bbox) / num_frames
        ret_frames = []
        for frame_idx in range(num_frames):
            frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
            frame[:] = self.bg_color
            current_bbox = start_bbox + bbox_increment * frame_idx
            current_bbox = current_bbox.astype(int)
            x, y, w, h = current_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            ret_frames.append(frame[..., ::-1])
        return ret_frames




class LayoutMaskAnnotator:
    def __init__(self, cfg, device=None):
        self.use_aug = cfg.get('USE_AUG', False)
        self.bg_color = cfg.get('BG_COLOR', [255, 255, 255])
        self.box_color = cfg.get('BOX_COLOR', [0, 0, 0])
        ram_tag_color_path = cfg.get('RAM_TAG_COLOR_PATH', None)
        self.color_dict = {'default': tuple(self.box_color)}
        if ram_tag_color_path is not None:
            lines = [id_name_color.strip().split('#;#') for id_name_color in open(ram_tag_color_path).readlines()]
            self.color_dict.update({id_name_color[1]: tuple(eval(id_name_color[2])) for id_name_color in lines})
        if self.use_aug:
            from .maskaug import MaskAugAnnotator
            self.maskaug_anno = MaskAugAnnotator(cfg={})


    def find_contours(self, mask):
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, canvas, contour, color):
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        canvas = cv2.drawContours(canvas, contour, -1, color, thickness=3)
        return canvas

    def forward(self, mask=None, color=None, label=None, mask_cfg=None):
        if not isinstance(mask, list):
            is_batch = False
            mask = [mask]
        else:
            is_batch = True

        if label is not None and label in self.color_dict:
            color = self.color_dict[label]
        elif color is not None:
            color = color
        else:
            color = self.color_dict['default']

        ret_data = []
        for sub_mask in mask:
            sub_mask = convert_to_numpy(sub_mask)
            if self.use_aug:
                sub_mask = self.maskaug_anno.forward(sub_mask, mask_cfg)
            canvas = np.ones((sub_mask.shape[0], sub_mask.shape[1], 3)) * 255
            contour = self.find_contours(sub_mask)
            frame = self.draw_contours(canvas, contour, color)
            ret_data.append(frame)

        if is_batch:
            return ret_data
        else:
            return ret_data[0]




class LayoutTrackAnnotator:
    def __init__(self, cfg, device=None):
        self.use_aug = cfg.get('USE_AUG', False)
        self.bg_color = cfg.get('BG_COLOR', [255, 255, 255])
        self.box_color = cfg.get('BOX_COLOR', [0, 0, 0])
        ram_tag_color_path = cfg.get('RAM_TAG_COLOR_PATH', None)
        self.color_dict = {'default': tuple(self.box_color)}
        if ram_tag_color_path is not None:
            lines = [id_name_color.strip().split('#;#') for id_name_color in open(ram_tag_color_path).readlines()]
            self.color_dict.update({id_name_color[1]: tuple(eval(id_name_color[2])) for id_name_color in lines})
        if self.use_aug:
            from .maskaug import MaskAugAnnotator
            self.maskaug_anno = MaskAugAnnotator(cfg={})
        from .inpainting import InpaintingVideoAnnotator
        self.inpainting_anno = InpaintingVideoAnnotator(cfg=cfg['INPAINTING'])

    def find_contours(self, mask):
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, canvas, contour, color):
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        canvas = cv2.drawContours(canvas, contour, -1, color, thickness=3)
        return canvas

    def forward(self, color=None, mask_cfg=None, frames=None, video=None, mask=None, bbox=None, label=None, caption=None, mode=None):
        inp_data = self.inpainting_anno.forward(frames, video, mask, bbox, label, caption, mode)
        inp_masks = inp_data['masks']

        label = label[0] if label is not None and isinstance(label, list) else label
        if label is not None and label in self.color_dict:
            color = self.color_dict[label]
        elif color is not None:
            color = color
        else:
            color = self.color_dict['default']

        num_frames = len(inp_masks)
        ret_data = []
        for i in range(num_frames):
            sub_mask = inp_masks[i]
            if self.use_aug and mask_cfg is not None:
                sub_mask = self.maskaug_anno.forward(sub_mask, mask_cfg)
            canvas = np.ones((sub_mask.shape[0], sub_mask.shape[1], 3)) * 255
            contour = self.find_contours(sub_mask)
            frame = self.draw_contours(canvas, contour, color)
            ret_data.append(frame)

        return ret_data


