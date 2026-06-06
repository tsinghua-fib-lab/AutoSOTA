# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random

import cv2
import numpy as np

from .utils import convert_to_numpy


class RegionCanvasAnnotator:
    def __init__(self, cfg, device=None):
        self.scale_range = cfg.get('SCALE_RANGE', [0.75, 1.0])
        self.canvas_value = cfg.get('CANVAS_VALUE', 255)
        self.use_resize = cfg.get('USE_RESIZE', True)
        self.use_canvas = cfg.get('USE_CANVAS', True)
        self.use_aug = cfg.get('USE_AUG', False)
        if self.use_aug:
            from .maskaug import MaskAugAnnotator
            self.maskaug_anno = MaskAugAnnotator(cfg={})

    def forward(self, image, mask, mask_cfg=None):

        image = convert_to_numpy(image)
        mask = convert_to_numpy(mask)
        image_h, image_w = image.shape[:2]

        if self.use_aug:
            mask = self.maskaug_anno.forward(mask, mask_cfg)

        # get region with white bg
        image[np.array(mask) == 0] = self.canvas_value
        x, y, w, h = cv2.boundingRect(mask)
        region_crop = image[y:y + h, x:x + w]

        if self.use_resize:
            # resize region
            scale_min, scale_max = self.scale_range
            scale_factor = random.uniform(scale_min, scale_max)
            new_w, new_h = int(image_w * scale_factor), int(image_h * scale_factor)
            obj_scale_factor = min(new_w/w, new_h/h)

            new_w = int(w * obj_scale_factor)
            new_h = int(h * obj_scale_factor)
            region_crop_resized = cv2.resize(region_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            region_crop_resized = region_crop

        if self.use_canvas:
            # plot region into canvas
            new_canvas = np.ones_like(image) * self.canvas_value
            max_x = max(0, image_w - new_w)
            max_y = max(0, image_h - new_h)
            new_x = random.randint(0, max_x)
            new_y = random.randint(0, max_y)

            new_canvas[new_y:new_y + new_h, new_x:new_x + new_w] = region_crop_resized
        else:
            new_canvas = region_crop_resized
        return new_canvas