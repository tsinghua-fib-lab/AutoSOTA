# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.


import random
from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .utils import convert_to_numpy



class MaskAugAnnotator:
    def __init__(self, cfg, device=None):
        # original / original_expand / hull / hull_expand / bbox / bbox_expand
        self.mask_cfg = cfg.get('MASK_CFG', [{"mode": "original", "proba": 0.1},
                                             {"mode": "original_expand", "proba": 0.1},
                                             {"mode": "hull", "proba": 0.1},
                                             {"mode": "hull_expand", "proba":0.1, "kwargs": {"expand_ratio": 0.2}},
                                             {"mode": "bbox", "proba": 0.1},
                                             {"mode": "bbox_expand", "proba": 0.1, "kwargs": {"min_expand_ratio": 0.2, "max_expand_ratio": 0.5}}])

    def forward(self, mask, mask_cfg=None):
        mask_cfg = mask_cfg if mask_cfg is not None else self.mask_cfg
        if not isinstance(mask, list):
            is_batch = False
            masks = [mask]
        else:
            is_batch = True
            masks = mask

        mask_func = self.get_mask_func(mask_cfg)
        # print(mask_func)
        aug_masks = []
        for submask in masks:
            mask = convert_to_numpy(submask)
            valid, large, h, w, bbox = self.get_mask_info(mask)
            # print(valid, large, h, w, bbox)
            if valid:
                mask = mask_func(mask, bbox, h, w)
            else:
                mask = mask.astype(np.uint8)
            aug_masks.append(mask)
        return  aug_masks if is_batch else aug_masks[0]

    def get_mask_info(self, mask):
        h, w = mask.shape
        locs = mask.nonzero()
        valid = True
        if len(locs) < 1 or locs[0].shape[0] < 1 or locs[1].shape[0] < 1:
            valid = False
            return valid, False, h, w, [0, 0, 0, 0]

        left, right = np.min(locs[1]), np.max(locs[1])
        top, bottom = np.min(locs[0]), np.max(locs[0])
        bbox = [left, top, right, bottom]

        large = False
        if (right - left + 1) * (bottom - top + 1) > 0.9 * h * w:
            large = True
        return valid, large, h, w, bbox

    def get_expand_params(self, mask_kwargs):
        if 'expand_ratio' in mask_kwargs:
            expand_ratio = mask_kwargs['expand_ratio']
        elif 'min_expand_ratio' in mask_kwargs and 'max_expand_ratio' in mask_kwargs:
            expand_ratio = random.uniform(mask_kwargs['min_expand_ratio'], mask_kwargs['max_expand_ratio'])
        else:
            expand_ratio = 0.3

        if 'expand_iters' in mask_kwargs:
            expand_iters = mask_kwargs['expand_iters']
        else:
            expand_iters = random.randint(1, 10)

        if 'expand_lrtp' in mask_kwargs:
            expand_lrtp = mask_kwargs['expand_lrtp']
        else:
            expand_lrtp = [random.random(), random.random(), random.random(), random.random()]

        return expand_ratio, expand_iters, expand_lrtp

    def get_mask_func(self, mask_cfg):
        if not isinstance(mask_cfg, list):
            mask_cfg = [mask_cfg]
        probas = [item['proba'] if 'proba' in item else 1.0 / len(mask_cfg) for item in mask_cfg]
        sel_mask_cfg = random.choices(mask_cfg, weights=probas, k=1)[0]
        mode = sel_mask_cfg['mode'] if 'mode' in sel_mask_cfg else 'original'
        mask_kwargs = sel_mask_cfg['kwargs'] if 'kwargs' in sel_mask_cfg else {}

        if mode == 'random':
            mode = random.choice(['original', 'original_expand', 'hull', 'hull_expand', 'bbox', 'bbox_expand'])
        if mode == 'original':
            mask_func = partial(self.generate_mask)
        elif mode == 'original_expand':
            expand_ratio, expand_iters, expand_lrtp = self.get_expand_params(mask_kwargs)
            mask_func = partial(self.generate_mask, expand_ratio=expand_ratio, expand_iters=expand_iters, expand_lrtp=expand_lrtp)
        elif mode == 'hull':
            clockwise = random.choice([True, False]) if 'clockwise' not in mask_kwargs else mask_kwargs['clockwise']
            mask_func = partial(self.generate_hull_mask, clockwise=clockwise)
        elif mode == 'hull_expand':
            expand_ratio, expand_iters, expand_lrtp = self.get_expand_params(mask_kwargs)
            clockwise = random.choice([True, False]) if 'clockwise' not in mask_kwargs else mask_kwargs['clockwise']
            mask_func = partial(self.generate_hull_mask, clockwise=clockwise, expand_ratio=expand_ratio, expand_iters=expand_iters, expand_lrtp=expand_lrtp)
        elif mode == 'bbox':
            mask_func = partial(self.generate_bbox_mask)
        elif mode == 'bbox_expand':
            expand_ratio, expand_iters, expand_lrtp = self.get_expand_params(mask_kwargs)
            mask_func = partial(self.generate_bbox_mask, expand_ratio=expand_ratio, expand_iters=expand_iters, expand_lrtp=expand_lrtp)
        else:
            raise NotImplementedError
        return mask_func


    def generate_mask(self, mask, bbox, h, w, expand_ratio=None, expand_iters=None, expand_lrtp=None):
        bin_mask = mask.astype(np.uint8)
        if expand_ratio:
            bin_mask = self.rand_expand_mask(bin_mask, bbox, h, w, expand_ratio, expand_iters, expand_lrtp)
        return bin_mask


    @staticmethod
    def rand_expand_mask(mask, bbox, h, w, expand_ratio=None, expand_iters=None, expand_lrtp=None):
        expand_ratio = 0.3 if expand_ratio is None else expand_ratio
        expand_iters = random.randint(1, 10) if expand_iters is None else expand_iters
        expand_lrtp = [random.random(), random.random(), random.random(), random.random()] if expand_lrtp is None else expand_lrtp
        # print('iters', expand_iters, 'expand_ratio', expand_ratio, 'expand_lrtp', expand_lrtp)
        # mask = np.squeeze(mask)
        left, top, right, bottom = bbox
        # mask expansion
        box_w = (right - left + 1) * expand_ratio
        box_h = (bottom - top + 1) * expand_ratio
        left_, right_ = int(expand_lrtp[0] * min(box_w, left / 2) / expand_iters), int(
            expand_lrtp[1] * min(box_w, (w - right) / 2) / expand_iters)
        top_, bottom_ = int(expand_lrtp[2] * min(box_h, top / 2) / expand_iters), int(
            expand_lrtp[3] * min(box_h, (h - bottom) / 2) / expand_iters)
        kernel_size = max(left_, right_, top_, bottom_)
        if kernel_size > 0:
            kernel = np.zeros((kernel_size * 2, kernel_size * 2), dtype=np.uint8)
            new_left, new_right = kernel_size - right_, kernel_size + left_
            new_top, new_bottom = kernel_size - bottom_, kernel_size + top_
            kernel[new_top:new_bottom + 1, new_left:new_right + 1] = 1
            mask = mask.astype(np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=expand_iters).astype(np.uint8)
            # mask = new_mask - (mask / 2).astype(np.uint8)
        # mask = np.expand_dims(mask, axis=-1)
        return mask


    @staticmethod
    def _convexhull(image, clockwise):
        contours, hierarchy = cv2.findContours(image, 2, 1)
        cnt = np.concatenate(contours)  # merge all regions
        hull = cv2.convexHull(cnt, clockwise=clockwise)
        hull = np.squeeze(hull, axis=1).astype(np.float32).tolist()
        hull = [tuple(x) for x in hull]
        return hull  # b, 1, 2

    def generate_hull_mask(self, mask, bbox, h, w, clockwise=None, expand_ratio=None, expand_iters=None, expand_lrtp=None):
        clockwise = random.choice([True, False]) if clockwise is None else clockwise
        hull = self._convexhull(mask, clockwise)
        mask_img = Image.new('L', (w, h), 0)
        pt_list = hull
        mask_img_draw = ImageDraw.Draw(mask_img)
        mask_img_draw.polygon(pt_list, fill=255)
        bin_mask = np.array(mask_img).astype(np.uint8)
        if expand_ratio:
            bin_mask = self.rand_expand_mask(bin_mask, bbox, h, w, expand_ratio, expand_iters, expand_lrtp)
        return bin_mask


    def generate_bbox_mask(self, mask, bbox, h, w, expand_ratio=None, expand_iters=None, expand_lrtp=None):
        left, top, right, bottom = bbox
        bin_mask = np.zeros((h, w), dtype=np.uint8)
        bin_mask[top:bottom + 1, left:right + 1] = 255
        if expand_ratio:
            bin_mask = self.rand_expand_mask(bin_mask, bbox, h, w, expand_ratio, expand_iters, expand_lrtp)
        return bin_mask