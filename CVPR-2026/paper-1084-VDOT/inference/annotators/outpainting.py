# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import random
from abc import ABCMeta

import numpy as np
import torch
from PIL import Image, ImageDraw
from .utils import convert_to_pil, get_mask_box


class OutpaintingAnnotator:
    def __init__(self, cfg, device=None):
        self.mask_blur = cfg.get('MASK_BLUR', 0)
        self.random_cfg = cfg.get('RANDOM_CFG', None)
        self.return_mask = cfg.get('RETURN_MASK', False)
        self.return_source = cfg.get('RETURN_SOURCE', True)
        self.keep_padding_ratio = cfg.get('KEEP_PADDING_RATIO', 8)
        self.mask_color = cfg.get('MASK_COLOR', 0)

    def forward(self,
                image,
                expand_ratio=0.3,
                mask=None,
                direction=['left', 'right', 'up', 'down'],
                return_mask=None,
                return_source=None,
                mask_color=None):
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = return_source if return_source is not None else self.return_source
        mask_color = mask_color if mask_color is not None else self.mask_color
        image = convert_to_pil(image)
        if self.random_cfg:
            direction_range = self.random_cfg.get(
                'DIRECTION_RANGE', ['left', 'right', 'up', 'down'])
            ratio_range = self.random_cfg.get('RATIO_RANGE', [0.0, 1.0])
            direction = random.sample(
                direction_range,
                random.choice(list(range(1,
                                         len(direction_range) + 1))))
            expand_ratio = random.uniform(ratio_range[0], ratio_range[1])

        if mask is None:
            init_image = image
            src_width, src_height = init_image.width, init_image.height
            left = int(expand_ratio * src_width) if 'left' in direction else 0
            right = int(expand_ratio * src_width) if 'right' in direction else 0
            up = int(expand_ratio * src_height) if 'up' in direction else 0
            down = int(expand_ratio * src_height) if 'down' in direction else 0
            tar_width = math.ceil(
                (src_width + left + right) /
                self.keep_padding_ratio) * self.keep_padding_ratio
            tar_height = math.ceil(
                (src_height + up + down) /
                self.keep_padding_ratio) * self.keep_padding_ratio
            if left > 0:
                left = left * (tar_width - src_width) // (left + right)
            if right > 0:
                right = tar_width - src_width - left
            if up > 0:
                up = up * (tar_height - src_height) // (up + down)
            if down > 0:
                down = tar_height - src_height - up
            if mask_color is not None:
                img = Image.new('RGB', (tar_width, tar_height),
                                color=mask_color)
            else:
                img = Image.new('RGB', (tar_width, tar_height))
            img.paste(init_image, (left, up))
            mask = Image.new('L', (img.width, img.height), 'white')
            draw = ImageDraw.Draw(mask)

            draw.rectangle(
                (left + (self.mask_blur * 2 if left > 0 else 0), up +
                 (self.mask_blur * 2 if up > 0 else 0), mask.width - right -
                 (self.mask_blur * 2 if right > 0 else 0) - 1, mask.height - down -
                 (self.mask_blur * 2 if down > 0 else 0) - 1),
                fill='black')
        else:
            bbox = get_mask_box(np.array(mask))
            if bbox is None:
                img = image
                mask = mask
                init_image = image
            else:
                mask = Image.new('L', (image.width, image.height), 'white')
                mask_zero = Image.new('L',
                                      (bbox[2] - bbox[0], bbox[3] - bbox[1]),
                                      'black')
                mask.paste(mask_zero, (bbox[0], bbox[1]))
                crop_image = image.crop(bbox)
                init_image = Image.new('RGB', (image.width, image.height),
                                       'black')
                init_image.paste(crop_image, (bbox[0], bbox[1]))
                img = image
        if return_mask:
            if return_source:
                ret_data = {
                    'src_image': np.array(init_image),
                    'image': np.array(img),
                    'mask': np.array(mask)
                }
            else:
                ret_data = {'image': np.array(img), 'mask': np.array(mask)}
        else:
            if return_source:
                ret_data = {
                    'src_image': np.array(init_image),
                    'image': np.array(img)
                }
            else:
                ret_data = np.array(img)
        return ret_data



class OutpaintingInnerAnnotator:
    def __init__(self, cfg, device=None):
        self.mask_blur = cfg.get('MASK_BLUR', 0)
        self.random_cfg = cfg.get('RANDOM_CFG', None)
        self.return_mask = cfg.get('RETURN_MASK', False)
        self.return_source = cfg.get('RETURN_SOURCE', True)
        self.keep_padding_ratio = cfg.get('KEEP_PADDING_RATIO', 8)
        self.mask_color = cfg.get('MASK_COLOR', 0)

    def forward(self,
                image,
                expand_ratio=0.3,
                direction=['left', 'right', 'up', 'down'],
                return_mask=None,
                return_source=None,
                mask_color=None):
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = return_source if return_source is not None else self.return_source
        mask_color = mask_color if mask_color is not None else self.mask_color
        image = convert_to_pil(image)
        if self.random_cfg:
            direction_range = self.random_cfg.get(
                'DIRECTION_RANGE', ['left', 'right', 'up', 'down'])
            ratio_range = self.random_cfg.get('RATIO_RANGE', [0.0, 1.0])
            direction = random.sample(
                direction_range,
                random.choice(list(range(1,
                                         len(direction_range) + 1))))
            expand_ratio = random.uniform(ratio_range[0], ratio_range[1])

        init_image = image
        src_width, src_height = init_image.width, init_image.height
        left = int(expand_ratio * src_width) if 'left' in direction else 0
        right = int(expand_ratio * src_width) if 'right' in direction else 0
        up = int(expand_ratio * src_height) if 'up' in direction else 0
        down = int(expand_ratio * src_height) if 'down' in direction else 0

        crop_left = left
        crop_right = src_width - right
        crop_up = up
        crop_down = src_height - down
        crop_box = (crop_left, crop_up, crop_right, crop_down)
        cropped_image = init_image.crop(crop_box)
        if mask_color is not None:
            img = Image.new('RGB', (src_width, src_height), color=mask_color)
        else:
            img = Image.new('RGB', (src_width, src_height))

        paste_x = left
        paste_y = up
        img.paste(cropped_image, (paste_x, paste_y))

        mask = Image.new('L', (img.width, img.height), 'white')
        draw = ImageDraw.Draw(mask)

        x0 = paste_x + (self.mask_blur * 2 if left > 0 else 0)
        y0 = paste_y + (self.mask_blur * 2 if up > 0 else 0)
        x1 = paste_x + cropped_image.width - (self.mask_blur * 2 if right > 0 else 0)
        y1 = paste_y + cropped_image.height - (self.mask_blur * 2 if down > 0 else 0)
        draw.rectangle((x0, y0, x1, y1), fill='black')

        if return_mask:
            if return_source:
                ret_data = {
                    'src_image': np.array(init_image),
                    'image': np.array(img),
                    'mask': np.array(mask)
                }
            else:
                ret_data = {'image': np.array(img), 'mask': np.array(mask)}
        else:
            if return_source:
                ret_data = {
                    'src_image': np.array(init_image),
                    'image': np.array(img)
                }
            else:
                ret_data = np.array(img)
        return ret_data





class OutpaintingVideoAnnotator(OutpaintingAnnotator):

    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.key_map = {
            "src_image": "src_images",
            "image" : "frames",
            "mask": "masks"
        }

    def forward(self, frames,
                      expand_ratio=0.3,
                      mask=None,
                      direction=['left', 'right', 'up', 'down'],
                      return_mask=None,
                      return_source=None,
                      mask_color=None):
        ret_frames = None
        for frame in frames:
            anno_frame = super().forward(frame, expand_ratio=expand_ratio, mask=mask, direction=direction, return_mask=return_mask, return_source=return_source, mask_color=mask_color)
            if isinstance(anno_frame, dict):
                ret_frames = {} if ret_frames is None else ret_frames
                for key, val in anno_frame.items():
                    new_key = self.key_map[key]
                    if new_key in ret_frames:
                        ret_frames[new_key].append(val)
                    else:
                        ret_frames[new_key] = [val]
            else:
                ret_frames = [] if ret_frames is None else ret_frames
                ret_frames.append(anno_frame)
        return ret_frames


class OutpaintingInnerVideoAnnotator(OutpaintingInnerAnnotator):

    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.key_map = {
            "src_image": "src_images",
            "image" : "frames",
            "mask": "masks"
        }

    def forward(self, frames,
                      expand_ratio=0.3,
                      direction=['left', 'right', 'up', 'down'],
                      return_mask=None,
                      return_source=None,
                      mask_color=None):
        ret_frames = None
        for frame in frames:
            anno_frame = super().forward(frame, expand_ratio=expand_ratio, direction=direction, return_mask=return_mask, return_source=return_source, mask_color=mask_color)
            if isinstance(anno_frame, dict):
                ret_frames = {} if ret_frames is None else ret_frames
                for key, val in anno_frame.items():
                    new_key = self.key_map[key]
                    if new_key in ret_frames:
                        ret_frames[new_key].append(val)
                    else:
                        ret_frames[new_key] = [val]
            else:
                ret_frames = [] if ret_frames is None else ret_frames
                ret_frames.append(anno_frame)
        return ret_frames
