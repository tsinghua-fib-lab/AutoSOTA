# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from scipy import ndimage

from .utils import convert_to_numpy


class MaskDrawAnnotator:
    def __init__(self, cfg, device=None):
        self.mode = cfg.get('MODE', 'maskpoint')
        self.return_dict = cfg.get('RETURN_DICT', True)
        assert self.mode in ['maskpoint', 'maskbbox', 'mask', 'bbox']

    def forward(self,
                mask=None,
                image=None,
                bbox=None,
                mode=None,
                return_dict=None):
        mode = mode if mode is not None else self.mode
        return_dict = return_dict if return_dict is not None else self.return_dict

        mask = convert_to_numpy(mask) if mask is not None else None
        image = convert_to_numpy(image) if image is not None else None

        mask_shape = mask.shape
        if mode == 'maskpoint':
            scribble = mask.transpose(1, 0)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            out_mask = np.zeros(mask_shape, dtype=np.uint8)
            hull = ConvexHull(centers)
            hull_vertices = centers[hull.vertices]
            rr, cc = polygon(hull_vertices[:, 1], hull_vertices[:, 0], mask_shape)
            out_mask[rr, cc] = 255
        elif mode == 'maskbbox':
            scribble = mask.transpose(1, 0)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            out_mask = np.zeros(mask_shape, dtype=np.uint8)
            out_mask[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1] = 255
            if image is not None:
                out_image = image[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1]
        elif mode == 'bbox':
            if isinstance(bbox, list):
                bbox = np.array(bbox)
            x_min, y_min, x_max, y_max = bbox
            out_mask = np.zeros(mask_shape, dtype=np.uint8)
            out_mask[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1] = 255
            if image is not None:
                out_image = image[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1]
        elif mode == 'mask':
            out_mask = mask
        else:
            raise NotImplementedError

        if return_dict:
            if image is not None:
                return {"image": out_image, "mask": out_mask}
            else:
                return {"mask": out_mask}
        else:
            if image is not None:
                return out_image, out_mask
            else:
                return out_mask