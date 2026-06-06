# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class RescueDataset(BaseSegDataset):
    METAINFO = dict(classes = ('impervious_surface', 'building',  'vegetation',
                 'car', 'clutter'),
    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]])

    def __init__(self, **kwargs):
        super(RescueDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_lab.png',
            reduce_zero_label=True,
            **kwargs)