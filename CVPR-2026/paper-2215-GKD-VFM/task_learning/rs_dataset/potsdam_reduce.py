# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class PotsdamReduceDataset(BaseSegDataset):
    METAINFO = dict(classes = ('impervious_surface', 'building', 'low_vegetation', 'tree',
                 'car', 'clutter'),
    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]])

    def __init__(self, **kwargs):
        super(PotsdamReduceDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)