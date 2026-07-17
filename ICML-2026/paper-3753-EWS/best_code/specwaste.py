# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SpectralWasteDataset(CustomDataset):
    """SpectralWasteDataset dataset.

    In SpectralWasteDataset, 
    background is included in 7 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        CLASSES=("background", "film", "basket", "cardboard", "video_tape", "filament", "bag"),
        PALETTE=[[0, 0, 0], [218, 247, 6], [51, 221, 255], [52, 50, 221], [202, 152, 195], [0, 128, 0], [255, 165, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
