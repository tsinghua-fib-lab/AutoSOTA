# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from easydict import EasyDict

######################### Common #########################
#------------------------ image ------------------------#
image_plain_anno = EasyDict()
image_plain_anno.NAME = "PlainImageAnnotator"
image_plain_anno.INPUTS = {"image": None}
image_plain_anno.OUTPUTS = {"image": None}

image_mask_plain_anno = EasyDict()
image_mask_plain_anno.NAME = "PlainMaskAnnotator"
image_mask_plain_anno.INPUTS = {"mask": None}
image_mask_plain_anno.OUTPUTS = {"mask": None}

image_maskaug_plain_anno = EasyDict()
image_maskaug_plain_anno.NAME = "PlainMaskAugAnnotator"
image_maskaug_plain_anno.INPUTS = {"mask": None}
image_maskaug_plain_anno.OUTPUTS = {"mask": None}

image_maskaug_invert_anno = EasyDict()
image_maskaug_invert_anno.NAME = "PlainMaskAugInvertAnnotator"
image_maskaug_invert_anno.INPUTS = {"mask": None}
image_maskaug_invert_anno.OUTPUTS = {"mask": None}

image_maskaug_anno = EasyDict()
image_maskaug_anno.NAME = "MaskAugAnnotator"
image_maskaug_anno.INPUTS = {"mask": None, 'mask_cfg': None}
image_maskaug_anno.OUTPUTS = {"mask": None}

image_mask_draw_anno = EasyDict()
image_mask_draw_anno.NAME = "MaskDrawAnnotator"
image_mask_draw_anno.INPUTS = {"mask": None, 'image': None, 'bbox': None, 'mode': None}
image_mask_draw_anno.OUTPUTS = {"mask": None}

image_maskaug_region_random_anno = EasyDict()
image_maskaug_region_random_anno.NAME = "RegionCanvasAnnotator"
image_maskaug_region_random_anno.SCALE_RANGE = [ 0.5, 1.0 ]
image_maskaug_region_random_anno.USE_AUG = True
image_maskaug_region_random_anno.INPUTS = {"mask": None, 'image': None, 'bbox': None, 'mode': None}
image_maskaug_region_random_anno.OUTPUTS = {"mask": None}

image_maskaug_region_crop_anno = EasyDict()
image_maskaug_region_crop_anno.NAME = "RegionCanvasAnnotator"
image_maskaug_region_crop_anno.SCALE_RANGE = [ 0.5, 1.0 ]
image_maskaug_region_crop_anno.USE_AUG = True
image_maskaug_region_crop_anno.USE_RESIZE = False
image_maskaug_region_crop_anno.USE_CANVAS = False
image_maskaug_region_crop_anno.INPUTS = {"mask": None, 'image': None, 'bbox': None, 'mode': None}
image_maskaug_region_crop_anno.OUTPUTS = {"mask": None}


#------------------------ video ------------------------#
video_plain_anno = EasyDict()
video_plain_anno.NAME = "PlainVideoAnnotator"
video_plain_anno.INPUTS = {"frames": None}
video_plain_anno.OUTPUTS = {"frames": None}

video_mask_plain_anno = EasyDict()
video_mask_plain_anno.NAME = "PlainMaskVideoAnnotator"
video_mask_plain_anno.INPUTS = {"masks": None}
video_mask_plain_anno.OUTPUTS = {"masks": None}

video_maskaug_plain_anno = EasyDict()
video_maskaug_plain_anno.NAME = "PlainMaskAugVideoAnnotator"
video_maskaug_plain_anno.INPUTS = {"masks": None}
video_maskaug_plain_anno.OUTPUTS = {"masks": None}

video_maskaug_invert_anno = EasyDict()
video_maskaug_invert_anno.NAME = "PlainMaskAugInvertVideoAnnotator"
video_maskaug_invert_anno.INPUTS = {"masks": None}
video_maskaug_invert_anno.OUTPUTS = {"masks": None}

video_mask_expand_anno = EasyDict()
video_mask_expand_anno.NAME = "ExpandMaskVideoAnnotator"
video_mask_expand_anno.INPUTS = {"masks": None}
video_mask_expand_anno.OUTPUTS = {"masks": None}

video_maskaug_anno = EasyDict()
video_maskaug_anno.NAME = "MaskAugAnnotator"
video_maskaug_anno.INPUTS = {"mask": None, 'mask_cfg': None}
video_maskaug_anno.OUTPUTS = {"mask": None}

video_maskaug_layout_anno = EasyDict()
video_maskaug_layout_anno.NAME = "LayoutMaskAnnotator"
video_maskaug_layout_anno.RAM_TAG_COLOR_PATH = "models/VACE-Annotators/layout/ram_tag_color_list.txt"
video_maskaug_layout_anno.USE_AUG = True
video_maskaug_layout_anno.INPUTS = {"mask": None, 'mask_cfg': None}
video_maskaug_layout_anno.OUTPUTS = {"mask": None}


#------------------------ prompt ------------------------#
prompt_plain_anno = EasyDict()
prompt_plain_anno.NAME = "PlainPromptAnnotator"
prompt_plain_anno.INPUTS = {"prompt": None}
prompt_plain_anno.OUTPUTS = {"prompt": None}