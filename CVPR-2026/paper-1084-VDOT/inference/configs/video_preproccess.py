# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from easydict import EasyDict


######################### V2V - Control #########################
#------------------------ Depth ------------------------#
video_depth_anno = EasyDict()
video_depth_anno.NAME = "DepthVideoAnnotator"
video_depth_anno.PRETRAINED_MODEL = "models/VACE-Annotators/depth/dpt_hybrid-midas-501f0c75.pt"
video_depth_anno.INPUTS = {"frames": None}
video_depth_anno.OUTPUTS = {"frames": None}

#------------------------ Depth ------------------------#
video_depthv2_anno = EasyDict()
video_depthv2_anno.NAME = "DepthV2VideoAnnotator"
video_depthv2_anno.PRETRAINED_MODEL = "models/VACE-Annotators/depth/depth_anything_v2_vitl.pth"
video_depthv2_anno.INPUTS = {"frames": None}
video_depthv2_anno.OUTPUTS = {"frames": None}

#------------------------ Flow ------------------------#
video_flow_anno = EasyDict()
video_flow_anno.NAME = "FlowVisAnnotator"
video_flow_anno.PRETRAINED_MODEL = "models/VACE-Annotators/flow/raft-things.pth"
video_flow_anno.INPUTS = {"frames": None}
video_flow_anno.OUTPUTS = {"frames": None}

#------------------------ Gray ------------------------#
video_gray_anno = EasyDict()
video_gray_anno.NAME = "GrayVideoAnnotator"
video_gray_anno.INPUTS = {"frames": None}
video_gray_anno.OUTPUTS = {"frames": None}

#------------------------ Pose ------------------------#
video_pose_anno = EasyDict()
video_pose_anno.NAME = "PoseBodyFaceVideoAnnotator"
video_pose_anno.DETECTION_MODEL = "models/VACE-Annotators/pose/yolox_l.onnx"
video_pose_anno.POSE_MODEL = "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx"
video_pose_anno.INPUTS = {"frames": None}
video_pose_anno.OUTPUTS = {"frames": None}

#------------------------ Pose ------------------------#
video_pose_body_anno = EasyDict()
video_pose_body_anno.NAME = "PoseBodyVideoAnnotator"
video_pose_body_anno.DETECTION_MODEL = "models/VACE-Annotators/pose/yolox_l.onnx"
video_pose_body_anno.POSE_MODEL = "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx"
video_pose_body_anno.INPUTS = {"frames": None}
video_pose_body_anno.OUTPUTS = {"frames": None}

#------------------------ Scribble ------------------------#
video_scribble_anno = EasyDict()
video_scribble_anno.NAME = "ScribbleVideoAnnotator"
video_scribble_anno.PRETRAINED_MODEL = "models/VACE-Annotators/scribble/anime_style/netG_A_latest.pth"
video_scribble_anno.INPUTS = {"frames": None}
video_scribble_anno.OUTPUTS = {"frames": None}


######################### R2V/MV2V - Extension #########################
# The 'mode' can be selected from options "firstframe", "lastframe", "firstlastframe"(needs image_2), "firstclip", "lastclip", "firstlastclip"(needs frames_2).
# "frames" refers to processing a video clip; 'image' refers to processing a single image.
# #------------------------ FrameRefExtract ------------------------#
video_framerefext_anno = EasyDict()
video_framerefext_anno.NAME = "FrameRefExtractAnnotator"
video_framerefext_anno.INPUTS = {"frames": None, "ref_cfg": None, "ref_num": None}
video_framerefext_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ FrameRefExp ------------------------#
video_framerefexp_anno = EasyDict()
video_framerefexp_anno.NAME = "FrameRefExpandAnnotator"
video_framerefexp_anno.INPUTS = {"image": None, "image_2": None, "mode": None, "expand_num": 80}
video_framerefexp_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ FrameRefExp ------------------------#
video_cliprefexp_anno = EasyDict()
video_cliprefexp_anno.NAME = "FrameRefExpandAnnotator"
video_cliprefexp_anno.INPUTS = {"frames": None, "frames_2": None, "mode": None, "expand_num": 80}
video_cliprefexp_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ FirstFrameRef ------------------------#
video_firstframeref_anno = EasyDict()
video_firstframeref_anno.NAME = "FrameRefExpandAnnotator"
video_firstframeref_anno.MODE = "firstframe"
video_firstframeref_anno.INPUTS = {"image": None, "mode": "firstframe", "expand_num": 80}
video_firstframeref_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ LastFrameRef ------------------------#
video_lastframeref_anno = EasyDict()
video_lastframeref_anno.NAME = "FrameRefExpandAnnotator"
video_lastframeref_anno.MODE = "lastframe"
video_lastframeref_anno.INPUTS = {"image": None, "mode": "lastframe", "expand_num": 80}
video_lastframeref_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ FirstlastFrameRef ------------------------#
video_firstlastframeref_anno = EasyDict()
video_firstlastframeref_anno.NAME = "FrameRefExpandAnnotator"
video_firstlastframeref_anno.MODE = "firstlastframe"
video_firstlastframeref_anno.INPUTS = {"image": None, "image_2": None, "mode": "firstlastframe", "expand_num": 80}
video_firstlastframeref_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ FirstClipRef ------------------------#
video_firstclipref_anno = EasyDict()
video_firstclipref_anno.NAME = "FrameRefExpandAnnotator"
video_firstclipref_anno.MODE = "firstclip"
video_firstclipref_anno.INPUTS = {"frames": None, "mode": "firstclip", "expand_num": 80}
video_firstclipref_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ LastClipRef ------------------------#
video_lastclipref_anno = EasyDict()
video_lastclipref_anno.NAME = "FrameRefExpandAnnotator"
video_lastclipref_anno.MODE = "lastclip"
video_lastclipref_anno.INPUTS = {"frames": None, "mode": "lastclip", "expand_num": 80}
video_lastclipref_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ FirstlastClipRef ------------------------#
video_firstlastclipref_anno = EasyDict()
video_firstlastclipref_anno.NAME = "FrameRefExpandAnnotator"
video_firstlastclipref_anno.MODE = "firstlastclip"
video_firstlastclipref_anno.INPUTS = {"frames": None, "frames_2": None, "mode": "firstlastclip", "expand_num": 80}
video_firstlastclipref_anno.OUTPUTS = {"frames": None, "masks": None}



######################### MV2V - Repaint - Inpainting #########################
#------------------------ Inpainting ------------------------#
video_inpainting_anno = EasyDict()
video_inpainting_anno.NAME = "InpaintingVideoAnnotator"
video_inpainting_anno.MODE = "all"
video_inpainting_anno.SALIENT = {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"}
video_inpainting_anno.GDINO = {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                               "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                               "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"}
video_inpainting_anno.SAM2 = {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                              "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}
video_inpainting_anno.INPUTS = {"frames": None, "video": None, "mask": None, "bbox": None, "label": None, "caption": None, "mode": None}
video_inpainting_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ InpaintingMask ------------------------#
video_inpainting_mask_anno = EasyDict()
video_inpainting_mask_anno.NAME = "InpaintingVideoAnnotator"
video_inpainting_mask_anno.MODE = "mask"
video_inpainting_mask_anno.INPUTS = {"frames": None, "mask": None, "mode": "mask"}
video_inpainting_mask_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ InpaintingBbox ------------------------#
video_inpainting_bbox_anno = EasyDict()
video_inpainting_bbox_anno.NAME = "InpaintingVideoAnnotator"
video_inpainting_bbox_anno.MODE = "bbox"
video_inpainting_bbox_anno.INPUTS = {"frames": None, "bbox": None, "mode": "bbox"}
video_inpainting_bbox_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ InpaintingMasktrack ------------------------#
video_inpainting_masktrack_anno = EasyDict()
video_inpainting_masktrack_anno.NAME = "InpaintingVideoAnnotator"
video_inpainting_masktrack_anno.MODE = "masktrack"
video_inpainting_masktrack_anno.SAM2 = {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                        "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}
video_inpainting_masktrack_anno.INPUTS = {"video": None, "mask": None, "mode": "masktrack"}
video_inpainting_masktrack_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ InpaintingBboxtrack ------------------------#
video_inpainting_bboxtrack_anno = EasyDict()
video_inpainting_bboxtrack_anno.NAME = "InpaintingVideoAnnotator"
video_inpainting_bboxtrack_anno.MODE = "bboxtrack"
video_inpainting_bboxtrack_anno.SAM2 = {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                        "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}
video_inpainting_bboxtrack_anno.INPUTS = {"video": None, "bbox": None, "mode": "bboxtrack"}
video_inpainting_bboxtrack_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ InpaintingLabel ------------------------#
video_inpainting_label_anno = EasyDict()
video_inpainting_label_anno.NAME = "InpaintingVideoAnnotator"
video_inpainting_label_anno.MODE = "label"
video_inpainting_label_anno.GDINO = {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                     "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                     "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"}
video_inpainting_label_anno.SAM2 = {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                    "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}
video_inpainting_label_anno.INPUTS = {"video": None, "label": None, "mode": "label"}
video_inpainting_label_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ InpaintingCaption ------------------------#
video_inpainting_caption_anno = EasyDict()
video_inpainting_caption_anno.NAME = "InpaintingVideoAnnotator"
video_inpainting_caption_anno.MODE = "caption"
video_inpainting_caption_anno.GDINO = {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                     "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                     "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"}
video_inpainting_caption_anno.SAM2 = {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                    "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}
video_inpainting_caption_anno.INPUTS = {"video": None, "caption": None, "mode": "caption"}
video_inpainting_caption_anno.OUTPUTS = {"frames": None, "masks": None}


######################### MV2V - Repaint - Outpainting #########################
#------------------------ Outpainting ------------------------#
# The 'direction' can be selected from options "left", "right", "up", "down".
video_outpainting_anno = EasyDict()
video_outpainting_anno.NAME = "OutpaintingVideoAnnotator"
video_outpainting_anno.RETURN_MASK = True
video_outpainting_anno.KEEP_PADDING_RATIO = 1
video_outpainting_anno.MASK_COLOR = 'gray'
video_outpainting_anno.INPUTS = {"frames": None, "direction": ['left', 'right'], 'expand_ratio': 0.25}
video_outpainting_anno.OUTPUTS = {"frames": None, "masks": None}

video_outpainting_inner_anno = EasyDict()
video_outpainting_inner_anno.NAME = "OutpaintingInnerVideoAnnotator"
video_outpainting_inner_anno.RETURN_MASK = True
video_outpainting_inner_anno.KEEP_PADDING_RATIO = 1
video_outpainting_inner_anno.MASK_COLOR = 'gray'
video_outpainting_inner_anno.INPUTS = {"frames": None, "direction": ['left', 'right'], 'expand_ratio': 0.25}
video_outpainting_inner_anno.OUTPUTS = {"frames": None, "masks": None}



######################### V2V - Control - Motion #########################
#------------------------ LayoutBbox ------------------------#
video_layout_bbox_anno = EasyDict()
video_layout_bbox_anno.NAME = "LayoutBboxAnnotator"
video_layout_bbox_anno.FRAME_SIZE = [720, 1280]  # [H, W]
video_layout_bbox_anno.NUM_FRAMES = 81
video_layout_bbox_anno.RAM_TAG_COLOR_PATH = "models/VACE-Annotators/layout/ram_tag_color_list.txt"
video_layout_bbox_anno.INPUTS = {'bbox': None, 'label': None}  # label is optional
video_layout_bbox_anno.OUTPUTS = {"frames": None}

#------------------------ LayoutTrack ------------------------#
video_layout_track_anno = EasyDict()
video_layout_track_anno.NAME = "LayoutTrackAnnotator"
video_layout_track_anno.USE_AUG = True  # ['original', 'original_expand', 'hull', 'hull_expand', 'bbox', 'bbox_expand']
video_layout_track_anno.RAM_TAG_COLOR_PATH = "models/VACE-Annotators/layout/ram_tag_color_list.txt"
video_layout_track_anno.INPAINTING = {"MODE": "all",
                                      "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                                      "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                                "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                                "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                                      "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                               "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}
# video_layout_track_anno.INPUTS = {"video": None, 'label': None, "mask": None, "mode": "masktrack"}
# video_layout_track_anno.INPUTS = {"video": None, "label": None, "bbox": None, "mode": "bboxtrack", "mask_cfg": {"mode": "hull"}}
# video_layout_track_anno.INPUTS = {"video": None, "label": None, "mode": "label", "mask_cfg": {"mode": "bbox_expand", "kwargs": {'expand_ratio': 0.2, 'expand_iters': 5}}}
# video_layout_track_anno.INPUTS = {"video": None, "label": None, "caption": None, "mode": "caption", "mask_cfg": {"mode": "original_expand", "kwargs": {'expand_ratio': 0.2, 'expand_iters': 5}}}
video_layout_track_anno.INPUTS = {"video": None, 'label': None, "mode": None, "mask": None, "bbox": None, "caption": None, "mask_cfg": None}
video_layout_track_anno.OUTPUTS = {"frames": None}