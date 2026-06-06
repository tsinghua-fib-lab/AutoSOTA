# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from easydict import EasyDict

#------------------------ CompositionBase ------------------------#
comp_anno = EasyDict()
comp_anno.NAME = "CompositionAnnotator"
comp_anno.INPUTS = {"process_type_1": None, "process_type_2": None, "frames_1": None, "frames_2": None, "masks_1": None, "masks_2": None}
comp_anno.OUTPUTS = {"frames": None, "masks": None}

#------------------------ ReferenceAnything ------------------------#
comp_refany_anno = EasyDict()
comp_refany_anno.NAME = "ReferenceAnythingAnnotator"
comp_refany_anno.SUBJECT = {"MODE": "all", "USE_AUG": True, "USE_CROP": True, "ROI_ONLY": True,
                            "INPAINTING": {"MODE": "all",
                                           "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                            "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                      "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                      "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                            "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                     "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}}
comp_refany_anno.INPUTS = {"images": None, "mode": None, "mask_cfg": None}
comp_refany_anno.OUTPUTS = {"images": None}


#------------------------ AnimateAnything ------------------------#
comp_aniany_anno = EasyDict()
comp_aniany_anno.NAME = "AnimateAnythingAnnotator"
comp_aniany_anno.POSE = {"DETECTION_MODEL": "models/VACE-Annotators/pose/yolox_l.onnx",
                         "POSE_MODEL": "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx"}
comp_aniany_anno.REFERENCE = {"MODE": "all", "USE_AUG": True, "USE_CROP": True, "ROI_ONLY": True,
                              "INPAINTING": {"MODE": "all",
                                             "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                              "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                        "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                        "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                              "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                       "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}}
comp_aniany_anno.INPUTS = {"frames": None, "images": None, "mode": None, "mask_cfg": None}
comp_aniany_anno.OUTPUTS = {"frames": None, "images": None}


#------------------------ SwapAnything ------------------------#
comp_swapany_anno = EasyDict()
comp_swapany_anno.NAME = "SwapAnythingAnnotator"
comp_swapany_anno.REFERENCE = {"MODE": "all", "USE_AUG": True, "USE_CROP": True, "ROI_ONLY": True,
                              "INPAINTING": {"MODE": "all",
                                             "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                              "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                        "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                        "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                              "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                       "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}}
comp_swapany_anno.INPAINTING = {"MODE": "all",
                                "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                                "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                          "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                          "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                                "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                         "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}
comp_swapany_anno.INPUTS = {"frames": None, "video": None, "images": None, "mask": None, "bbox": None, "label": None, "caption": None, "mode": None, "mask_cfg": None}
comp_swapany_anno.OUTPUTS = {"frames": None, "images": None, "masks": None}



#------------------------ ExpandAnything ------------------------#
comp_expany_anno = EasyDict()
comp_expany_anno.NAME = "ExpandAnythingAnnotator"
comp_expany_anno.REFERENCE = {"MODE": "all", "USE_AUG": True, "USE_CROP": True, "ROI_ONLY": True,
                              "INPAINTING": {"MODE": "all",
                                             "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                              "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                        "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                        "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                              "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                       "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}}
comp_expany_anno.OUTPAINTING = {"RETURN_MASK": True, "KEEP_PADDING_RATIO": 1, "MASK_COLOR": "gray"}
comp_expany_anno.FRAMEREF = {}
comp_expany_anno.INPUTS = {"images": None, "mode": None, "mask_cfg": None, "direction": None, "expand_ratio": None, "expand_num": None}
comp_expany_anno.OUTPUTS = {"frames": None, "images": None, "masks": None}


#------------------------ MoveAnything ------------------------#
comp_moveany_anno = EasyDict()
comp_moveany_anno.NAME = "MoveAnythingAnnotator"
comp_moveany_anno.LAYOUTBBOX = {"RAM_TAG_COLOR_PATH": "models/VACE-Annotators/layout/ram_tag_color_list.txt"}
comp_moveany_anno.INPUTS = {"image": None, "bbox": None, "label": None, "expand_num": None}
comp_moveany_anno.OUTPUTS = {"frames": None, "masks": None}