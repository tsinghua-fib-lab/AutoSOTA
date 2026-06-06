# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from easydict import EasyDict

######################### Control #########################
#------------------------ Depth ------------------------#
image_depth_anno = EasyDict()
image_depth_anno.NAME = "DepthAnnotator"
image_depth_anno.PRETRAINED_MODEL = "models/VACE-Annotators/depth/dpt_hybrid-midas-501f0c75.pt"
image_depth_anno.INPUTS = {"image": None}
image_depth_anno.OUTPUTS = {"image": None}

#------------------------ Depth ------------------------#
image_depthv2_anno = EasyDict()
image_depthv2_anno.NAME = "DepthV2Annotator"
image_depthv2_anno.PRETRAINED_MODEL = "models/VACE-Annotators/depth/depth_anything_v2_vitl.pth"
image_depthv2_anno.INPUTS = {"image": None}
image_depthv2_anno.OUTPUTS = {"image": None}

#------------------------ Gray ------------------------#
image_gray_anno = EasyDict()
image_gray_anno.NAME = "GrayAnnotator"
image_gray_anno.INPUTS = {"image": None}
image_gray_anno.OUTPUTS = {"image": None}

#------------------------ Pose ------------------------#
image_pose_anno = EasyDict()
image_pose_anno.NAME = "PoseBodyFaceAnnotator"
image_pose_anno.DETECTION_MODEL = "models/VACE-Annotators/pose/yolox_l.onnx"
image_pose_anno.POSE_MODEL = "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx"
image_pose_anno.INPUTS = {"image": None}
image_pose_anno.OUTPUTS = {"image": None}

#------------------------ Scribble ------------------------#
image_scribble_anno = EasyDict()
image_scribble_anno.NAME = "ScribbleAnnotator"
image_scribble_anno.PRETRAINED_MODEL = "models/VACE-Annotators/scribble/anime_style/netG_A_latest.pth"
image_scribble_anno.INPUTS = {"image": None}
image_scribble_anno.OUTPUTS = {"image": None}

#------------------------ Outpainting ------------------------#
image_outpainting_anno = EasyDict()
image_outpainting_anno.NAME = "OutpaintingAnnotator"
image_outpainting_anno.RETURN_MASK = True
image_outpainting_anno.KEEP_PADDING_RATIO = 1
image_outpainting_anno.MASK_COLOR = 'gray'
image_outpainting_anno.INPUTS = {"image": None, "direction": ['left', 'right'], 'expand_ratio': 0.25}
image_outpainting_anno.OUTPUTS = {"image": None, "mask": None}




######################### R2V - Subject #########################
#------------------------ Face ------------------------#
image_face_anno = EasyDict()
image_face_anno.NAME = "FaceAnnotator"
image_face_anno.MODEL_NAME = "antelopev2"
image_face_anno.PRETRAINED_MODEL = "models/VACE-Annotators/face/"
image_face_anno.RETURN_RAW = False
image_face_anno.MULTI_FACE = False
image_face_anno.INPUTS = {"image": None}
image_face_anno.OUTPUTS = {"image": None}

#------------------------ FaceMask ------------------------#
image_face_mask_anno = EasyDict()
image_face_mask_anno.NAME = "FaceAnnotator"
image_face_mask_anno.MODEL_NAME = "antelopev2"
image_face_mask_anno.PRETRAINED_MODEL = "models/VACE-Annotators/face/"
image_face_mask_anno.MULTI_FACE = False
image_face_mask_anno.RETURN_RAW = False
image_face_mask_anno.RETURN_DICT = True
image_face_mask_anno.RETURN_MASK = True
image_face_mask_anno.INPUTS = {"image": None}
image_face_mask_anno.OUTPUTS = {"image": None, "mask": None}

#------------------------ Salient ------------------------#
image_salient_anno = EasyDict()
image_salient_anno.NAME = "SalientAnnotator"
image_salient_anno.NORM_SIZE = [320, 320]
image_salient_anno.RETURN_IMAGE = True
image_salient_anno.USE_CROP = True
image_salient_anno.PRETRAINED_MODEL = "models/VACE-Annotators/salient/u2net.pt"
image_salient_anno.INPUTS = {"image": None}
image_salient_anno.OUTPUTS = {"image": None}

#------------------------ Inpainting ------------------------#
image_inpainting_anno = EasyDict()
image_inpainting_anno.NAME = "InpaintingAnnotator"
image_inpainting_anno.MODE = "all"
image_inpainting_anno.USE_AUG = True
image_inpainting_anno.SALIENT = {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"}
image_inpainting_anno.GDINO = {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                               "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                               "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"}
image_inpainting_anno.SAM2 = {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                              "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}
# image_inpainting_anno.INPUTS = {"image": None, "mode": "salient"}
# image_inpainting_anno.INPUTS = {"image": None, "mask": None, "mode": "mask"}
# image_inpainting_anno.INPUTS = {"image": None, "bbox": None, "mode": "bbox"}
image_inpainting_anno.INPUTS = {"image": None, "mode": "salientmasktrack", "mask_cfg": None}
# image_inpainting_anno.INPUTS = {"image": None, "mode": "salientbboxtrack"}
# image_inpainting_anno.INPUTS = {"image": None, "mask": None, "mode": "masktrack"}
# image_inpainting_anno.INPUTS = {"image": None, "bbox": None, "mode": "bboxtrack"}
# image_inpainting_anno.INPUTS = {"image": None, "label": None, "mode": "label"}
# image_inpainting_anno.INPUTS = {"image": None, "caption": None, "mode": "caption"}
image_inpainting_anno.OUTPUTS = {"image": None, "mask": None}


#------------------------ Subject ------------------------#
image_subject_anno = EasyDict()
image_subject_anno.NAME = "SubjectAnnotator"
image_subject_anno.MODE = "all"
image_subject_anno.USE_AUG = True
image_subject_anno.USE_CROP = True
image_subject_anno.ROI_ONLY = True
image_subject_anno.INPAINTING = {"MODE": "all",
                                 "SALIENT": {"PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt"},
                                 "GDINO": {"TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                                           "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                                           "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth"},
                                 "SAM2": {"CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                          "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'}}
# image_subject_anno.INPUTS = {"image": None, "mode": "salient"}
# image_subject_anno.INPUTS = {"image": None, "mask": None, "mode": "mask"}
# image_subject_anno.INPUTS = {"image": None, "bbox": None, "mode": "bbox"}
# image_subject_anno.INPUTS = {"image": None, "mode": "salientmasktrack"}
# image_subject_anno.INPUTS = {"image": None, "mode": "salientbboxtrack"}
# image_subject_anno.INPUTS = {"image": None, "mask": None, "mode": "masktrack"}
# image_subject_anno.INPUTS = {"image": None, "bbox": None, "mode": "bboxtrack"}
# image_subject_anno.INPUTS = {"image": None, "label": None, "mode": "label"}
# image_subject_anno.INPUTS = {"image": None, "caption": None, "mode": "caption"}
image_subject_anno.INPUTS = {"image": None, "mode": None, "mask": None, "bbox": None, "label": None, "caption": None, "mask_cfg": None}
image_subject_anno.OUTPUTS = {"image": None, "mask": None}