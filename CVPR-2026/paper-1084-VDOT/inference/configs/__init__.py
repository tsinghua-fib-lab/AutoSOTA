# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from .video_preproccess import video_depth_anno, video_depthv2_anno, video_flow_anno, video_gray_anno, video_pose_anno, video_pose_body_anno, video_scribble_anno
from .video_preproccess import video_framerefext_anno, video_firstframeref_anno, video_lastframeref_anno, video_firstlastframeref_anno, video_firstclipref_anno, video_lastclipref_anno, video_firstlastclipref_anno, video_framerefexp_anno, video_cliprefexp_anno
from .video_preproccess import video_inpainting_mask_anno, video_inpainting_bbox_anno, video_inpainting_masktrack_anno, video_inpainting_bboxtrack_anno, video_inpainting_label_anno, video_inpainting_caption_anno, video_inpainting_anno
from .video_preproccess import video_outpainting_anno, video_outpainting_inner_anno
from .video_preproccess import video_layout_bbox_anno, video_layout_track_anno
from .image_preproccess import image_face_anno, image_salient_anno, image_subject_anno, image_face_mask_anno
from .image_preproccess import image_inpainting_anno, image_outpainting_anno
from .image_preproccess import image_depth_anno, image_gray_anno, image_pose_anno, image_scribble_anno
from .common_preproccess import image_plain_anno, image_mask_plain_anno, image_maskaug_plain_anno, image_maskaug_invert_anno, image_maskaug_anno, video_mask_plain_anno, video_maskaug_plain_anno, video_plain_anno, video_maskaug_invert_anno, video_mask_expand_anno, prompt_plain_anno, video_maskaug_anno, video_maskaug_layout_anno, image_mask_draw_anno, image_maskaug_region_random_anno, image_maskaug_region_crop_anno
from .prompt_preprocess import prompt_extend_ltx_en_anno, prompt_extend_wan_zh_anno, prompt_extend_wan_en_anno, prompt_extend_wan_zh_ds_anno, prompt_extend_wan_en_ds_anno, prompt_extend_ltx_en_ds_anno
from .composition_preprocess import comp_anno, comp_refany_anno, comp_aniany_anno, comp_swapany_anno, comp_expany_anno, comp_moveany_anno

VACE_IMAGE_PREPROCCESS_CONFIGS = {
    'image_plain': image_plain_anno,
    'image_face': image_face_anno,
    'image_salient': image_salient_anno,
    'image_inpainting': image_inpainting_anno,
    'image_reference': image_subject_anno,
    'image_outpainting': image_outpainting_anno,
    'image_depth': image_depth_anno,
    'image_gray': image_gray_anno,
    'image_pose': image_pose_anno,
    'image_scribble': image_scribble_anno
}

VACE_IMAGE_MASK_PREPROCCESS_CONFIGS = {
    'image_mask_plain': image_mask_plain_anno,
    'image_mask_seg': image_inpainting_anno,
    'image_mask_draw': image_mask_draw_anno,
    'image_mask_face': image_face_mask_anno
}

VACE_IMAGE_MASKAUG_PREPROCCESS_CONFIGS = {
    'image_maskaug_plain': image_maskaug_plain_anno,
    'image_maskaug_invert': image_maskaug_invert_anno,
    'image_maskaug': image_maskaug_anno,
    'image_maskaug_region_random': image_maskaug_region_random_anno,
    'image_maskaug_region_crop': image_maskaug_region_crop_anno
}


VACE_VIDEO_PREPROCCESS_CONFIGS = {
    'plain': video_plain_anno,
    'depth': video_depth_anno,
    'depthv2': video_depthv2_anno,
    'flow': video_flow_anno,
    'gray': video_gray_anno,
    'pose': video_pose_anno,
    'pose_body': video_pose_body_anno,
    'scribble': video_scribble_anno,
    'framerefext': video_framerefext_anno,
    'frameref': video_framerefexp_anno,
    'clipref': video_cliprefexp_anno,
    'firstframe': video_firstframeref_anno,
    'lastframe': video_lastframeref_anno,
    "firstlastframe": video_firstlastframeref_anno,
    'firstclip': video_firstclipref_anno,
    'lastclip': video_lastclipref_anno,
    'firstlastclip': video_firstlastclipref_anno,
    'inpainting': video_inpainting_anno,
    'inpainting_mask': video_inpainting_mask_anno,
    'inpainting_bbox': video_inpainting_bbox_anno,
    'inpainting_masktrack': video_inpainting_masktrack_anno,
    'inpainting_bboxtrack': video_inpainting_bboxtrack_anno,
    'inpainting_label': video_inpainting_label_anno,
    'inpainting_caption': video_inpainting_caption_anno,
    'outpainting': video_outpainting_anno,
    'outpainting_inner': video_outpainting_inner_anno,
    'layout_bbox': video_layout_bbox_anno,
    'layout_track': video_layout_track_anno,
}

VACE_VIDEO_MASK_PREPROCCESS_CONFIGS = {
    # 'mask_plain': video_mask_plain_anno,
    'mask_expand': video_mask_expand_anno,
    'mask_seg': video_inpainting_anno,
}

VACE_VIDEO_MASKAUG_PREPROCCESS_CONFIGS = {
    'maskaug_plain': video_maskaug_plain_anno,
    'maskaug_invert': video_maskaug_invert_anno,
    'maskaug': video_maskaug_anno,
    'maskaug_layout': video_maskaug_layout_anno
}

VACE_COMPOSITION_PREPROCCESS_CONFIGS = {
    'composition': comp_anno,
    'reference_anything': comp_refany_anno,
    'animate_anything': comp_aniany_anno,
    'swap_anything': comp_swapany_anno,
    'expand_anything': comp_expany_anno,
    'move_anything': comp_moveany_anno
}


VACE_PREPROCCESS_CONFIGS = {**VACE_IMAGE_PREPROCCESS_CONFIGS, **VACE_VIDEO_PREPROCCESS_CONFIGS, **VACE_COMPOSITION_PREPROCCESS_CONFIGS}

VACE_PROMPT_CONFIGS = {
    'plain': prompt_plain_anno,
    'wan_zh': prompt_extend_wan_zh_anno,
    'wan_en': prompt_extend_wan_en_anno,
    'wan_zh_ds': prompt_extend_wan_zh_ds_anno,
    'wan_en_ds': prompt_extend_wan_en_ds_anno,
    'ltx_en': prompt_extend_ltx_en_anno,
    'ltx_en_ds': prompt_extend_ltx_en_ds_anno
}


VACE_CONFIGS = {
    "prompt": VACE_PROMPT_CONFIGS,
    "image": VACE_IMAGE_PREPROCCESS_CONFIGS,
    "image_mask": VACE_IMAGE_MASK_PREPROCCESS_CONFIGS,
    "image_maskaug": VACE_IMAGE_MASKAUG_PREPROCCESS_CONFIGS,
    "video": VACE_VIDEO_PREPROCCESS_CONFIGS,
    "video_mask": VACE_VIDEO_MASK_PREPROCCESS_CONFIGS,
    "video_maskaug": VACE_VIDEO_MASKAUG_PREPROCCESS_CONFIGS,
}