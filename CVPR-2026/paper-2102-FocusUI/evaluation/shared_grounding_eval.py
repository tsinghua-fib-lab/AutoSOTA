"""
Shared helpers for grounding evaluation scripts.

This module provides common utilities for all benchmark evaluation scripts:
- Model loading and inference setup
- Bounding box normalization and overlap detection
- Metric formatting and computation
- Patch score overlay visualization
"""
import os
from functools import partial
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from transformers import AutoProcessor

from focusui.constants import (
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    assistant_starter_guiactor,
    assistant_starter_guiactor_qwen3vl,
    grounding_system_message_guiactor_qwen25vl,
    grounding_system_message_guiactor_qwen3vl,
)
from focusui.inference import (
    ForceFollowTokensLogitsProcessor,
    inference_focusui_token_select,
)
from focusui.visualization import vis_patch_saliency_heatmap_overlay, draw_point, draw_bbox


def load_model_and_inference(
    model_name_or_path: str,
    model_type: str,
    device: str,
    apply_visual_token_select: bool = True,
    visual_reduct_ratio: float = 0.5,
):
    """Load model, tokenizer, processor, grounding message, inference fn, logits processor, patch size.

    Supported model_type values:
      - focusui_3b, focusui_7b, focusui_qwen3vl_2b, focusui_qwen3vl_4b
    Returns:
      (model, tokenizer, data_processor, grounding_system_message, inference_fn, logits_processor, image_patch_size)
    """

    data_processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = data_processor.tokenizer

    image_patch_size = 14

    if model_type in ["focusui_3b", "focusui_7b"]:
        from focusui.modeling_focusui_qwen25vl import (
            FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer,
        )

        model = FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",
        ).eval()
        model.apply_visual_token_select = apply_visual_token_select
        model.visual_reduct_ratio = visual_reduct_ratio
        grounding_system_message = grounding_system_message_guiactor_qwen25vl
        inference_fn = partial(
            inference_focusui_token_select, assistant_starter=assistant_starter_guiactor,
        )
        image_patch_size = 14

    elif model_type in ["focusui_qwen3vl_2b", "focusui_qwen3vl_4b"]:
        from focusui.modeling_focusui_qwen3vl import (
            FocusUI_Qwen3VLForConditionalGenerationWithPointer,
        )

        model = FocusUI_Qwen3VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",
        ).eval()
        model.apply_visual_token_select = apply_visual_token_select
        model.visual_reduct_ratio = visual_reduct_ratio
        grounding_system_message = grounding_system_message_guiactor_qwen3vl
        inference_fn = partial(
            inference_focusui_token_select, assistant_starter=assistant_starter_guiactor_qwen3vl
        )
        image_patch_size = 16
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    logits_processor = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]],
    )

    return (
        model,
        tokenizer,
        data_processor,
        grounding_system_message,
        inference_fn,
        logits_processor,
        image_patch_size,
    )


def format_cell(cell: Any) -> str:
    """Format value for metric table cell."""
    if isinstance(cell, float):
        return f"{cell*100:.2f}"
    if cell is None:
        return "N/A"
    return str(cell)


def compute_mean(examples: List[Dict], key: str):
    """Safe mean over list of example dicts; returns None if empty."""
    if not examples:
        return None
    return sum(ex.get(key, 0) for ex in examples) / len(examples)


def normalize_bbox(bbox_x1y1x2y2: Sequence[Union[int, float]], img_width: Union[int, float], img_height: Union[int, float]) -> Tuple[float, float, float, float]:
    """Normalize a bbox in xyxy format to [0,1] and ensure x1<=x2, y1<=y2.

    If the bbox already looks normalized (all coords in [0,1]), it is returned (with ordering fixed).
    """
    x1, y1, x2, y2 = bbox_x1y1x2y2
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    # ensure ordering
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # already normalized?
    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        return x1, y1, x2, y2

    w = float(img_width) if float(img_width) != 0 else 1.0
    h = float(img_height) if float(img_height) != 0 else 1.0
    return x1 / w, y1 / h, x2 / w, y2 / h


def do_boxes_overlap(box1, box2):
    """
    Check if two boxes overlap.
    
    Each box is represented as a tuple: (x1, y1, x2, y2)
    Where (x1, y1) is the top-left and (x2, y2) is the bottom-right corner.
    """
    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check for no overlap
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True


def save_patch_saliency_heatmap_overlay(
    *,
    screenshot: object,
    pred: dict,
    save_name: str,
    image_patch_size: int,
    out_dir: str = "./",
    alpha: float = 0.5,
    cmap_name: str = "jet",
) -> str:
    """Optionally save patch_score_pred overlay. Returns save path."""

    patch_score_pred = pred.get("patch_score_pred", None)
    image_grid_thw = pred.get("image_grid_thw", None)
    _, h_grid, w_grid = map(int, image_grid_thw)
    h_grid_half, w_grid_half = h_grid // 2, w_grid // 2
    out_h, out_w = h_grid * image_patch_size, w_grid * image_patch_size

    overlay_img = vis_patch_saliency_heatmap_overlay(
        screenshot_path=screenshot,
        ps_1d=patch_score_pred,
        h_grid_half=h_grid_half,
        w_grid_half=w_grid_half,
        out_w=out_w,
        out_h=out_h,
        alpha=alpha,
        cmap_name=cmap_name,
    )
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, save_name)
    overlay_img.save(save_path)
    return save_path

