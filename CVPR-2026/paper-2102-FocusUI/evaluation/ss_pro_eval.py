"""
ScreenSpot-Pro evaluation script for FocusUI grounding.

Evaluates element grounding performance on ScreenSpot-Pro benchmark,
which contains professional software UI screenshots across categories:
Dev, Creative, CAD, Scientific, Office, and OS.
"""
import argparse
import json
import os
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

from evaluation.shared_grounding_eval import (
    compute_mean,
    do_boxes_overlap,
    format_cell,
    load_model_and_inference,
    normalize_bbox,
    save_patch_saliency_heatmap_overlay,
)

# Default patch size (overridden by loader)
IMAGE_PATCH_SIZE = 14


def evaluate(
    model_name_or_path: str,
    model_type: str,
    data_fn: str,
    image_dir: str,
    use_placeholder: bool,
    topk: int,
    device: str = "cuda:0",
    args=None,
) -> List[Dict]:
    """Run ScreenSpot-Pro evaluation and return per-example results.

    Removed options: selection_method, eval_sample, shuffle_data.
    """
    model, tokenizer, data_processor, grounding_system_message, inference_fn, logits_processor, IMAGE_PATCH_SIZE_LOADED = load_model_and_inference(
        model_name_or_path,
        model_type,
        device,
        getattr(args, "apply_visual_token_select", True),
        getattr(args, "visual_reduct_ratio", 0.5),
    )
    global IMAGE_PATCH_SIZE
    IMAGE_PATCH_SIZE = IMAGE_PATCH_SIZE_LOADED

    # Load dataset
    with open(data_fn, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {data_fn}")

    results: List[Dict] = []
    overlay_out_dir = os.path.join(args.save_path, "saliency_heatmaps")
    os.makedirs(overlay_out_dir, exist_ok=True)

    for example in tqdm(data, total=len(data)):
        ele = {
            "file_name": example["img_filename"],
            "ui_type": example["ui_type"],
            "group": example["group"],
            "platform": example["platform"],
            "application": example["application"],
            "id": example["id"],
            "instruction": example["instruction"],
            "img_size": example["img_size"],
            "bbox_x1y1x2y2": normalize_bbox(
                example["bbox"], example["img_size"][0], example["img_size"][1]
            ),
            "hit_top1": 0,
            "overlap_top1": 0,
            "hit_topk": 0,
            "overlap_topk": 0,
        }

        image_path = os.path.join(image_dir, example["img_filename"])
        image = Image.open(image_path)

        # Multi-scale TTA with confidence-weighted ensemble (3 scales)
        w, h = image.size
        scale_configs = [
            (image, 1.0),
            (image.resize((int(w * 1.5), int(h * 1.5)), Image.LANCZOS), 1.5),
        ]

        all_points = []
        all_values = []
        all_confidences = []
        pred_original = None

        for scaled_image, scale_factor in scale_configs:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": scaled_image},
                        {"type": "text", "text": example["instruction"]},
                    ],
                },
            ]

            with torch.no_grad():
                pred = inference_fn(
                    conversation=conversation,
                    model=model,
                    tokenizer=tokenizer,
                    data_processor=data_processor,
                    logits_processor=logits_processor,
                    use_placeholder=use_placeholder,
                    topk=topk,
                )

            pts = pred["topk_points"]
            vals = pred["topk_values"]
            all_points.append(pts)
            all_values.append(vals)
            # Confidence = max region score for this scale
            confidence = vals[0] if len(vals) > 0 else 0.0
            all_confidences.append(confidence)
            if scale_factor == 1.0:
                pred_original = pred

        # Confidence-weighted ensemble
        total_conf = sum(all_confidences) + 1e-8
        weights = [c / total_conf for c in all_confidences]

        topk_points = []
        topk_values = []
        max_len = max(len(pts) for pts in all_points)
        for k in range(max_len):
            wsum_x = 0.0
            wsum_y = 0.0
            wsum_v = 0.0
            wsum = 0.0
            for pts, vals, w in zip(all_points, all_values, weights):
                if k < len(pts):
                    wsum_x += pts[k][0] * w
                    wsum_y += pts[k][1] * w
                    wsum_v += vals[k] * w
                    wsum += w
            if wsum > 0:
                topk_points.append((wsum_x / wsum, wsum_y / wsum))
                topk_values.append(wsum_v / wsum)

        gt_bbox = ele["bbox_x1y1x2y2"]
        ele["topk_points"], ele["topk_values"], ele["gt_bbox"] = topk_points, topk_values, gt_bbox

        # compute the metrics
        px, py = topk_points[0]
        x1, y1, x2, y2 = gt_bbox

        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1

        w, h = example["img_size"]
        pred_bbox = [
            px - IMAGE_PATCH_SIZE / w,
            py - IMAGE_PATCH_SIZE / h,
            px + IMAGE_PATCH_SIZE / w,
            py + IMAGE_PATCH_SIZE / h,
        ]
        if do_boxes_overlap(pred_bbox, gt_bbox):
            ele["overlap_top1"] = 1
            ele["overlap_topk"] = 1

        for px, py in topk_points[1:]:
            if (x1 <= px <= x2) and (y1 <= py <= y2):
                ele["hit_topk"] = 1
            pred_bbox = [
                px - IMAGE_PATCH_SIZE / w,
                py - IMAGE_PATCH_SIZE / h,
                px + IMAGE_PATCH_SIZE / w,
                py + IMAGE_PATCH_SIZE / h,
            ]
            if do_boxes_overlap(pred_bbox, gt_bbox):
                ele["overlap_topk"] = 1

        # Optionally save patch_saliency_heatmap_overlay (use first scale pred)
        if args.save_saliency_heatmaps:
            base_name = os.path.splitext(os.path.basename(ele["file_name"]))[0]
            save_name = f"overlay_ps_pred_sspro_{base_name}.png"
            overlays_saved = save_patch_saliency_heatmap_overlay(
                screenshot=image,
                pred=pred_original if pred_original is not None else {},
                out_dir=overlay_out_dir,
                save_name=save_name,
                image_patch_size=IMAGE_PATCH_SIZE,
            )

        results.append(ele)

    return results


def get_metric(list_of_examples, groups=["Dev", "Creative", "CAD", "Scientific", "Office", "OS"], ui_types=["text", "icon"]):
    """
    Computes metrics over a list of examples and prints/plots a table.

    Each element in list_of_examples is a dict containing:
        - "group": Group name (e.g., "Dev", "Creative", etc.)
        - "ui_type": UI type (e.g., "text", "icon")
        - "hit_top1", "overlap_top1", "hit_topk", "overlap_topk": binary (0 or 1)

    The final table has columns for each group broken down by UI type (plus a group-average)
    and overall columns ("All-text", "All-icon", "All-average").

    The rows of the table are:
        - hit_top1
        - overlap_top1
        - hit_topk
        - overlap_topk
    """

    # List of metric keys to compute.
    metrics = ["hit_top1", "overlap_top1", "hit_topk", "overlap_topk"]

    # Prepare results dictionary: structure {metric: {column_name: value}}.
    results = {metric: {} for metric in metrics}

    # Compute metrics for each group broken down by UI type.
    for group in groups:
        # Filter examples for the current group.
        group_examples = [ex for ex in list_of_examples if ex.get("group") == group]
        for ui in ui_types:
            # Filter further for the specific UI type.
            group_ui_examples = [ex for ex in group_examples if ex.get("ui_type") == ui]
            col_name = f"{group}-{ui}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(group_ui_examples, metric)

        # Compute group-average (all UI types for this group).
        col_name_avg = f"{group}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(group_examples, metric)

    # Compute overall metrics for each UI type across all groups.
    for ui in ui_types:
        ui_examples = [ex for ex in list_of_examples if ex.get("ui_type") == ui]
        col_name = f"All-{ui}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(ui_examples, metric)

    # Compute overall average across all examples.
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)

    # Define the order of columns.
    columns_order = []
    for group in groups:
        for ui in ui_types:
            columns_order.append(f"{group}-{ui}")
        columns_order.append(f"{group}-avg")
    for ui in ui_types:
        columns_order.append(f"All-{ui}")
    columns_order.append("All-avg")

    # ------------- Print Table to Console -------------
    # Prepare header row.
    header = [""] + columns_order
    # Calculate column widths for console printing.
    col_widths = [max(len(col), 12) for col in header]

    # Print header.
    header_line = " | ".join(word.ljust(width) for word, width in zip(header, col_widths))
    separator_line = "-+-".join("-" * width for width in col_widths)
    print(header_line)
    print(separator_line)

    for metric in metrics:
        row = [metric]
        for col in columns_order:
            val = results[metric].get(col)
            row.append(format_cell(val))
        row_line = " | ".join(word.ljust(width) for word, width in zip(row, col_widths))
        print(row_line)

    # ------------- Print Tab-delimited Version (for Excel Copy-Paste) -------------
    metric_info = "Tab-delimited Table for Excel:\n"
    # Header row.
    header_tab = "\t".join([""] + columns_order)
    metric_info += header_tab + "\n"
    # Each row.
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    return metric_info, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="focusui_3b")
    parser.add_argument("--model_name_or_path", type=str, default="checkpoints/focusui_3b")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--data_path", type=str, default="./dataset/ScreenSpot-Pro")
    parser.add_argument("--topk", type=int, default=3, help="Topk")
    parser.add_argument(
        "--no-placeholder",
        dest="use_placeholder",
        action="store_false",
        help="Disable the placeholder",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    # FocusUI controls
    parser.add_argument("--apply_visual_token_select", dest="apply_visual_token_select", action="store_true")
    parser.add_argument("--no-apply_visual_token_select", dest="apply_visual_token_select", action="store_false")
    parser.add_argument("--visual_reduct_ratio", type=float, default=0.5)

    parser.set_defaults(use_placeholder=True)
    parser.set_defaults(apply_visual_token_select=True)
    parser.set_defaults(save_saliency_heatmaps=False)
    args = parser.parse_args()

    image_dir = os.path.join(args.data_path, "images")
    data_fn = os.path.join(args.data_path, "annotations/all.json")
    os.makedirs(args.save_path, exist_ok=True)

    pred_path = os.path.join(args.save_path, "screenspot-Pro_all_preds.json")
    metric_path = os.path.join(args.save_path, "screenspot-Pro_all_preds.txt")
    metric_json_path = os.path.join(args.save_path, "screenspot-Pro_all_metrics.json")

    print(f"Evaluating {args.model_name_or_path}...")
    results = evaluate(
        args.model_name_or_path,
        args.model_type,
        data_fn,
        image_dir,
        args.use_placeholder,
        args.topk,
        args.device,
        args,
    )

    with open(pred_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} predictions to {pred_path}")

    metric_info, metrics = get_metric(results)
    with open(metric_path, "w") as f:
        f.write(metric_info)
    print(f"Saved metric to {metric_path}")
    with open(metric_json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metric to {metric_json_path}")
