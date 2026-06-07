"""
UI-Vision evaluation script for FocusUI grounding.

Evaluates element grounding performance on UI-Vision benchmark,
which contains diverse UI screenshots with text and icon elements.
"""
import argparse
import json
import os
from typing import Dict, List

import torch
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
    data_path: str,
    use_placeholder: bool,
    topk: int,
    device: str = "cuda:0",
    args=None,
) -> List[Dict]:
    """Run UI-Vision evaluation and return per-example results."""
    global IMAGE_PATCH_SIZE
    (
        model,
        tokenizer,
        data_processor,
        grounding_system_message,
        inference_fn,
        logits_processor_pointer,
        IMAGE_PATCH_SIZE,
    ) = load_model_and_inference(
        model_name_or_path,
        model_type,
        device,
        getattr(args, "apply_visual_token_select", True),
        getattr(args, "visual_reduct_ratio", 0.5),
    )
    print(f"Loaded model from {model_name_or_path}")

    # Load dataset: support HuggingFace path or local JSON directory/file
    json_filename = "element_grounding_all.json"
    json_path = os.path.join(data_path, "annotations", json_filename)
    
    dataset = []

    with open(json_path, "r") as f:
        items = json.load(f)

        for it in items:
            dataset.append({
                "image_path": it["image_path"],
                "image_size": it["image_size"],
                "instruction": it['prompt_to_evaluate'],
                "bbox": it["bbox"],
                "element_type": it["element_type"],
                "group": it["group"],
            })


    results = []
    overlay_out_dir = os.path.join(args.save_path, "saliency_heatmaps")
    os.makedirs(overlay_out_dir, exist_ok=True)

    for example in tqdm(dataset, total=len(dataset)):
        image_path = os.path.join(data_path, "images", example["image_path"])
        instruction = example["instruction"]
        element_type = example["element_type"]
        group = example["group"]
        img_w, img_h = example["image_size"]
        raw_gt_bbox = example["bbox"]

        x1, y1, x2, y2 = raw_gt_bbox[:4]
        gt_bbox_norm = normalize_bbox([x1, y1, x2, y2], img_w, img_h)


        ele = {
            "image_path": image_path,
            "instruction": instruction,
            "img_size": (img_w, img_h),
            "gt_bbox": gt_bbox_norm,
            "group": group,
            "element_type": element_type,
            "hit_top1": 0,
            "overlap_top1": 0,
            "hit_topk": 0,
            "overlap_topk": 0,
        }

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": grounding_system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        with torch.no_grad():
            pred = inference_fn(
                conversation=conversation,
                model=model,
                tokenizer=tokenizer,
                data_processor=data_processor,
                logits_processor=logits_processor_pointer,
                use_placeholder=use_placeholder,
                topk=topk,
            )

        topk_points = pred["topk_points"]
        topk_values = pred["topk_values"]
        gt_bbox = ele["gt_bbox"]

        # compute the metrics
        px, py = topk_points[0]
        x1, y1, x2, y2 = gt_bbox

        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1

        w, h = img_w, img_h
        pred_bbox = [
            px - IMAGE_PATCH_SIZE / w,
            py - IMAGE_PATCH_SIZE / h,
            px + IMAGE_PATCH_SIZE / w,
            py + IMAGE_PATCH_SIZE / h,
        ]

        ele["topk_points"] = topk_points
        ele["topk_values"] = topk_values
        ele["topk_pred_bboxes"] = [pred_bbox]

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

            ele["topk_pred_bboxes"].append(pred_bbox)

        # Optionally save patch_saliency_heatmap_overlay
        if args.save_saliency_heatmaps:
            base_name = example["image_path"].replace("/", "-").split(".")[0]
            save_name = f"overlay_ps_pred_uivision_{base_name}.png"
            overlays_saved = save_patch_saliency_heatmap_overlay(
                screenshot=image_path,
                pred=pred,
                out_dir=overlay_out_dir,
                save_name=save_name,
                image_patch_size=IMAGE_PATCH_SIZE,
            )

        results.append(ele)

    return results


def get_metric(list_of_examples):
    """
    Computes metrics over a list of examples and prints/plots a table.

    Each element in list_of_examples is a dict containing:
        - "hit_top1", "overlap_top1", "hit_topk", "overlap_topk": binary (0 or 1)

    The final table has columns for each group (plus a group-average)
    and overall columns ("All-average").

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

    # Compute overall average across all examples.
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)

    # Define the order of columns.
    columns_order = []
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


def get_group_metric(list_of_examples):
    """
    Computes metrics over a list of examples grouped by the "group" key and prints/plots a table.

    Each element in list_of_examples is a dict containing:
        - "group": Group name (string)
        - "hit_top1", "overlap_top1", "hit_topk", "overlap_topk": binary (0 or 1)

    Columns per group are broken down by UI type plus a group-average, and overall columns
    ("All-text", "All-icon", "All-avg").
    """

    metrics = ["hit_top1", "overlap_top1", "hit_topk", "overlap_topk"]

    # Derive group list from data
    groups = sorted({ex.get("group", "Unknown") for ex in list_of_examples})

    results = {metric: {} for metric in metrics}

    # Per-group metrics (with UI type breakdown)
    for grp in groups:
        grp_examples = [ex for ex in list_of_examples if ex.get("group", "Unknown") == grp]

        # Group-average across UI types
        col_name_avg = f"{grp}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(grp_examples, metric)


    # Overall average across all examples
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)

    # Column order
    columns_order = []
    for grp in groups:
        columns_order.append(f"{grp}-avg")
    columns_order.append("All-avg")

    # ------------- Print Table to Console -------------
    header = [""] + columns_order
    col_widths = [max(len(col), 12) for col in header]

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
    metric_info = "Tab-delimited Table for Excel (By Group):\n"
    header_tab = "\t".join([""] + columns_order)
    metric_info += header_tab + "\n"
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    return metric_info, results


"""
# cd to project root directory
python eval/ui_vision_eval.py --save_path <path_to_save_results>
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="focusui_3b")
    parser.add_argument("--model_name_or_path", type=str, default="checkpoints/focusui_3b")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--data_path", type=str, default="./dataset/ui_benchmarks/ui-vision")
    parser.add_argument("--topk", type=int, default=3, help="Topk")
    parser.add_argument(
        "--no-placeholder",
        dest="use_placeholder",
        action="store_false",
        help="Disable the placeholder",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.set_defaults(use_placeholder=True)

    # FocusUI controls
    parser.add_argument("--apply_visual_token_select", dest="apply_visual_token_select", action="store_true")
    parser.add_argument("--no-apply_visual_token_select", dest="apply_visual_token_select", action="store_false")
    parser.add_argument("--visual_reduct_ratio", type=float, default=0.5)
    parser.set_defaults(apply_visual_token_select=True)
    parser.set_defaults(save_saliency_heatmaps=False)

    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pred_path = f"{save_path}/uivision_preds.json"
    metric_path = f"{save_path}/uivision_metrics.txt"
    metric_json_path = f"{save_path}/uivision_metrics.json"

    print(f"Evaluating {args.model_name_or_path}...")
    results = evaluate(
        args.model_name_or_path,
        args.model_type,
        args.data_path,
        args.use_placeholder,
        args.topk,
        args.device,
        args,
    )
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} predictions to {pred_path}")

    # if not os.path.exists(metric_path):
    metric_info_domtype, results_domtype = get_metric(results)
    metric_info_group, results_group = get_group_metric(results)

    combined_info = metric_info_group + "\n\n" + metric_info_domtype
    with open(metric_path, "w") as f:
        f.write(combined_info)
    print(f"Saved metric to {metric_path}")
    
    combined_json = {"ByGroup": results_group, "ByDomainType": results_domtype}
    with open(metric_json_path, "w") as f:
        json.dump(combined_json, f, indent=4)
    print(f"Saved metric to {metric_json_path}")
