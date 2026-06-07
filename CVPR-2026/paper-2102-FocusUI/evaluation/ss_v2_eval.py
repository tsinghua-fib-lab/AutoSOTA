"""
ScreenSpot-v2 evaluation script for FocusUI grounding.

Evaluates element grounding performance on ScreenSpot-v2 benchmark,
which contains UI screenshots across domains: mobile, desktop, and web.
"""
import argparse
import json
import os
from typing import Dict, List

import torch
from datasets import load_dataset
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

    dataset = load_dataset(data_path)["test"]
    domain_dict = {
        "windows": "desktop",
        "macos": "desktop",
        "ios": "mobile",
        "android": "mobile",
        "tool": "web",
        "shop": "web",
        "gitlab": "web",
        "forum": "web",
    }

    results = []

    overlay_out_dir = os.path.join(args.save_path, "saliency_heatmaps")
    os.makedirs(overlay_out_dir, exist_ok=True)

    for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        ele = {
            "file_name": example["file_name"],
            "data_type": example["data_type"],
            "domain": domain_dict[example["data_source"]],
            "instruction": example["instruction"],
            "img_size": example["image"].size,
            "bbox_x1y1x2y2": normalize_bbox(
                example["bbox"], example["image"].size[0], example["image"].size[1]
            ),
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
                    {"type": "image", "image": example["image"]},
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
                logits_processor=logits_processor_pointer,
                use_placeholder=use_placeholder,
                topk=topk,
            )

        topk_points = pred["topk_points"]
        gt_bbox = ele["bbox_x1y1x2y2"]
        topk_values = pred["topk_values"]
        ele["topk_points"] = topk_points
        ele["topk_values"] = topk_values
        ele["gt_bbox"] = gt_bbox

        # compute the metrics
        px, py = topk_points[0]
        x1, y1, x2, y2 = gt_bbox

        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1

        w, h = example["image"].size
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

        # Optionally save patch_saliency_heatmap_overlay
        if args.save_saliency_heatmaps:
            base_name = os.path.splitext(os.path.basename(ele.get("file_name", f"sample_{idx}")))[0]
            save_name = f"overlay_ps_pred_ssv2_{base_name}.png"
            overlays_saved = save_patch_saliency_heatmap_overlay(
                screenshot=example["image"],
                pred=pred,
                out_dir=overlay_out_dir,
                save_name=save_name,
                image_patch_size=IMAGE_PATCH_SIZE,
            )

        results.append(ele)

    return results


def get_metric(list_of_examples, domains=["mobile", "desktop", "web"], data_types=["text", "icon"]):
    """
    Computes metrics over a list of examples and prints/plots a table.

    Each element in list_of_examples is a dict containing:
        - "domain": Domain name (e.g., "web", "mobile", "desktop")
        - "data_type": Data type (e.g., "text", "icon")
        - "hit_top1", "overlap_top1", "hit_topk", "overlap_topk": binary (0 or 1)

    The final table has columns for each domain broken down by UI type (plus a domain-average)
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
    for domain in domains:
        # Filter examples for the current group.
        domain_examples = [ex for ex in list_of_examples if ex.get("domain") == domain]
        for data_type in data_types:
            # Filter further for the specific UI type.
            domain_data_type_examples = [ex for ex in domain_examples if ex.get("data_type") == data_type]
            col_name = f"{domain}-{data_type}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(domain_data_type_examples, metric)

        # Compute domain-average (all UI types for this domain).
        col_name_avg = f"{domain}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(domain_examples, metric)

    # Compute overall metrics for each UI type across all domains.
    for data_type in data_types:
        data_type_examples = [ex for ex in list_of_examples if ex.get("data_type") == data_type]
        col_name = f"All-{data_type}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(data_type_examples, metric)

    # Compute overall average across all examples.
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)

    # Define the order of columns.
    columns_order = []
    for domain in domains:
        for data_type in data_types:
            columns_order.append(f"{domain}-{data_type}")
        columns_order.append(f"{domain}-avg")
    for data_type in data_types:
        columns_order.append(f"All-{data_type}")
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


"""
# cd to project root directory
python eval/screenSpot_v2_tokenselect.py --save_path <path_to_save_results>
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="focusui_3b")
    parser.add_argument("--model_name_or_path", type=str, default="checkpoints/focusui_3b")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--data_path", type=str, default="./dataset/ScreenSpot-V2")
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
    parser.set_defaults(save_saliency_heatmaps=False)
    parser.set_defaults(apply_visual_token_select=True)

    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pred_path = f"{save_path}/screenspot_v2_all_preds.json"
    metric_path = f"{save_path}/screenspot_v2_all_metrics.txt"
    metric_json_path = f"{save_path}/screenspot_v2_all_metrics.json"

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
    metric_info, results = get_metric(results)
    with open(metric_path, "w") as f:
        f.write(metric_info)
    print(f"Saved metric to {metric_path}")
    with open(metric_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved metric to {metric_json_path}")
