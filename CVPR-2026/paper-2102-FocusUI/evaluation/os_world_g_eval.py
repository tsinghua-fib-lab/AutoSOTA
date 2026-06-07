"""
OSWorld-G evaluation script for FocusUI grounding.

Evaluates element grounding performance on OSWorld-G benchmark,
which contains OS-level UI screenshots with support for both
bounding box and polygon ground truth annotations.
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from evaluation.shared_grounding_eval import (
    do_boxes_overlap,
    format_cell,
    load_model_and_inference,
    normalize_bbox,
    save_patch_saliency_heatmap_overlay,
)

# Default patch size (overridden by loader)
IMAGE_PATCH_SIZE = 14


# =================================
# Helper Functions for OSWorld-G
# =================================

def normalize_point(
    px: float, py: float, img_width: int, img_height: int
) -> Tuple[float, float]:
    """Normalize a single point to [0,1]. If already normalized, return as-is."""
    if 0 <= px <= 1 and 0 <= py <= 1:
        return px, py
    return px / img_width, py / img_height


def point_in_bbox(px: float, py: float, bbox_x1y1x2y2: Tuple[float, ...]) -> bool:
    """Check if a point is inside a bounding box."""
    x1, y1, x2, y2 = bbox_x1y1x2y2
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def _point_on_segment(px, py, x1, y1, x2, y2, eps=1e-9):
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= px <= max(x1, x2) + eps
        and min(y1, y2) - eps <= py <= max(y1, y2) + eps
    )


def point_in_polygon(px, py, polygon):
    """
    Ray-casting point-in-polygon test.
    Accepts either a flat list [x1, y1, x2, y2, ...] or a list of (x, y) tuples.
    """
    if not polygon:
        return False

    # Normalize input to flat list [x1, y1, x2, y2, ...]
    if isinstance(polygon[0], (tuple, list)):
        flat = []
        for x, y in polygon:
            flat.extend([x, y])
        polygon = flat

    x, y = px, py
    n = len(polygon) // 2
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i

    return inside


def _segments_intersect(p1, p2, q1, q2, eps=1e-12):
    """
    Inclusive segment intersection test (counts touching/collinear overlap as intersecting).
    p1, p2, q1, q2 are (x,y) tuples in normalized coords.
    """

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    # Proper intersection
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    # Collinear / endpoint cases (inclusive)
    if abs(o1) <= eps and _point_on_segment(q1[0], q1[1], p1[0], p1[1], p2[0], p2[1], eps):
        return True
    if abs(o2) <= eps and _point_on_segment(q2[0], q2[1], p1[0], p1[1], p2[0], p2[1], eps):
        return True
    if abs(o3) <= eps and _point_on_segment(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1], eps):
        return True
    if abs(o4) <= eps and _point_on_segment(p2[0], p2[1], q1[0], q1[1], q2[0], q2[1], eps):
        return True

    return False


def do_boxes_polygon_overlap(bbox_x1y1x2y2, polygon_xy):
    """
    Check if an axis-aligned rectangle overlaps a polygon.

    Args:
        bbox_x1y1x2y2: [x1, y1, x2, y2] in normalized coords (0-1). Ordering is normalized if needed.
        polygon_xy: list of (x, y) tuples in normalized coords, implicitly closed.

    Returns True if any overlap exists, including boundary touching.
    """
    if not polygon_xy or len(polygon_xy) < 3:
        return False

    x1, y1, x2, y2 = bbox_x1y1x2y2
    # ensure ordering (robust to any input ordering)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # 1) Any polygon vertex inside the rectangle?
    for (px, py) in polygon_xy:
        if point_in_bbox(px, py, (x1, y1, x2, y2)):
            return True

    # 2) Any rectangle corner inside the polygon?
    rect_corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    for (rx, ry) in rect_corners:
        if point_in_polygon(rx, ry, polygon_xy):
            return True

    # 3) Any edge intersection between rectangle and polygon?
    rect_edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    n = len(polygon_xy)
    for i in range(n):
        p_start = polygon_xy[i]
        p_end = polygon_xy[(i + 1) % n]
        for e_start, e_end in rect_edges:
            if _segments_intersect(p_start, p_end, e_start, e_end):
                return True

    return False

def parse_and_normalize_gt(box_type, coords, img_width, img_height):
    """
    Decide GT type and normalize to [0,1].
    - box_type == 'refusal': special case
    - box_type == 'bbox': coords as [x1, y1, w, h]
    - box_type == 'polygon': coords as flat [x1, y1, x2, y2, ...]
    Returns:
        {"type": "bbox", "bbox": (x1,y1,x2,y2)} OR
        {"type": "polygon", "polygon": [x1,y1,x2,y2, ...]} OR
        {"type": "refusal", "bbox": [0,0,0,0]}
    """
    coords = list(coords)

    if box_type == "refusal":
        return {"type": "refusal", "bbox": [0, 0, 0, 0]}
    elif box_type == "bbox":
        x1, y1, w, h = coords
        rect = normalize_bbox([x1, y1, x1 + w, y1 + h], img_width, img_height)
        return {"type": "bbox", "bbox": rect}
    elif box_type == "polygon":
        pts = []
        for i in range(0, len(coords), 2):
            x, y = coords[i], coords[i + 1]
            x, y = normalize_point(x, y, img_width, img_height)
            pts.extend([x, y])
        return {"type": "polygon", "polygon": pts}
    else:
        raise ValueError(f"Invalid box type: {box_type}. Expected one of: 'refusal', 'bbox', 'polygon'.")


def hit_gt(px, py, gt):
    """Route to the correct test based on GT type."""
    if gt["type"] == "bbox":
        return point_in_bbox(px, py, gt["bbox"])
    elif gt["type"] == "polygon":
        return point_in_polygon(px, py, gt["polygon"])
    elif gt["type"] == "refusal":
        return (px < 0 and py < 0)

def overlap_gt(pred_bbox, gt):
    if gt["type"] == "bbox":
        return do_boxes_overlap(pred_bbox, gt["bbox"])
    elif gt["type"] == "polygon":
        poly = gt["polygon"]
        # Convert flat list to list of (x, y) for overlap helper
        if poly and not isinstance(poly[0], (tuple, list)):
            poly_xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        else:
            poly_xy = poly
        return do_boxes_polygon_overlap(pred_bbox, poly_xy)
    elif gt["type"] == "refusal":
        return False


def evaluate(
    model_name_or_path: str,
    model_type: str,
    data_path: str,
    use_placeholder: bool,
    topk: int,
    device: str = "cuda:0",
    args=None,
) -> List[Dict]:
    """Run OSWorld-G evaluation and return per-example results."""
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
    if args.pure_grounding_eval:
        json_path = os.path.join(data_path, "OSWorld-G_refined.json")
    else:
        json_path = os.path.join(data_path, "OSWorld-G.json")
    with open(json_path, "r") as f:
        items = json.load(f)

    ds = []
    
    for it in items:
        img_path_abs = os.path.join(data_path, "images", it["image_path"])

        # skip "refusal" since we evaluate grounding action only
        # if it["box_type"] == "refusal":
        #     continue

        ds.append({
            "id": it["id"],
            "image_path": img_path_abs,
            "image_size": it["image_size"],
            "instruction": it["instruction"],
            "box_type": it["box_type"],
            "box_coordinates": it["box_coordinates"],
            "data_types": it["GUI_types"],
            "file_name": os.path.basename(img_path_abs),
        })

    dataset = ds

    results = []

    overlay_out_dir = os.path.join(args.save_path, "saliency_heatmaps")
    os.makedirs(overlay_out_dir, exist_ok=True)

    for example in tqdm(dataset, total=len(dataset)):
        # Image and size
        image_input = example["image_path"]
        img_w, img_h = example["image_size"]
        file_name = example["file_name"]
        instruction = example["instruction"]
        data_types = example["data_types"]

        # Ground-truth coordinates: box_coordinates
        bbox_gt = parse_and_normalize_gt(example["box_type"], example["box_coordinates"], img_w, img_h)

        ele = {
            "id": example["id"],
            "file_name": file_name,
            "image_path": image_input,
            "instruction": instruction,
            "img_size": (img_w, img_h),
            "data_types": data_types,
            "box_type": bbox_gt["type"],
            "gt_bbox": bbox_gt.get("bbox", []),
            "gt_polygon": bbox_gt.get("polygon", []),
            "hit_top1": 0,
            "hit_topk": 0,
            "overlap_top1": 0,
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
                    {"type": "image", "image": image_input},
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

        ele["topk_points"] = topk_points
        ele["topk_values"] = topk_values

        px0, py0 = topk_points[0]
        if hit_gt(px0, py0, bbox_gt):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1
        
        pred_bbox = [
            px0 - IMAGE_PATCH_SIZE / img_w,
            py0 - IMAGE_PATCH_SIZE / img_h,
            px0 + IMAGE_PATCH_SIZE / img_w,
            py0 + IMAGE_PATCH_SIZE / img_h,
        ]
        ele["topk_pred_bboxes"] = [pred_bbox]

        if overlap_gt(pred_bbox, bbox_gt):
            ele["overlap_top1"] = 1
            ele["overlap_topk"] = 1


        for px_k, py_k in topk_points[1:]:
            if hit_gt(px_k, py_k, bbox_gt):
                ele["hit_topk"] = 1
            
            pred_bbox = [
                px_k - IMAGE_PATCH_SIZE / img_w,
                py_k - IMAGE_PATCH_SIZE / img_h,
                px_k + IMAGE_PATCH_SIZE / img_w,
                py_k + IMAGE_PATCH_SIZE / img_h,
            ]
            if overlap_gt(pred_bbox, bbox_gt):
                ele["overlap_topk"] = 1

            ele["topk_pred_bboxes"].append(pred_bbox)
            
        
        # Optionally save patch_saliency_heatmap_overlay
        if args.save_saliency_heatmaps:
            base_name = os.path.splitext(os.path.basename(ele["file_name"]))[0]
            save_name = f"overlay_ps_pred_osworldg_{base_name}.png"
            overlays_saved = save_patch_saliency_heatmap_overlay(
                screenshot=image_input,
                pred=pred,
                out_dir=overlay_out_dir,
                save_name=save_name,
                image_patch_size=IMAGE_PATCH_SIZE,
            )

        results.append(ele)

    return results


def get_metric(list_of_examples, buckets):
    """
    Compute OS-World-G success rates by:
        1) per data type (as before)
        2) per capability group
    """

    id_to_groups = {}
    # Build id_to_groups from buckets["classified"][group] -> list of items with "id"
    # id_to_groups is a dictionary of id -> [group_names]
    if isinstance(buckets, dict):
        classified = buckets["classified"]
        if isinstance(classified, dict):
            for group_name, items in classified.items():
                for it in items:
                    ex_id = it["id"]
                    if ex_id not in id_to_groups:
                        id_to_groups[ex_id] = []
                    id_to_groups[ex_id].append(group_name)

    # Per-type (GUI_types) metrics
    per_type = defaultdict(lambda: {
        "count": 0,
        "sum_top1": 0,
        "sum_topk": 0,
        "sum_overlap_top1": 0,
        "sum_overlap_topk": 0,
    })
    overall_type = {
        "count": 0,
        "sum_top1": 0,
        "sum_topk": 0,
        "sum_overlap_top1": 0,
        "sum_overlap_topk": 0,
    }

    # Per-group (from classification_result.json) metrics
    per_group = defaultdict(lambda: {
        "count": 0,
        "sum_top1": 0,
        "sum_topk": 0,
        "sum_overlap_top1": 0,
        "sum_overlap_topk": 0,
    })
    overall_group = {
        "count": 0,
        "sum_top1": 0,
        "sum_topk": 0,
        "sum_overlap_top1": 0,
        "sum_overlap_topk": 0,
    }

    for ex in list_of_examples:
        acc1 = ex['hit_top1']
        acck = ex['hit_topk']
        overlap1 = ex.get('overlap_top1', 0)
        overlapk = ex.get('overlap_topk', 0)
        types_raw = ex.get('data_types', [])
        types = types_raw if isinstance(types_raw, list) else []

        # Per-type aggregation (always include)
        overall_type["count"] += 1
        overall_type["sum_top1"] += acc1
        overall_type["sum_topk"] += acck
        overall_type["sum_overlap_top1"] += overlap1
        overall_type["sum_overlap_topk"] += overlapk

        for t in types:
            d = per_type[t]
            d["count"] += 1
            d["sum_top1"] += acc1
            d["sum_topk"] += acck
            d["sum_overlap_top1"] += overlap1
            d["sum_overlap_topk"] += overlapk

        # Per-group aggregation (skip if id not mapped)
        ex_id = ex["id"]
        grps = id_to_groups.get(ex_id, [])

        if grps:
            for grp in grps:
                g = per_group[grp]
                g["count"] += 1
                g["sum_top1"] += acc1
                g["sum_topk"] += acck
                g["sum_overlap_top1"] += overlap1
                g["sum_overlap_topk"] += overlapk

            overall_group["count"] += 1
            overall_group["sum_top1"] += acc1
            overall_group["sum_topk"] += acck
            overall_group["sum_overlap_top1"] += overlap1
            overall_group["sum_overlap_topk"] += overlapk
        
    metrics = ["hit_top1", "overlap_top1", "hit_topk", "overlap_topk", "Count"]
    
    def safe_rate(s, c):
        return (s / c) if c > 0 else 0.0

    # ---- Per-Group Results ----
    columns_group = sorted(per_group.keys()) + ["All"]
    results_group = {m: {} for m in metrics}
    for g in sorted(per_group.keys()):
        c = per_group[g]["count"]
        results_group["hit_top1"][g] = safe_rate(per_group[g]["sum_top1"], c)
        results_group["overlap_top1"][g] = safe_rate(per_group[g]["sum_overlap_top1"], c)
        results_group["hit_topk"][g] = safe_rate(per_group[g]["sum_topk"], c)
        results_group["overlap_topk"][g] = safe_rate(per_group[g]["sum_overlap_topk"], c)
        results_group["Count"][g] = c

    results_group["hit_top1"]["All"] = safe_rate(overall_group["sum_top1"], overall_group["count"])
    results_group["overlap_top1"]["All"] = safe_rate(overall_group["sum_overlap_top1"], overall_group["count"])
    results_group["hit_topk"]["All"] = safe_rate(overall_group["sum_topk"], overall_group["count"])
    results_group["overlap_topk"]["All"] = safe_rate(overall_group["sum_overlap_topk"], overall_group["count"])
    results_group["Count"]["All"] = overall_group["count"]

    # Print console table (Per Group)
    if columns_group:
        header_group = [""] + columns_group
        col_widths_group = [max(len(col), 12) for col in header_group]
        header_line_g = " | ".join(word.ljust(width) for word, width in zip(header_group, col_widths_group))
        separator_line_g = "-+-".join("-" * width for width in col_widths_group)
        print(header_line_g)
        print(separator_line_g)
        for metric in metrics:
            row = [metric]
            for col in columns_group:
                val = results_group[metric].get(col)
                row.append(format_cell(val))
            row_line_g = " | ".join(word.ljust(width) for word, width in zip(row, col_widths_group))
            print(row_line_g)

    # ------------- Tab-delimited (Per Group) -------------
    metric_info = "Tab-delimited Table for Excel (By Group):\n"
    if columns_group:
        header_tab_g = "\t".join([""] + columns_group)
        metric_info += header_tab_g + "\n"
        for metric in metrics:
            row = [metric] + [format_cell(results_group[metric].get(col)) for col in columns_group]
            metric_info += ("\t".join(row) + "\n")
    else:
        metric_info += "(No group mappings found)\n"
    
    # Combine results
    results_all = {"ByGroup": results_group}
    # results_all = {"ByGroup": results_group, "ByType": results_type}
    return metric_info, results_all


"""
# cd to project root directory
python eval/os_world_g_eval.py --save_path <path_to_save_results>
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="focusui_3b")
    parser.add_argument("--model_name_or_path", type=str, default="checkpoints/focusui_3b")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--data_path", type=str, default="./dataset/OSWorld-G")
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

    parser.add_argument("--pure_grounding_eval", action="store_true", default=True, help="Pure grounding evaluation mode")

    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pred_path = f"{save_path}/osworld_g_preds.json"
    metric_path = f"{save_path}/osworld_g_metrics.txt"
    metric_json_path = f"{save_path}/osworld_g_metrics.json"

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

    # Try to load buckets.json for capability grouping
    class_buckets = json.load(open(os.path.join(args.data_path, "classification_result.json")))

    metric_info, results = get_metric(
        results,
        buckets=class_buckets,
    )

    with open(metric_path, "w") as f:
        f.write(metric_info)
    print(f"Saved metric to {metric_path}")

    with open(metric_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved metric to {metric_json_path}")
