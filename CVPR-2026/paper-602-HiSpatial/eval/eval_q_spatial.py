"""Evaluate HiSpatial on the Q-Spatial benchmark (QSpatial+ and QSpatial-ScanNet)."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import argparse
import json
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from hispatial.inference import MoGeProcessor, HiSpatialPredictor


def _load_rgb_and_xyz_from_scannet(image_path, scannet_images_dir):
    """Load RGB image and compute XYZ point cloud from ScanNet depth + intrinsics.

    Args:
        image_path: Path in dataset (e.g. ``.../scene0015_00/color/0.jpg``).
        scannet_images_dir: Root directory containing ScanNet scene folders.

    Returns:
        Tuple of (PIL Image, dict with ``points`` and ``mask`` tensors).
    """
    color_dir, fname = os.path.split(image_path)
    scan_name = os.path.basename(os.path.dirname(color_dir))
    base_dir = os.path.join(scannet_images_dir, scan_name)
    frame_id = os.path.splitext(fname)[0]

    depth_path = os.path.join(base_dir, "depth", f"{frame_id}.png")
    intrinsics_path = os.path.join(base_dir, "intrinsics_depth.npy")
    image_path = os.path.join(base_dir, "color", f"{frame_id}.jpg")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"RGB not found: {image_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")

    rgb_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    K = np.load(intrinsics_path)

    # ScanNet depth is in millimeters
    depth = depth / 1000.0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    mask = (Z > 0.01) & (Z < 250.0)
    xyz_map = np.stack([X, Y, Z], axis=-1)

    mask = torch.from_numpy(mask)
    xyz_map = torch.from_numpy(xyz_map).float()

    return pil_image, {"points": xyz_map, "mask": mask}


def test_qspatial_plus(vlm_model_path, save_path, gpu_rank=0):
    """Evaluate on the QSpatial+ split (estimated depth)."""
    processor = MoGeProcessor(device_name=f"cuda:{gpu_rank}")
    model_wrapper = HiSpatialPredictor(model_load_path=vlm_model_path, gpu_rank=gpu_rank)

    dataset = load_dataset("andrewliao11/Q-Spatial-Bench")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for index, item in enumerate(dataset["QSpatial_plus"]):
        image = item["image"]
        xyz_values = processor.apply_transform(image)
        model_out = model_wrapper.query(image=image, prompt=item["question"], xyz_values=xyz_values)

        answer_value = item["answer_value"]
        if item["answer_unit"] == "centimeter":
            answer_value = answer_value / 100.0

        with open(f"{save_path}_plus.jsonl", "a") as f:
            f.write(json.dumps({
                "idx": f"{index:04d}",
                "question": item["question"],
                "model_output": model_out,
                "gt_value": answer_value,
            }) + "\n")


def test_qspatial_scannet(vlm_model_path, save_path, scannet_images_dir, use_gt_depth=False, gpu_rank=0):
    """Evaluate on the QSpatial-ScanNet split (ground-truth or estimated depth)."""
    processor = MoGeProcessor(device_name=f"cuda:{gpu_rank}")
    model_wrapper = HiSpatialPredictor(model_load_path=vlm_model_path, gpu_rank=gpu_rank)

    dataset = load_dataset("andrewliao11/Q-Spatial-Bench")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if "QSpatial_scannet" not in dataset:
        raise KeyError("Dataset missing 'QSpatial_scannet' split.")

    for index, item in enumerate(dataset["QSpatial_scannet"]):
        try:
            if "image_path" not in item:
                raise KeyError(f"Item {index} has no 'image_path' key.")

            pil_image, xyz_dict = _load_rgb_and_xyz_from_scannet(item["image_path"], scannet_images_dir)

            if not use_gt_depth:
                xyz_dict = processor.infer_depth(pil_image)

            model_out = model_wrapper.query(image=pil_image, prompt=item["question"], xyz_dict=xyz_dict)

            answer_value = item.get("answer_value")
            if answer_value is not None and item.get("answer_unit") == "centimeter":
                answer_value = answer_value / 100.0

            with open(f"{save_path}_scannet.jsonl", "a") as f:
                f.write(json.dumps({
                    "idx": f"{index:04d}",
                    "image_path": item["image_path"],
                    "question": item["question"],
                    "model_output": model_out,
                    "gt_value": answer_value,
                    "gt_unit": "meter" if item.get("answer_unit") == "centimeter" else item.get("answer_unit"),
                }) + "\n")
        except Exception as e:
            print(f"Error processing item {index}: {e}")
            continue


def calc_acc(json_path):
    """Compute accuracy: prediction within [0.5x, 2x] of ground truth."""
    with open(json_path, "r") as f:
        qa_data = [json.loads(line) for line in f]

    total = len(qa_data)
    correct = 0
    for item in qa_data:
        match = re.search(r"[-+]?\d*\.\d+|\d+", item["model_output"])
        if not match:
            continue
        model_out = float(match.group())
        gt_value = float(item["gt_value"])
        if 0.5 * gt_value < model_out < 2.0 * gt_value:
            correct += 1

    return correct / total if total > 0 else 0.0, correct, total


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HiSpatial on Q-Spatial Bench")
    parser.add_argument("--vlm_model_path", type=str, required=True, help="Path to VLM checkpoint (weights.pt)")
    parser.add_argument("--save_path", type=str, required=True, help="Path prefix to save results")
    parser.add_argument("--scannet_images_dir", type=str, required=True, help="Root directory of ScanNet scene images")
    parser.add_argument("--use_gt_depth", action="store_true", help="Use ground-truth depth for ScanNet split")
    parser.add_argument("--gpu_rank", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_args()

    test_qspatial_plus(
        vlm_model_path=args.vlm_model_path,
        save_path=args.save_path,
        gpu_rank=args.gpu_rank,
    )
    test_qspatial_scannet(
        vlm_model_path=args.vlm_model_path,
        save_path=args.save_path,
        scannet_images_dir=args.scannet_images_dir,
        use_gt_depth=args.use_gt_depth,
        gpu_rank=args.gpu_rank,
    )

    acc_plus, correct_plus, total_plus = calc_acc(f"{args.save_path}_plus.jsonl")
    acc_scannet, correct_scannet, total_scannet = calc_acc(f"{args.save_path}_scannet.jsonl")

    with open(f"{args.save_path}_accuracy.json", "w") as f:
        json.dump({
            "QSpatial_plus": {
                "total_questions": total_plus,
                "correct_answers": correct_plus,
                "accuracy": acc_plus,
            },
            "QSpatial_scannet": {
                "total_questions": total_scannet,
                "correct_answers": correct_scannet,
                "accuracy": acc_scannet,
            },
            "accuracy": (correct_plus + correct_scannet) / (total_plus + total_scannet)
                if (total_plus + total_scannet) > 0 else 0.0,
        }, f, indent=4)


if __name__ == "__main__":
    main()
