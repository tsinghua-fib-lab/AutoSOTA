"""Evaluate HiSpatial on CV-Bench (2D Relation + 3D)."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import argparse
import json
import os
import re

import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from hispatial.inference import MoGeProcessor, HiSpatialPredictor


def load_cv_bench_3d():
    ds = load_dataset("nyu-visionx/CV-Bench", "3D")
    return ds["test"]


def load_cv_bench_2d_relation():
    ds = load_dataset("nyu-visionx/CV-Bench", "2D")
    return [item for item in ds["test"] if item["task"] == "Relation"]


def test_cv_bench_3d(vlm_model_path, save_path, gpu_rank=0):
    """Evaluate on CV-Bench 3D split (depth + distance)."""
    cv_bench_3d = load_cv_bench_3d()
    processor = MoGeProcessor(device_name=f"cuda:{gpu_rank}")
    model_wrapper = HiSpatialPredictor(model_load_path=vlm_model_path, gpu_rank=gpu_rank)

    overall_correct = 0
    overall_total = 0

    for i in tqdm(range(len(cv_bench_3d)), desc="CV-Bench 3D"):
        image = cv_bench_3d[i]["image"].convert("RGB")
        prompt = cv_bench_3d[i]["prompt"]

        xyz_values = processor.apply_transform(image)
        image_np = np.array(image)

        result = model_wrapper.query(image=image_np, prompt=prompt, xyz_values=xyz_values)

        gt = cv_bench_3d[i]["answer"].replace("(", "").replace(")", "").replace(" ", "")
        result = result.replace("<eos>", "").replace(" ", "").replace(")", "").replace("(", "")

        if gt in result:
            overall_correct += 1
        overall_total += 1

    with open(f"{save_path}_3d_accuracy.json", "w") as f:
        json.dump({
            "total_questions": overall_total,
            "correct_answers": overall_correct,
            "accuracy": overall_correct / overall_total if overall_total > 0 else 0.0,
        }, f, indent=4)


def test_cv_bench_2d_relation(vlm_model_path, save_path, gpu_rank=0):
    """Evaluate on CV-Bench 2D Relation split."""
    cv_bench_2d = load_cv_bench_2d_relation()
    processor = MoGeProcessor(device_name=f"cuda:{gpu_rank}")
    model_wrapper = HiSpatialPredictor(model_load_path=vlm_model_path, gpu_rank=gpu_rank)

    overall_correct = 0
    overall_total = 0

    for i in tqdm(range(len(cv_bench_2d)), desc="CV-Bench 2D Relation"):
        image = cv_bench_2d[i]["image"].convert("RGB")
        prompt = cv_bench_2d[i]["prompt"]

        xyz_values = processor.apply_transform(image)
        image_np = np.array(image)

        result = model_wrapper.query(image=image_np, prompt=prompt, xyz_values=xyz_values)

        gt = cv_bench_2d[i]["answer"].replace("(", "").replace(")", "").replace(" ", "")
        result = result.replace("<eos>", "").replace(" ", "").replace(")", "").replace("(", "")

        if gt in result:
            overall_correct += 1
        overall_total += 1

    with open(f"{save_path}_relation_accuracy.json", "w") as f:
        json.dump({
            "total_questions": overall_total,
            "correct_answers": overall_correct,
            "accuracy": overall_correct / overall_total if overall_total > 0 else 0.0,
        }, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HiSpatial on CV-Bench")
    parser.add_argument("--vlm_model_path", type=str, required=True, help="Path to VLM checkpoint (weights.pt)")
    parser.add_argument("--save_path", type=str, required=True, help="Path prefix to save results")
    parser.add_argument("--gpu_rank", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_args()
    test_cv_bench_3d(
        vlm_model_path=args.vlm_model_path,
        save_path=args.save_path,
        gpu_rank=args.gpu_rank,
    )
    test_cv_bench_2d_relation(
        vlm_model_path=args.vlm_model_path,
        save_path=args.save_path,
        gpu_rank=args.gpu_rank,
    )


if __name__ == "__main__":
    main()
