"""Evaluate HiSpatial on the RoboSpatial-Home benchmark."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from hispatial.inference import MoGeProcessor, HiSpatialPredictor


def test(vlm_model_path, save_path, gpu_rank=0):
    processor = MoGeProcessor(device_name=f"cuda:{gpu_rank}")
    model_wrapper = HiSpatialPredictor(model_load_path=vlm_model_path, gpu_rank=gpu_rank)

    dataset = load_dataset("chanhee-luke/RoboSpatial-Home")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    correct_num = 0
    total_num = 0

    for item in dataset["configuration"]:
        image = item["img"]
        xyz_values = processor.apply_transform(image)

        model_out = model_wrapper.query(image=image, prompt=item["question"], xyz_values=xyz_values)

        answer_lower = item["answer"].lower()
        output_lower = model_out.lower()
        if (answer_lower in output_lower
                or ("true" in output_lower and answer_lower == "yes")
                or ("false" in output_lower and answer_lower == "no")):
            correct_num += 1
        total_num += 1

    with open(f"{save_path}_accuracy.json", "w") as f:
        json.dump({
            "total_questions": total_num,
            "correct_answers": correct_num,
            "accuracy": correct_num / total_num,
        }, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HiSpatial on RoboSpatial-Home")
    parser.add_argument("--vlm_model_path", type=str, required=True, help="Path to VLM checkpoint (weights.pt)")
    parser.add_argument("--save_path", type=str, required=True, help="Path prefix to save results")
    parser.add_argument("--gpu_rank", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_args()
    test(
        vlm_model_path=args.vlm_model_path,
        save_path=args.save_path,
        gpu_rank=args.gpu_rank,
    )


if __name__ == "__main__":
    main()
