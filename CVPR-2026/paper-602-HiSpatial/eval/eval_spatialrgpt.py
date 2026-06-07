"""Evaluate HiSpatial on the SpatialRGPT benchmark."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import argparse
import ast
import json
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from hispatial.inference import MoGeProcessor, HiSpatialPredictor

def eval_model(vlm_model_path, answers_file, gpu_rank=0):
    """Run evaluation on SpatialRGPT-Bench and save per-sample predictions."""
    processor = MoGeProcessor(device_name=f"cuda:{gpu_rank}")
    model_wrapper = HiSpatialPredictor(model_load_path=vlm_model_path, gpu_rank=gpu_rank)

    ds = load_dataset("a8cheng/SpatialRGPT-Bench")
    benchmark = ds["val"]

    with open(answers_file, "w") as ans_file:
        for data in benchmark:
            image_pil = data["image"]
            image_rgb = np.array(image_pil)
            xyz_values_orig = processor.apply_transform(image_pil)
            # Flip augmentation: run MoGe on horizontally flipped image, unflip XYZ, average
            image_pil_flipped = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            xyz_values_flipped = processor.apply_transform(image_pil_flipped)
            xyz_values_unflipped = torch.flip(xyz_values_flipped, [2])
            xyz_values_unflipped[0] = -xyz_values_unflipped[0]
            xyz_values = (xyz_values_orig + xyz_values_unflipped) / 2.0

            qa_info = ast.literal_eval(data["qa_info"])
            if qa_info["type"] != 'quantitative':
                continue
            conversations = ast.literal_eval(data["conversations"])
            bbox = ast.literal_eval(data["bbox"])

            class_names = qa_info["class"]
            bbox_color = ["red", "blue", "green"]
            obj_refer = [
                f"the {name} (highlighted by a {bbox_color[i]} box)"
                for i, name in enumerate(class_names)
            ]

            for i, (x1, y1, x2, y2) in enumerate(bbox):
                color = [(255, 0, 0), (0, 0, 255), (0, 255, 0)][i] if i < 3 else (255, 255, 0)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)

            input_text = conversations[0]["value"]
            for obj in obj_refer:
                input_text = input_text.replace("<mask>", obj, 1)

            model_out = model_wrapper.query(image=image_rgb, prompt=input_text, xyz_values=xyz_values)

            ans_file.write(json.dumps({
                "question_id": data["id"],
                "question": input_text,
                "pred": model_out,
                "gt": conversations[1]["value"],
                "qa_info": qa_info,
            }) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HiSpatial on SpatialRGPT-Bench")
    parser.add_argument("--vlm_model_path", type=str, required=True, help="Path to VLM checkpoint (weights.pt)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save predictions (.jsonl)")
    parser.add_argument("--gpu_rank", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_args()
    eval_model(
        vlm_model_path=args.vlm_model_path,
        answers_file=args.save_path,
        gpu_rank=args.gpu_rank,
    )


if __name__ == "__main__":
    main()
