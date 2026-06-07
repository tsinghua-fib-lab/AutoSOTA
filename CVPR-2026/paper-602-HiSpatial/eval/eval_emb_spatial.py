"""Evaluate HiSpatial on the EmbSpatial benchmark."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import argparse
import base64
import json
import os
from io import BytesIO

from PIL import Image
from hispatial.inference import MoGeProcessor, HiSpatialPredictor


def test(vlm_model_path, save_path, benchmark_path, gpu_rank=0):
    processor = MoGeProcessor(device_name=f"cuda:{gpu_rank}")
    model_wrapper = HiSpatialPredictor(model_load_path=vlm_model_path, gpu_rank=gpu_rank)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(benchmark_path, "r") as f:
        bench_raw = json.load(f)

    correct_num = 0
    total_num = 0

    for data in bench_raw:
        image_data = base64.b64decode(data["image"])
        image = Image.open(BytesIO(image_data))

        question = data["question"]
        answer_options = data["answer_options"]
        model_input = question
        for letter, answer in zip("ABCDEF"[:len(answer_options)], answer_options):
            model_input += f"\n{letter}. {answer}"

        gt_answer = "ABCDEF"[data["answer"]]

        xyz_values = processor.apply_transform(image)
        model_out = model_wrapper.query(image=image, prompt=model_input, xyz_values=xyz_values)
        model_out = model_out[:2]

        if gt_answer in model_out:
            correct_num += 1
        total_num += 1

    with open(f"{save_path}_accuracy.json", "w") as f:
        json.dump({
            "total_questions": total_num,
            "correct_answers": correct_num,
            "accuracy": correct_num / total_num,
        }, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HiSpatial on EmbSpatial")
    parser.add_argument("--vlm_model_path", type=str, required=True, help="Path to VLM checkpoint (weights.pt)")
    parser.add_argument("--save_path", type=str, required=True, help="Path prefix to save results")
    parser.add_argument("--benchmark_path", type=str, required=True, help="Path to embspatial_bench2.json")
    parser.add_argument("--gpu_rank", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_args()
    test(
        vlm_model_path=args.vlm_model_path,
        save_path=args.save_path,
        benchmark_path=args.benchmark_path,
        gpu_rank=args.gpu_rank,
    )


if __name__ == "__main__":
    main()
