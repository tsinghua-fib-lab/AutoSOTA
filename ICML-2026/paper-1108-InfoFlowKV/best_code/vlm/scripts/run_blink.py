"""Run baseline inference on BLINK dataset."""

import argparse
import yaml
import torch
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import get_dataset
from inference import run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Model config
    model_name = cfg["model"]
    cache_dir = cfg["cache_dir"]

    # Dataset config
    dataset_name = cfg["dataset"]
    num_samples = cfg.get("num_samples")
    dataset_dir = cfg.get("dataset_dir")
    output_dir = cfg.get("output_dir", "./output")

    # Generation config
    max_new_tokens = cfg.get("max_new_tokens", 128)

    print("=" * 60)
    print("Baseline Inference for Qwen3-VL")
    print("=" * 60)

    print(f"\nLoading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    # Load dataset
    dataset = get_dataset(dataset_name, dataset_dir=dataset_dir, output_dir=output_dir)
    if num_samples:
        dataset.data = dataset.data.select(range(min(num_samples, len(dataset))))

    print(f"Dataset: {dataset_name}, samples: {len(dataset)}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(output_dir, f"{dataset_name}_baseline", timestamp)
    os.makedirs(output_subdir, exist_ok=True)

    # Run inference
    results, correct, total = [], 0, 0

    for i in tqdm(range(len(dataset)), desc="Processing"):
        sample = dataset[i]
        batch = [sample]

        predictions, _, _ = run_inference(
            model, processor, batch, dataset,
            max_new_tokens=max_new_tokens,
            attention_patch=None,
        )

        pred = predictions[0]
        results.append(dataset.get_result_dict(sample, pred))

        if dataset.check_correct(sample, pred):
            correct += 1
        total += 1

        if i < 3 or (i + 1) % 50 == 0:
            print(f"\n[{sample['idx']}] Answer: {sample['answer']} | Pred: {pred[:60]}...")

    # Save results
    accuracy = correct / total * 100 if total else 0

    with open(os.path.join(output_subdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_subdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"Results saved to {output_subdir}")


if __name__ == "__main__":
    main()
