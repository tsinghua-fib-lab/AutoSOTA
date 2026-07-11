"""Evaluation script: Load trained 2l1h64d3lr01drop model and evaluate on Unique Copy."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer
import torch
import numpy as np
from patching_utils import set_seed
from patching_data import *
from train_new_models import customCollator, compute_metrics
import json
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/repo/share/saved_models/unique_copy-@2l1h64d3lr01drop")
    parser.add_argument("--test_num", type=int, default=2000)
    args = parser.parse_args()

    set_seed(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))

    task = "unique_copy"
    test_length_ranges = [(0, 50), (51, 100), (101, 150)]
    max_test_length = test_length_ranges[-1][1]
    test_num = args.test_num

    print("=" * 60)
    print("Evaluating model on Unique Copy task")
    print("Model path: {}".format(args.model_path))
    print("Test ranges: {}".format(test_length_ranges))
    print("Samples per range: {}".format(test_num))
    print("=" * 60)

    # Tokenizer and test datasets
    tokenizer = get_tokenizer_for_task(task, max_test_length)
    print("Vocab size: {}".format(len(tokenizer)))

    test_dataset = {}
    for test_range in test_length_ranges:
        key = "len{}-{}".format(test_range[0], test_range[1])
        test_dataset[key] = EvalDataset(
            get_dataset_for_task(task, tokenizer, test_range, -1, {"period_for_data": 3}), test_num
        )

    # Load model
    print("\nLoading model from {}...".format(args.model_path))
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {:,}".format(total_params))

    # Setup trainer for evaluation
    per_device_bz = 64 // torch.cuda.device_count() if torch.cuda.is_available() else 64
    n_positions = max_test_length * 2 + 3

    training_args = TrainingArguments(
        output_dir="/tmp/eval_temp",
        per_device_eval_batch_size=per_device_bz,
        report_to="none",
        seed=0,
    )

    data_collator = customCollator(tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Evaluate on all three length ranges
    print("\n--- Evaluation Results ---")
    results = {}
    for test_range in test_length_ranges:
        key = "len{}-{}".format(test_range[0], test_range[1])
        result = trainer.evaluate(eval_dataset=test_dataset[key])
        acc = result.get("eval_{}_acc".format(key), result.get("eval_acc", 0))
        metric_name = "Task Acc in < {}".format(test_range[1]) if test_range[0] == 0 else "Task Acc in [{}-{}]".format(test_range[0], test_range[1])
        results[metric_name] = acc
        print("  {}: {:.4f} ({:.1f}%)".format(metric_name, acc, acc * 100))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, val in results.items():
        print("  {}: {:.4f} ({:.1f}%)".format(name, val, val * 100))

    # Compare with paper targets
    paper_targets = {
        "Task Acc in < 50": 100.0,
        "Task Acc in [51-100]": 100.0,
        "Task Acc in [101-150]": 99.2,
    }
    print("\n--- Comparison with Paper (Table 1, Appendix E) ---")
    for name, val in results.items():
        target = paper_targets.get(name, 0)
        status = "MATCH" if val * 100 >= target else "BELOW"
        print("  {}: ours={:.1f}%, paper={:.1f}% [{}]".format(name, val * 100, target, status))

    return results


if __name__ == "__main__":
    main()
