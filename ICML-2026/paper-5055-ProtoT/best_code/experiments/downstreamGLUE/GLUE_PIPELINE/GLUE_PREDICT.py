#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from glue_data_utils import GLUE_TASKS, GLUEDataset, create_glue_collate_fn
from GLUE_PIPELINE.GLUE_CONFIG import (
    BATCH_SIZE,
    DATA_CACHE,
    LABEL_MAPPINGS,
    MODEL_SPECS,
    OUTPUT_ROOT,
    SUBMISSION_ROOT,
    TASK_TO_SUBMISSION_FILE,
    TASK_TO_TEST_SPLIT,
    TOKENIZER_PATH,
)
from GLUE_PIPELINE.GLUE_MODELS import build_glue_model

DEFAULT_PRED_TASKS = [
    "cola",
    "sst2",
    "mrpc",
    "qqp",
    "stsb",
    "mnli",
    "mnli-mm",
    "qnli",
    "rte",
    "wnli",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Create GLUE submissions for a fine-tuned model")
    parser.add_argument("--model", required=True, choices=MODEL_SPECS.keys())
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_PRED_TASKS)
    parser.add_argument("--checkpoint_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--submission_root", type=Path, default=SUBMISSION_ROOT)
    parser.add_argument("--tokenizer_path", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--data_cache", type=Path, default=DATA_CACHE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--include_ax", action="store_true", help="Also create AX.tsv using the MNLI checkpoint")
    return parser.parse_args()

def load_trained_model(model_key: str, task_dir: Path, tokenizer_path: Path, task_name: str, device: torch.device):
    if not task_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing for task {task_name}: {task_dir}")

    config_path = task_dir / "training_args.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type", MODEL_SPECS[model_key].model_type)
    model_path = Path(config.get("model_path", MODEL_SPECS[model_key].checkpoint))
    tokenizer_override = Path(config.get("tokenizer_path", tokenizer_path))

    model, tokenizer, _ = build_glue_model(model_type, model_path, tokenizer_override, _base_task(task_name))
    state_dict = torch.load(task_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer

def _base_task(task: str) -> str:
    return "mnli" if task in {"mnli-mm", "ax"} else task

def load_split(task: str, tokenizer, split: str, max_length: int, cache_dir: Path):
    dataset_name = "mnli" if task in {"mnli", "mnli-mm"} else task
    if task == "ax":
        dataset_name = "ax"
    dataset = load_dataset("glue", dataset_name, cache_dir=cache_dir)
    if split == "test":
        split_name = TASK_TO_TEST_SPLIT[task]
    else:
        split_name = GLUE_TASKS[_base_task(task)]["eval_split"]
    raw_split = dataset[split_name]
    glued = GLUEDataset(raw_split, tokenizer, max_length=max_length, task_name=_base_task(task))
    return glued

def predict(model, dataloader: DataLoader, task: str, device: torch.device) -> List[float]:
    outputs: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)["logits"].detach().cpu()
            if GLUE_TASKS[_base_task(task)]["is_regression"]:
                outputs.extend(logits.squeeze().float().tolist())
            else:
                outputs.extend(torch.argmax(logits, dim=-1).tolist())
    return outputs

def convert_to_labels(task: str, predictions: List[float]) -> List[str]:
    base = _base_task(task)
    if GLUE_TASKS[base]["is_regression"]:
        def _clip(val: float) -> float:
            if task == "stsb":
                return max(0.0, min(5.0, val))
            return val

        return [f"{_clip(float(p)):.4f}" for p in predictions]
    mapping = LABEL_MAPPINGS.get(task) or LABEL_MAPPINGS.get(base)
    if mapping:
        return [mapping[int(p)] for p in predictions]
    return [str(int(p)) for p in predictions]

def write_submission(file_path: Path, labels: List[str]):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write("index\tprediction\n")
        for idx, label in enumerate(labels):
            f.write(f"{idx}\t{label}\n")

def main():
    args = parse_args()
    device = torch.device(args.device)
    model_spec = MODEL_SPECS[args.model]
    tasks = list(args.tasks)
    if args.include_ax and "ax" not in tasks:
        tasks.append("ax")

    submission_dir = args.submission_root / args.model
    submission_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    for task in tasks:
        if task == "mnli-mm":
            source_task = "mnli"
        elif task == "ax":
            source_task = "mnli"
        else:
            source_task = task
        task_dir = args.checkpoint_root / model_spec.key / source_task
        model, tokenizer = load_trained_model(args.model, task_dir, args.tokenizer_path, task, device)

        dataset = load_split(task, tokenizer, args.split, args.max_length, args.data_cache)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=create_glue_collate_fn())
        raw_predictions = predict(model, dataloader, task, device)
        labels = convert_to_labels(task, raw_predictions)

        filename = TASK_TO_SUBMISSION_FILE.get(task, f"{task}.tsv")
        output_file = submission_dir / filename
        write_submission(output_file, labels)
        generated_files.append(output_file)
        print(f"Saved {task} predictions to {output_file}")

    archive_path = submission_dir / f"{args.model}_{args.split}_submission.zip"
    import zipfile

    with zipfile.ZipFile(archive_path, "w") as zf:
        for file_path in generated_files:
            zf.write(file_path, arcname=file_path.name)
    print(f"Submission archive ready: {archive_path}")

if __name__ == "__main__":
    main()
