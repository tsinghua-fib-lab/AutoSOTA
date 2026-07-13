#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from glue_data_utils import (
    GLUE_TASKS,
    EarlyStopping,
    compute_glue_metrics,
    create_glue_collate_fn,
    load_glue_data,
)

from GLUE_PIPELINE.GLUE_CONFIG import (
    BATCH_SIZE,
    DATA_CACHE,
    DEFAULT_GLUE_TASKS,
    EARLY_STOPPING_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MAX_SEQ_LEN,
    MODEL_SPECS,
    OUTPUT_ROOT,
    TOKENIZER_PATH,
    WEIGHT_DECAY,
)
from GLUE_PIPELINE.GLUE_MODELS import GLUEModelWrapper, build_glue_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune custom checkpoints on GLUE")
    parser.add_argument("--model", required=True, choices=MODEL_SPECS.keys(), help="Model key to run")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=DEFAULT_GLUE_TASKS,
        help="Subset of GLUE tasks to train (defaults to full list)",
    )
    parser.add_argument("--tokenizer_path", type=Path, default=TOKENIZER_PATH, help="Tokenizer JSON path")
    parser.add_argument("--data_cache", type=Path, default=DATA_CACHE, help="GLUE cache directory")
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT, help="Where to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Override warmup ratio")
    parser.add_argument("--max_length", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--early_delta", type=float, default=EARLY_STOPPING_DELTA)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _should_skip_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    n = name.lower()
    if param.ndim <= 1 or n.endswith("bias") or ".bias" in n or "norm" in n:
        return True
    proto_keys = ("p_tables", "alpha", "tau", "tau_read", "logit_beta")
    if any(k in n for k in proto_keys):
        return True
    mamba_keys = ("dt_proj", "dt_bias", "conv1d.bias", "norm_f")
    if any(k in n for k in mamba_keys):
        return True
    delta_keys = ("delta", "gate")
    if any(k in n for k in delta_keys):
        return True
    return False

def build_optimizer(model: GLUEModelWrapper, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if _should_skip_weight_decay(name, param) else decay).append(param)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)

def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def evaluate(model: GLUEModelWrapper, dataloader: DataLoader, task: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, labels, losses = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = model(input_ids, labels=batch_labels)
            losses.append(outputs["loss"].item())
            logits = outputs["logits"].detach().cpu()
            if GLUE_TASKS[task]["is_regression"]:
                preds.extend(logits.squeeze().float().tolist())
                labels.extend(batch_labels.squeeze().float().cpu().tolist())
            else:
                preds.extend(torch.argmax(logits, dim=-1).tolist())
                labels.extend(batch_labels.cpu().tolist())
    metrics = compute_glue_metrics(task, np.array(preds), np.array(labels))
    metrics["eval_loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics

def main():
    args = parse_args()
    set_seed(args.seed)

    spec = MODEL_SPECS[args.model]
    tokenizer_path = args.tokenizer_path
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    model_path = spec.checkpoint
    lr = args.lr or spec.lr
    warmup_ratio = args.warmup_ratio or spec.warmup_ratio

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    for task in args.tasks:
        if task not in GLUE_TASKS:
            raise ValueError(f"Unsupported GLUE task: {task}")
        task_dir = args.output_root / spec.key / task
        task_dir.mkdir(parents=True, exist_ok=True)

        logger = _create_logger(task_dir / f"{task}.log")
        logger.info("Starting task %s for model %s", task, spec.key)
        model, tokenizer, _ = build_glue_model(spec.model_type, model_path, tokenizer_path, task)
        model.to(device)

        train_dataset, dev_dataset = load_glue_data(
            task,
            tokenizer,
            args.max_length,
            cache_dir=args.data_cache,
            seed=args.seed,
        )
        collate_fn = create_glue_collate_fn()
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

        total_steps = len(train_loader) * args.epochs
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        optimizer = build_optimizer(model, lr=lr, weight_decay=args.weight_decay)
        scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
        stopper = EarlyStopping(patience=args.patience, min_delta=args.early_delta, maximize=True)

        best_metric = float("-inf")
        history: List[Dict[str, float]] = []
        best_model_path = task_dir / "best_model.pt"
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                global_step += 1
                if global_step % args.log_steps == 0:
                    logger.info(
                        "Epoch %d step %d/%d | loss %.4f | lr %.2e",
                        epoch,
                        step,
                        len(train_loader),
                        running_loss / step,
                        optimizer.param_groups[0]["lr"],
                    )

            dev_metrics = evaluate(model, dev_loader, task, device)
            main_metric = _select_main_metric(task, dev_metrics)
            history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader), **dev_metrics})
            logger.info("Epoch %d dev metrics: %s", epoch, json.dumps(dev_metrics, ensure_ascii=False))

            if main_metric > best_metric:
                best_metric = main_metric
                torch.save(model.state_dict(), best_model_path)
                _write_json(
                    {
                        "task": task,
                        "model": spec.key,
                        "metric": best_metric,
                        "metrics": dev_metrics,
                        "epoch": epoch,
                        "lr": lr,
                        "warmup_ratio": warmup_ratio,
                    },
                    task_dir / "best_dev_metrics.json",
                )

            if stopper(main_metric):
                logger.info("Early stopping triggered after epoch %d", epoch)
                break

        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            final_metrics = evaluate(model, dev_loader, task, device)
        else:
            final_metrics = history[-1] if history else {"status": "no-training"}

        _write_json(history, task_dir / "training_history.json")
        summary = {
            "model": spec.key,
            "model_type": spec.model_type,
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "task": task,
            "best_metric": best_metric,
            "final_dev_metrics": final_metrics,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": lr,
            "warmup_ratio": warmup_ratio,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
        }
        _write_json(summary, task_dir / "training_args.json")
        logger.info("Task %s finished. Best metric %.4f", task, best_metric)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _select_main_metric(task: str, metrics: Dict[str, float]) -> float:
    target = GLUE_TASKS[task]["metric"]
    if target == "pearson_spearman":
        return metrics.get("pearson_spearman", 0.0)
    if target == "matthews_corrcoef":
        return metrics.get("matthews_corrcoef", 0.0)
    if target == "f1":
        return metrics.get("f1", 0.0)
    return metrics.get("accuracy", 0.0)

def _write_json(payload, path: Path):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

if __name__ == "__main__":
    main()
