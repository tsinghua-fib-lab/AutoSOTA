"""
Real focal loss for finetuned models (ICML rebuttal).

FL(p_t) = -(1 - p_t)^gamma * log(p_t)

For finetuned models (roberta_finetune, resnet_finetune), we modify the loss
function directly in the training loop. This is true focal loss, not a
two-stage approximation.

For frozen-embedding models (tfidf, histgbm, logreg, resnet50), use
run_focal_loss.py which does two-stage reweighting (the only option for sklearn).

Usage:
    python rebuttal/training/focal_loss/run_focal_finetune.py --datasets turkey --gammas 2 --seeds 42
    python rebuttal/training/focal_loss/run_focal_finetune.py --datasets jigsaw turkey inaturalist
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
os.chdir(PROJECT_ROOT)

from data import load_dataset
from core.seed import set_seed

SEEDS = [42, 123, 456, 0, 1, 7, 13, 99, 2024, 314]

DATASET_MODEL_MAP = {
    'jigsaw': 'roberta_finetune',
    'turkey': 'resnet_finetune',
    'inaturalist': 'resnet_finetune',
}


def focal_loss(logits, labels, gamma):
    """
    Focal loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Equivalent to: focal_weight * CE, where focal_weight = (1 - p_t)^gamma
    and p_t = probability assigned to the true class.
    """
    ce = F.cross_entropy(logits, labels, reduction='none')
    p_t = torch.exp(-ce)
    return ((1 - p_t) ** gamma) * ce


def train_resnet_focal(dataset, gamma, seed):
    """Train ResNet with focal loss. Self-contained — no monkey-patching."""
    from torchvision import transforms, models
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    # Reuse ImageDataset from source
    from models.resnet_finetune import ImageDataset

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = dataset.train.X
    if isinstance(X_train, np.ndarray):
        X_train = X_train.tolist()

    # Transforms (same as source)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
    ])

    train_dataset = ImageDataset(X_train, dataset.train.y, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # Build model (same as source)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training loop — FOCAL LOSS instead of CE
    model.train()
    for epoch in range(10):
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = focal_loss(outputs, labels, gamma).mean()  # <-- THE ONLY CHANGE
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")
        scheduler.step()

    # Predict on val and test
    model.eval()
    results = []
    for split_name, split_data in [('val', dataset.val), ('test', dataset.test)]:
        if split_data is None:
            continue
        X_eval = split_data.X if isinstance(split_data.X, list) else split_data.X.tolist()
        eval_dataset = ImageDataset(X_eval, np.zeros(len(X_eval)), transform=eval_transform)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)

        all_probs = []
        with torch.no_grad():
            for batch in eval_loader:
                outputs = model(batch["pixel_values"].to(device))
                probs = torch.softmax(outputs, dim=-1)
                all_probs.append(probs.cpu().numpy())

        proba = np.concatenate(all_probs)
        preds = proba.argmax(axis=1)

        for i in range(len(split_data.y)):
            idx = split_data.indices[i] if split_data.indices is not None else i
            results.append({
                'instance_id': int(idx), 'split': split_name,
                'y_true': int(split_data.y[i]), 'y_pred': int(preds[i]),
                'y_proba_0': float(proba[i, 0]), 'y_proba_1': float(proba[i, 1]),
                'delta_signed': float(split_data.delta[i]),
                'abs_delta': float(split_data.abs_delta[i]),
                'y_star': int(split_data.y[i]),
            })

    return results


def train_roberta_focal(dataset, gamma, seed):
    """Train RoBERTa with focal loss. Self-contained — no monkey-patching."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from models.roberta_finetune import TextDataset, WeightedTrainer

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    train_dataset = TextDataset(dataset.train.X, dataset.train.y, tokenizer, max_len=128)

    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # Subclass WeightedTrainer to use focal loss
    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            inputs.pop("idx", None)
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            loss = focal_loss(logits.view(-1, model.config.num_labels), labels.view(-1), gamma)
            loss = loss.mean()

            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir="outputs/focal_roberta_temp",
        overwrite_output_dir=True,
        eval_strategy="no", save_strategy="no",
        logging_strategy="steps", logging_steps=100,
        num_train_epochs=3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=2e-5, weight_decay=0.01,
        gradient_accumulation_steps=4,
        bf16=torch.cuda.is_available(),
        seed=seed, report_to="none", dataloader_num_workers=4,
    )

    trainer = FocalTrainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()

    # Predict on val and test
    results = []
    for split_name, split_data in [('val', dataset.val), ('test', dataset.test)]:
        if split_data is None:
            continue
        eval_dataset = TextDataset(split_data.X, np.zeros(len(split_data.X), dtype=int), tokenizer, max_len=128)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=128, shuffle=False)

        model.eval()
        all_probs = []
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        proba = np.concatenate(all_probs)
        preds = proba.argmax(axis=1)

        for i in range(len(split_data.y)):
            idx = split_data.indices[i] if split_data.indices is not None else i
            results.append({
                'instance_id': int(idx), 'split': split_name,
                'y_true': int(split_data.y[i]), 'y_pred': int(preds[i]),
                'y_proba_0': float(proba[i, 0]), 'y_proba_1': float(proba[i, 1]),
                'delta_signed': float(split_data.delta[i]),
                'abs_delta': float(split_data.abs_delta[i]),
                'y_star': int(split_data.y[i]),
            })

    return results


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description='Real focal loss for finetuned models')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_MODEL_MAP.keys()))
    parser.add_argument('--gammas', nargs='+', type=int, default=[2])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--output-dir', type=str, default='rebuttal/predictions')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for dataset_name in args.datasets:
        model_name = DATASET_MODEL_MAP[dataset_name]
        print(f"\n=== {dataset_name} / {model_name} (Real Focal Loss) ===")

        for gamma in args.gammas:
            for seed in args.seeds:
                out_path = output_dir / dataset_name / f"{model_name}_classification_focal{gamma}_s{seed}.csv"
                if out_path.exists():
                    print(f"  SKIP: {out_path} exists")
                    continue

                print(f"  gamma={gamma}, seed={seed}")
                try:
                    dataset = load_dataset(dataset_name, val_size=0.1, test_size=0.1, seed=seed)

                    if 'roberta' in model_name:
                        results = train_roberta_focal(dataset, gamma, seed)
                    elif 'resnet' in model_name:
                        results = train_resnet_focal(dataset, gamma, seed)
                    else:
                        raise ValueError(f"No focal finetune for {model_name}")

                    df = pd.DataFrame(results)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out_path, index=False)
                    print(f"  Saved: {out_path} ({len(df)} rows)")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == '__main__':
    main()
