from __future__ import annotations
from typing import Tuple, Any, List, Dict

from transformers import DebertaV2Model, AutoTokenizer
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm


class FocalLoss(nn.Module):
    """Focal Loss for multi-label binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy negatives and focuses training on hard borderline cases.
    alpha balances positive/negative class imbalance (typically 1-3 correct
    models out of 7 per query).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MLPClassifier(nn.Module):
    """MLP head for LLM routing: predicts per-model correctness probability."""
    def __init__(self, input_dim: int, hidden_size: int, num_models: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_models),
        )

    def forward(self, x):
        return self.net(x)


class MLPModule(nn.Module):
    """MLP Router: DeBERTa encoder + MLP classifier head.

    Paper description: "A multi-layer perceptron (MLP) classifier is trained as the router
    to predict model performance score. The architecture consists of a hidden layer with 256 units.
    Training is performed using BCEWithLogitsLoss with a batch size of 32 and a learning rate of
    10^-4 for 100 epochs."
    """
    model_name = "mlp"

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.seed = args.seed if hasattr(args, 'seed') else 42

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            truncation_side='left',
            padding=True,
            use_fast=False
        )
        self.backbone = DebertaV2Model.from_pretrained(args.model_path).to(device)
        self.backbone.eval()  # Freeze backbone

        # MLP head (initialized after seeing first data)
        self.classifier = None
        self.num_models = None

        # Training hyperparams
        self.hidden_size = getattr(args, 'hidden_size', 256)
        self.lr = getattr(args, 'lr', 1e-4)
        self.epochs = getattr(args, 'epoch', 100)
        self.weight_decay = getattr(args, 'weight_decay', 0.01)
        self.train_bs = getattr(args, 'train_bs', 32)

    def load_datasets(self, train_paths, cal_paths, test_paths, answer_path, data_types):
        def parse_paths(x):
            if x is None:
                return []
            if isinstance(x, str):
                return [p.strip() for p in x.split(",") if p.strip()]
            return list(x)

        train_list = parse_paths(train_paths)
        train_ds_list = []
        for idx, p in enumerate(train_list):
            ds = RouterMLPDataset(self.args, p, device=self.device, split_type="train", dataset_id=idx)
            train_ds_list.append(ds)
        train_data = ConcatDataset(train_ds_list) if len(train_ds_list) > 1 else (train_ds_list[0] if train_ds_list else None)

        cal_data = RouterMLPDataset(self.args, cal_paths, device=self.device, split_type="cal")
        test_data = RouterMLPDataset(self.args, test_paths, device=self.device, split_type="test", answer_path=answer_path)
        return train_data, cal_data, test_data

    def forward(self, **inputs):
        """Get logits from the MLP classifier. Input: {"embedding": tensor [B, D]}"""
        x = inputs["embedding"].to(self.device)
        logits = self.classifier(x)
        return logits, None

    def fit(self, device: str, train_dataset) -> None:
        """Train the MLP classifier head."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Collect embeddings and labels
        X_train, Y_train = [], []
        for item, scores, _, _, _ in train_dataset:
            X_train.append(item['embedding'])
            Y_train.append(scores)
        X_train = torch.stack([x for x in X_train]).to(device)
        Y_train = torch.stack([y for y in Y_train]).to(device)

        self.num_models = Y_train.shape[1]
        input_dim = X_train.shape[1]

        # Initialize classifier
        self.classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_models=self.num_models,
        ).to(device)

        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        n_samples = len(X_train)
        batch_size = min(self.train_bs, n_samples)

        print(f"[MLP] Training on {n_samples} samples, {self.num_models} models, "
              f"input_dim={input_dim}, hidden={self.hidden_size}, "
              f"epochs={self.epochs}, lr={self.lr}, bs={batch_size}")

        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                x_batch = X_train[idx]
                y_batch = Y_train[idx]

                logits = self.classifier(x_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/n_batches:.6f}")

        self.classifier.eval()
        print(f"[MLP] Training complete.")


class RouterMLPDataset(Dataset):
    """Dataset for MLP router. Uses the same data format as RouterKNNDataset."""

    def __init__(self, args, data_path: str, device: str = "cuda",
                 batch_size: int = 256, size: int = None,
                 dataset_id: int = 0, answer_path: str = None,
                 split_type: str = "train"):
        self.args = args
        self.device = device
        self.dataset_id = dataset_id
        self.split_type = split_type

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if size:
            while len(self.data) < size:
                self.data.extend(self.data)
            self.data = self.data[:size]

        # Detect data format
        first_sample = self.data[0]
        forced_format = getattr(args, "data_format", None)
        if forced_format is not None:
            forced_format = forced_format.strip().lower()
            if forced_format not in ["label", "score"]:
                raise ValueError(f"Unsupported data_format: {forced_format}")
            self.data_format = forced_format
        else:
            if "outputs" in first_sample and isinstance(first_sample["outputs"], list):
                self.data_format = "label"
            else:
                self.data_format = "score"

        if self.data_format == "label":
            self.router_node = [out["model"] for out in first_sample["outputs"]]
        else:
            self.router_node = list(first_sample["score"].keys())

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, truncation_side="left", padding=True, use_fast=False
        )
        self.backbone = DebertaV2Model.from_pretrained(args.model_path).to(device)
        self.backbone.eval()

        self.aug_dict = {}
        if split_type == "test" and answer_path is not None:
            with open(answer_path, "r", encoding="utf-8") as f:
                aug_data = json.load(f)
            for item in aug_data:
                self.aug_dict[item["question"]] = {
                    out["model"]: out["pred_text"] for out in item["outputs"]
                }

        self._encode_all_embeddings(batch_size=batch_size)

    def _encode_all_embeddings(self, batch_size=256):
        questions = [sample["question"] for sample in self.data]
        embeddings = []
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Encoding {self.split_type}"):
            batch_texts = questions[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.backbone(**inputs)
                sentence_emb = outputs.last_hidden_state[:, 0, :]
                embeddings.append(sentence_emb.cpu())
        self.embeddings = torch.cat(embeddings, dim=0)

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample["question"]

        if self.data_format == "label":
            outputs = sample["outputs"]
            model_names = [out["model"] for out in outputs]
            score_tensor = torch.tensor([out["label"] for out in outputs], dtype=torch.float)
            if self.split_type in ["train", "cal"]:
                answer_dict = {out["model"]: out.get("pred_text", "") for out in outputs}
            else:
                answer_dict = self.aug_dict.get(question, {out["model"]: out.get("pred_text", "") for out in outputs})
        else:
            scores = sample["score"]
            model_names = list(scores.keys())
            score_tensor = torch.tensor([scores[m] for m in model_names], dtype=torch.float)
            if self.split_type in ["train", "cal"]:
                answer_dict = {m: "" for m in model_names}
            else:
                answer_dict = self.aug_dict.get(question, {m: "" for m in model_names})

        inputs = {"embedding": self.embeddings[index]}
        return inputs, score_tensor, question, model_names, answer_dict

    def __len__(self):
        return len(self.data)

    def router_node(self):
        return self.router_node
