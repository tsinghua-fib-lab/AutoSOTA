from __future__ import annotations
from typing import Tuple, Any, List, Dict

from transformers import DebertaV2Model, AutoTokenizer, AutoModel
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import ConcatDataset

class KnnModule(nn.Module):
    model_name = "knn"
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.knn_model = NearestNeighbors(n_neighbors=args.knearest, metric='cosine')
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            truncation_side='left',
            padding=True,
            use_fast=False
        )
        self.backbone = DebertaV2Model.from_pretrained(args.model_path).to(device)
        self.backbone.eval()
        self.seed = 42
        np.random.seed(self.seed)
    
    def load_datasets(self, train_paths, cal_paths, test_paths, answer_path, data_types):
        # print("[KNN] Loading datasets via factory...")
        def parse_paths(x):
            if x is None:
                return []
            if isinstance(x, str):
                return [p.strip() for p in x.split(",") if p.strip()]
            return list(x)
        train_list = parse_paths(train_paths)
        train_ds_list = []
        for idx, p in enumerate(train_list):
            ds = RouterKNNDataset(self.args, p, device=self.device, split_type="train", dataset_id=idx)
            train_ds_list.append(ds)
        train_data = ConcatDataset(train_ds_list) if len(train_ds_list) > 1 else (train_ds_list[0] if train_ds_list else None)

        cal_data   = RouterKNNDataset(self.args, cal_paths, device=self.device, split_type="cal")
        test_data  = RouterKNNDataset(self.args, test_paths, device=self.device, split_type="test", answer_path=answer_path)
        return train_data, cal_data, test_data
    
    def forward(self, **inputs):
        """
        Interface compatible with BaseConformalPredictor:
        inputs: embedding tensor with shape [batch_size, embed_dim]
        
        Returns (logits, None), logits shape [batch_size, n_models]
        """
        np.random.seed(self.seed)
        if self.knn_model is None:
            raise ValueError("KNN index not built. Call fit_knn() first.")
        x = inputs["embedding"]  # [B, D]
        distances, indices = self.knn_model.kneighbors(x)
        num_test = x.shape[0]
        num_llms = self.y_train.shape[1]
        predicted_probs = np.zeros((num_test, num_llms))
        for i in range(num_test):
            neighbor_indices = indices[i]  # indices of the k nearest neighbors for test inquiry i.
            # Average the correctness labels across these neighbors for each LLM.
            predicted_probs[i] = np.mean(self.y_train[neighbor_indices], axis=0)
        return torch.tensor(predicted_probs, dtype=torch.float, device=self.device), None

    def fit(self, device: str, train_dataset) -> None:
        np.random.seed(self.seed)
        X_train, Y_train = [], []
        for item, scores, _, _, _ in train_dataset:
            X_train.append(item['embedding'])
            Y_train.append(scores)
        X_train = np.stack([x.numpy() for x in X_train])
        Y_train = np.stack([y.numpy() for y in Y_train])
        
        self.y_train = Y_train  # [N, M]
        self.knn_model.fit(X_train)
    
    # ========= Helper: evaluation (aligned with evaluation(...)) =========
    def _evaluate(self, router_model, dataset_paths: List[str], dataset_types: List[str],
                  tokenizer, batch_size: int, device: str) -> Dict[str, Tuple[float, float]]:
        pass
    


class RouterKNNDataset(Dataset):
    """
    Unified Dataset class for KNN Router:
      - Reads JSON data files
      - Generates and caches sentence embeddings (via DeBERTa)
      - Returns embedding, score, label, etc.
      
    Supports two data formats:
      1. RouterDC format: {"question": "", "scores": {model: score}}
      2. Label format: {"question": "", "outputs": [{"model": "", "label": 0/1}]}
    """

    def __init__(self,
                 args,
                 data_path: str,
                 device: str = "cuda",
                 batch_size: int = 256,
                 size: int = None,
                 dataset_id: int = 0,
                 answer_path: str = None,
                 split_type: str = "train"):
        """
        Args:
            args: argument object, must include model_path, optional use_label_data
            data_path: data file path (.json)
            device: GPU or CPU
            batch_size: batch size for embedding generation
            size: optional sample size limit
            dataset_id: dataset id
            answer_path: answer path for test split only
            split_type: train / cal / test
        """
        self.args = args
        self.device = device
        self.dataset_id = dataset_id
        self.split_type = split_type
        self.use_label_data = getattr(args, 'use_label_data', False)

        # === 1) Load main data ===
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if size:
            while len(self.data) < size:
                self.data.extend(self.data)
            self.data = self.data[:size]

        # Detect data format
        first_sample = self.data[0]
        forced_format = getattr(args, "data_format", None)  # can be set via argparse --data_format
        if forced_format is not None:
            forced_format = forced_format.strip().lower()
            if forced_format not in ["label", "score"]:
                raise ValueError(f"Unsupported data_format: {forced_format}")
            self.data_format = forced_format
        else:
            # Auto-detect
            
            if "outputs" in first_sample and isinstance(first_sample["outputs"], list):
                self.data_format = "label"
            else:
                self.data_format = "score"
        if self.data_format == "label":
            self.router_node = [out["model"] for out in first_sample["outputs"]]
        else:
            self.router_node = list(first_sample["score"].keys())

        # === 2) Initialize tokenizer and backbone ===
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            truncation_side="left",
            padding=True,
            use_fast=False
        )
        self.backbone = DebertaV2Model.from_pretrained(args.model_path).to(device)
        self.backbone.eval()

        # === 3) If test split, load augmented answers ===
        self.aug_dict = {}
        if split_type == "test" and answer_path is not None:
            with open(answer_path, "r", encoding="utf-8") as f:
                aug_data = json.load(f)
            for item in aug_data:
                self.aug_dict[item["question"]] = {
                    out["model"]: out["pred_text"] for out in item["outputs"]
                }

        # === 4) Batch-generate embeddings and cache ===
        # print(f"[RouterKNNDataset] Encoding {split_type} set...")
        self._encode_all_embeddings(batch_size=batch_size)

    # ----------------------------------------------------------------------
    def _encode_all_embeddings(self, batch_size=256):
        """Batch-generate all sentence embeddings and cache."""
        questions = [sample["question"] for sample in self.data]
        embeddings = []

        for i in tqdm(range(0, len(questions), batch_size), desc=f"Encoding {self.split_type}"):
            batch_texts = questions[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.backbone(**inputs)
                # Use [CLS] embedding
                sentence_emb = outputs.last_hidden_state[:, 0, :]
                embeddings.append(sentence_emb.cpu())

        self.embeddings = torch.cat(embeddings, dim=0)

    # ----------------------------------------------------------------------
    def __getitem__(self, index):
        """Return a single sample."""
        if isinstance(index, slice):
            # Support dataset[start:end]
            indices = range(*index.indices(len(self.data)))
            return [self.__getitem__(i) for i in indices]
        sample = self.data[index]
        question = sample["question"]
        
        # Get scores/labels based on data format
        if self.data_format == "label":
            # New format: extract labels from outputs
            outputs = sample["outputs"]
            model_names = [out["model"] for out in outputs]
            # Use label (0/1) as score
            score_tensor = torch.tensor([out["label"] for out in outputs], dtype=torch.float)
            # Extract answers from outputs
            if self.split_type in ["train", "cal"]:
                answer_dict = {out["model"]: out.get("pred_text", "") for out in outputs}
            else:
                answer_dict = self.aug_dict.get(question, {out["model"]: out.get("pred_text", "") for out in outputs})
        else:
            # Old format: use scores
            scores = sample["score"]
            model_names = list(scores.keys())
            score_tensor = torch.tensor([scores[m] for m in model_names], dtype=torch.float)
            # Train/cal answers empty; test answers from aug_dict
            if self.split_type in ["train", "cal"]:
                answer_dict = {m: "" for m in model_names}
            else:
                answer_dict = self.aug_dict.get(question, {m: "" for m in model_names})

        inputs = {"embedding": self.embeddings[index]}
        return inputs, score_tensor, question, model_names, answer_dict

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    # ----------------------------------------------------------------------
    def router_node(self):
        return self.router_node

    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        