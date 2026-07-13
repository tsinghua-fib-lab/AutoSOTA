from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd


@dataclass
class RuleInfo:
    text: str
    source_round: int
    train_accuracy: Optional[float] = None
    train_f1: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_f1: Optional[float] = None
    train_coverage: Optional[float] = None
    val_coverage: Optional[float] = None


@dataclass
class RuleFeatureStore:
    samples: pd.DataFrame
    label_column: str
    val_samples: Optional[pd.DataFrame] = None
    rules: List[RuleInfo] = field(default_factory=list)
    features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    val_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def add_rules(
        self,
        new_rules: List[str],
        train_vectors: List[List[int]],
        round_idx: int,
        val_vectors: Optional[List[List[int]]] = None,
        train_metrics: Optional[List[Dict[str, float]]] = None,
        val_metrics: Optional[List[Dict[str, float]]] = None,
    ):
        if not new_rules:
            return
        if train_metrics is not None and len(train_metrics) != len(new_rules):
            raise ValueError("train_metrics must align with new_rules length")
        if val_metrics is not None and len(val_metrics) != len(new_rules):
            raise ValueError("val_metrics must align with new_rules length")
        new_rule_infos: List[RuleInfo] = []
        for idx, rule in enumerate(new_rules):
            metrics: Dict[str, float] = train_metrics[idx] if train_metrics is not None else {}
            val_values: Dict[str, float] = val_metrics[idx] if val_metrics is not None else {}
            new_rule_infos.append(
                RuleInfo(
                    text=rule,
                    source_round=round_idx,
                    train_accuracy=metrics.get("accuracy"),
                    train_f1=metrics.get("f1"),
                    train_coverage=metrics.get("coverage"),
                    val_accuracy=val_values.get("accuracy"),
                    val_f1=val_values.get("f1"),
                    val_coverage=val_values.get("coverage"),
                )
            )
        train_matrix = np.array(train_vectors, dtype=np.int8)
        if train_matrix.ndim != 2:
            raise ValueError("Expected 2D train rule vectors")
        train_matrix = train_matrix.reshape(len(new_rules), len(self.samples))
        if self.features.size == 0:
            self.features = train_matrix
        else:
            self.features = np.vstack([self.features, train_matrix])

        if self.val_samples is not None:
            if val_vectors is None:
                raise ValueError("Validation samples provided but val_vectors missing")
            val_matrix = np.array(val_vectors, dtype=np.int8)
            if val_matrix.ndim != 2:
                raise ValueError("Expected 2D validation rule vectors")
            val_matrix = val_matrix.reshape(len(new_rules), len(self.val_samples))
            if self.val_features.size == 0:
                self.val_features = val_matrix
            else:
                self.val_features = np.vstack([self.val_features, val_matrix])
        self.rules.extend(new_rule_infos)

    def prune_rules(self, max_rules: Optional[int], metric: str = "f1") -> int:
        if max_rules is None or max_rules <= 0:
            return 0
        if len(self.rules) <= max_rules:
            return 0
        metric_key = metric.lower()
        if metric_key not in {"accuracy", "acc", "f1"}:
            raise ValueError(f"Unsupported prune metric '{metric}'. Use 'accuracy' or 'f1'.")

        scores: List[float] = []
        for info in self.rules:
            if metric_key in {"accuracy", "acc"}:
                value = info.val_accuracy if info.val_accuracy is not None else info.train_accuracy
            else:
                value = info.val_f1 if info.val_f1 is not None else info.train_f1
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = float("-inf")
            scores.append(value)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda item: item[1])  # ascending, so lowest first
        num_to_drop = len(self.rules) - max_rules
        drop_indices = {idx for idx, _ in indexed_scores[:num_to_drop]}
        if not drop_indices:
            return 0

        keep_indices = [idx for idx in range(len(self.rules)) if idx not in drop_indices]
        self.rules = [self.rules[idx] for idx in keep_indices]
        if self.features.size > 0:
            self.features = self.features[keep_indices, :]
        if self.val_features.size > 0:
            self.val_features = self.val_features[keep_indices, :]
        return len(drop_indices)

    def to_sample_feature_matrix(self, split: str = "train") -> np.ndarray:
        if split == "train":
            if self.features.size == 0:
                return np.zeros((len(self.samples), 0))
            return self.features.T
        if split == "val":
            if self.val_samples is None:
                raise ValueError("Validation samples not initialized for feature store")
            if self.val_features.size == 0:
                return np.zeros((len(self.val_samples), 0))
            return self.val_features.T
        raise ValueError(f"Unsupported split '{split}' for feature matrix")

    def feature_slice(self, sample_indices: List[int]) -> tuple[np.ndarray, List[int]]:
        if self.features.size == 0 or not sample_indices:
            return np.zeros((len(self.rules), 0)), []
        valid_indices = [idx for idx in sample_indices if 0 <= idx < self.features.shape[1]]
        if not valid_indices:
            return np.zeros((len(self.rules), 0)), []
        slice_matrix = self.features[:, valid_indices]
        return slice_matrix, valid_indices

    def dump_rules(self, path: str):
        payload = [
            {
                "rule": info.text,
                "round": info.source_round,
                "train_accuracy": info.train_accuracy,
                "train_f1": info.train_f1,
                "train_coverage": info.train_coverage,
                "val_accuracy": info.val_accuracy,
                "val_f1": info.val_f1,
                "val_coverage": info.val_coverage,
            }
            for info in self.rules
        ]
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
