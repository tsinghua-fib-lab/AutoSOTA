# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loading utilities for encoder training.

Loads benchmark question embeddings and model scores from CSV files
for cross-benchmark encoder training.

Terminology:
    - target_benchmark: The benchmark we want to evaluate on (via BQ sampling).
    - target_model: The specific model to evaluate on the target benchmark.
    - train_benchmarks: Benchmarks used for encoder training.

Settings:
    - New Model: Target benchmark has other models' scores. Target model is new.
    - New Benchmark: Target benchmark is new (no scores available from any model).
    - New Pair: Both target model and benchmark are known, just this pair is new.

Migrated from ``colabs/data_loader.py``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Models to exclude from training by default
DEFAULT_EXCLUDE_MODELS = []


def load_benchmark_data(
    benchmark_name: str,
    data_dir: str = "data",
    embedding_model: str = "text-embedding-3-large",
    exclude_models: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load data for a single benchmark.

    Args:
        benchmark_name: Benchmark name (e.g., ``"gsm8k"``, ``"svamp"``).
        data_dir: Directory containing CSV and embedding files.
        embedding_model: Embedding model name for filename lookup.
        exclude_models: Models to exclude from training data.

    Returns:
        ``(embeddings, scores, model_names)`` where embeddings is
        ``(m, d)``, scores is ``(m, n)`` (1=error, 0=correct).
    """
    if exclude_models is None:
        exclude_models = DEFAULT_EXCLUDE_MODELS

    csv_path = Path(data_dir) / f"{benchmark_name}_predictions.csv"
    df = pd.read_csv(csv_path)

    # Extract model names and scores
    model_columns = [col for col in df.columns if col.startswith("label_")]
    model_names = [col.replace("label_", "") for col in model_columns]

    labels = []
    for name in model_names:
        y_raw = df[f"label_{name}"].values
        # Use raw labels: 1=error, 0=correct.
        # For DICES/DICES_T2I: continuous ratings → binarise at 0.5,
        # then invert so 1=unsafe/poor (error) and 0=safe/good (correct).
        # For other datasets: raw labels are already 1=error, 0=correct.
        if benchmark_name in ("dices", "dices_t2i"):
            y_error = (y_raw < 0.5).astype(float)
        else:
            y_error = y_raw
        labels.append(y_error)
    labels = np.column_stack(labels)  # (m, n) — 1=error, 0=correct

    # Filter excluded models
    if exclude_models:
        keep = [i for i, m in enumerate(model_names) if m not in exclude_models]
        excluded = [m for m in model_names if m in exclude_models]
        if excluded:
            print(f"  Excluding models: {excluded}")
        model_names = [model_names[i] for i in keep]
        labels = labels[:, keep]

    # Load embeddings
    slug = embedding_model.replace("/", "_").replace("-", "_")
    emb_path = Path(data_dir) / f"{benchmark_name}_embeddings_{slug}.npy"
    if not emb_path.exists():
        emb_path = Path(data_dir) / f"{benchmark_name}_embeddings.npy"

    if emb_path.exists():
        embeddings = np.load(emb_path).astype(np.float32)
        print(f"  Loaded embeddings from {emb_path}")
    else:
        raise FileNotFoundError(
            f"Embeddings not found at {emb_path}. "
            f"Generate with: python generate_question_embeddings.py"
        )

    return embeddings, labels, model_names


def load_all_benchmarks(
    benchmarks: List[str],
    data_dir: str = "data",
    embedding_model: str = "text-embedding-3-large",
    exclude_models: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Load data for multiple benchmarks.

    Args:
        benchmarks: List of benchmark names.
        data_dir: Data directory path.
        embedding_model: Embedding model name.
        exclude_models: Models to exclude.

    Returns:
        Dict mapping benchmark name → ``{"embeddings", "scores", "model_names"}``.
    """
    if exclude_models is None:
        exclude_models = DEFAULT_EXCLUDE_MODELS

    benchmark_data = {}
    for name in benchmarks:
        try:
            embeddings, labels, model_names = load_benchmark_data(
                name, data_dir=data_dir,
                embedding_model=embedding_model,
                exclude_models=exclude_models,
            )
            benchmark_data[name] = {
                "embeddings": embeddings,
                "labels": labels,
                "model_names": model_names,
            }
            print(f"  {name}: {embeddings.shape[0]} questions, "
                  f"{labels.shape[1]} models, dim={embeddings.shape[1]}")
        except Exception as e:
            print(f"  Could not load {name}: {e}")

    return benchmark_data


def get_model_index_by_name(
    benchmark_data: Dict, benchmark_name: str, model_name: str,
) -> int:
    """Get model index by name in a benchmark.

    Returns -1 if not found.
    """
    data = benchmark_data.get(benchmark_name)
    if data is None:
        return -1
    model_names = data.get("model_names", [])
    if model_name in model_names:
        return model_names.index(model_name)
    return -1


def split_train_and_target(
    benchmark_data: Dict,
    target_benchmark: str,
    target_model: str,
    setting: str = "new_pair",
) -> Tuple[Dict, Optional[Dict]]:
    """Split benchmark data into training set and target evaluation info.

    Separates the target model's scores for BQ evaluation and prepares
    training data according to the experiment setting:

    - **New Pair**: Other models' data on the target benchmark IS included.
    - **New Benchmark**: The target benchmark is completely excluded.
    - **New Model**: Target model's data is excluded from ALL benchmarks.

    Args:
        benchmark_data: Dict of all benchmark data.
        target_benchmark: Benchmark to evaluate on.
        target_model: Model name to evaluate.
        setting: Type of holdout experiment (new_pair, new_benchmark, new_model).

    Returns:
        ``(train_data, target_info)`` where train_data contains the training
        benchmarks and target_info contains the target model's scores.
    """
    # Start with all benchmarks except target benchmark
    train_data = {
        k: v for k, v in benchmark_data.items() if k != target_benchmark
    }

    # New Model setting: Exclude target model from ALL other benchmarks as well!
    if setting == "new_model":
        print(f"Applying New Model exclusion: dropping '{target_model}' columns from all training benchmarks.")
        for k, v in train_data.items():
            model_names = v.get("model_names", [])
            if target_model in model_names:
                idx = model_names.index(target_model)
                v["labels"] = np.delete(v["labels"], idx, axis=1)
                v["model_names"] = [m for m in model_names if m != target_model]

    target_data = benchmark_data.get(target_benchmark)
    if target_data is None:
        print(f"Warning: target benchmark '{target_benchmark}' not found!")
        return train_data, None

    # Resolve target model to index
    model_names = target_data.get("model_names", [])
    if target_model not in model_names:
        raise ValueError(
            f"Model {target_model!r} not found in '{target_benchmark}'. "
            f"Available: {model_names}"
        )
    target_model_idx = model_names.index(target_model)

    target_info = {
        "benchmark": target_benchmark,
        "embeddings": target_data["embeddings"],
        "labels": target_data["labels"][:, target_model_idx],
        "model_name": target_data["model_names"][target_model_idx],
        "model_index": target_model_idx,
    }

    if setting in ("new_pair", "new_model"):
        # New Pair / New Model: keep other models' data on target benchmark
        train_labels = np.delete(target_data["labels"], target_model_idx, axis=1)
        train_data[target_benchmark] = {
            "embeddings": target_data["embeddings"],
            "labels": train_labels,
            "model_names": [
                m for i, m in enumerate(target_data["model_names"])
                if i != target_model_idx
            ],
        }
        setting_name = "New Pair" if setting == "new_pair" else "New Model"
    else:
        # New Benchmark setting: fully exclude target benchmark
        setting_name = "New Benchmark"

    print(f"Target: {target_benchmark}, Model: {target_info['model_name']}")
    print(f"Setting: {setting_name}")
    print(f"Target set: {target_info['embeddings'].shape[0]} questions")
    print(f"Training on {len(train_data)} benchmarks")

    return train_data, target_info


# Backward-compatible aliases for old code
prepare_target_split = split_train_and_target


def prepare_holdout_split(
    benchmark_data: Dict,
    holdout_benchmark: str,
    target_model: Union[int, str] = 0,
) -> Tuple[Dict, Optional[Dict]]:
    """Deprecated: Use split_train_and_target() instead.

    This preserves the old behavior (New Pair setting) for backward
    compatibility with existing experiment scripts.
    """
    # Resolve int index to model name for the new API
    if isinstance(target_model, int):
        data = benchmark_data.get(holdout_benchmark)
        if data is not None:
            target_model = data["model_names"][target_model]

    return split_train_and_target(
        benchmark_data,
        target_benchmark=holdout_benchmark,
        target_model=target_model,
        include_target_benchmark_in_training=True,
    )

