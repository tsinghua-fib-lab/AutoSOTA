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

"""Data loading utilities for active evaluation.

Provides functions to load prediction CSVs, text embeddings, and prepare
train/test splits for Bayesian Quadrature sampling.
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _default_data_dir() -> str:
    """Resolve the default data directory (data/)."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
    )


def load_predictions(
    dataset_name: str, data_dir: str = None
) -> pd.DataFrame:
    """Load predictions CSV for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'svamp', 'strategyqa').
        data_dir: Directory containing CSV files. Defaults to ``data/``.

    Returns:
        DataFrame with model prediction columns (``label_<model>``).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if data_dir is None:
        data_dir = _default_data_dir()
    csv_path = os.path.join(data_dir, f"{dataset_name}_predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_embeddings(
    dataset_name: str,
    data_dir: str = None,
    embedding_model: str = "text_embedding_3_large",
) -> np.ndarray:
    """Load pre-computed text embeddings for a dataset.

    Args:
        dataset_name: Name of the dataset.
        data_dir: Directory containing ``.npy`` files. Defaults to ``data/``.
        embedding_model: Embedding model name suffix.

    Returns:
        Embeddings array of shape ``(n_samples, n_features)``.
    """
    if data_dir is None:
        data_dir = _default_data_dir()
    if dataset_name == "gqa":
        embedding_path = os.path.join(data_dir, "gqa_embeddings.npy")
    else:
        embedding_path = os.path.join(
            data_dir, f"{dataset_name}_embeddings_{embedding_model}.npy"
        )
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    return np.load(embedding_path)


def extract_model_predictions(
    df: pd.DataFrame, dataset_name: str = None
) -> Tuple[np.ndarray, List[str]]:
    """Extract a binary prediction matrix from a predictions DataFrame.

    Labels use the convention: **1=error, 0=correct** (measuring failure rate).

    For DICES/DICES-T2I datasets, continuous ratings are binarised at 0.5,
    then inverted so 1=unsafe/poor (error) and 0=safe/good (correct).
    For other datasets, ``label_`` columns are already error indicators
    and are used directly.

    Args:
        df: Predictions DataFrame with ``label_<model>`` columns.
        dataset_name: Dataset name for special-case handling.

    Returns:
        ``(prediction_matrix, model_names)`` where ``prediction_matrix`` has
        shape ``(n_samples, n_models)`` with values in {0, 1}
        (1=error, 0=correct).
    """
    model_columns = [col for col in df.columns if col.startswith("label_")]
    model_names = [col.replace("label_", "") for col in model_columns]

    model_data = {}
    for model_name in model_names:
        y_labels = df[f"label_{model_name}"].values
        # Use raw labels: 1=error, 0=correct.
        # For DICES/DICES_T2I: continuous ratings, binarise at 0.5,
        # then invert so 1=unsafe/poor and 0=safe/good.
        # For other datasets: raw labels are already 1=error, 0=correct.
        if dataset_name in ("dices", "dices_t2i"):
            y_error = (y_labels < 0.5).astype(float)
        else:
            y_error = y_labels
        model_data[model_name] = y_error

    prediction_matrix = np.column_stack([model_data[n] for n in model_names])
    return prediction_matrix, model_names


def setup_train_test_split(
    prediction_matrix: np.ndarray,
    target_model: Union[int, str],
    pretrain_indices: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split prediction matrix into pre-training features and target model test data.

    Args:
        prediction_matrix: Full prediction matrix ``(n_samples, n_models)``.
        target_model: Name or index of the model to target for testing.
        pretrain_indices: Optional list of model indices to use for pre-training.
            If ``None``, uses all models except the target.
        model_names: List of model names (required when *target_model* is a
            string).

    Returns:
        ``(pretrain_matrix, test_x, test_y, prior_mean, prior_cov)``
    """
    # Resolve target model to index
    if isinstance(target_model, str):
        if model_names is None:
            raise ValueError(
                "model_names must be provided when target_model is a string"
            )
        if target_model not in model_names:
            raise ValueError(
                f"Model {target_model!r} not found. Available: {model_names}"
            )
        target_model_index = model_names.index(target_model)
    else:
        target_model_index = int(target_model)

    test_y = prediction_matrix[:, target_model_index]

    if pretrain_indices is not None:
        pretrain_matrix = prediction_matrix[:, pretrain_indices]
    else:
        pretrain_matrix = np.delete(
            prediction_matrix,
            slice(target_model_index, target_model_index + 1),
            axis=1,
        )

    u = np.mean(pretrain_matrix, axis=1)
    S = np.cov(pretrain_matrix)
    if S.ndim == 0:
        S = np.array([[S]])

    test_x = pretrain_matrix.T
    n_pretrain = test_x.shape[0]
    if n_pretrain > 1:
        test_x = (test_x - u) / np.sqrt(n_pretrain - 1)
    else:
        test_x = test_x - u

    return pretrain_matrix, test_x, test_y, u, S
