import os
import numpy as np
import torch
from appfl.misc.data import Dataset


def get_b2ai_voice(client_id: int, dataset_path: str, **kwargs):
    """
    Load pre-partitioned B2AI-Voice spectrogram features for a given FL client.

    4 training clients (single-cohort disease groups):
      0 - Voice Disorders      (94 participants,  3261 recordings)
      1 - Neurological         (69 participants,  3498 recordings)
      2 - Mood / Psychiatric   (22 participants,   670 recordings)
      3 - Respiratory          (77 participants,  2289 recordings)

    Shared validation set (all clients):
      Multi-cohort / Controls  (175 participants, 6863 recordings)
      Loaded from client_4/data.npz — not used for training.

    Features: 402-dim vector (mean + std pooled from 201-freq spectrogram over time).
    Labels:   0 = Female, 1 = Male  (sex_at_birth from phenotype.tsv)

    Returns:
        (train_dataset, val_dataset) as appfl.misc.data.Dataset objects
    """
    # Load training data for this client (all recordings used for training)
    train_path = os.path.join(dataset_path, f"client_{client_id}", "data.npz")
    train_cache = np.load(train_path)
    X_train = train_cache["X"].astype(np.float32)  # (N, 402)
    y_train = train_cache["y"].astype(np.int64)  # (N,)

    # Load shared validation set from multi-cohort / controls (client_4)
    val_path = os.path.join(dataset_path, "client_4", "data.npz")
    val_cache = np.load(val_path)
    X_val = val_cache["X"].astype(np.float32)
    y_val = val_cache["y"].astype(np.int64)

    # Normalize using train statistics only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_dataset = Dataset(
        torch.FloatTensor(X_train),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = Dataset(
        torch.FloatTensor(X_val),
        torch.tensor(y_val, dtype=torch.long),
    )
    return train_dataset, val_dataset
