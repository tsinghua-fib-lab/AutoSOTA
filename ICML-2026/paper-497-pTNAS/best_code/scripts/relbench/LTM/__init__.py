"""
TP-BERTa Package

This package provides three main interfaces:

1. get_embeddings: Runtime embedding extraction (returns numpy array)
   - Auto-generates and cleans up feature_names.json
   - Use for runtime processing (e.g., process_relbench.py)

2. preprocess: Offline preprocessing (saves CSV files with embeddings)
   - Saves train.csv, val.csv, test.csv with embeddings
   - Use for batch preprocessing

3. train: Train prediction head on preprocessed embeddings
   - Loads embeddings from CSV files
   - Trains TP-BERTa style head
"""

from .get_embeddings import get_embeddings
from .process_tables import preprocess
from .train import train_prediction_head as train

__all__ = [
    'get_embeddings',      # Runtime embedding extraction
    'preprocess',          # Offline preprocessing (saves CSV)
    'train',               # Train prediction head
]

