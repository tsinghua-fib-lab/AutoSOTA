"""
Text embedding model for classification and regression.

Uses frozen HuggingFace transformer embeddings (e.g., RoBERTa, BERT)
with linear heads (LogisticRegression for classification, Ridge for regression).
"""

import os
from typing import Union, Optional, List, Literal
import numpy as np
import torch
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, Ridge

from models.base import BaseModel

# Quieter tokenizers when forking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class TextEmbedModel(BaseModel):
    """
    Frozen transformer embeddings + linear head.

    Encodes text using a pretrained HuggingFace model (frozen weights),
    then trains a linear classifier (LogisticRegression) or regressor (Ridge).

    Supports mean pooling or CLS token pooling for sentence embeddings.

    Example:
        >>> model = TextEmbedModel(
        ...     hf_model='roberta-base',
        ...     task='classification',
        ...     pooling='mean'
        ... )
        >>> model.fit(train_texts, train_labels, sample_weight=weights)
        >>> predictions = model.predict(test_texts)
    """

    def __init__(
        self,
        task: str = 'classification',
        hf_model: str = 'roberta-base',
        pooling: Literal['mean', 'cls'] = 'mean',
        max_length: int = 128,
        batch_size: int = 64,
        # Classification params
        logreg_C: float = 1.0,
        logreg_max_iter: int = 1000,
        logreg_solver: str = 'saga',
        # Regression params
        ridge_alpha: float = 1.0,
        # Device
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize text embedding model.

        Args:
            task: 'classification' or 'regression'
            hf_model: HuggingFace model name (e.g., 'roberta-base', 'bert-base-uncased')
            pooling: Pooling strategy - 'mean' (average all tokens) or 'cls' (use [CLS] token)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for encoding
            logreg_C: Inverse regularization strength for LogisticRegression
            logreg_max_iter: Max iterations for LogisticRegression
            logreg_solver: Solver for LogisticRegression ('saga', 'lbfgs', etc.)
            ridge_alpha: Regularization strength for Ridge regression
            device: Device to use ('cuda', 'cpu', 'mps'). If None, auto-detects.
        """
        super().__init__(task=task)

        self.hf_model = hf_model
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.logreg_C = logreg_C
        self.logreg_max_iter = logreg_max_iter
        self.logreg_solver = logreg_solver
        self.ridge_alpha = ridge_alpha

        # Device setup
        if device is None:
            self.device = self._get_device()
        else:
            self.device = torch.device(device)

        # Will be initialized in fit()
        self.tokenizer_ = None
        self.encoder_ = None
        self.head_ = None

        # Cache for embeddings (optional optimization)
        self._cache_train_embeddings = None

    def _get_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _initialize_encoder(self):
        """Lazy-load HuggingFace model and tokenizer."""
        if self.tokenizer_ is None or self.encoder_ is None:
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer_ = AutoTokenizer.from_pretrained(self.hf_model)
            self.encoder_ = AutoModel.from_pretrained(self.hf_model).to(self.device)
            self.encoder_.eval()  # Frozen embeddings

    def _embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to fixed-size embeddings.

        Args:
            texts: List of text strings
            show_progress: Whether to show tqdm progress bar

        Returns:
            Embeddings array of shape (N, hidden_size)
        """
        self._initialize_encoder()

        out_vecs = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + self.batch_size]

                # Tokenize
                toks = self.tokenizer_(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}

                # Forward pass
                outputs = self.encoder_(**toks)
                h = getattr(outputs, "last_hidden_state", None)
                if h is None:  # Fallback for some model architectures
                    h = self.encoder_.base_model(**toks).last_hidden_state

                # Pool to sentence embedding
                if self.pooling == "cls":
                    vec = h[:, 0, :]  # [CLS] token
                else:  # mean pooling
                    mask = toks["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                    summed = (h * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1)
                    vec = summed / counts

                out_vecs.append(vec.detach().cpu().numpy())

        if not out_vecs:
            # Empty input case
            hidden_size = self.encoder_.config.hidden_size
            return np.zeros((0, hidden_size), dtype=np.float32)

        return np.concatenate(out_vecs, axis=0)

    def fit(
        self,
        X: Union[List[str], np.ndarray],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'TextEmbedModel':
        """
        Fit text embedding model.

        Args:
            X: Text data (list of strings or array of strings)
            y: Target labels (classification) or values (regression)
            sample_weight: Optional per-example weights

        Returns:
            self (fitted model)
        """
        # Convert to list of strings
        if isinstance(X, np.ndarray):
            X = X.tolist()
        X = [str(text) for text in X]
        y = np.asarray(y)

        # Embed texts
        X_vec = self._embed_texts(X, show_progress=True)

        # Cache embeddings (useful if refitting with different weights)
        self._cache_train_embeddings = X_vec

        # Initialize and fit head based on task
        if self.task == 'classification':
            self.head_ = LogisticRegression(
                C=self.logreg_C,
                max_iter=self.logreg_max_iter,
                solver=self.logreg_solver,
                n_jobs=-1,
            )
        else:  # regression
            self.head_ = Ridge(
                alpha=self.ridge_alpha,
                random_state=0,
            )

        # Fit with optional sample weights
        if sample_weight is not None:
            self.head_.fit(X_vec, y, sample_weight=sample_weight)
        else:
            self.head_.fit(X_vec, y)

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """
        Make predictions on text data.

        Args:
            X: Text data (list of strings or array of strings)

        Returns:
            Predictions (binary labels for classification, continuous for regression)
        """
        self._check_fitted()

        # Convert to list of strings
        if isinstance(X, np.ndarray):
            X = X.tolist()
        X = [str(text) for text in X]

        # Embed and predict
        X_vec = self._embed_texts(X, show_progress=True)
        return self.head_.predict(X_vec)

    def predict_proba(
        self,
        X: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """
        Get predicted class probabilities (classification only).

        Args:
            X: Text data

        Returns:
            Class probabilities, shape (N, 2)
        """
        if self.task != 'classification':
            raise NotImplementedError("predict_proba only available for classification")

        self._check_fitted()

        # Convert to list of strings
        if isinstance(X, np.ndarray):
            X = X.tolist()
        X = [str(text) for text in X]

        # Embed and get probabilities
        X_vec = self._embed_texts(X, show_progress=True)
        return self.head_.predict_proba(X_vec)

    def get_params(self) -> dict:
        """Get model hyperparameters."""
        params = super().get_params()
        params.update({
            'hf_model': self.hf_model,
            'pooling': self.pooling,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
        })
        if self.task == 'classification':
            params.update({
                'logreg_C': self.logreg_C,
                'logreg_max_iter': self.logreg_max_iter,
                'logreg_solver': self.logreg_solver,
            })
        else:
            params.update({
                'ridge_alpha': self.ridge_alpha,
            })
        return params

    def get_embedding_dim(self) -> Optional[int]:
        """
        Get embedding dimensionality (requires encoder to be initialized).

        Returns:
            Embedding dimension or None if encoder not initialized
        """
        if self.encoder_ is None:
            return None
        return self.encoder_.config.hidden_size

    def clear_cache(self):
        """Clear cached training embeddings to free memory."""
        self._cache_train_embeddings = None
