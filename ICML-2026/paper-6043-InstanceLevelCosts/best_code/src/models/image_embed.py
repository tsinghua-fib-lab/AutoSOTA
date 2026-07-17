"""
Image embedding model for classification and regression.

Uses frozen torchvision CNN embeddings (e.g., ResNet50) with linear heads
(LogisticRegression for classification, Ridge for regression).
"""

from pathlib import Path
from typing import Union, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as tvm

from sklearn.linear_model import LogisticRegression, Ridge

from models.base import BaseModel


class ImagePathDataset(Dataset):
    """Simple dataset that loads images from paths."""

    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


class ImageEmbedModel(BaseModel):
    """
    Frozen CNN embeddings + linear head.

    Extracts features using a pretrained torchvision model (frozen weights),
    then trains a linear classifier (LogisticRegression) or regressor (Ridge).

    Currently supports ResNet50 architecture with 2048-d features.

    Example:
        >>> model = ImageEmbedModel(
        ...     arch='resnet50',
        ...     task='classification'
        ... )
        >>> # X is list of image paths
        >>> model.fit(image_paths, labels, sample_weight=weights)
        >>> predictions = model.predict(test_image_paths)
    """

    def __init__(
        self,
        task: str = 'classification',
        arch: str = 'resnet50',
        batch_size: int = 64,
        num_workers: int = 0,
        # Classification params
        logreg_C: float = 1.0,
        logreg_max_iter: int = 1000,
        logreg_solver: str = 'lbfgs',
        # Regression params
        ridge_alpha: float = 1.0,
        # Device
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize image embedding model.

        Args:
            task: 'classification' or 'regression'
            arch: Architecture name (currently only 'resnet50' supported)
            batch_size: Batch size for feature extraction
            num_workers: Number of DataLoader workers
            logreg_C: Inverse regularization strength for LogisticRegression
            logreg_max_iter: Max iterations for LogisticRegression
            logreg_solver: Solver for LogisticRegression ('lbfgs', 'saga', etc.)
            ridge_alpha: Regularization strength for Ridge regression
            device: Device to use ('cuda', 'cpu', 'mps'). If None, auto-detects.
        """
        super().__init__(task=task)

        if arch != 'resnet50':
            raise ValueError(f"Only 'resnet50' is currently supported, got '{arch}'")

        self.arch = arch
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        self.encoder_ = None
        self.head_ = None
        self.transform_ = None

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
        """Lazy-load torchvision model and transforms."""
        if self.encoder_ is None:
            # Load ResNet50 with pretrained weights
            try:
                # torchvision >= 0.13 (new API)
                weights = tvm.ResNet50_Weights.DEFAULT
                model = tvm.resnet50(weights=weights)
            except Exception:
                # Fallback to old API
                model = tvm.resnet50(pretrained=True)

            # Remove classification head to get 2048-d features
            model.fc = torch.nn.Identity()
            model.eval()  # Frozen embeddings
            model.to(self.device)

            self.encoder_ = model

        if self.transform_ is None:
            # ImageNet normalization (standard for ResNet50)
            self.transform_ = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _embed_images(
        self,
        image_paths: List[Union[str, Path]],
    ) -> np.ndarray:
        """
        Extract CNN features from images.

        Args:
            image_paths: List of paths to image files

        Returns:
            Feature embeddings array of shape (N, 2048)
        """
        self._initialize_encoder()

        # Convert to Path objects
        image_paths = [Path(p) for p in image_paths]

        # Create dataset and loader
        dataset = ImagePathDataset(image_paths, self.transform_)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == 'cuda'
        )

        # Extract features
        feats = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                z = self.encoder_(batch)  # [B, 2048]
                feats.append(z.cpu().numpy())

        if not feats:
            # Empty input case
            return np.zeros((0, 2048), dtype=np.float32)

        return np.concatenate(feats, axis=0)

    def fit(
        self,
        X: Union[List[str], List[Path], np.ndarray],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'ImageEmbedModel':
        """
        Fit image embedding model.

        Args:
            X: Image paths (list of strings/Paths or array)
            y: Target labels (classification) or values (regression)
            sample_weight: Optional per-example weights

        Returns:
            self (fitted model)
        """
        # Convert to list of paths
        if isinstance(X, np.ndarray):
            X = X.tolist()
        y = np.asarray(y)

        # Extract features
        X_vec = self._embed_images(X)

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
        X: Union[List[str], List[Path], np.ndarray],
    ) -> np.ndarray:
        """
        Make predictions on images.

        Args:
            X: Image paths (list of strings/Paths or array)

        Returns:
            Predictions (binary labels for classification, continuous for regression)
        """
        self._check_fitted()

        # Convert to list of paths
        if isinstance(X, np.ndarray):
            X = X.tolist()

        # Extract features and predict
        X_vec = self._embed_images(X)
        return self.head_.predict(X_vec)

    def predict_proba(
        self,
        X: Union[List[str], List[Path], np.ndarray],
    ) -> np.ndarray:
        """
        Get predicted class probabilities (classification only).

        Args:
            X: Image paths

        Returns:
            Class probabilities, shape (N, 2)
        """
        if self.task != 'classification':
            raise NotImplementedError("predict_proba only available for classification")

        self._check_fitted()

        # Convert to list of paths
        if isinstance(X, np.ndarray):
            X = X.tolist()

        # Extract features and get probabilities
        X_vec = self._embed_images(X)
        return self.head_.predict_proba(X_vec)

    def get_params(self) -> dict:
        """Get model hyperparameters."""
        params = super().get_params()
        params.update({
            'arch': self.arch,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
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

    def get_feature_dim(self) -> int:
        """
        Get feature dimensionality.

        Returns:
            Feature dimension (2048 for ResNet50)
        """
        return 2048

    def clear_cache(self):
        """Clear cached training embeddings to free memory."""
        self._cache_train_embeddings = None
