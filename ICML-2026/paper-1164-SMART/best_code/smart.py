"""SMART core."""

from __future__ import annotations

import copy
import random
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import CharbonnierSoftECE


ArrayLike = Union[np.ndarray, torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_tensor(x: ArrayLike, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _validate_logits_and_labels(logits: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [n_samples, n_classes], got {tuple(logits.shape)}")
    if logits.size(1) < 2:
        raise ValueError("SMART requires at least two classes to compute the top-two margin")
    if labels is not None:
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape [n_samples], got {tuple(labels.shape)}")
        if labels.size(0) != logits.size(0):
            raise ValueError("logits and labels must contain the same number of samples")


def compute_margins(logits: ArrayLike) -> ArrayLike:
    original_is_tensor = isinstance(logits, torch.Tensor)
    original_device = logits.device if original_is_tensor else torch.device("cpu")
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32, device=original_device)
    _validate_logits_and_labels(logits_tensor)

    top2 = torch.topk(logits_tensor, k=2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    if original_is_tensor:
        return margins
    return margins.cpu().numpy()


class MarginTemperatureNet(nn.Module):
    def __init__(self, hidden_dim: int = 16, nlayers: int = 2, min_temperature: float = 0.1) -> None:
        super().__init__()
        if nlayers < 1:
            raise ValueError("nlayers must be at least 1")

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1, hidden_dim))
        for _ in range(nlayers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.min_temperature = min_temperature

    def forward(self, margins: torch.Tensor) -> torch.Tensor:
        x = margins.view(-1, 1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return F.softplus(self.layers[-1](x)).view(-1) + self.min_temperature


class SMART:
    """Sample Margin-Aware Recalibration of Temperature."""

    def __init__(
        self,
        hidden_dim: int = 16,
        nlayers: int = 2,
        lr: float = 5e-3,
        epochs: int = 2000,
        loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "smooth_soft_ece",
        n_bins: int = 15,
        seed: int = 1,
        batch_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        min_temperature: float = 0.1,
        sigma: float = 0.05,
        delta: float = 1e-3,
        early_stopping: bool = True,
        patience: int = 200,
        min_delta: float = 1e-4,
        normalize_margins: bool = True,
        verbose: bool = False,
        sigma_anneal_start: float = 0.0,
        sigma_anneal_end: float = 0.0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.lr = lr
        self.epochs = epochs
        self.loss_name = loss if isinstance(loss, str) else loss.__class__.__name__
        self.n_bins = n_bins
        self.seed = seed
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.min_temperature = min_temperature
        self.sigma = sigma
        self.delta = delta
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.normalize_margins = normalize_margins
        self.verbose = verbose
        self.sigma_anneal_start = sigma_anneal_start
        self.sigma_anneal_end = sigma_anneal_end

        set_seed(seed)
        self.model = MarginTemperatureNet(hidden_dim, nlayers, min_temperature).to(self.device)
        self.loss_fn = self._build_loss(loss)
        self.margin_mean: float = 0.0
        self.margin_std: float = 1.0
        self.loss_history: List[float] = []
        self.is_fitted = False

    def _build_loss(
        self,
        loss: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if callable(loss) and not isinstance(loss, str):
            return loss

        name = str(loss).lower().replace("-", "_")
        if name in {"smooth_soft_ece", "smoothsoftece", "charbonnier_softece", "charbonnier_soft_ece"}:
            return CharbonnierSoftECE(n_bins=self.n_bins, sigma=self.sigma, delta=self.delta)
        raise ValueError(
            "Unsupported loss. Use 'smooth_soft_ece', 'charbonnier_soft_ece', "
            "or pass a callable."
        )

    def _normalized_margins(self, logits: torch.Tensor, *, fit: bool) -> torch.Tensor:
        margins = torch.topk(logits, k=2, dim=1).values
        margins = margins[:, 0] - margins[:, 1]

        if not self.normalize_margins:
            if fit:
                self.margin_mean = 0.0
                self.margin_std = 1.0
            return margins

        if fit:
            self.margin_mean = float(margins.mean().item())
            self.margin_std = float(margins.std(unbiased=False).clamp_min(1e-8).item())
        return (margins - self.margin_mean) / self.margin_std

    def fit(self, val_logits: ArrayLike, val_labels: ArrayLike) -> "SMART":
        set_seed(self.seed)
        logits = _to_tensor(val_logits, dtype=torch.float32, device=self.device)
        labels = _to_tensor(val_labels, dtype=torch.long, device=self.device)
        _validate_logits_and_labels(logits, labels)

        margins = self._normalized_margins(logits, fit=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n_samples = logits.size(0)
        batch_size = self.batch_size or n_samples
        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0
        self.loss_history = []

        for epoch in range(self.epochs):
            # Sigma annealing: linearly interpolate from start to end over training
            if self.sigma_anneal_end > 0:
                progress = epoch / max(1, self.epochs - 1)
                current_sigma = self.sigma_anneal_start + (self.sigma_anneal_end - self.sigma_anneal_start) * progress
                self.loss_fn.sigma = current_sigma
            permutation = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                idx = permutation[start : start + batch_size]
                batch_logits = logits[idx]
                batch_labels = labels[idx]
                batch_margins = margins[idx]

                temperatures = self.model(batch_margins)
                scaled_logits = batch_logits / temperatures.view(-1, 1)
                loss = self.loss_fn(scaled_logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * idx.numel()

            epoch_loss /= n_samples
            self.loss_history.append(epoch_loss)
            if epoch_loss < best_loss - self.min_delta:
                best_loss = epoch_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"epoch {epoch + 1:4d}/{self.epochs}: loss={epoch_loss:.6f}")

            if self.early_stopping and epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(
                        f"early stopping at epoch {epoch + 1}; "
                        f"best loss={best_loss:.6f}"
                    )
                break

        self.model.load_state_dict(best_state)
        self.is_fitted = True
        return self

    def temperatures(self, logits: ArrayLike) -> ArrayLike:
        if not self.is_fitted:
            raise RuntimeError("Call fit(val_logits, val_labels) before predicting temperatures.")

        original_is_tensor = isinstance(logits, torch.Tensor)
        original_device = logits.device if original_is_tensor else torch.device("cpu")
        logits_tensor = _to_tensor(logits, dtype=torch.float32, device=self.device)
        _validate_logits_and_labels(logits_tensor)

        with torch.no_grad():
            margins = self._normalized_margins(logits_tensor, fit=False)
            temps = self.model(margins)

        if original_is_tensor:
            return temps.to(original_device)
        return temps.cpu().numpy()

    def calibrate(self, logits: ArrayLike, return_logits: bool = False) -> ArrayLike:
        if not self.is_fitted:
            raise RuntimeError("Call fit(val_logits, val_labels) before calibrating logits.")

        original_is_tensor = isinstance(logits, torch.Tensor)
        original_device = logits.device if original_is_tensor else torch.device("cpu")
        logits_tensor = _to_tensor(logits, dtype=torch.float32, device=self.device)
        _validate_logits_and_labels(logits_tensor)

        with torch.no_grad():
            margins = self._normalized_margins(logits_tensor, fit=False)
            temperatures = self.model(margins)
            scaled_logits = logits_tensor / temperatures.view(-1, 1)
            output = scaled_logits if return_logits else F.softmax(scaled_logits, dim=1)

        if original_is_tensor:
            return output.to(original_device)
        return output.cpu().numpy()


__all__ = ["SMART", "MarginTemperatureNet", "compute_margins", "set_seed"]
