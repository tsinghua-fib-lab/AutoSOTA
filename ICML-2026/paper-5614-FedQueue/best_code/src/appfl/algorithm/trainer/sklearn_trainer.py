import time
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from omegaconf import DictConfig
from typing import Any, Optional, Union

from appfl.algorithm.trainer.base_trainer import BaseTrainer


# ── shared param-conversion helpers ───────────────────────────────────────────


def _sklearn_get_state(sklearn_model) -> "OrderedDict[str, torch.Tensor]":
    """Extract a fitted sklearn estimator's learned arrays as an OrderedDict of tensors."""
    state = OrderedDict()
    for attr in sorted(dir(sklearn_model)):
        if not attr.endswith("_") or attr.startswith("__"):
            continue
        val = getattr(sklearn_model, attr)
        if isinstance(val, np.ndarray):
            state[attr] = torch.from_numpy(val.copy()).float()
        elif isinstance(val, list) and val and isinstance(val[0], np.ndarray):
            for i, arr in enumerate(val):
                state[f"{attr}.{i}"] = torch.from_numpy(arr.copy()).float()
    return state


def _sklearn_set_state(sklearn_model, state_dict: dict) -> None:
    """Write tensors from state_dict back into a sklearn estimator's attributes."""
    for key, tensor in state_dict.items():
        arr = tensor.numpy()
        if "." in key:
            attr, idx = key.rsplit(".", 1)
            lst = getattr(sklearn_model, attr, None)
            if isinstance(lst, list):
                lst[int(idx)] = arr.astype(lst[int(idx)].dtype)
        else:
            existing = getattr(sklearn_model, key, None)
            if isinstance(existing, np.ndarray):
                setattr(sklearn_model, key, arr.astype(existing.dtype))


# ── server-side nn.Module wrapper ─────────────────────────────────────────────


class SklearnModelWrapper(nn.Module):
    """
    Wraps a **fitted** sklearn estimator as a ``nn.Module`` so APPFL's
    server-side infrastructure (FedAvgAggregator, scheduler, …) can handle it
    without modification.

    ``state_dict()`` and ``load_state_dict()`` operate on the sklearn model's
    learned numpy arrays rather than ``nn.Parameter`` tensors.  Because the
    wrapper has no ``nn.Parameter`` objects, ``named_parameters()`` yields
    nothing — FedAvg falls through to its direct-averaging path, which is the
    correct behaviour for sklearn parameters.

    Usage (model factory file)::

        model = SGDClassifier(...)
        model.partial_fit(X_dummy, y_dummy, classes=all_classes)
        return SklearnModelWrapper(model)

    The same factory file is used by both server and client.
    ``SklearnTrainer`` automatically unwraps the model on the client side.
    """

    def __init__(self, sklearn_model):
        super().__init__()
        self.sklearn_model = sklearn_model

    def forward(self, x):
        return x

    def state_dict(self, **kwargs):
        return _sklearn_get_state(self.sklearn_model)

    def load_state_dict(self, state_dict, strict=True):
        _sklearn_set_state(self.sklearn_model, state_dict)


# ── client-side trainer ────────────────────────────────────────────────────────


class SklearnTrainer(BaseTrainer):
    """
    Trainer for FL clients that wraps a scikit-learn estimator.

    Parameter conversion (numpy ↔ torch tensor) lets the server's FedAvg
    aggregator treat sklearn params the same way it treats a PyTorch state_dict.

    Supports any sklearn estimator whose learned weights are numpy arrays stored
    as attributes ending with '_' (SGDClassifier, LogisticRegression, GaussianNB,
    MLPClassifier, …).

    If ``model`` is a ``SklearnModelWrapper`` (i.e. the same factory file is
    used for both server and client), the wrapper is automatically unwrapped so
    the trainer always operates on the raw sklearn estimator.

    train_configs keys:
        mode               - "fit" or "partial_fit" (default "partial_fit")
        num_local_epochs   - partial_fit passes per round (default 1)
        partial_fit_classes- class list required by some estimators on first call
        do_validation      - validate after training (default False)
        do_pre_validation  - validate before training, skipped in round 0 (default False)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )
        # Unwrap SklearnModelWrapper if the same factory file is used for server+client.
        if isinstance(self.model, SklearnModelWrapper):
            self.model = self.model.sklearn_model

        # No DataLoader needed — sklearn works directly on numpy arrays.
        # train_dataset / val_dataset can be a torch Dataset or a (X, y) numpy tuple.
        # Mark as fitted if the model already has learned parameters (pre-fitted init model).
        self._is_fitted = hasattr(self.model, "coef_") or hasattr(self.model, "theta_")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_numpy(self, dataset):
        """Convert dataset to (X, y) numpy arrays.

        Accepts:
        - a (X, y) tuple/list of numpy arrays — used as-is
        - a torch Dataset — items are stacked into arrays (no DataLoader needed)
        """
        if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            X, y = dataset
            return np.asarray(X), np.asarray(y)
        # torch Dataset / NumpyDataset: index all items directly.
        X = np.stack([np.asarray(dataset[i][0]) for i in range(len(dataset))])
        y = np.array([np.asarray(dataset[i][1]) for i in range(len(dataset))])
        return X, y

    # ── BaseTrainer interface ─────────────────────────────────────────────────

    def train(self, **kwargs):
        if "round" in kwargs:
            self.round = kwargs["round"]
        self.val_results = {"round": self.round + 1}

        mode = self.train_configs.get("mode", "partial_fit")
        classes = self.train_configs.get("partial_fit_classes", None)
        do_validation = (
            self.train_configs.get("do_validation", False)
            and self.val_dataset is not None
        )
        do_pre_validation = (
            self.train_configs.get("do_pre_validation", False)
            and self.val_dataset is not None
            and self._is_fitted
        )

        title = (
            ["Round", "Pre-Val Acc", "Time", "Train Acc", "Val Acc"]
            if do_pre_validation
            else ["Round", "Time", "Train Acc", "Val Acc"]
        )
        if self.round == 0:
            self.logger.log_title(title)
        self.logger.set_title(title)

        if do_pre_validation:
            pre_val_acc = self._validate()
            self.val_results["pre_val_accuracy"] = pre_val_acc

        X, y = self._to_numpy(self.train_dataset)
        # Match X dtype to coef_ to avoid sklearn's internal dataset type mismatch
        # (SequentialDataset64 vs ArrayDataset32, or vice versa).
        if hasattr(self.model, "coef_"):
            X = X.astype(self.model.coef_.dtype)

        t0 = time.time()
        if mode == "fit":
            self.model.fit(X, y)
        else:
            for _ in range(self.train_configs.get("num_local_epochs", 1)):
                kw = (
                    {"classes": np.array(classes)}
                    if classes and not self._is_fitted
                    else {}
                )
                self.model.partial_fit(X, y, **kw)
        self._is_fitted = True

        train_acc = float(np.mean(self.model.predict(X) == y)) * 100
        val_acc = self._validate() if do_validation else None

        if do_pre_validation:
            self.logger.log_content(
                [
                    self.round,
                    f"{pre_val_acc:.1f}%",
                    time.time() - t0,
                    f"{train_acc:.1f}%",
                    f"{val_acc:.1f}%" if val_acc is not None else "-",
                ]
            )
        else:
            self.logger.log_content(
                [
                    self.round,
                    time.time() - t0,
                    f"{train_acc:.1f}%",
                    f"{val_acc:.1f}%" if val_acc is not None else "-",
                ]
            )

        self.val_results["train_accuracy"] = train_acc
        if val_acc is not None:
            self.val_results["val_accuracy"] = val_acc

        self.round += 1
        self.model_state = _sklearn_get_state(self.model)

    def get_parameters(self) -> Union[dict, tuple]:
        if not hasattr(self, "model_state"):
            self.model_state = _sklearn_get_state(self.model)
        return (
            (self.model_state, self.val_results)
            if hasattr(self, "val_results")
            else self.model_state
        )

    def load_parameters(self, params) -> None:
        if isinstance(params, dict) and params:
            _sklearn_set_state(self.model, params)
            self._is_fitted = True

    def _validate(self) -> float:
        X, y = self._to_numpy(self.val_dataset)
        return float(np.mean(self.model.predict(X) == y)) * 100
