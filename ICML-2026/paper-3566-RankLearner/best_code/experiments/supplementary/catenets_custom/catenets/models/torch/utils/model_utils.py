"""
Model utils shared across different nets
"""
# Author: Alicia Curth, Bogdan Cebere
from typing import Any, Optional

import torch
from sklearn.model_selection import train_test_split

import catenets.logger as log
from catenets.models.constants import DEFAULT_SEED, DEFAULT_VAL_SPLIT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_STRING = "training"
VALIDATION_STRING = "validation"


def make_val_split(
    X: torch.Tensor,
    y: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
) -> Any:
    if val_split_prop == 0:
        # return original data
        if w is None:
            return X, y, X, y, TRAIN_STRING

        return X, y, w, X, y, w, TRAIN_STRING

    X = X.cpu()
    y = y.cpu()
    # make actual split
    if w is None:
        X_t, X_val, y_t, y_val = train_test_split(
            X, y, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return (
            X_t.to(DEVICE),
            y_t.to(DEVICE),
            X_val.to(DEVICE),
            y_val.to(DEVICE),
            VALIDATION_STRING,
        )

    w = w.cpu()
    if stratify_w:
        # split to stratify by group
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X,
            y,
            w,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X, y, w, test_size=val_split_prop, random_state=seed, shuffle=True
        )

    return (
        X_t.to(DEVICE),
        y_t.to(DEVICE),
        w_t.to(DEVICE),
        X_val.to(DEVICE),
        y_val.to(DEVICE),
        w_val.to(DEVICE),
        VALIDATION_STRING,
    )


def train_wrapper(
    estimator: Any,
    X: torch.Tensor,
    y: torch.Tensor,
    **kwargs: Any,
) -> None:
    if hasattr(estimator, "train"):
        log.debug(f"Train PyTorch network {estimator}")
        estimator.fit(X, y, **kwargs)
    elif hasattr(estimator, "fit"):
        log.debug(f"Train sklearn estimator {estimator}")
        estimator.fit(X.detach().cpu().numpy(), y.detach().cpu().numpy())
    else:
        raise NotImplementedError(f"Invalid estimator for the {estimator}")


def predict_wrapper(estimator: Any, X: torch.Tensor) -> torch.Tensor:
    if hasattr(estimator, "forward"):
        return estimator(X)
    elif hasattr(estimator, "predict_proba"):
        X_np = X.detach().cpu().numpy()
        no_event_proba = estimator.predict_proba(X_np)[:, 0]  # no event probability

        return torch.Tensor(no_event_proba)
    elif hasattr(estimator, "predict"):
        X_np = X.detach().cpu().numpy()
        no_event_proba = estimator.predict(X_np)

        return torch.Tensor(no_event_proba)
    else:
        raise NotImplementedError(f"Invalid estimator for the {estimator}")
