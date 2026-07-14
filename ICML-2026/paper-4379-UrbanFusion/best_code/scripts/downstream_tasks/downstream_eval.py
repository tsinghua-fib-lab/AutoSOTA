#!/usr/bin/env python3
"""
Description: Implementation of downstream evaluation pipeline for
regression and classification tasks using MLP and Ridge models.
"""

import os
import random
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna.trial import Trial
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
SEED = 42


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str = "classification",
    ) -> None:
        """
        Simple MLP model for regression or classification tasks.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of output classes (for classification) or 1
            (for regression).
        task_type : str
            Type of task, either 'classification' or 'regression'.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, output_dim)
        self.task_type = task_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim) for classification
            or (batch_size,) for regression.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.output(x)
        return out if self.task_type == "classification" else out.squeeze()


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min") -> None:
        """
        Early stopping mechanism to stop training when a monitored metric
        has stopped improving.

        Parameters
        ----------
        patience : int
            Number of epochs with no improvement after which training will
            be stopped.
        mode : str
            One of {"min", "max"}. In "min" mode, training will stop when
            the quantity monitored has stopped decreasing; in "max" mode it
            will stop when the quantity monitored has stopped increasing.
        """
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.early_stop = False

    def step(self, score: float, model: nn.Module) -> None:
        """
        Update the early stopping state with the current score.

        Parameters
        ----------
        score : float
            The current score (e.g., validation loss or accuracy).
        model : nn.Module
            The model being trained.
        """
        if (
            self.best_score is None
            or (self.mode == "min" and score < self.best_score)
            or (self.mode == "max" and score > self.best_score)
        ):
            self.best_score = score
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model: nn.Module) -> None:
        """
        Restore the model state from the best state.

        Parameters
        ----------
        model : nn.Module
            The model to restore the state for.
        """
        model.load_state_dict(self.best_state)


def cosine_scheduler(
    optimizer: optim.Optimizer, epochs: int
) -> optim.lr_scheduler.CosineAnnealingLR:
    """
    Creates a cosine annealing learning rate scheduler.

    Parameters
    ----------
    optimizer : optim.Optimizer
        The optimizer for which to schedule the learning rate.
    epochs : int
        Total number of epochs for training.

    Returns
    -------
    optim.lr_scheduler.CosineAnnealingLR
        A cosine annealing learning rate scheduler.
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epochs: int,
    patience: int = 10,
    mode: str = "min",  # 'min' for loss, 'max' for accuracy, etc.
    task_type: str = "classification",
    metric_name: str = "accuracy",
    class_weights: torch.Tensor = None,
) -> tuple:
    """
    Train the MLP model with early stopping using a selectable metric.

    Parameters
    ----------
    model : nn.Module
        The MLP model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : nn.Module
        Loss function to use for training.
    optimizer : optim.Optimizer
        Optimizer for updating model weights.
    scheduler : optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    epochs : int
        Number of epochs to train for.
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    mode : str
        One of {"min", "max"}. In "min" mode, training will stop when
        the quantity monitored has stopped decreasing; in "max" mode it
        will stop when the quantity monitored has stopped increasing.
    task_type : str
        Type of task, either 'classification' or 'regression'.
    metric_name : str
        Name of the metric to monitor for early stopping.
    class_weights : torch.Tensor
        Class weights for the loss function (if applicable).

    Returns
    -------
    tuple
        The trained model and the best validation score.
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, mode=mode)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        # Validation Metric Selection
        if val_loader is not None:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    output = model(xb)
                    preds.append(output.cpu())
                    targets.append(yb.cpu())
            preds = torch.cat(preds).numpy()
            targets = torch.cat(targets).numpy()
            if task_type == "classification":
                if metric_name == "accuracy":
                    val_metric = accuracy_score(
                        targets, np.argmax(preds, axis=1)
                    )
                elif metric_name == "f1":
                    val_metric = f1_score(
                        targets, np.argmax(preds, axis=1), average="macro"
                    )
                elif metric_name == "f1_weighted":
                    val_metric = f1_score(
                        targets, np.argmax(preds, axis=1), average="weighted"
                    )
                elif metric_name == "log_loss":
                    logits = torch.tensor(preds, device=device)
                    targets_t = torch.tensor(
                        targets, device=device, dtype=torch.long
                    )
                    ce = nn.CrossEntropyLoss(
                        weight=class_weights, reduction="mean"
                    )
                    val_metric = ce(logits, targets_t).item()
                else:
                    val_metric = balanced_accuracy_score(
                        targets, np.argmax(preds, axis=1)
                    )
            else:
                val_metric = mean_squared_error(targets, preds)
            early_stopping.step(val_metric, model)
            if early_stopping.early_stop:
                break
        else:
            val_metric = None

    if val_loader is not None:
        early_stopping.restore(model)
        return model, early_stopping.best_score
    else:
        return model, None


def evaluate_mlp(
    model: nn.Module, test_loader: DataLoader, task_type: str
) -> dict:
    """
    Evaluate the MLP model on the test dataset.

    Parameters
    ----------
    model : nn.Module
        The trained MLP model.
    test_loader : DataLoader
        DataLoader for the test dataset.
    task_type : str
        Type of task, either 'classification' or 'regression'.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            output = model(xb).cpu()
            preds.append(output)
            targets.append(yb)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    if task_type == "regression":
        return {
            "mse": mean_squared_error(targets, preds),
            "r2": r2_score(targets, preds),
        }
    else:
        y_pred = np.argmax(preds, axis=1)
        return {
            "accuracy": accuracy_score(targets, y_pred),
            "f1": f1_score(targets, y_pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(targets, y_pred),
            "f1_weighted": f1_score(targets, y_pred, average="weighted"),
        }


def tune_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model_type: str,
    task_type: str,
    input_dim: int,
    output_dim: int,
    epochs: int,
    metric_name: str,
    class_weights: torch.Tensor = None,
    seed: int = 0,
    alpha_range: tuple[float, float] = (1e-4, 100.0),
    lr_range: tuple[float, float] = (1e-5, 1e-1),
    weight_decay_range: tuple[float, float] = (1e-6, 1e-1),
) -> dict:
    """
    Tune hyperparameters using Optuna.

    For classification tasks, the metric to optimize is determined by
    `metric_name` and can be 'accuracy', 'f1', or 'balanced_accuracy'.
    For regression tasks, mean squared error (MSE) is always optimized.

    Parameters
    ----------
    X_train : torch.Tensor
        Training feature data.
    y_train : torch.Tensor
        Training target data.
    X_val : torch.Tensor
        Validation feature data.
    y_val : torch.Tensor
        Validation target data.
    model_type : str
        The type of model to tune (e.g., 'mlp', 'ridge', etc.).
    task_type : str
        The type of task, either 'classification' or 'regression'.
    input_dim : int
        The number of input features.
    output_dim : int
        The number of output targets or classes.
    epochs : int
        Number of training epochs for each trial.
    metric_name : str
        The metric to optimize for classification tasks ('accuracy', 'f1', or
        'balanced_accuracy').
        Ignored for regression tasks.
    class_weights : torch.Tensor, optional
        Class weights for handling class imbalance in classification tasks.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    alpha_range : tuple[float, float], optional
        Range for the alpha hyperparameter in Ridge regression.
        Default is (1e-4, 100.0).
    lr_range : tuple[float, float], optional
        Range for the learning rate hyperparameter in MLP.
        Default is (1e-5, 1e-1).
    weight_decay_range : tuple[float, float], optional
        Range for the weight decay hyperparameter in MLP.
        Default is (1e-6, 1e-1).

    Returns
    -------
    study : optuna.Study
        The Optuna study object containing the optimization results.
    best_params : dict
        The best hyperparameters found during the optimization.
    best_score : float
        The best score achieved on the validation set.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def objective(trial: Trial):
        if model_type == "mlp":
            lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
            wd = trial.suggest_float(
                "weight_decay",
                weight_decay_range[0],
                weight_decay_range[1],
                log=True,
            )
            model = MLP(input_dim, output_dim, task_type).to(device)
            if task_type == "regression":
                criterion = nn.MSELoss()
            else:
                weight = (
                    class_weights.to(device)
                    if class_weights is not None
                    else None
                )
                criterion = nn.CrossEntropyLoss(weight=weight)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=wd
            )
            scheduler = cosine_scheduler(optimizer, epochs)
            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_train).float(),
                    (
                        torch.tensor(y_train).long()
                        if task_type == "classification"
                        else torch.tensor(y_train).float()
                    ),
                ),
                batch_size=64,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val).float(),
                    (
                        torch.tensor(y_val).long()
                        if task_type == "classification"
                        else torch.tensor(y_val).float()
                    ),
                ),
                batch_size=64,
            )
            # Define early stopping
            patience = 10  # you can make this a parameter!
            if metric_name == "log_loss" or task_type == "regression":
                mode = "min"
            else:
                mode = "max"
            early_stopping = EarlyStopping(patience=patience, mode=mode)

            for epoch in range(epochs):
                # Training loop
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                # Validation
                model.eval()
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        out = model(xb).cpu()
                        val_preds.append(out)
                        val_targets.append(yb)
                val_preds = torch.cat(val_preds).numpy()
                val_targets = torch.cat(val_targets).numpy()
                if np.any(np.isnan(val_preds)) or np.any(
                    np.isnan(val_targets)
                ):
                    raise optuna.exceptions.TrialPruned()
                if task_type == "classification":
                    if metric_name == "accuracy":
                        y_pred = np.argmax(val_preds, axis=1)
                        val_score = accuracy_score(val_targets, y_pred)
                    elif metric_name == "f1":
                        y_pred = np.argmax(val_preds, axis=1)
                        val_score = f1_score(
                            val_targets, y_pred, average="macro"
                        )
                    elif metric_name == "f1_weighted":
                        y_pred = np.argmax(val_preds, axis=1)
                        val_score = f1_score(
                            val_targets, y_pred, average="weighted"
                        )
                    elif metric_name == "log_loss":
                        logits = torch.tensor(val_preds, device=device)
                        targets_t = torch.tensor(
                            val_targets, device=device, dtype=torch.long
                        )
                        ce = nn.CrossEntropyLoss(
                            weight=class_weights, reduction="mean"
                        )
                        val_score = ce(logits, targets_t).item()
                    else:
                        y_pred = np.argmax(val_preds, axis=1)
                        val_score = balanced_accuracy_score(
                            val_targets, y_pred
                        )
                else:
                    val_score = mean_squared_error(val_targets, val_preds)

                # Step early stopping
                early_stopping.step(val_score, model)
                if early_stopping.early_stop:
                    break

            # Restore best weights
            early_stopping.restore(model)

            # At the end, compute score on val set using best model
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    out = model(xb).cpu()
                    preds.append(out)
                    targets.append(yb)
            preds = torch.cat(preds).numpy()
            targets = torch.cat(targets).numpy()
            if task_type == "classification":
                y_pred = np.argmax(preds, axis=1)
                extra_metrics = {
                    "accuracy": accuracy_score(targets, y_pred),
                    "validation_balanced_accuracy": balanced_accuracy_score(
                        targets, y_pred
                    ),
                    "validation_f1_weighted": f1_score(
                        targets, y_pred, average="weighted"
                    ),
                    "validation_f1_macro": f1_score(
                        targets, y_pred, average="macro"
                    ),
                    "validation_log_loss": log_loss(targets, preds),
                }

                # Store these extra metrics with the trial (for later inspection)
                for k, v in extra_metrics.items():
                    trial.set_user_attr(k, v)

                if metric_name == "accuracy":
                    score = accuracy_score(targets, y_pred)
                elif metric_name == "f1":
                    score = f1_score(targets, y_pred, average="macro")
                elif metric_name == "f1_weighted":
                    score = f1_score(targets, y_pred, average="weighted")
                elif metric_name == "log_loss":
                    logits = torch.tensor(preds, device=device)
                    targets_t = torch.tensor(
                        targets, device=device, dtype=torch.long
                    )
                    ce = nn.CrossEntropyLoss(
                        weight=class_weights, reduction="mean"
                    )
                    score = ce(logits, targets_t).item()
                    return score
                else:
                    score = balanced_accuracy_score(targets, y_pred)
                return 1.0 - score
            else:
                return mean_squared_error(targets, preds)
        else:
            alpha = trial.suggest_float(
                "alpha", alpha_range[0], alpha_range[1], log=True
            )
            if task_type == "classification":
                model = LogisticRegression(
                    C=1.0 / alpha,
                    max_iter=1000,
                    random_state=0,
                    class_weight=class_weights,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                if class_weights is not None:
                    sample_weights = [class_weights[y] for y in y_val]
                else:
                    sample_weights = None
                probs = model.predict_proba(X_val)
                extra_metrics = {
                    "validation_accuracy": accuracy_score(y_val, preds),
                    "validation_balanced_accuracy": balanced_accuracy_score(
                        y_val, preds
                    ),
                    "validation_f1_weighted": f1_score(
                        y_val, preds, average="weighted"
                    ),
                    "validation_f1_macro": f1_score(
                        y_val, preds, average="macro"
                    ),
                    "validation_log_loss": log_loss(
                        y_val, probs, sample_weight=sample_weights
                    ),
                }

                # Store these extra metrics with the trial (for later inspection)
                for k, v in extra_metrics.items():
                    trial.set_user_attr(k, v)
                if metric_name == "accuracy":
                    score = accuracy_score(y_val, preds)
                elif metric_name == "f1":
                    score = f1_score(y_val, preds, average="macro")
                elif metric_name == "f1_weighted":
                    score = f1_score(y_val, preds, average="weighted")
                elif metric_name == "log_loss":
                    if class_weights is not None:
                        sample_weights = [class_weights[y] for y in y_val]
                    else:
                        sample_weights = None
                    score = log_loss(
                        y_val, probs, sample_weight=sample_weights
                    )
                    return score
                else:
                    score = balanced_accuracy_score(y_val, preds)
                return 1.0 - score
            elif model_type == "kernel_ridge":
                alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1], log=True)
                kernel = trial.suggest_categorical("kernel", ["rbf", "cosine"])
                gamma = trial.suggest_float("gamma", 1e-5, 1e2, log=True)
                n_components = trial.suggest_int("n_components", 500, 2000)
                kwargs = {"kernel": kernel, "n_components": n_components, "random_state": SEED}
                if kernel == "rbf":
                    kwargs["gamma"] = gamma
                feature_map = Nystroem(**kwargs)
                X_train_mapped = feature_map.fit_transform(X_train)
                X_val_mapped = feature_map.transform(X_val)
                model = Ridge(alpha=alpha)
                model.fit(X_train_mapped, y_train)
                preds = model.predict(X_val_mapped)
                return mean_squared_error(y_val, preds)
            else:
                model_cls = Ridge
                model = model_cls(alpha=alpha)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                return mean_squared_error(y_val, preds)

    n_jobs = 1 if model_type == "mlp" else -1

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective, n_trials=100, show_progress_bar=False, n_jobs=n_jobs
    )
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    if task_type == "classification":
        if metric_name == "log_loss":
            best_score = best_value
        else:
            best_score = 1.0 - best_value
        accuracy = best_trial.user_attrs.get("validation_accuracy", None)
        balanced_accuracy = best_trial.user_attrs.get(
            "validation_balanced_accuracy", None
        )
        f1_weighted = best_trial.user_attrs.get("validation_f1_weighted", None)
        f1_macro = best_trial.user_attrs.get("validation_f1_macro", None)
        return {
            "best_params": best_params,
            "best_score": best_score,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
        }

    else:
        best_score = best_value

    return {"best_params": best_params, "best_score": best_score}


def standardize_and_rescale_last_n(
    X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray, n_new: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize the last `n_new` columns of each split, then (if there are
    other features) rescale them to match the global std of the original
    features.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature data of shape (n_samples, n_features).
    X_test : np.ndarray
        Test feature data of shape (n_samples, n_features).
    X_val : np.ndarray
        Validation feature data of shape (n_samples, n_features).
    n_new : int
        Number of last columns to standardize and rescale.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the standardized and rescaled training,
        test, and validation feature data.
    """
    n_total = X_train.shape[1]
    if n_new <= 0 or n_new > n_total:
        raise ValueError(
            "`n_new` must be between 1 and total number of columns"
        )

    new_slice = slice(n_total - n_new, None)

    scaler = StandardScaler()
    X_train[:, new_slice] = scaler.fit_transform(X_train[:, new_slice])
    X_test[:, new_slice] = scaler.transform(X_test[:, new_slice])
    X_val[:, new_slice] = scaler.transform(X_val[:, new_slice])

    if n_total > n_new:
        orig_std = X_train[:, : n_total - n_new].std()
        X_train[:, new_slice] *= orig_std
        X_test[:, new_slice] *= orig_std
        X_val[:, new_slice] *= orig_std

    return X_train, X_test, X_val


def standardize_all(
    X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize all columns (zero mean, unit variance) using statistics from X_train.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature data of shape (n_samples, n_features).
    X_test : np.ndarray
        Test feature data of shape (n_samples, n_features).
    X_val : np.ndarray
        Validation feature data of shape (n_samples, n_features).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Standardized training, test, and validation data.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_test, X_val


def standardize_except_onehot(
    X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray, n_one_hot: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize all columns except the last `n_one_hot` (assumed to be one-hot encoded).
    Standardization is done using statistics from X_train only.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature data of shape (n_samples, n_features).
    X_test : np.ndarray
        Test feature data of shape (n_samples, n_features).
    X_val : np.ndarray
        Validation feature data of shape (n_samples, n_features).
    n_one_hot : int
        Number of one-hot encoded features at the end that should not be standardized.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Transformed training, test, and validation data.
    """
    if n_one_hot > 0:
        X_train_cont = X_train[:, :-n_one_hot]
        X_test_cont = X_test[:, :-n_one_hot]
        X_val_cont = X_val[:, :-n_one_hot]

        X_train_oh = X_train[:, -n_one_hot:]
        X_test_oh = X_test[:, -n_one_hot:]
        X_val_oh = X_val[:, -n_one_hot:]
    else:
        X_train_cont = X_train
        X_test_cont = X_test
        X_val_cont = X_val
        X_train_oh = X_test_oh = X_val_oh = np.empty((X_train.shape[0], 0))

    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_cont)
    X_test_cont = scaler.transform(X_test_cont)
    X_val_cont = scaler.transform(X_val_cont)

    X_train_new = np.hstack([X_train_cont, X_train_oh])
    X_test_new = np.hstack([X_test_cont, X_test_oh])
    X_val_new = np.hstack([X_val_cont, X_val_oh])

    return X_train_new, X_test_new, X_val_new


def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    model_type: str,
    epochs: int = 40,
    metric_name: str = "accuracy",
    train_split: float = 0.6,
    validation_split: float = 0.2,
    class_weights: torch.Tensor = None,
    standardize: int = None,
    alpha_range: tuple[float, float] = (1e-4, 100.0),
    lr_range: tuple[float, float] = (1e-5, 1e-1),
    weight_decay_range: tuple[float, float] = (1e-6, 1e-1),
    dataset_name: str = None,
) -> pd.DataFrame:
    """
    Run the entire evaluation pipeline for a given dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature data of shape (n_samples, n_features).
    y : np.ndarray
        Target data of shape (n_samples,).
    task_type : str
        Type of task, either 'classification' or 'regression'.
    model_type : str
        Type of model to use, either 'mlp' or 'ridge'.
    epochs : int, optional
        Number of training epochs. Default is 40.
    metric_name : str, optional
        Metric to optimize for classification tasks. Can be 'accuracy',
        'f1', or 'balanced_accuracy'. Default is 'accuracy'.
    train_split : float, optional
        Proportion of data to use for training. Default is 0.6.
    validation_split : float, optional
        Proportion of data to use for validation. Default is 0.2.
    class_weights : torch.Tensor, optional
        Class weights for handling class imbalance in classification tasks.
        Default is None (no class weights).
    standardize : int, optional
        N new features to standardize. If None, no standardization is applied.
        If True, standardize all features. If an integer, standardize the last
        `n_new` features. Default is None.
    alpha_range : tuple[float, float], optional
        Range for the alpha hyperparameter in Ridge regression. Default is (1e-4, 100.0).
    lr_range : tuple[float, float], optional
        Range for the learning rate hyperparameter in MLP. Default is (1e-5, 1e-1).
    weight_decay_range : tuple[float, float], optional
        Range for the weight decay hyperparameter in MLP. Default is (1e-6, 1e-1).
    dataset_name : str, optional
        Name of the dataset being evaluated. Used for specific handling
        of certain datasets (e.g., 'landuse_eu_fine_in_region').
        Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the evaluation metrics for each trial.
        Columns depend on the task type:
        - For classification: 'accuracy', 'f1', 'balanced_accuracy'.
        - For regression: 'mse', 'r2'.
    """
    metrics_list = []

    # Set a fixed seed for consistent tuning and data splits
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if task_type == "classification":
        uniq_labels, y = np.unique(y, return_inverse=True)
        y = y.astype(np.int64)
        label_map = {old: new for new, old in enumerate(uniq_labels)}
    else:
        label_map = None
    stratify_flag = y if task_type == "classification" else None

    test_size = 1 - train_split - validation_split
    val_size = validation_split / (train_split + validation_split)

    # Single train/val/test split for tuning and evaluation
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=stratify_flag
    )
    stratify_2 = y_trainval if task_type == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=SEED,
        stratify=stratify_2,
    )
    if dataset_name == "landuse_eu_f_in_region":
        idx12_train = np.where(y_train == 12)[0]
        idx12_val = np.where(y_val == 12)[0]
        if len(idx12_train) > 0 and len(idx12_val) == 0:
            chosen = idx12_train[0]
            X_swap = X_train[chosen : chosen + 1]
            y_swap = y_train[chosen : chosen + 1]
            X_train = np.delete(X_train, chosen, axis=0)
            y_train = np.delete(y_train, chosen, axis=0)
            X_val = np.concatenate([X_val, X_swap], axis=0)
            y_val = np.concatenate([y_val, y_swap], axis=0)

    if task_type == "classification":
        if class_weights is None:
            # no weighting
            class_weights = None
        elif isinstance(class_weights, torch.Tensor):
            # user supplied a ready-to-use tensor
            if model_type == "mlp":
                class_weights = class_weights.to(device)
            else:
                class_weights = {
                    i: w for i, w in enumerate(class_weights.cpu().numpy())
                }
        elif (
            isinstance(class_weights, str)
            and class_weights.lower() == "inverse_square_root"
        ):
            # compute 1/√freq weights on the *training* set
            uniq, counts = np.unique(y_train, return_counts=True)
            weight_arr = np.zeros(len(np.unique(y_train)), dtype=np.float32)
            for cls, cnt in zip(uniq, counts):
                weight_arr[cls] = 1.0 / np.sqrt(cnt)
            if model_type == "mlp":
                class_weights = torch.tensor(
                    weight_arr, dtype=torch.float32, device=device
                )
            else:
                class_weights = {
                    cls: float(w) for cls, w in enumerate(weight_arr)
                }
        else:
            raise ValueError(
                "class_weights must be None, a torch.Tensor, or the string "
                '"inverse_square_root"'
            )
    else:
        # regression – ignore any class_weights argument
        class_weights = None

    if standardize is not None:
        if isinstance(standardize, bool) and standardize:
            X_train, X_test, X_val = standardize_all(X_train, X_test, X_val)
        elif isinstance(standardize, int) and standardize > 0:
            # standardize last n columns
            X_train, X_test, X_val = standardize_except_onehot(
                X_train, X_test, X_val, n_one_hot=standardize
            )
    input_dim = X.shape[1]
    output_dim = len(np.unique(y)) if task_type == "classification" else 1

    # Count samples per class in each split
    if task_type == "classification":
        train_counts = np.bincount(y_train, minlength=output_dim)
        val_counts = np.bincount(y_val, minlength=output_dim)
        test_counts = np.bincount(y_test, minlength=output_dim)
        print(
            f"Train class counts: {dict(enumerate(train_counts))}, "
            f"Val class counts: {dict(enumerate(val_counts))}, "
            f"Test class counts: {dict(enumerate(test_counts))}"
        )

    # Tune hyperparameters on the fixed train/val split
    tuning_results = tune_model(
        X_train,
        y_train,
        X_val,
        y_val,
        model_type,
        task_type,
        input_dim,
        output_dim,
        epochs,
        metric_name,
        seed=SEED,
        class_weights=class_weights,
        alpha_range=alpha_range,
        lr_range=lr_range,
        weight_decay_range=weight_decay_range,
    )
    best_params = tuning_results["best_params"]
    best_score = tuning_results["best_score"]

    if model_type == "mlp":
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [0]

    # Create DataLoaders for final training and testing
    for seed in seeds:
        # Seed RNGs for model init and DataLoader shuffle
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # confert array to float
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        train_dataset = TensorDataset(
            torch.tensor(X_train).float(),
            (
                torch.tensor(y_train).long()
                if task_type == "classification"
                else torch.tensor(y_train).float()
            ),
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        val_dataset = TensorDataset(
            torch.tensor(X_val).float(),
            (
                torch.tensor(y_val).long()
                if task_type == "classification"
                else torch.tensor(y_val).float()
            ),
        )
        val_loader = DataLoader(val_dataset, batch_size=64)

        test_dataset = TensorDataset(
            torch.tensor(X_test).float(),
            (
                torch.tensor(y_test).long()
                if task_type == "classification"
                else torch.tensor(y_test).float()
            ),
        )
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Train and evaluate
        if model_type == "mlp":
            model = MLP(input_dim, output_dim, task_type).to(device)
            if task_type == "regression":
                criterion = nn.MSELoss()
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"],
            )
            scheduler = cosine_scheduler(optimizer, epochs)

            if task_type == "classification":
                if metric_name == "log_loss":
                    mode = "min"
                else:
                    mode = "max"
            else:
                mode = "min"

            model, val_score = train_mlp(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
                patience=10,
                mode=mode,
                task_type=task_type,
                metric_name=metric_name,
                class_weights=class_weights,
            )
            metric = evaluate_mlp(model, test_loader, task_type)
            metric["hyperparams"] = best_params
            metric["seed"] = seed
            if task_type == "classification":
                metric[f"val_{metric_name}"] = best_score
                metric["validation_accuracy"] = tuning_results.get(
                    "accuracy", None
                )
                metric["validation_balanced_accuracy"] = tuning_results.get(
                    "balanced_accuracy", None
                )
                metric["validation_f1_weighted"] = tuning_results.get(
                    "f1_weighted", None
                )
                metric["validation_f1_macro"] = tuning_results.get(
                    "f1_macro", None
                )
            elif task_type == "regression":
                metric["val_mse"] = best_score

        else:
            if task_type == "classification":
                model = LogisticRegression(
                    C=1.0 / best_params["alpha"],
                    random_state=seed,
                    max_iter=1000,
                    class_weight=class_weights,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metric = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, average="macro"),
                    "f1_weighted": f1_score(
                        y_test, y_pred, average="weighted"
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_test, y_pred
                    ),
                    f"val_{metric_name}": best_score,
                    "validation_accuracy": tuning_results.get(
                        "accuracy", None
                    ),
                    "validation_balanced_accuracy": tuning_results.get(
                        "balanced_accuracy", None
                    ),
                    "validation_f1_weighted": tuning_results.get(
                        "f1_weighted", None
                    ),
                    "validation_f1_macro": tuning_results.get(
                        "f1_macro", None
                    ),
                    "hyperparams": best_params,
                    "seed": seed,
                }
            elif model_type == "kernel_ridge":
                kernel = best_params.get("kernel", "rbf")
                kwargs = {
                    "kernel": kernel,
                    "n_components": best_params["n_components"],
                    "random_state": SEED,
                }
                if kernel == "rbf" and "gamma" in best_params:
                    kwargs["gamma"] = best_params["gamma"]
                feature_map = Nystroem(**kwargs)
                X_train_mapped = feature_map.fit_transform(X_train)
                X_test_mapped = feature_map.transform(X_test)
                model = Ridge(alpha=best_params["alpha"])
                model.fit(X_train_mapped, y_train)
                preds = model.predict(X_test_mapped)
                metric = {
                    "mse": mean_squared_error(y_test, preds),
                    "r2": r2_score(y_test, preds),
                    "val_mse": best_score,
                    "hyperparams": best_params,
                    "seed": seed,
                }
            else:
                model_cls = Ridge
                model = model_cls(alpha=best_params["alpha"])
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metric = {
                    "mse": mean_squared_error(y_test, preds),
                    "r2": r2_score(y_test, preds),
                    "val_mse": best_score,
                    "hyperparams": best_params,
                    "seed": seed,
                }

        metrics_list.append(metric)

    print(
        f"{model_type.upper()} ({task_type}) — Aggregated Results "
        f"({metric_name} optimization):"
    )

    print(metrics_list)

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df
