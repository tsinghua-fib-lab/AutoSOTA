import random
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sbi.diagnostics.lc2st import LC2ST
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from utils.misc import (
    rbf_kernel_matrix,
    t_stat_scorer,
)
from utils.networks import DefaultMLP, LinearClassifier, SmallMLP

# Registry allowing string choice
_MODEL_REGISTRY = {
    "mlp": DefaultMLP,
    "mlp_small": SmallMLP,
    "linear": LinearClassifier,
}



def compute_lc2st(
    estimator,
    x_tilde,
    ntraj,
    n_train_cal_lc2st,
    y_o,
    idx,
    theta_cal_train,
    y_cal,
):
    """
    Computes the lc2st value for a given set of inputs.

    Args:
        estimators (dict): Dictionary containing the "npe" estimator.
        x_tilde (Tensor): Input tensor.
        ntraj (int): Stride to index the input tensor.
        n_train_cal_lc2st (int): Number of training samples for calibration.
        y_o (Tensor): Observation tensor.
        idx (int): Index to select specific data.
        theta_cal_train (Tensor): Calibration training tensor.
        y_cal (Tensor): Calibration data tensor.

    Returns:
        float: lc2st value.
    """
    # Generate training predictions
    preds_train = estimator.flow(x_tilde[::ntraj][:n_train_cal_lc2st]).sample()

    # Extract observation at index
    xs_star = y_o[idx]

    # Generate posterior samples
    post_samples_star_fmcpe = estimator.flow(x_tilde[::ntraj][idx]).sample((200,))

    # Compute lc2st
    _, l_c2st_val = lc2st(
        theta_cal_train.to("cpu"),
        y_cal.to("cpu"),
        preds_train.to("cpu"),
        xs_star.to("cpu"),
        post_samples_star_fmcpe.to("cpu"),
    )

    return l_c2st_val


def classifier_two_samples_test(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    cv: str = "StratifiedKFold",
) -> float:
    """Classifier-based 2-sample test returning accuracy

    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClassifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionality of the
    samples X and Y.

    Args:
        X: Sample 1
        Y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples
        cv: Cross-validation strategy. Either 'KFold' or 'StratifiedKFold'

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    if cv == "KFold":
        shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    elif cv == "StratifiedKFold":
        shuffle = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"Unknown cross-validation strategy: {cv}")
    if scoring == "t_stat":
        scoring = t_stat_scorer
    elif scoring == "accuracy":
        scoring = "accuracy"
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.mean(scores)
    if scoring == "accuracy":
        return scores
    else:
        return -1.0 * scores


def stein_discrepancy(theta: torch.Tensor, score_fn, lengthscale: float = 1.0) -> float:
    """
    Compute Kernel Stein Discrepancy for a set of samples and target score function.

    Args:
        theta: Tensor of shape (N, D), the predicted samples
        score_fn: Callable that returns ∇ log p(theta), shape (N, D)
        lengthscale: RBF kernel lengthscale

    Returns:
        Scalar KSD value.
    """
    N, D = theta.shape
    theta.requires_grad_(True)
    score = score_fn(theta)  # (N, D)
    theta = theta.detach()  # Detach to avoid gradient tracking

    K, sq_dist = rbf_kernel_matrix(theta, lengthscale)  # (N, N)
    K_grad = (
        -(theta.unsqueeze(1) - theta.unsqueeze(0)) / lengthscale**2 * K.unsqueeze(-1)
    )  # (N, N, D)
    K_hess_trace = (D / lengthscale**2 - sq_dist / lengthscale**4) * K  # (N, N)

    s_i_dot_s_j = score @ score.T  # (N, N)
    s_i_dot_grad_j = torch.einsum("ik,ijk->ij", score, K_grad)  # (N, N)
    s_j_dot_grad_i = torch.einsum("jk,ijk->ij", score, K_grad)  # (N, N), symmetric
    U = s_i_dot_s_j * K + s_i_dot_grad_j + s_j_dot_grad_i + K_hess_trace

    ksd_squared = U.mean()
    theta.requires_grad_(False)
    theta.grad = None
    return ksd_squared.item()


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


def lc2st(
    theta_cal: Tensor,
    y_cal: Tensor,
    preds_fmcpe: Tensor,
    xs_star: Tensor,
    post_samples_star_fmcpe: Tensor,
):
    lc2st_fmcpe = LC2ST(
        thetas=theta_cal,
        xs=y_cal,
        posterior_samples=preds_fmcpe,
        classifier="mlp",
        num_ensemble=1,
    )
    _ = lc2st_fmcpe.train_under_null_hypothesis()  # over 100 trials under (H0)
    _ = lc2st_fmcpe.train_on_observed_data()  # on observed data

    conf = 0.05
    p_value_fmcpe_list = []
    reject_fmcpe_list = []
    for i in tqdm(range(len(xs_star)), desc="LC2ST on test set"):
        # Compute LC2ST scores for FMCPE
        probs_fmcpe, scores_fmcpe = lc2st_fmcpe.get_scores(
            theta_o=post_samples_star_fmcpe[i, :],
            x_o=xs_star[i],
            return_probs=True,
            trained_clfs=lc2st_fmcpe.trained_clfs,
        )
        p_value_fmcpe = lc2st_fmcpe.p_value(post_samples_star_fmcpe[i], xs_star[i])
        reject_fmcpe = lc2st_fmcpe.reject_test(post_samples_star_fmcpe[i], xs_star[i], alpha=conf)
        p_value_fmcpe_list.append(p_value_fmcpe)
        reject_fmcpe_list.append(reject_fmcpe)

    p_value_fmcpe_mean = np.mean(p_value_fmcpe_list)
    reject_fmcpe_mean = np.mean(reject_fmcpe_list)
    return (
        p_value_fmcpe_mean,
        reject_fmcpe_mean,
    )


def acauc_rope(true_thetas, theta_pred):
    """
    Compute the Average Coverage AUC (ACAUC) metric using sbi's Posterior.
    """
    m = theta_pred.shape[0]
    cred_levels = []
    N = len(true_thetas)
    for i, true_theta in enumerate(true_thetas):
        samples = theta_pred[:, i, :].squeeze(1)  # shape (m, D)
        sorted_samples, _ = torch.sort(samples, dim=0)
        ranks = (sorted_samples <= true_theta[None, :]).sum(dim=0) / m
        cred_levels.append(ranks)
    cred_levels = torch.cat(cred_levels, dim=0)  # shape (N, D)
    sorted_cred_levels, _ = torch.sort(cred_levels, dim=0)  # shape (N, D)
    calibration = sorted_cred_levels - torch.linspace(0, 1, N)[:, None]
    return torch.mean(calibration)


def acauc(true_thetas, posterior_samples):
    """Compute ACAUC as defined in Appendix J of the RoPE paper.

    For each (observation, dimension) pair, computes the empirical CDF rank
    of the true parameter under the posterior samples, then derives the
    minimum credible level that would include it.

    ACAUC = mean over all (i, j) of (|2*rank - 1| - 0.5)

    Args:
        true_thetas: torch.Tensor of shape (N, D)
            True parameter values.
        posterior_samples: torch.Tensor of shape (M, N, D)
            Posterior samples (M per observation per dimension).

    Returns:
        ACAUC scalar (float). 0 = perfect calibration,
        >0 = overconfident, <0 = underconfident.
    """
    # ranks: fraction of samples <= true value, shape (N, D)
    ranks = (posterior_samples <= true_thetas.unsqueeze(0)).float().mean(dim=0)
    # Minimum credible level containing the true value
    c = torch.abs(2 * ranks - 1)
    return (c - 0.5).mean().item()


def mse(theta_true: torch.Tensor, theta_pred: torch.Tensor) -> float:
    """
    Compute average MSE between predicted samples and true parameters.

    Args:
        theta_true: torch.Tensor of shape (N, D)
            True parameter values.
        theta_pred: torch.Tensor of shape (M, N, D)
            Predicted samples (M draws for each of N observations).

    Returns:
        mse: float, average mean squared error across all samples, N, D
    """
    # Expand theta_true to match shape (M, N, D)
    expanded_true = theta_true.unsqueeze(0).expand_as(theta_pred)

    # Compute squared error
    sq_error = (theta_pred - expanded_true) ** 2

    # Average over all dimensions, data points, and samples
    mse = sq_error.mean().item()
    return mse


# ---------- Main function (PyTorch) ----------
def classifier_two_samples_test_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",  # "accuracy" or "t_stat"
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    cv: str = "StratifiedKFold",  # "KFold" or "StratifiedKFold"
    model: Union[
        str, Callable
    ] = "mlp",  # str to pick from registry or callable returning nn.Module
    model_kwargs: Optional[Dict] = None,  # kwargs passed when constructing the model
    training_kwargs: Optional[Dict] = None,  # dict for epochs, batch_size, lr, weight_decay, device
) -> float:
    """
    PyTorch version of classifier two-sample test.

    Args:
        X, Y: torch.Tensor inputs of shape (n_samples, dim). CPU/GPU tensors accepted.
        model: "mlp" (default), "mlp_small", "linear", or a callable/class that accepts (input_dim, **model_kwargs)
        model_kwargs: kwargs forwarded to model constructor
        training_kwargs: {
            "epochs": int (default 100),
            "batch_size": int (default 128),
            "lr": float (default 1e-3),
            "weight_decay": float (default 0.0),
            "device": "cpu" or "cuda" or None (auto),
            "verbose": bool (default False)
        }
    Returns:
        float: mean accuracy (if scoring=="accuracy") or -mean_score (for other scoring to match your original sign convention)
    """
    # --- defaults and reproducibility ---
    if model_kwargs is None:
        model_kwargs = {}
    if training_kwargs is None:
        training_kwargs = {}
    epochs = int(training_kwargs.get("epochs", 100))
    batch_size = int(training_kwargs.get("batch_size", 128))
    lr = float(training_kwargs.get("lr", 1e-3))
    weight_decay = float(training_kwargs.get("weight_decay", 0.0))
    verbose = bool(training_kwargs.get("verbose", False))
    device = training_kwargs.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # --- Preprocessing ---
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        # avoid division by zero
        X_std = torch.where(X_std == 0, torch.ones_like(X_std), X_std)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X = X + noise_scale * torch.randn_like(X)
        Y = Y + noise_scale * torch.randn_like(Y)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    ndim = X_np.shape[1]

    # --- Build labels and data container ---
    data = np.concatenate((X_np, Y_np), axis=0)
    labels = np.concatenate(
        (np.zeros(X_np.shape[0], dtype=int), np.ones(Y_np.shape[0], dtype=int)), axis=0
    )

    # --- CV splitter ---
    if cv == "KFold":
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    elif cv == "StratifiedKFold":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"Unknown cross-validation strategy: {cv}")

    fold_scores = []

    # --- iterate folds ---
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(data, labels)):
        # build datasets
        X_train = torch.from_numpy(data[train_idx]).float()
        y_train = torch.from_numpy(labels[train_idx]).long()
        X_test = torch.from_numpy(data[test_idx]).float()
        y_test = torch.from_numpy(labels[test_idx]).long()

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # --- construct model ---
        if isinstance(model, str):
            if model not in _MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model string '{model}'. Known: {list(_MODEL_REGISTRY.keys())}"
                )
            model_cls = _MODEL_REGISTRY[model]
            net = model_cls(ndim, **model_kwargs) if model != "linear" else model_cls(ndim)
        elif callable(model):
            # If user passed a callable/class, attempt to instantiate with (input_dim, **model_kwargs)
            try:
                net = model(ndim, **model_kwargs)
            except TypeError:
                # fallback: assume model() returns an nn.Module without needing input_dim
                net = model()
        else:
            raise ValueError("model must be a string key or a callable/class building an nn.Module")

        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # --- train ---
        net.train()
        tbar = tqdm(range(epochs), desc=f"Fold {fold_idx} Training", disable=not verbose)
        for epoch in tbar:
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * xb.size(0)
            # optionally verbose
            current_acc = float((logits.argmax(dim=1) == yb).float().mean())
            tbar.set_postfix(
                loss=f"{epoch_loss / len(train_ds):.4f}", acc=f"{100 * current_acc:.2f} %"
            )

        # --- eval on test set ---
        net.eval()
        all_preds = []
        all_probs = []
        all_true = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = net(xb)
                probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # prob of class 1
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                all_probs.append(probs)
                all_preds.append(preds)
                all_true.append(yb.numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_true = np.concatenate(all_true, axis=0)

        # compute fold score
        if scoring == "accuracy":
            acc = float((all_preds == all_true).mean())
            fold_scores.append(acc)
        elif scoring == "t_stat":
            # t_stat defined as MSE of predicted probability away from 0.5
            # Using mean squared deviation (MSE) across test samples
            t = float(np.mean((all_probs - 0.5) ** 2))
            fold_scores.append(t)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

    mean_score = float(np.mean(fold_scores))
    if scoring == "accuracy":
        return mean_score
    else:
        # mirror original behavior (they returned -1.0 * scores for non-accuracy)
        return -1.0 * mean_score
