"""
Reproduction evaluation script for paper "A Strictly Proper Scoring Rule and
a Calibration Metric for Interval-Censored Data Analysis" (ICML 2026).

Evaluates NN-IC-Log model on breastCancer dataset from icensBKL package
using 5-fold cross-validation with SIC-Log and IC-Cal metrics.

Usage: python3 eval_breastCancer.py
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import rdata
import os
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from cenreg.pytorch.loss_cdf import NegativeLogLikelihoodInterval
from cenreg.distribution.cdf import CumulativeDist
import cenreg.metric.cdf as metric_cdf
import cenreg.metric.quantile as metric_quantile


class MLP2Layer(nn.Module):
    """MLP with 2 hidden layers (3-layer MLP) + dropout 0.5 + ReLU + Softmax."""

    def __init__(self, input_len, output_len, num_neuron):
        super().__init__()
        self.fc1 = nn.Linear(input_len, num_neuron)
        self.fc2 = nn.Linear(num_neuron, num_neuron)
        self.fc3 = nn.Linear(num_neuron, output_len)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def load_breast_cancer_data():
    """Load and preprocess the breastCancer dataset from icensBKL."""
    rda_path = "/autosota_cache/tmp/icensBKL/icensBKL/data/breastCancer.rda"
    parsed = rdata.read_rda(rda_path)
    df = parsed["breastCancer"]

    lb = df["low"].values.astype(float)
    ub = df["upp"].values.astype(float)
    lb = np.where(np.isnan(lb), 0.0, lb)
    ub = np.where(np.isnan(ub), np.inf, ub)

    treat = df["treat"].values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    X = encoder.fit_transform(treat).astype(np.float32)

    return X, lb, ub


def build_bins(lb_train, ub_train, num_bins=None):
    """Build Turnbull-adaptive bin boundaries from training interval data.
    
    Collects all unique L/R endpoints (Turnbull intervals), then optionally
    reduces to `num_bins` by merging adjacent smallest intervals.
    """
    # Collect all unique finite endpoints (Turnbull approach)
    all_points = np.unique(np.concatenate([
        lb_train,
        ub_train[ub_train != np.inf]
    ]))
    if all_points[0] > 0.0:
        all_points = np.append([0.0], all_points)
    
    if num_bins is None or len(all_points) - 1 <= num_bins:
        return all_points
    
    # Need to merge: compute interval widths, merge smallest
    intervals = np.diff(all_points)
    # Keep merging smallest intervals until we have num_bins
    n_to_merge = len(all_points) - 1 - num_bins
    if n_to_merge <= 0:
        return all_points
    
    # Merge smallest adjacent intervals
    merged = list(all_points)
    for _ in range(n_to_merge):
        widths = np.diff(merged)
        # Find smallest interval (not including the last one edge case)
        min_idx = np.argmin(widths)
        # Remove the boundary between min_idx and min_idx+1
        merged.pop(min_idx + 1)
    
    return np.array(merged)


def train_one_fold(X_train, lb_train, ub_train, y_bins, device, seed):
    """Train NN-IC-Log model for one fold with CDSC early stopping."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Split training data into train/val for CDSC
    n_total = len(X_train)
    n_val = max(1, int(n_total * CDSC_VAL_SPLIT))
    n_train = n_total - n_val
    indices = np.random.RandomState(seed).permutation(n_total)
    train_idx_cdsc = indices[:n_train]
    val_idx_cdsc = indices[n_train:]

    X_tr = X_train[train_idx_cdsc]
    lb_tr = lb_train[train_idx_cdsc]
    ub_tr = ub_train[train_idx_cdsc]
    X_val = X_train[val_idx_cdsc]
    lb_val = lb_train[val_idx_cdsc]
    ub_val = ub_train[val_idx_cdsc]

    num_bins = len(y_bins) - 1
    model = MLP2Layer(X_train.shape[1], num_bins, HIDDEN_SIZE).to(device)
    loss_fn = NegativeLogLikelihoodInterval(
        torch.tensor(y_bins, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(lb_tr, dtype=torch.float32),
        torch.tensor(ub_tr, dtype=torch.float32))
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_lb = torch.tensor(lb_val, dtype=torch.float32).to(device)
    val_tensor_ub = torch.tensor(ub_val, dtype=torch.float32).to(device)

    best_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(N_EPOCHS):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for x_batch, lb_batch, ub_batch in train_loader:
            x_batch = x_batch.to(device)
            lb_batch = lb_batch.to(device)
            ub_batch = ub_batch.to(device)
            optimizer.zero_grad()
            pred_batch = model(x_batch)
            loss = loss_fn.loss(pred_batch, lb_batch, ub_batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        avg_train_loss = loss_sum / n_batches if n_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(val_tensor_x)
            val_loss = loss_fn.loss(val_pred, val_tensor_lb, val_tensor_ub).mean().item()
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # CDSC: correlation-based early stopping
        if epoch >= CDSC_MIN_EPOCHS and epoch >= CDSC_WINDOW:
            recent_train = train_losses[-CDSC_WINDOW:]
            recent_val = val_losses[-CDSC_WINDOW:]
            if np.std(recent_train) > 1e-8 and np.std(recent_val) > 1e-8:
                corr, _ = pearsonr(recent_train, recent_val)
                if corr < CDSC_THRESHOLD:
                    break

        # Fallback: patience-based stopping
        if epoch - best_epoch > 1000:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_fold(models, X_test, lb_test, ub_test, y_bins, device):
    """Evaluate ensemble of trained models on test fold."""
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t)
        preds.append(pred.detach().cpu().numpy())
    # Average softmax outputs across ensemble
    pred_np = np.mean(preds, axis=0)
    cum_pred_np = np.cumsum(pred_np, axis=1)
    dist = CumulativeDist(y_bins, cum_p=cum_pred_np)

    ic_nll = metric_cdf.negative_log_likelihood_interval(dist, lb_test, ub_test)
    sic_log = ic_nll.mean()

    ic_cal = metric_quantile.ic_calibration(dist, lb_test, ub_test, p=2.0)

    return float(sic_log), float(ic_cal)


# =============================================================================
# Configuration (matching rubric specification)
# =============================================================================
SEED = 0
N_FOLDS = 5
N_EPOCHS = 5000
BATCH_SIZE = 128
HIDDEN_SIZE = 32
LEARNING_RATE = 0.1
MAX_GRAD_NORM = 1.0
ENSEMBLE_SEEDS = [0]
NUM_BINS = 10
CDSC_WINDOW = 10
CDSC_THRESHOLD = -0.2
CDSC_MIN_EPOCHS = 100
CDSC_VAL_SPLIT = 0.2
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Fix seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
# Main evaluation
# =============================================================================
print("=" * 70)
print("Reproduction: NN-IC-Log on breastCancer (icensBKL)")
print("Model: 3-layer MLP, hidden_size=32, dropout=0.5, ReLU, Adam lr=0.01")
print("Loss: SIC-Log (Negative Log-Likelihood Interval)")
print("=" * 70)

X, lb, ub = load_breast_cancer_data()
print("Dataset: {} samples, {} feature(s)".format(X.shape[0], X.shape[1]))
print("Interval types: left-cens={}, right-cens={}, strict-interval={}".format(
    np.sum(lb == 0), np.sum(ub == np.inf), np.sum((lb > 0) & (ub < np.inf))))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

sic_log_results = []
ic_cal_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    lb_train, lb_test = lb[train_idx], lb[test_idx]
    ub_train, ub_test = ub[train_idx], ub[test_idx]

    y_bins = build_bins(lb_train, ub_train, NUM_BINS)
    print("\nFold {}/{}: {} bins, ensemble={} seeds".format(fold + 1, N_FOLDS, len(y_bins) - 1, len(ENSEMBLE_SEEDS)))

    # Train ensemble of models with different seeds
    fold_models = []
    for es in ENSEMBLE_SEEDS:
        ensemble_seed = SEED + fold * 100 + es
        model = train_one_fold(X_train, lb_train, ub_train, y_bins, DEVICE, ensemble_seed)
        fold_models.append(model)
    sic_log, ic_cal = evaluate_fold(fold_models, X_test, lb_test, ub_test, y_bins, DEVICE)

    sic_log_results.append(sic_log)
    ic_cal_results.append(ic_cal)
    print("  SIC-Log = {:.4f}, IC-Cal = {:.6f}".format(sic_log, ic_cal))

# =============================================================================
# Results
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
sic_log_mean = np.mean(sic_log_results)
sic_log_std = np.std(sic_log_results)
ic_cal_mean = np.mean(ic_cal_results)
ic_cal_std = np.std(ic_cal_results)

print("SIC-Log: {:.4f} +/- {:.4f}".format(sic_log_mean, sic_log_std))
print("  Per-fold: {}".format([round(x, 4) for x in sic_log_results]))
print("IC-Cal:  {:.6f} +/- {:.6f}".format(ic_cal_mean, ic_cal_std))
print("  Per-fold: {}".format([round(x, 6) for x in ic_cal_results]))

print("\nPaper values:")
print("  SIC-Log: 1.5343,  CI=[1.3771, 1.6914]")
print("  IC-Cal:  0.008892, CI=[0.003479, 0.014304]")

sic_ok = 1.3771 <= sic_log_mean <= 1.6914
ic_ok = 0.003479 <= ic_cal_mean <= 0.014304
print("\nSIC-Log within CI: {}".format(sic_ok))
print("IC-Cal within CI:  {}".format(ic_ok))
