import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import rdata
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from cenreg.pytorch.mlp import MLP
from cenreg.pytorch.loss_cdf import NegativeLogLikelihoodInterval
from cenreg.distribution.cdf import CumulativeDist
import cenreg.metric.cdf as metric_cdf
import cenreg.metric.quantile as metric_quantile


# ===========================================================================
# Config
# ===========================================================================
SEED = 0
N_FOLDS = 5
N_EPOCHS = 3000
BATCH_SIZE = 128
HIDDEN_SIZE = 32
LEARNING_RATE = 0.01
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEP = "=" * 60

print("Device:", DEVICE)
print("Config: hidden_size={}, lr={}, epochs={}, folds={}, seed={}".format(
    HIDDEN_SIZE, LEARNING_RATE, N_EPOCHS, N_FOLDS, SEED))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ===========================================================================
# Load icensBKL breastCancer data
# ===========================================================================
rda_path = "/autosota_cache/tmp/icensBKL/icensBKL/data/breastCancer.rda"
parsed = rdata.read_rda(rda_path)
df = parsed["breastCancer"]
print("Loaded breastCancer: {} samples".format(df.shape[0]))

# Process interval bounds
lb = df["low"].values.astype(float)
ub = df["upp"].values.astype(float)
lb = np.where(np.isnan(lb), 0.0, lb)
ub = np.where(np.isnan(ub), np.inf, ub)

print("Data: Left-cens={}, Right-cens={}, Interval={}".format(
    np.sum(lb == 0), np.sum(ub == np.inf), np.sum((lb > 0) & (ub < np.inf))))

# One-hot encode treatment
treat = df["treat"].values.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False, drop="first")
X = encoder.fit_transform(treat).astype(np.float32)
print("Features: {} dims".format(X.shape[1]))


# ===========================================================================
# 5-Fold Cross-Validation
# ===========================================================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

sic_log_scores = []
ic_cal_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print("\n" + SEP)
    print("Fold {}/{}".format(fold + 1, N_FOLDS))
    print(SEP)

    X_train, X_test = X[train_idx], X[test_idx]
    lb_train, lb_test = lb[train_idx], lb[test_idx]
    ub_train, ub_test = ub[train_idx], ub[test_idx]

    print("Train: {}, Test: {}".format(len(X_train), len(X_test)))

    # Determine bin boundaries from training data
    finite_ub_train = ub_train[np.isfinite(ub_train)]
    all_vals = np.concatenate([lb_train, finite_ub_train]) if len(finite_ub_train) > 0 else lb_train.copy()
    y_bins = np.unique(all_vals)
    if y_bins[0] > 0.0:
        y_bins = np.append([0.0], y_bins)
    max_val = np.max(finite_ub_train) if len(finite_ub_train) > 0 else y_bins[-1]
    y_bins = np.append(y_bins, max_val + 1.0)
    num_bins = len(y_bins) - 1
    print("Bins: {}".format(num_bins))

    # Simple dataloader without multiprocessing
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    lb_train_t = torch.tensor(lb_train, dtype=torch.float32)
    ub_train_t = torch.tensor(ub_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_t, lb_train_t, ub_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Model
    model = MLP(X_train.shape[1], num_bins, HIDDEN_SIZE).to(DEVICE)
    loss_fn = NegativeLogLikelihoodInterval(torch.tensor(y_bins, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training with early stopping on training loss
    best_loss = float("inf")
    best_epoch = 0
    patience = 500
    for epoch in range(N_EPOCHS):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for x_batch, lb_batch, ub_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            lb_batch = lb_batch.to(DEVICE)
            ub_batch = ub_batch.to(DEVICE)
            optimizer.zero_grad()
            pred_batch = model(x_batch)
            loss = loss_fn.loss(pred_batch, lb_batch, ub_batch).mean()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        avg_loss = loss_sum / n_batches if n_batches > 0 else 0.0
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print("  Epoch {}/{}, Loss: {:.4f}, Best: {:.4f} @ epoch {}".format(
                epoch + 1, N_EPOCHS, avg_loss, best_loss, best_epoch + 1))
        
        # Early stopping
        if epoch - best_epoch > patience:
            print("  Early stopping at epoch {}".format(epoch + 1))
            break

    print("  Final: Loss={:.4f}, Best={:.4f} @ epoch {}".format(avg_loss, best_loss, best_epoch + 1))

    # Evaluation
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = model(X_test_t)

    pred_np = pred.detach().cpu().numpy()
    cum_pred_np = np.cumsum(pred_np, axis=1)
    dist = CumulativeDist(y_bins, cum_p=cum_pred_np)

    # SIC-Log
    ic_nll = metric_cdf.negative_log_likelihood_interval(dist, lb_test, ub_test)
    sic_log = ic_nll.mean()
    sic_log_scores.append(sic_log)
    print("  SIC-Log: {:.6f}".format(sic_log))

    # IC-Cal with p=2
    ic_cal = metric_quantile.ic_calibration(dist, lb_test, ub_test, p=2.0)
    ic_cal_scores.append(ic_cal)
    print("  IC-Cal:  {:.6f}".format(ic_cal))


# ===========================================================================
# Final Results
# ===========================================================================
print("\n" + SEP)
print("FINAL RESULTS (5-fold CV on breastCancer)")
print(SEP)
sic_log_mean = np.mean(sic_log_scores)
sic_log_std = np.std(sic_log_scores)
ic_cal_mean = np.mean(ic_cal_scores)
ic_cal_std = np.std(ic_cal_scores)

print("SIC-Log: {:.4f} +/- {:.4f}".format(sic_log_mean, sic_log_std))
print("IC-Cal:  {:.6f} +/- {:.6f}".format(ic_cal_mean, ic_cal_std))
print("Per-fold SIC-Log: {}".format([round(float(x), 4) for x in sic_log_scores]))
print("Per-fold IC-Cal:  {}".format([round(float(x), 6) for x in ic_cal_scores]))

print("\n" + SEP)
print("COMPARISON WITH RUBRIC")
print(SEP)
print("SIC-Log: obtained={:.4f}, paper=1.5343, CI=[1.3771, 1.6914]".format(sic_log_mean))
print("IC-Cal:  obtained={:.6f}, paper=0.008892, CI=[0.003479, 0.014304]".format(ic_cal_mean))

sic_in_ci = 1.3771 <= sic_log_mean <= 1.6914
ic_cal_in_ci = 0.003479 <= ic_cal_mean <= 0.014304
print("SIC-Log in CI: {}".format(sic_in_ci))
print("IC-Cal in CI:  {}".format(ic_cal_in_ci))
