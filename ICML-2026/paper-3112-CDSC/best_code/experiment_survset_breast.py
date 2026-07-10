import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from SurvSet.data import SurvLoader
from cenreg.pytorch.loss_cdf import NegativeLogLikelihoodInterval
from cenreg.distribution.cdf import CumulativeDist
import cenreg.metric.cdf as metric_cdf
import cenreg.metric.quantile as metric_quantile


class MLP2Layer(nn.Module):
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


SEED = 0
N_FOLDS = 5
N_EPOCHS = 5000
BATCH_SIZE = 128
HIDDEN_SIZE = 32
LEARNING_RATE = 0.01
DEVICE = "cuda:0"
SEP = "=" * 60

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Load SurvSet breast dataset
loader = SurvLoader()
df, _ = loader.load_dataset(ds_name="breast").values()
print("Loaded SurvSet breast: {} samples".format(df.shape[0]))

# Encode features (following notebook)
cols_cont = [col for col in df.columns if col.startswith("num_")]
if len(cols_cont) > 0:
    df[cols_cont] = df[cols_cont].fillna(df[cols_cont].median())
    mm = MinMaxScaler()
    df[cols_cont] = mm.fit_transform(df[cols_cont])

cols_cat = [col for col in df.columns if col.startswith("fac_")]
df = pd.get_dummies(df, columns=cols_cat, drop_first=True, dtype="float32")

# Create interval-censored format (following notebook pattern exactly)
mask_exact = df["event"] == 1
lb = df["time"].values.copy().astype(float)
ub = df["time"].values.copy().astype(float)
lb[mask_exact] -= 0.00001
ub[~mask_exact] = np.inf

feature_cols = [c for c in df.columns if c not in ["pid", "time", "event"]]
X = df[feature_cols].values.astype(np.float32)
print("Features: {} dims".format(X.shape[1]))
print("Exact events: {}, Right-censored: {}".format(mask_exact.sum(), (~mask_exact).sum()))

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

    # Bin boundaries - following notebook pattern
    y_bins = np.unique(np.concatenate([lb_train, ub_train[ub_train != np.inf]]))
    if y_bins[0] == -np.inf:
        y_bins = y_bins[1:]
    if y_bins[0] > 0.0:
        y_bins = np.append([0.0], y_bins)
    if y_bins[-1] == np.inf:
        y_bins = y_bins[:-1]
    num_bins = len(y_bins) - 1
    print("Bins: {}".format(num_bins))

    # Verify bins are sorted and unique
    assert np.all(np.diff(y_bins) > 0), "Bins not strictly increasing!"

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(lb_train, dtype=torch.float32),
        torch.tensor(ub_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = MLP2Layer(X_train.shape[1], num_bins, HIDDEN_SIZE).to(DEVICE)
    loss_fn = NegativeLogLikelihoodInterval(torch.tensor(y_bins, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float("inf")
    best_epoch = 0
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
            print("  Epoch {}, Loss: {:.4f}, Best: {:.4f} @ {}".format(epoch+1, avg_loss, best_loss, best_epoch+1))
        if epoch - best_epoch > 1000:
            print("  Early stop @ epoch {}".format(epoch+1))
            break

    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = model(X_test_t)
    pred_np = pred.detach().cpu().numpy()
    cum_pred_np = np.cumsum(pred_np, axis=1)
    dist = CumulativeDist(y_bins, cum_p=cum_pred_np)

    ic_nll = metric_cdf.negative_log_likelihood_interval(dist, lb_test, ub_test)
    sic_log = ic_nll.mean()
    sic_log_scores.append(sic_log)
    ic_cal = metric_quantile.ic_calibration(dist, lb_test, ub_test, p=2.0)
    ic_cal_scores.append(ic_cal)
    print("  SIC-Log: {:.4f}, IC-Cal: {:.6f}".format(sic_log, ic_cal))

print("\n" + SEP)
print("FINAL RESULTS (SurvSet breast, hidden_layers=2, lr=0.01)")
print(SEP)
sic_log_mean = np.mean(sic_log_scores)
sic_log_std = np.std(sic_log_scores)
ic_cal_mean = np.mean(ic_cal_scores)
ic_cal_std = np.std(ic_cal_scores)
print("SIC-Log: {:.4f} +/- {:.4f}".format(sic_log_mean, sic_log_std))
print("IC-Cal:  {:.6f} +/- {:.6f}".format(ic_cal_mean, ic_cal_std))
print("Paper SIC-Log: 1.5343, CI=[1.3771, 1.6914]")
print("Paper IC-Cal:  0.008892, CI=[0.003479, 0.014304]")
print("SIC-Log in CI: {}".format(1.3771 <= sic_log_mean <= 1.6914))
print("IC-Cal in CI:  {}".format(0.003479 <= ic_cal_mean <= 0.014304))
