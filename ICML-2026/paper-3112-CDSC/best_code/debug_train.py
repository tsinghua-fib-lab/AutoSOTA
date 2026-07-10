import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from cenreg.pytorch.loss_cdf import NegativeLogLikelihoodInterval
from cenreg.pytorch.mlp import MLP
import rdata
from sklearn.preprocessing import OneHotEncoder

# Load data
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

# Bin boundaries
finite_ub = ub[np.isfinite(ub)]
all_vals = np.concatenate([lb, finite_ub])
y_bins = np.unique(all_vals)
if y_bins[0] > 0.0:
    y_bins = np.append([0.0], y_bins)
y_bins = np.append(y_bins, np.max(finite_ub) + 1.0)

print("y_bins[:10]:", y_bins[:10])
print("num bins:", len(y_bins) - 1)
print("Finite ub range:", finite_ub.min(), "-", finite_ub.max())

X_t = torch.tensor(X, dtype=torch.float32)
lb_t = torch.tensor(lb, dtype=torch.float32)
ub_t = torch.tensor(ub, dtype=torch.float32)
dataset = TensorDataset(X_t, lb_t, ub_t)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

device = "cuda:0"
y_bins_t = torch.tensor(y_bins, dtype=torch.float32).to(device)

# Try with different learning rates
for lr in [0.001, 0.01, 0.1, 1.0]:
    torch.manual_seed(42)
    model = MLP(1, len(y_bins)-1, 32).to(device)
    loss_fn = NegativeLogLikelihoodInterval(y_bins_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(50):
        model.train()
        for x_batch, lb_batch, ub_batch in loader:
            x_batch = x_batch.to(device)
            lb_batch = lb_batch.to(device)
            ub_batch = ub_batch.to(device)
            optimizer.zero_grad()
            pred_batch = model(x_batch)
            loss = loss_fn.loss(pred_batch, lb_batch, ub_batch).mean()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    
    print("lr={:.3f}: epoch1={:.4f}, epoch10={:.4f}, epoch50={:.4f}, min={:.4f}, max={:.4f}".format(
        lr, losses[0], losses[9], losses[-1], min(losses), max(losses)))
