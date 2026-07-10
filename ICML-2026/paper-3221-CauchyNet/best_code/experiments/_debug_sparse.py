import json, math, time, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

with open("best_config_gap_filling.py") as f:
    code = f.read()
main_idx = code.find("\ndef main()")
exec(code[:main_idx])

torch.manual_seed(10)
np.random.seed(10)

tX, tY, vX, vY, teX, teY = build_data(seed=10)
train_loader = DataLoader(TensorDataset(tX, tY), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(vX, vY), batch_size=32, shuffle=False)
test_data = (teX, teY)

# Test with very large sparse_penalty to verify it works
model = CauchyNet(hidden_size=64, r_re=2.5, r_im=0.4, seed=0)
model = model.to(device)
opt = optim.Adam(model.parameters(), lr=5e-2)
crit = nn.MSELoss()

# One forward pass to set _last_activated
xb, yb = next(iter(train_loader))
xb, yb = xb.to(device), yb.to(device)
out = model(xb)
print("has _last_activated:", hasattr(model, "_last_activated"))
print("_last_activated shape:", model._last_activated.shape)
print("_last_activated real norm:", torch.norm(model._last_activated.real, p=1).item())

yr, yi = out
loss_mse = crit(yr, yb).item()
loss_total = crit(yr, yb)
sp = 0.01
loss_total = loss_total + sp * torch.norm(model._last_activated.real, p=1)
print("MSE loss:", loss_mse)
print("Sparse term:", (sp * torch.norm(model._last_activated.real, p=1)).item())
print("Total loss:", loss_total.item())
