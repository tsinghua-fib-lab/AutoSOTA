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

# Test different gradient clipping thresholds
for max_norm in [0.5, 1.0, 5.0, 10.0, None]:
    errs_all = []
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        model = CauchyNet(hidden_size=64, r_re=2.5, r_im=0.4, seed=s)
        # Need to modify train_score_one to accept max_norm
        # For now, manually test by modifying the function

# Actually, let me just create a custom training function
def train_with_clip(model, train_loader, val_loader, test_data, epochs=3000, lr=5e-2, max_norm=1.0):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=150)
    cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs-150, eta_min=lr*1e-3)
    sch = optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[150])
    crit = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train(True)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                yr, yi = out
                loss = crit(yr, yb)
            else:
                loss = crit(out, yb)
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            opt.step()
        sch.step()
        model.train(False)
        with torch.no_grad():
            vl = 0.0
            n = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                if isinstance(out, tuple): out = out[0]
                vl += crit(out, yb).item(); n += 1
            vl /= n
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)
    test_X, test_Y = test_data
    test_X, test_Y = test_X.to(device), test_Y.to(device)
    with torch.no_grad():
        out = model(test_X)
        if isinstance(out, tuple): out = out[0]
        preds = out.cpu().numpy().flatten()
    targs = test_Y.cpu().numpy().flatten()
    err = np.abs(preds - targs)
    return err

for max_norm in [0.5, 1.0, 5.0, 10.0, None]:
    errs_all = []
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        model = CauchyNet(hidden_size=64, r_re=2.5, r_im=0.4, seed=s)
        err = train_with_clip(model, train_loader, val_loader, test_data, max_norm=max_norm)
        errs_all.extend(err.tolist())
    errs = np.array(errs_all)
    label = "None" if max_norm is None else str(max_norm)
    print("max_norm=%5s: mean=%.5f median=%.5f max=%.5f" % (label, errs.mean(), np.median(errs), errs.max()))
