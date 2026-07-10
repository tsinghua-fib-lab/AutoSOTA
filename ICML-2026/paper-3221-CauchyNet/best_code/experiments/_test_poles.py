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

best_mean = float("inf")
best_config = None

for r_re in [1.5, 2.0, 2.5, 3.0, 3.5]:
    for r_im in [0.2, 0.4, 0.6, 0.8]:
        errs_all = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            model = CauchyNet(hidden_size=64, r_re=r_re, r_im=r_im, seed=s)
            result = train_score_one("poles_%.1f_%.1f_s%d" % (r_re, r_im, s), model, train_loader, val_loader, test_data, epochs=3000, lr=5e-2, imag_pen=0.0, log=False, use_cosine=True, warmup_epochs=150, grad_clip=0.5)
            errs_all.extend(result["errs"])
        errs = np.array(errs_all)
        mean_val = errs.mean()
        print("(%.1f, %.1f): mean=%.5f median=%.5f max=%.5f" % (r_re, r_im, mean_val, np.median(errs), errs.max()))
        if mean_val < best_mean:
            best_mean = mean_val
            best_config = (r_re, r_im)

print("\nBest: (%.1f, %.1f) with mean=%.5f" % (best_config[0], best_config[1], best_mean))
