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

for lr in [1e-2, 3e-2, 5e-2, 8e-2, 1e-1]:
    errs_all = []
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        model = CauchyNet(hidden_size=64, r_re=2.5, r_im=0.4, seed=s)
        result = train_score_one("lr%.2f_s%d" % (lr, s), model, train_loader, val_loader, test_data, epochs=3000, lr=lr, imag_pen=0.0, log=False, use_cosine=True, warmup_epochs=150, grad_clip=0.5)
        errs_all.extend(result["errs"])
    errs = np.array(errs_all)
    print("lr=%.3f: mean=%.5f median=%.5f max=%.5f" % (lr, errs.mean(), np.median(errs), errs.max()))
