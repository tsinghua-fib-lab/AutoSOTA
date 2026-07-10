import json, math, time, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

with open(os.path.join(os.path.dirname(__file__), "best_config_gap_filling.py")) as f:
    code = f.read()
main_idx = code.find("\ndef main()")
exec(code[:main_idx])

torch.manual_seed(10)
np.random.seed(10)

tX, tY, vX, vY, teX, teY = build_data(seed=10)
train_loader = DataLoader(TensorDataset(tX, tY), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(vX, vY), batch_size=32, shuffle=False)
test_data = (teX, teY)

t_start = time.time()
model = CauchyNet(hidden_size=64, r_re=2.5, r_im=0.4, seed=0)
result = train_score_one("CauchyNet", model, train_loader, val_loader, test_data, epochs=3000, lr=5e-2, imag_pen=0.0, log=True)
t_elapsed = time.time() - t_start
print("1 seed MAE mean=%.5f, median=%.5f, max=%.5f, time=%.1fs" % (result["mae_mean"], result["mae_median"], result["mae_max"], t_elapsed))
