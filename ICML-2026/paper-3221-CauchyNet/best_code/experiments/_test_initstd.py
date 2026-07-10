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

class CauchyNetTunable(CauchyNet):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, r_re=2.5, r_im=0.4, seed=0, init_std=0.1):
        super().__init__(input_size, hidden_size, output_size, r_re, r_im, seed)
        # Override lambda_ init
        self.lambda_ = nn.Parameter(
            torch.normal(mean=0.0, std=init_std, size=(hidden_size, output_size), dtype=torch.cfloat))

# We need to also change the normalization in forward
# Actually, let me just directly test with different init_std values
# by modifying the model after creation

for init_std in [0.05, 0.1, 0.15, 0.2, 0.3]:
    errs_all = []
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        model = CauchyNetTunable(hidden_size=64, r_re=2.5, r_im=0.4, seed=s, init_std=init_std)
        result = train_score_one("is%.2f_s%d" % (init_std, s), model, train_loader, val_loader, test_data, epochs=3000, lr=5e-2, imag_pen=0.0, log=False, use_cosine=True, warmup_epochs=150, grad_clip=0.5)
        errs_all.extend(result["errs"])
    errs = np.array(errs_all)
    print("init_std=%.2f: mean=%.5f median=%.5f max=%.5f" % (init_std, errs.mean(), np.median(errs), errs.max()))
