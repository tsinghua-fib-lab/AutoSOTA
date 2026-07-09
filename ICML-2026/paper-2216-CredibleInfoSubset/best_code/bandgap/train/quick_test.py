import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import random
from dataset.dataset import CompositionDataset
from model.model_mine import BandModelSE, combined_vae_evidential_loss_SE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(0)

full_df = pd.read_csv("./data/bandgap.csv")
hse_df = full_df[full_df.state == 0].reset_index(drop=True)
HSE_dataset = CompositionDataset(hse_df, "material formula", "Band_gap", "state")
GGA_df = full_df[full_df.state == 1].reset_index(drop=True)
GGA_dataset = CompositionDataset(GGA_df, "material formula", "Band_gap", "state")

hse_len = len(HSE_dataset)
train_hse = int(0.8 * hse_len)
val_hse = int(0.1 * hse_len)
test_hse = hse_len - train_hse - val_hse

gga_len = len(GGA_dataset)
train_gga = int(0.8 * gga_len)
val_gga = int(0.1 * gga_len)

train_HSE, val_HSE, test_HSE = random_split(HSE_dataset, [train_hse, val_hse, test_hse])
train_GGA, val_GGA, _ = random_split(GGA_dataset, [train_gga, val_gga, gga_len - train_gga - val_gga])

from torch.utils.data import ConcatDataset
train_dataset = ConcatDataset([train_HSE, train_GGA])
val_dataset = ConcatDataset([val_HSE, val_GGA])
test_loader = DataLoader(test_HSE, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test (HSE only): {len(test_HSE)}")
print(f"HSE: {hse_len} (train={train_hse}, val={val_hse}, test={test_hse})")
print(f"GGA: {gga_len} (train={train_gga}, val={val_gga})")

model = BandModelSE().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
import time
t0 = time.time()
model.train()
for batch_idx, batch in enumerate(train_loader):
    x_comp = batch["x_comp"].to(device)
    x_total = batch["x_total_feats"].to(device)
    x_state = batch["state"].to(device)
    y = batch["y_bandgap"].to(device)
    optimizer.zero_grad()
    loss, loss_dict = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y)
    loss.backward()
    optimizer.step()
t1 = time.time()
print(f"1 epoch time: {t1-t0:.2f}s ({len(train_loader)} batches)")
print("Pipeline test PASSED!")
