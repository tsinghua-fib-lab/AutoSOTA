"""Reproduction evaluation for Paper 2216: Credible Information Subset Decomposition
TF-Bandgap, 2-fidelity (GGA+HSE), seed=1024"""
import os, sys
sys.path.append("/repo/bandgap")
os.chdir("/repo/bandgap")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import random, json
from dataset.dataset import CompositionDataset
from model.model_mine import BandModelSE, combined_vae_evidential_loss_SE
from scipy.stats import kendalltau

device = torch.device("cuda")
print("Device:", device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# Settings matching paper: lr=1e-4, batch_size=128, Adam, patience=50 (extended to 100 for stability)
SEED = 1024
MAX_EPOCHS = 300
PATIENCE = 100
WARMUP_EPOCHS = 50  # CODE-01: linear warmup epochs for alpha_abs/alpha_rank
KL_WEIGHT = 1e-3  # KL weight (default; ALGO-01 showed this is near-optimal)
LR = 1e-4
BATCH_SIZE = 128

set_seed(SEED)

# Load TF-Bandgap data
full_df = pd.read_csv("./data/bandgap.csv")
hse_df = full_df[full_df.state == 0].reset_index(drop=True)
HSE_dataset = CompositionDataset(hse_df, "material formula", "Band_gap", "state")
GGA_df = full_df[full_df.state == 1].reset_index(drop=True)
GGA_dataset = CompositionDataset(GGA_df, "material formula", "Band_gap", "state")

hse_len = len(HSE_dataset)
gga_len = len(GGA_dataset)

# Split: 80/10/10 HSE train/val/test, 80/10/10 GGA train/val (GGA test unused)
train_hse = int(0.8 * hse_len)
val_hse = int(0.1 * hse_len)
test_hse = hse_len - train_hse - val_hse
train_gga = int(0.8 * gga_len)
val_gga = int(0.1 * gga_len)

train_HSE, val_HSE, test_HSE = random_split(HSE_dataset, [train_hse, val_hse, test_hse])
train_GGA, val_GGA, _ = random_split(GGA_dataset, [train_gga, val_gga, gga_len-train_gga-val_gga])

train_dataset = ConcatDataset([train_HSE, train_GGA])
val_dataset = ConcatDataset([val_HSE, val_GGA])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_HSE, batch_size=BATCH_SIZE, shuffle=False)

print("HSE: train={}, val={}, test={}".format(train_hse, val_hse, test_hse))
print("GGA: train={}, val={}".format(train_gga, val_gga))
print("Total: train={}, val={}, test={}".format(len(train_dataset), len(val_dataset), len(test_HSE)))

model = BandModelSE().to(device)
n_params = sum(p.numel() for p in model.parameters())
print("Model params: {:,}".format(n_params))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val_mae = float("inf")
patience_counter = 0
best_state = None
best_epoch = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    model.train()
    for batch in train_loader:
        x_comp = batch["x_comp"].to(device)
        x_total = batch["x_total_feats"].to(device)
        x_state = batch["state"].to(device)
        y = batch["y_bandgap"].to(device)
        optimizer.zero_grad()
        epoch_frac = min(1.0, epoch / WARMUP_EPOCHS)
        cur_alpha_abs = 0.1 + 0.9 * epoch_frac
        cur_alpha_rank = 5e-4 + 4.5e-3 * epoch_frac
        loss, _ = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y,
            kl_weight=KL_WEIGHT, alpha_abs=cur_alpha_abs, alpha_rank=cur_alpha_rank)
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            x_comp = batch["x_comp"].to(device)
            x_total = batch["x_total_feats"].to(device)
            x_state = batch["state"].to(device)
            y = batch["y_bandgap"].to(device)
            _, ld = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y, kl_weight=KL_WEIGHT)
            val_preds.append(ld["pred"].detach().cpu().ravel())
            val_trues.append(y.detach().cpu().ravel())
    val_mae = float(np.mean(np.abs(np.concatenate(val_preds) - np.concatenate(val_trues))))
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stop at epoch {}".format(epoch))
            break
    
    if epoch % 50 == 0 or epoch == 1:
        print("Epoch {:3d}: val_mae={:.4f}, best={:.4f} @ epoch {}".format(
            epoch, val_mae, best_val_mae, best_epoch))

# Final test evaluation
model.load_state_dict(best_state)
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for batch in test_loader:
        x_comp = batch["x_comp"].to(device)
        x_total = batch["x_total_feats"].to(device)
        x_state = batch["state"].to(device)
        y = batch["y_bandgap"].to(device)
        _, ld = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y, kl_weight=KL_WEIGHT)
        all_preds.append(ld["pred"].detach().cpu().ravel())
        all_trues.append(y.detach().cpu().ravel())

all_preds = np.concatenate(all_preds)
all_trues = np.concatenate(all_trues)

mae = float(np.mean(np.abs(all_preds - all_trues)))
rmse = float(np.sqrt(np.mean((all_preds - all_trues)**2)))
tau, _ = kendalltau(all_trues, all_preds)
tau = float(tau)

print("\n" + "=" * 60)
print("REPRODUCTION RESULTS - Paper 2216")
print("=" * 60)
print("Dataset: TF-Bandgap (2-fidelity GGA+HSE)")
print("Seed: {}".format(SEED))
print("Best epoch: {}".format(best_epoch))
print("MAE:   {:.6f}  (paper: 0.57 +/- 0.012)".format(mae))
print("RMSE:  {:.6f}  (paper: 0.78 +/- 0.018)".format(rmse))
print("tau_b: {:.6f}  (paper: 0.68 +/- 0.027)".format(tau))
print("=" * 60)

results = {
    "paper_id": 2216,
    "dataset": "TF-Bandgap",
    "fidelity": "GGA+HSE",
    "seed": SEED,
    "best_epoch": best_epoch,
    "MAE": mae,
    "RMSE": rmse,
    "tau_b": tau,
}
print(json.dumps(results, indent=2))

# Save
with open("./reproduction_result.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to ./reproduction_result.json")
