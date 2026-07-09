import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Subset
import random
import torch.nn.functional as F
from dataset.dataset import CompositionDataset
from model.model_mine import BandModelSE, combined_vae_evidential_loss_SE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from scipy.stats import kendalltau

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(1023)

def train_one_epoch(model, loader, optimizer):
    model.train()
    epoch_loss = 0
    epoch_mae = 0
    epoch_loss_rank = 0
    for batch in loader:
        x_comp = batch["x_comp"].to(device)
        x_total = batch["x_total_feats"].to(device)
        x_state = batch["state"].to(device)
        y = batch["y_bandgap"].to(device)
        optimizer.zero_grad()
        loss, loss_dict = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_loss_rank += loss_dict["rank_loss"]
        epoch_mae  += loss_dict["mae"]
    avg_loss = epoch_loss / len(loader)
    avg_mae  = epoch_mae  / len(loader)
    avg_rank_loss = epoch_loss_rank/ len(loader)
    return avg_loss, avg_mae, avg_rank_loss

def eval_one_epoch(model, loader):
    model.eval()
    epoch_loss = 0
    epoch_mae = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in loader:
            x_comp = batch["x_comp"].to(device)
            x_total = batch["x_total_feats"].to(device)
            x_state = batch["state"].to(device)
            y = batch["y_bandgap"].to(device)
            loss, loss_dict = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y)
            epoch_loss += loss.item()
            epoch_mae  += loss_dict["mae"]
            pred = loss_dict["pred"]
            all_preds.append(pred.detach().cpu().ravel())
            all_trues.append(y.detach().cpu().ravel())
    avg_loss = epoch_loss / len(loader)
    avg_mae  = epoch_mae  / len(loader)
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    return avg_loss, avg_mae, all_preds, all_trues

def train_model(model=None, epochs=200, train_loader=None, val_loader=None,
                lr=1e-3, test_loader=None, save_path="./pt/best_model.pt", patience=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_mae = float("inf")
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_mae, avg_rank_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_mae, all_preds, all_trues = eval_one_epoch(model, val_loader)
        tau, _ = kendalltau(all_trues, all_preds)
        if epoch % 50 == 0 or epoch == 1:
            print("Epoch {:3d}: Train MAE={:.4f} | Val MAE={:.4f} | Val Tau={:.4f}".format(
                epoch, train_mae, val_mae, tau), flush=True)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered (patience={})".format(patience))
                break
        if test_loader is not None and epoch % 50 == 0:
            test_model(model, test_loader)
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

def test_model(model, test_loader):
    model.eval()
    test_loss, test_mae, all_preds, all_trues = eval_one_epoch(model, test_loader)
    tau, _ = kendalltau(all_trues, all_preds)
    mae = float(np.mean(np.abs(all_preds - all_trues)))
    rmse = float(np.sqrt(np.mean((all_preds - all_trues)**2)))
    print("TEST: MAE={:.6f} RMSE={:.6f} tau_b={:.6f}".format(mae, rmse, float(tau)), flush=True)
    return mae, rmse, float(tau)

# Run exactly like the original methodmine.py
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
train_GGA, val_GGA, test_GGA = random_split(GGA_dataset, [train_gga, val_gga, gga_len - train_gga - val_gga])

from torch.utils.data import ConcatDataset
train_dataset = ConcatDataset([train_HSE, train_GGA])
val_dataset = ConcatDataset([val_HSE, val_GGA])
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_HSE, batch_size=batch_size, shuffle=False)

print("HSE: train={}, val={}, test={}".format(train_hse, val_hse, test_hse))
print("GGA: train={}, val={}".format(train_gga, val_gga))
print("Train total={}, Val total={}, Test={}".format(len(train_dataset), len(val_dataset), len(test_HSE)))

model = BandModelSE().to(device)
best_model = train_model(model, epochs=200, train_loader=train_loader,
                         val_loader=val_loader, lr=1e-4,
                         test_loader=test_loader, save_path="./pt/best_model_orig.pt", patience=100)
mae, rmse, tau = test_model(best_model, test_loader)
print("\nFinal: MAE={:.6f} RMSE={:.6f} tau_b={:.6f}".format(mae, rmse, tau))
