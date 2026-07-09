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

def evaluate(model, loader):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            x_comp = batch["x_comp"].to(device)
            x_total = batch["x_total_feats"].to(device)
            x_state = batch["state"].to(device)
            y = batch["y_bandgap"].to(device)
            _, ld = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y)
            all_preds.append(ld["pred"].detach().cpu().ravel())
            all_trues.append(y.detach().cpu().ravel())
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    mae = float(np.mean(np.abs(all_preds - all_trues)))
    rmse = float(np.sqrt(np.mean((all_preds - all_trues)**2)))
    tau, _ = kendalltau(all_trues, all_preds)
    return mae, rmse, float(tau), all_preds, all_trues

# Load data
full_df = pd.read_csv("./data/bandgap.csv")
hse_df = full_df[full_df.state == 0].reset_index(drop=True)
HSE_dataset = CompositionDataset(hse_df, "material formula", "Band_gap", "state")
GGA_df = full_df[full_df.state == 1].reset_index(drop=True)
GGA_dataset = CompositionDataset(GGA_df, "material formula", "Band_gap", "state")

hse_len = len(HSE_dataset)
gga_len = len(GGA_dataset)

seeds = [0, 1, 64, 1023, 1024]
results = []

for seed in seeds:
    print("\n" + "="*60)
    print("Seed:", seed)
    set_seed(seed)
    
    train_hse = int(0.8 * hse_len)
    val_hse = int(0.1 * hse_len)
    test_hse = hse_len - train_hse - val_hse
    train_gga = int(0.8 * gga_len)
    val_gga = int(0.1 * gga_len)
    
    train_HSE, val_HSE, test_HSE = random_split(HSE_dataset, [train_hse, val_hse, test_hse])
    train_GGA, val_GGA, _ = random_split(GGA_dataset, [train_gga, val_gga, gga_len-train_gga-val_gga])
    
    train_dataset = ConcatDataset([train_HSE, train_GGA])
    val_dataset = ConcatDataset([val_HSE, val_GGA])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_HSE, batch_size=128, shuffle=False)
    
    model = BandModelSE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_mae = float("inf")
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, 301):
        model.train()
        for batch in train_loader:
            x_comp = batch["x_comp"].to(device)
            x_total = batch["x_total_feats"].to(device)
            x_state = batch["state"].to(device)
            y = batch["y_bandgap"].to(device)
            optimizer.zero_grad()
            loss, _ = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_comp = batch["x_comp"].to(device)
                x_total = batch["x_total_feats"].to(device)
                x_state = batch["state"].to(device)
                y = batch["y_bandgap"].to(device)
                _, ld = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y)
                val_preds.append(ld["pred"].detach().cpu().ravel())
                val_trues.append(y.detach().cpu().ravel())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_mae = float(np.mean(np.abs(val_preds - val_trues)))
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= 100:
                print("  Early stop at epoch", epoch)
                break
    
    model.load_state_dict(best_state)
    mae, rmse, tau, preds, trues = evaluate(model, test_loader)
    print("Seed {}: best_epoch={}, MAE={:.6f}, RMSE={:.6f}, tau_b={:.6f}".format(
        seed, best_epoch, mae, rmse, tau))
    results.append({"MAE": mae, "RMSE": rmse, "tau_b": tau, "seed": seed})

maes = [r["MAE"] for r in results]
rmses = [r["RMSE"] for r in results]
taus = [r["tau_b"] for r in results]
print("\n" + "="*60)
print("FINAL (300 epochs, patience=100)")
print("MAE:   {:.4f} +/- {:.4f}".format(np.mean(maes), np.std(maes, ddof=1)))
print("RMSE:  {:.4f} +/- {:.4f}".format(np.mean(rmses), np.std(rmses, ddof=1)))
print("tau_b: {:.4f} +/- {:.4f}".format(np.mean(taus), np.std(taus, ddof=1)))
print(json.dumps(dict(MAE_mean=float(np.mean(maes)), MAE_std=float(np.std(maes, ddof=1)),
                      RMSE_mean=float(np.mean(rmses)), RMSE_std=float(np.std(rmses, ddof=1)),
                      tau_b_mean=float(np.mean(taus)), tau_b_std=float(np.std(taus, ddof=1))), indent=2))
