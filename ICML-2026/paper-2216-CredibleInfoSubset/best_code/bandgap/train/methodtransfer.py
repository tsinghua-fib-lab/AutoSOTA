import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Subset
import random
import torch.nn.functional as F
from dataset.dataset import CompositionDataset
from model.model import BandModel, combined_vae_evidential_loss
from model.model_mfse import BandModelSE, combined_vae_evidential_loss_SE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(1023)


def train_one_epoch(model, loader, optimizer):
    model.train()
    epoch_loss = 0
    epoch_mae = 0

    for batch in loader:
        x_comp = batch["x_comp"].to(device)
        x_total = batch["x_total_feats"].to(device)
        y = batch["y_bandgap"].to(device)
        optimizer.zero_grad()
        loss, loss_dict = combined_vae_evidential_loss(model, x_comp, x_total, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_mae  += loss_dict["mae"]

    avg_loss = epoch_loss / len(loader)
    avg_mae  = epoch_mae  / len(loader)

    return avg_loss, avg_mae



def eval_one_epoch(model, loader, model_type="theory"):
    model.eval()
    epoch_loss = 0
    epoch_mae = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in loader:
            x_comp = batch["x_comp"].to(device)
            x_total = batch["x_total_feats"].to(device)
            y = batch["y_bandgap"].to(device)
            loss, loss_dict = combined_vae_evidential_loss(model, x_comp, x_total, y)
            epoch_loss += loss.item()
            epoch_mae  += loss_dict["mae"]
            pred = loss_dict["pred"]
            all_preds.append(pred.detach().cpu())
            all_trues.append(y.detach().cpu())

    avg_loss = epoch_loss / len(loader)
    avg_mae  = epoch_mae  / len(loader)

    return avg_loss, avg_mae,all_preds,all_trues


def train_model( model=None, epochs=30, train_loader=None, val_loader=None, lr=1e-3, save_path="./pt/best_model.pt", patience=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\n========= Epoch {epoch} =========")
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_mae, _, _ = eval_one_epoch(model, val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   MAE: {val_mae:.4f}", flush=True)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model (val MAE={val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered (patience={patience})")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model



from scipy.stats import kendalltau

def test_model(model, test_loader):
    model.eval()
    test_loss, test_mae, all_preds, all_trues = eval_one_epoch(model, test_loader)
    tau, _ = kendalltau(all_trues, all_preds)
    print(f"\n===== Test Loss = {test_loss:.4f} | Test MAE = {test_mae:.4f} | Kendall Tau = {tau:.4f}", flush=True)
    
    return test_loss, test_mae, tau

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True


def main():
    full_df = pd.read_csv("./data/bandgap.csv")

    # ===== GGA Pretrain =====
    gga_df = full_df[full_df.state == 1].reset_index(drop=True)
    GGA_dataset = CompositionDataset(gga_df, "material formula", "Band_gap", "state")
    N = len(GGA_dataset)
    train_len = int(0.8 * N)
    val_len   = int(0.1 * N)
    test_len  = N - train_len - val_len

    train_GGA, val_GGA, test_GGA = random_split(
        GGA_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(1023)
    )

    train_loader_gga = DataLoader(train_GGA, batch_size=128, shuffle=True)
    val_loader_gga = DataLoader(val_GGA, batch_size=128)

    model = BandModel().to(device)
    model = train_model(model, 200, train_loader_gga, val_loader_gga,
                        lr=1e-3, save_path="./pt/pretrain_gga.pt")

    # ===== HSE Fine-tune =====
    hse_df = full_df[full_df.state == 0].reset_index(drop=True)
    HSE_dataset = CompositionDataset(hse_df, "material formula", "Band_gap", "state")
    N = len(HSE_dataset)
    train_len = int(0.8 * N)
    val_len   = int(0.1 * N)
    test_len  = N - train_len - val_len

    train_HSE, val_HSE, test_HSE = random_split(
        HSE_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(1023)
    )


    train_loader_hse = DataLoader(train_HSE, batch_size=128, shuffle=True)
    val_loader_hse = DataLoader(val_HSE, batch_size=128)
    test_loader_hse = DataLoader(test_HSE, batch_size=128)

    freeze_module(model.mlp_emb)
    freeze_module(model.vae)
    unfreeze_module(model.predictor)

    model = train_model(model, 50, train_loader_hse, val_loader_hse,
                        lr=1e-4, save_path="./pt/finetune_hse.pt")

    test_model(model, test_loader_hse)





if __name__ == "__main__":
    main()
