import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import random
import json
from dataset.dataset import CompositionDataset
from model.model_mine import BandModelSE, combined_vae_evidential_loss_SE
from scipy.stats import kendalltau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

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
    avg_rank_loss = epoch_loss_rank / len(loader)
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
                lr=1e-4, test_loader=None, save_path="./pt/best_model.pt", patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_mae = float("inf")
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_mae, avg_rank_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_mae, all_preds, all_trues = eval_one_epoch(model, val_loader)
        tau, _ = kendalltau(all_trues, all_preds)
        if epoch % 20 == 0 or epoch == 1:
            print("Epoch {:3d}: Train Loss={:.4f} MAE={:.4f} RankLoss={:.4f} | Val MAE={:.4f} Tau={:.4f}".format(
                epoch, train_loss, train_mae, avg_rank_loss, val_mae, tau), flush=True)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping at epoch {} (patience={})".format(epoch, patience))
                break
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

def test_model(model, test_loader):
    model.eval()
    test_loss, test_mae, all_preds, all_trues = eval_one_epoch(model, test_loader)
    mae = float(np.mean(np.abs(all_preds - all_trues)))
    rmse = float(np.sqrt(np.mean((all_preds - all_trues) ** 2)))
    tau, _ = kendalltau(all_trues, all_preds)
    tau = float(tau)
    print("TEST: MAE={:.6f} RMSE={:.6f} tau_b={:.6f}".format(mae, rmse, tau), flush=True)
    return {"MAE": mae, "RMSE": rmse, "tau_b": tau}

def run_seed(seed, data_dir="./data"):
    sep = "=" * 60
    print("\n{}".format(sep))
    print("Running seed={}".format(seed))
    print(sep)
    set_seed(seed)

    full_df = pd.read_csv(os.path.join(data_dir, "bandgap.csv"))
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

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_HSE, batch_size=batch_size, shuffle=False)

    model = BandModelSE().to(device)
    save_path = "./pt/best_model_seed{}.pt".format(seed)
    os.makedirs("./pt", exist_ok=True)
    best_model = train_model(model, epochs=200, train_loader=train_loader,
                             val_loader=val_loader, lr=1e-4,
                             test_loader=None, save_path=save_path, patience=50)
    metrics = test_model(best_model, test_loader)
    metrics["seed"] = seed
    return metrics

if __name__ == "__main__":
    seeds = [0, 1, 64, 1023, 1024]
    all_results = []
    for seed in seeds:
        result = run_seed(seed)
        all_results.append(result)
        print("Seed {}: {}".format(seed, result))

    maes = [r["MAE"] for r in all_results]
    rmses = [r["RMSE"] for r in all_results]
    taus = [r["tau_b"] for r in all_results]

    mean_mae = np.mean(maes)
    std_mae = np.std(maes, ddof=1)
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses, ddof=1)
    mean_tau = np.mean(taus)
    std_tau = np.std(taus, ddof=1)

    sep = "=" * 60
    print("\n{}".format(sep))
    print("FINAL RESULTS (mean +/- std over {} seeds)".format(len(seeds)))
    print(sep)
    print("MAE:   {:.4f} +/- {:.4f}".format(mean_mae, std_mae))
    print("RMSE:  {:.4f} +/- {:.4f}".format(mean_rmse, std_rmse))
    print("tau_b: {:.4f} +/- {:.4f}".format(mean_tau, std_tau))

    summary = {
        "paper_id": 2216,
        "dataset": "TF-Bandgap",
        "fidelity_setting": "GGA+HSE",
        "seeds": seeds,
        "individual_results": all_results,
        "MAE_mean": float(mean_mae),
        "MAE_std": float(std_mae),
        "RMSE_mean": float(mean_rmse),
        "RMSE_std": float(std_rmse),
        "tau_b_mean": float(mean_tau),
        "tau_b_std": float(std_tau),
    }
    print(json.dumps(summary, indent=2))

    with open("./results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Results saved to ./results_summary.json")
