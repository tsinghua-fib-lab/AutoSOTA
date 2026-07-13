"""
Multi-run wrapper for SENDAI Jr. reproduction.
Runs the model N times with different seeds and averaged metrics.
"""
import sys
import warnings
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import json
import argparse

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from SENDAI import (
    SHRED, DASHRED,
    pretrain_ssl, train_shred, train_dashred,
    load_data, fix_bad_frames, select_sensors,
    create_time_delay_dataset, SHREDDataset,
    compute_all_metrics,
    get_device, TimingLogger, count_parameters, print_parameter_summary, Tee,
)

def evaluate_dashred(model, dataset, scaler_state):
    device = get_device()
    model.eval()
    loader = DataLoader(dataset, batch_size=64)
    results = {"da": [], "targets": []}
    with torch.no_grad():
        for sensors, state, _ in loader:
            sensors = sensors.to(device)
            pred, _, _ = model(sensors, apply_transform=True)
            results["da"].append(pred.cpu().numpy())
            results["targets"].append(state.numpy())
    for k in results:
        results[k] = np.vstack(results[k])
    results["targets"] = scaler_state.inverse_transform(results["targets"])
    results["da"] = scaler_state.inverse_transform(results["da"])
    rmse = np.sqrt(np.mean((results["da"] - results["targets"])**2))
    return results, rmse

def run_single(seed, config):
    """Run a single SENDAI Jr. experiment with given seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = get_device()
    print("\n" + "=" * 70)
    print("SENDAI Jr. Run - Seed: {}".format(seed))
    print("=" * 70)
    
    data_path = Path(config["data_dir"]) / config["location"] / "processed"
    output_dir = Path(config["output_dir"]) / "seed_{}".format(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("[1] Loading data...")
    sim_raw, real_raw, metadata = load_data(data_path)
    
    # Fix bad frames
    sim_raw, n_fixed = fix_bad_frames(sim_raw)
    print("  Fixed {} frames".format(n_fixed))
    
    T_sim, H, W = sim_raw.shape
    state_dim = H * W
    
    # Select sensors
    print("[2] Selecting {} sensors...".format(config["n_sensors"]))
    sensor_locs = select_sensors(sim_raw, config["n_sensors"],
                                 strategy=config["sensor_strategy"], seed=seed)
    sensor_indices = sensor_locs[:, 0] * W + sensor_locs[:, 1]
    
    # Create datasets
    print("[3] Creating datasets...")
    sim_sensors, sim_states = create_time_delay_dataset(sim_raw, sensor_locs, config["lags"])
    real_sensors, real_states = create_time_delay_dataset(real_raw, sensor_locs, config["lags"])
    
    n_train_sim = int(len(sim_sensors) * 0.8)
    n_train_real = int(len(real_sensors) * 0.8)
    
    train_sim = SHREDDataset(sim_sensors[:n_train_sim], sim_states[:n_train_sim],
                             sensor_indices, fit_scaler=True)
    valid_sim = SHREDDataset(sim_sensors[n_train_sim:], sim_states[n_train_sim:],
                             sensor_indices, scaler_sensor=train_sim.scaler_sensor,
                             scaler_state=train_sim.scaler_state)
    train_real = SHREDDataset(real_sensors[:n_train_real], real_states[:n_train_real],
                              sensor_indices, scaler_sensor=train_sim.scaler_sensor,
                              scaler_state=train_sim.scaler_state)
    valid_real = SHREDDataset(real_sensors[n_train_real:], real_states[n_train_real:],
                              sensor_indices, scaler_sensor=train_sim.scaler_sensor,
                              scaler_state=train_sim.scaler_state)
    
    train_loader_sim = DataLoader(train_sim, batch_size=config["batch_size"], shuffle=True)
    valid_loader_sim = DataLoader(valid_sim, batch_size=config["batch_size"])
    train_loader_real = DataLoader(train_real, batch_size=config["batch_size"], shuffle=True)
    
    print("  Train sim: {}, Valid sim: {}".format(len(train_sim), len(valid_sim)))
    print("  Train real: {}, Valid real: {}".format(len(train_real), len(valid_real)))
    
    # Create SHRED model
    shred = SHRED(config["n_sensors"], config["lags"], config["hidden_size"], state_dim,
                  num_layers=config["num_lstm_layers"], decoder_layers=config["decoder_layers"],
                  H=config["H"], W=config["W"], use_inr_decoder=config["use_inr_decoder"])

    # SSL Pretraining
    if config.get("ssl_epochs", 0) > 0:
        print("[3.5] SSL Pretraining (masked reconstruction)...")
        shred = pretrain_ssl(shred, train_loader_sim,
                            epochs=config["ssl_epochs"],
                            mask_ratio=config["ssl_mask_ratio"],
                            lr=config["lr"],
                            H=config["H"], W=config["W"])

    # Stage 1: SHRED
    print("[4] Training SHRED...")
    shred = train_shred(shred, train_loader_sim, valid_loader_sim,
                        epochs=config["shred_epochs"], lr=config["lr"],
                        patience=config["shred_patience"],
                        H=config["H"], W=config["W"],
                        lambda_ssim=config["lambda_ssim"],
                        lambda_grad=config["lambda_grad"])
    
    # Stage 2: DA-SHRED
    print("[5] Training DA-SHRED...")
    dashred = DASHRED(shred, freeze_decoder=False).to(device)
    dashred = train_dashred(dashred, train_loader_sim, train_loader_real, sensor_indices,
                            epochs=config["dashred_epochs"], lr=config["lr"],
                            patience=config["dashred_patience"],
                            gan_epochs=config["gan_epochs"],
                            H=config["H"], W=config["W"],
                            lambda_ssim=config["lambda_ssim"],
                            lambda_grad=config["lambda_grad"])
    
    # Evaluate on validation set
    print("[6] Evaluating...")
    results_valid, rmse_valid = evaluate_dashred(dashred, valid_real, train_sim.scaler_state)
    metrics_valid = compute_all_metrics(results_valid["targets"], results_valid["da"], H, W)
    
    # Evaluate on full real dataset
    class TempDataset(Dataset):
        def __init__(self, s, st, c):
            self.s, self.st, self.c = s, st, c
        def __len__(self):
            return len(self.s)
        def __getitem__(self, i):
            return (torch.tensor(self.s[i], dtype=torch.float32),
                    torch.tensor(self.st[i], dtype=torch.float32),
                    torch.tensor(self.c[i], dtype=torch.float32))
    
    full_real = TempDataset(
        np.vstack([train_real.sensors_scaled, valid_real.sensors_scaled]),
        np.vstack([train_real.states_scaled, valid_real.states_scaled]),
        np.vstack([train_real.current_sensors_state_scale, valid_real.current_sensors_state_scale]))
    
    results_full, rmse_full = evaluate_dashred(dashred, full_real, train_sim.scaler_state)
    metrics_full = compute_all_metrics(results_full["targets"], results_full["da"], H, W)
    
    # Save model
    torch.save({
        "shred": shred.state_dict(),
        "dashred": dashred.state_dict(),
        "sensor_locs": sensor_locs,
    }, output_dir / "model_checkpoint.pt")
    
    # Save predictions
    np.savez(output_dir / "predictions_full.npz", **results_full, H=H, W=W)
    
    print("  Seed {} - Full RMSE: {:.4f}, Full SSIM: {:.4f}".format(
        seed, rmse_full, metrics_full["SSIM_mean"]))
    
    return {
        "seed": seed,
        "rmse_valid": float(rmse_valid),
        "rmse_full": float(rmse_full),
        "metrics_valid": metrics_valid,
        "metrics_full": metrics_full,
    }

def main():
    parser = argparse.ArgumentParser(description="SENDAI Jr. Multi-run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="Seeds for multiple runs")
    parser.add_argument("--data-dir", type=str, default="/repo/data",
                        help="Data directory")
    parser.add_argument("--location", type=str, default="western_us",
                        help="Location key")
    parser.add_argument("--output-dir", type=str, default="/repo/data/western_us/results",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with reduced epochs")
    args = parser.parse_args()
    
    config = {
        "data_dir": args.data_dir,
        "location": args.location,
        "output_dir": args.output_dir,
        "n_sensors": 64,
        "lags": 5,
        "sensor_strategy": "random",
        "hidden_size": 32,
        "decoder_layers": [256, 256],
        "num_lstm_layers": 2,
        "shred_epochs": 10 if args.quick else 800,
        "shred_patience": 30,
        "dashred_epochs": 10 if args.quick else 1500,
        "dashred_patience": 50,
        "gan_epochs": 10 if args.quick else 1000,
        "batch_size": 32,
        "lr": 1e-4,
        "lambda_ssim": 0.5,
        "lambda_grad": 0.1,
        "H": 64,
        "W": 64,
        "use_inr_decoder": True,
        "ssl_epochs": 80,
        "ssl_mask_ratio": 0.4,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print("  {}: {}".format(k, v))
    print("Seeds: {}".format(args.seeds))
    
    all_results = []
    for seed in args.seeds:
        result = run_single(seed, config)
        all_results.append(result)
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS ({} runs)".format(len(all_results)))
    print("=" * 70)
    
    rmse_vals = [r["rmse_full"] for r in all_results]
    ssim_vals = [r["metrics_full"]["SSIM_mean"] for r in all_results]
    mae_vals = [r["metrics_full"]["MAE"] for r in all_results]
    
    print("RMSE: mean={:.4f}, std={:.4f}, values={}".format(
        np.mean(rmse_vals), np.std(rmse_vals), [float("{:.4f}".format(v)) for v in rmse_vals]))
    print("SSIM: mean={:.4f}, std={:.4f}, values={}".format(
        np.mean(ssim_vals), np.std(ssim_vals), [float("{:.4f}".format(v)) for v in ssim_vals]))
    print("MAE:  mean={:.4f}, std={:.4f}, values={}".format(
        np.mean(mae_vals), np.std(mae_vals), [float("{:.4f}".format(v)) for v in mae_vals]))
    
    # Save summary
    summary = {
        "n_runs": len(all_results),
        "seeds": args.seeds,
        "config": {k: str(v) for k, v in config.items()},
        "rmse": {"mean": float(np.mean(rmse_vals)), "std": float(np.std(rmse_vals)),
                 "values": [float(v) for v in rmse_vals]},
        "ssim": {"mean": float(np.mean(ssim_vals)), "std": float(np.std(ssim_vals)),
                 "values": [float(v) for v in ssim_vals]},
        "mae": {"mean": float(np.mean(mae_vals)), "std": float(np.std(mae_vals)),
                "values": [float(v) for v in mae_vals]},
    }
    
    summary_path = Path(args.output_dir) / "multi_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to: {}".format(summary_path))
    
    # Save per-run details
    rows = []
    for r in all_results:
        rows.append({
            "seed": r["seed"],
            "RMSE_valid": r["rmse_valid"],
            "RMSE_full": r["rmse_full"],
            "SSIM_valid": r["metrics_valid"]["SSIM_mean"],
            "SSIM_full": r["metrics_full"]["SSIM_mean"],
            "MAE_valid": r["metrics_valid"]["MAE"],
            "MAE_full": r["metrics_full"]["MAE"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.output_dir) / "per_run_metrics.csv", index=False)
    print("Per-run metrics saved to: {}".format(Path(args.output_dir) / "per_run_metrics.csv"))

if __name__ == "__main__":
    main()
