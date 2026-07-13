"""
SENDAI Jr.: Low-frequency pathway only (DA-SHRED).

Suitable for landscapes with relatively homogeneous spatial structure.

Usage:
    python SENDAI_Jr_main.py

Configure paths and hyperparameters in the CONFIGURATION section below.
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

warnings.filterwarnings('ignore')

from SENDAI import (
    SHRED, DASHRED,
    train_shred, train_dashred, 
    load_data, fix_bad_frames, select_sensors,
    create_time_delay_dataset, SHREDDataset,
    compute_all_metrics, save_comprehensive_metrics,
    plot_reconstruction, plot_temporal,
    get_device, TimingLogger, count_parameters, print_parameter_summary, Tee,
)


# =============================================================================
# CONFIGURATION - Modify these for your setup
# =============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / 'data'
LOCATION = 'western_us' # Input target location

# Random seed
SEED = 42

# Model architecture
HIDDEN_SIZE = 32
DECODER_LAYERS = [256, 256]
NUM_LSTM_LAYERS = 2
DROPOUT = 0.1

# Sensor configuration
N_SENSORS = 64
LAGS = 5
SENSOR_STRATEGY = 'random'

# Training hyperparameters
SHRED_EPOCHS = 800
SHRED_PATIENCE = 30
DASHRED_EPOCHS = 1500
DASHRED_PATIENCE = 50
GAN_EPOCHS = 1000
BATCH_SIZE = 32
LR = 1e-4


# =============================================================================
# EVALUATION (DA-SHRED only, no HF)
# =============================================================================

def evaluate_dashred(model, dataset, scaler_state):
    """Evaluate DA-SHRED model."""
    device = get_device()
    model.eval()
    loader = DataLoader(dataset, batch_size=64)
    
    results = {'da': [], 'targets': []}
    
    with torch.no_grad():
        for sensors, state, _ in loader:
            sensors = sensors.to(device)
            pred, _, _ = model(sensors, apply_transform=True)
            results['da'].append(pred.cpu().numpy())
            results['targets'].append(state.numpy())
    
    for k in results:
        results[k] = np.vstack(results[k])
    
    # Inverse transform
    results['targets'] = scaler_state.inverse_transform(results['targets'])
    results['da'] = scaler_state.inverse_transform(results['da'])
    
    rmse = np.sqrt(np.mean((results['da'] - results['targets'])**2))
    
    return results, rmse


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    device = get_device()
    print(f"Using device: {device}")
    
    data_path = DATA_DIR / LOCATION / 'processed'
    output_dir = DATA_DIR / LOCATION / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    
    timer = TimingLogger()
    timer.start_total()
    all_params = OrderedDict()
    
    print("\n" + "="*70)
    print("SENDAI Jr. (DA-SHRED Only)")
    print("="*70)
    print(f"Location: {LOCATION}")
    
    # Load data
    print("\n[1] Loading data...")
    timer.start("1. Data Loading")
    sim_raw, real_raw, metadata = load_data(data_path)
    timer.stop("1. Data Loading")
    
    # Fix bad frames
    print("\n[2] Fixing bad frames...")
    timer.start("2. Bad Frame Fixing")
    sim_raw, n_fixed = fix_bad_frames(sim_raw)
    print(f"  Fixed {n_fixed} frames")
    timer.stop("2. Bad Frame Fixing")
    
    T_sim, H, W = sim_raw.shape
    state_dim = H * W
    
    # Sensors
    print(f"\n[3] Selecting {N_SENSORS} sensors...")
    timer.start("3. Sensor Selection")
    sensor_locs = select_sensors(sim_raw, N_SENSORS, strategy=SENSOR_STRATEGY, seed=SEED)
    sensor_indices = sensor_locs[:, 0] * W + sensor_locs[:, 1]
    timer.stop("3. Sensor Selection")
    
    # Datasets
    print("\n[4] Creating datasets...")
    timer.start("4. Dataset Creation")
    sim_sensors, sim_states = create_time_delay_dataset(sim_raw, sensor_locs, LAGS)
    real_sensors, real_states = create_time_delay_dataset(real_raw, sensor_locs, LAGS)
    
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
    
    train_loader_sim = DataLoader(train_sim, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_sim = DataLoader(valid_sim, batch_size=BATCH_SIZE)
    train_loader_real = DataLoader(train_real, batch_size=BATCH_SIZE, shuffle=True)
    timer.stop("4. Dataset Creation")
    
    print(f"  Train sim: {len(train_sim)}, Valid sim: {len(valid_sim)}")
    print(f"  Train real: {len(train_real)}, Valid real: {len(valid_real)}")
    
    # Stage 1: SHRED
    print("\n[5] Stage 1: Training SHRED...")
    timer.start("5. SHRED Training")
    shred = SHRED(N_SENSORS, LAGS, HIDDEN_SIZE, state_dim,
                  num_layers=NUM_LSTM_LAYERS, decoder_layers=DECODER_LAYERS)
    all_params['SHRED'] = count_parameters(shred, detailed=True)
    shred = train_shred(shred, train_loader_sim, valid_loader_sim,
                        epochs=SHRED_EPOCHS, lr=LR, patience=SHRED_PATIENCE)
    timer.stop("5. SHRED Training")
    
    # Stage 2: DA-SHRED
    print("\n[6] Stage 2: Training DA-SHRED...")
    timer.start("6. DA-SHRED Training")
    dashred = DASHRED(shred, freeze_decoder=False).to(device)
    all_params['DA-SHRED'] = count_parameters(dashred, detailed=True)
    dashred = train_dashred(dashred, train_loader_sim, train_loader_real, sensor_indices,
                            epochs=DASHRED_EPOCHS, lr=LR, patience=DASHRED_PATIENCE,
                            gan_epochs=GAN_EPOCHS)
    timer.stop("6. DA-SHRED Training")
    
    # Evaluation
    print("\n[7] Evaluating...")
    timer.start("7. Evaluation")
    
    results_valid, rmse_valid = evaluate_dashred(dashred, valid_real, train_sim.scaler_state)
    metrics_valid = compute_all_metrics(results_valid['targets'], results_valid['da'], H, W)
    
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
    metrics_full = compute_all_metrics(results_full['targets'], results_full['da'], H, W)
    
    all_metrics = {'Validation': metrics_valid, 'Full': metrics_full}
    timer.stop("7. Evaluation")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Validation RMSE: {rmse_valid:.4f}")
    print(f"  Full RMSE:       {rmse_full:.4f}")
    print(f"  Validation SSIM: {metrics_valid['SSIM_mean']:.4f}")
    print(f"  Full SSIM:       {metrics_full['SSIM_mean']:.4f}")
    
    # Save
    print("\n[8] Saving...")
    timer.start("8. Saving")
    
    # Metrics
    rows = []
    for dataset_name, metrics in all_metrics.items():
        rows.append({'Dataset': dataset_name, 'Model': 'DA-SHRED', **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'comprehensive_metrics.csv', index=False)
    print(f"  Saved: {output_dir / 'comprehensive_metrics.csv'}")
    
    np.savez(output_dir / "predictions_valid.npz", **results_valid, H=H, W=W)
    np.savez(output_dir / "predictions_full.npz", **results_full, H=H, W=W)
    
    # For visualization compatibility, add 'total' key (same as 'da' for Jr)
    results_valid['total'] = results_valid['da']
    results_full['total'] = results_full['da']
    
    plot_reconstruction(results_valid, sensor_locs, H, W, output_dir,
                        train_sim.scaler_state, valid_sim.states_scaled, Full_Sendai=False, suffix='_valid')
    plot_temporal(results_valid, output_dir, suffix='_valid')
    full_sim_states = np.vstack([train_sim.states_scaled, valid_sim.states_scaled])
    plot_reconstruction(results_full, sensor_locs, H, W, output_dir,
                        train_sim.scaler_state, full_sim_states, Full_Sendai=False, suffix='_full')
    plot_temporal(results_full, output_dir, suffix='_full')
    
    torch.save({
        'shred': shred.state_dict(),
        'dashred': dashred.state_dict(),
        'sensor_locs': sensor_locs,
    }, output_dir / 'model_checkpoint.pt')
    
    pd.DataFrame([timer.to_dict()]).to_csv(output_dir / 'timing.csv', index=False)
    timer.stop("8. Saving")
    
    timer.print_summary()
    print_parameter_summary(all_params)
    
    print(f"\nOutput: {output_dir}")
    print("\nDONE!")


if __name__ == "__main__":
    main()
