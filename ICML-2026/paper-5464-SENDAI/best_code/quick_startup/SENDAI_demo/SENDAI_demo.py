"""
SENDAI Quick Demo: Full hierarchical model (LF + HF pathways).

This demo uses pre-downloaded data from northwestern_china.
Run from the quick_startup/SENDAI/ directory.

Usage:
    cd quick_startup/SENDAI
    python SENDAI_demo.py
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
    SHRED, DASHRED, FullDASHRED,
    train_shred, train_dashred, train_hierarchical_hf, evaluate,
    load_data, detect_bad_frames, fix_bad_frames, select_sensors,
    create_time_delay_dataset, SHREDDataset,
    compute_all_metrics, save_comprehensive_metrics,
    plot_reconstruction, plot_temporal, plot_hf_analysis,
    get_device, TimingLogger, count_parameters, count_hf_layer_parameters,
    print_parameter_summary, Tee,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / 'data'
LOCATION = 'northwestern_china'

USE_INR = True
SEED = 42

# Model
HIDDEN_SIZE = 32
DECODER_LAYERS = [256, 256]
NUM_LSTM_LAYERS = 2
DROPOUT = 0.1

# Sensors
N_SENSORS = 64
LAGS = 5
SENSOR_STRATEGY = 'random'

# Training
SHRED_EPOCHS = 800
SHRED_PATIENCE = 30
DASHRED_EPOCHS = 1500
DASHRED_PATIENCE = 50
GAN_EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-4

# HF
LAMBDA_SPARSE = 0.05
MAX_TARGET_FREQ = 16
HF_WARMUP = 100
N_PEEL_LAYERS = 2

PEEL_CONFIG = [
    {'name': 'HF_Peel1', 'target_k': 3, 'epochs': 500,
     'lambda_sparse': 0.05, 'finetune_epochs': 200, 'finetune_lambda': 0.005},
    {'name': 'HF_Peel2', 'target_k': None, 'epochs': 500,
     'lambda_sparse': 0.05, 'finetune_epochs': 200, 'finetune_lambda': 0.005},
]

INR_CONFIG = {
    'pe_num_frequencies': 16, 'pe_max_frequency': 8.0, 'pe_include_input': True,
    'encoder_hidden': [128, 128], 'latent_dim': 64,
    'decoder_hidden': [256, 256, 128], 'activation': 'relu', 'omega_0': 30.0,
    'lambda_smooth': 0.1, 'smooth_type': 'laplacian',
}


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
    output_dir = SCRIPT_DIR / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    
    timer = TimingLogger()
    timer.start_total()
    all_params = OrderedDict()
    
    print("\n" + "="*70)
    print("SENDAI Demo")
    print("="*70)
    print(f"Location: {LOCATION}")
    print(f"INR Mode: {'ON' if USE_INR else 'OFF'}")
    
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
    
    # Full model
    full_model = FullDASHRED(dashred, N_SENSORS, state_dim, sensor_indices, H, W,
                             n_hf_layers=N_PEEL_LAYERS, use_inr=USE_INR,
                             inr_config=INR_CONFIG).to(device)
    all_params['Full Model'] = count_parameters(full_model, detailed=True)
    all_params['HF Layers'] = count_hf_layer_parameters(full_model)
    
    # Stage 3: HF Peeling
    print(f"\n[7] Stage 3: HF Peeling ({N_PEEL_LAYERS} layers)...")
    timer.start("7. HF Peeling")
    full_model, all_histories, all_discovered_freqs = train_hierarchical_hf(
        full_model, train_loader_real, H, W, peel_config=PEEL_CONFIG,
        max_freq=MAX_TARGET_FREQ, exclusion_radius=2.0,
        lambda_smooth=INR_CONFIG['lambda_smooth'],
        smooth_type=INR_CONFIG['smooth_type'], warmup=HF_WARMUP, base_lr=LR)
    timer.stop("7. HF Peeling")
    
    # Evaluation
    print("\n[8] Evaluating...")
    timer.start("8. Evaluation")
    
    results_valid, rmse_da_v, rmse_total_v = evaluate(
        full_model, valid_real, train_sim.scaler_state, use_hf=True)
    metrics_valid_da = compute_all_metrics(results_valid['targets'], results_valid['da'], H, W)
    metrics_valid_total = compute_all_metrics(results_valid['targets'], results_valid['total'], H, W)
    
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
    
    results_full, rmse_da_f, rmse_total_f = evaluate(
        full_model, full_real, train_sim.scaler_state, use_hf=True)
    metrics_full_da = compute_all_metrics(results_full['targets'], results_full['da'], H, W)
    metrics_full_total = compute_all_metrics(results_full['targets'], results_full['total'], H, W)
    
    all_metrics = {
        'Validation': {'DA-SHRED': metrics_valid_da, 'DA+HF': metrics_valid_total},
        'Full': {'DA-SHRED': metrics_full_da, 'DA+HF': metrics_full_total}
    }
    timer.stop("8. Evaluation")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Validation: DA={rmse_da_v:.4f}, DA+HF={rmse_total_v:.4f}")
    print(f"  Full:       DA={rmse_da_f:.4f}, DA+HF={rmse_total_f:.4f}")
    print(f"  Improvement: {(rmse_da_f - rmse_total_f) / rmse_da_f * 100:.1f}%")
    
    # Save
    print("\n[9] Saving...")
    timer.start("9. Saving")
    
    save_comprehensive_metrics(all_metrics, output_dir, use_inr=USE_INR)
    
    np.savez(output_dir / "predictions_valid.npz", **results_valid, H=H, W=W)
    np.savez(output_dir / "predictions_full.npz", **results_full, H=H, W=W)
    
    plot_reconstruction(results_valid, sensor_locs, H, W, output_dir,
                        train_sim.scaler_state, suffix='_valid')
    plot_temporal(results_valid, output_dir, suffix='_valid')
    plot_reconstruction(results_full, sensor_locs, H, W, output_dir,
                        train_sim.scaler_state, suffix='_full')
    plot_temporal(results_full, output_dir, suffix='_full')
    plot_hf_analysis(results_full, H, W, output_dir, suffix='_full')
    
    torch.save({
        'full_model': full_model.state_dict(),
        'sensor_locs': sensor_locs,
        'all_discovered_freqs': all_discovered_freqs,
        'peel_config': PEEL_CONFIG,
        'use_inr': USE_INR,
    }, output_dir / 'model_checkpoint.pt')
    
    pd.DataFrame([timer.to_dict()]).to_csv(output_dir / 'timing.csv', index=False)
    timer.stop("9. Saving")
    
    timer.print_summary()
    print_parameter_summary(all_params)
    
    print(f"\nOutput: {output_dir}")
    print("\nDONE!")


if __name__ == "__main__":
    main()
