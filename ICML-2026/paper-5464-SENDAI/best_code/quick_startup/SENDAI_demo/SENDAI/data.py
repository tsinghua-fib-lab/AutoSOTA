"""
Data loading and preprocessing for SENDAI.
"""

import numpy as np
import json
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_data(data_dir):
    """
    Load NDVI data from processed directory.
    
    Args:
        data_dir: Path to directory containing simulation_ndvi.npy, 
                  real_physics_ndvi.npy, and metadata.json
    
    Returns:
        sim_data: (T_sim, H, W) simulation period data
        real_data: (T_real, H, W) ground truth period data
        metadata: dict with data information
    """
    data_dir = Path(data_dir)
    
    sim_data = np.load(data_dir / 'simulation_ndvi.npy')
    real_data = np.load(data_dir / 'real_physics_ndvi.npy')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return sim_data, real_data, metadata


def detect_bad_frames(data, low_threshold=0.05, bad_ratio=0.3):
    """Detect frames with excessive low/invalid pixels."""
    bad_indices = []
    for t in range(len(data)):
        frame = data[t]
        low_ratio = np.sum(frame < low_threshold) / frame.size
        nan_ratio = np.sum(np.isnan(frame)) / frame.size
        if low_ratio > bad_ratio or nan_ratio > 0.1:
            bad_indices.append(t)
    return bad_indices


def fix_bad_frames(data, low_threshold=0.05, bad_ratio=0.3):
    """Fix bad frames via temporal interpolation."""
    data_fixed = data.copy()
    T = len(data)
    bad_indices = detect_bad_frames(data, low_threshold, bad_ratio)
    
    for t in bad_indices:
        t_prev, t_next = t - 1, t + 1
        while t_prev in bad_indices and t_prev > 0:
            t_prev -= 1
        while t_next in bad_indices and t_next < T - 1:
            t_next += 1
        
        if t_prev >= 0 and t_next < T and t_prev not in bad_indices and t_next not in bad_indices:
            alpha = (t - t_prev) / (t_next - t_prev)
            data_fixed[t] = (1 - alpha) * data[t_prev] + alpha * data[t_next]
        elif t_prev >= 0 and t_prev not in bad_indices:
            data_fixed[t] = data[t_prev]
        elif t_next < T and t_next not in bad_indices:
            data_fixed[t] = data[t_next]
    
    return data_fixed, len(bad_indices)


def select_sensors(data, n_sensors, strategy='random', seed=42):
    """
    Select sensor locations.
    
    Args:
        data: (T, H, W) spatiotemporal data
        n_sensors: number of sensors to place
        strategy: 'random', 'grid', or 'stratified'
        seed: random seed
    
    Returns:
        sensor_locs: (n_sensors, 2) array of (row, col) locations
    """
    np.random.seed(seed)
    T, H, W = data.shape
    
    if strategy == 'grid':
        n_side = int(np.sqrt(n_sensors))
        rows = np.linspace(2, H - 3, n_side, dtype=int)
        cols = np.linspace(2, W - 3, n_side, dtype=int)
        rr, cc = np.meshgrid(rows, cols)
        locs = np.column_stack([rr.ravel(), cc.ravel()])[:n_sensors]
    
    elif strategy == 'stratified':
        variance = np.var(data, axis=0)
        flat_var = variance.ravel()
        mask = np.zeros_like(flat_var, dtype=bool)
        for i in range(H):
            for j in range(W):
                if 2 <= i < H - 2 and 2 <= j < W - 2:
                    mask[i * W + j] = True
        weights = np.where(mask, flat_var + 1e-6, 0)
        weights = weights / weights.sum()
        indices = np.random.choice(H * W, size=n_sensors, replace=False, p=weights)
        locs = np.column_stack([indices // W, indices % W])
    
    else:  # random
        indices = np.random.choice(H * W, size=n_sensors, replace=False)
        locs = np.column_stack([indices // W, indices % W])
    
    return locs


def create_time_delay_dataset(data, sensor_locs, lags):
    """
    Create time-delay embedded dataset for SHRED.
    
    Args:
        data: (T, H, W) spatiotemporal data
        sensor_locs: (n_sensors, 2) sensor locations
        lags: number of time lags
    
    Returns:
        sensor_sequences: (N, lags, n_sensors) sensor time histories
        full_states: (N, H*W) flattened spatial states
    """
    T, H, W = data.shape
    n_sensors = len(sensor_locs)
    
    sensor_series = np.zeros((T, n_sensors))
    for i, (r, c) in enumerate(sensor_locs):
        sensor_series[:, i] = data[:, r, c]
    
    N = T - lags
    sensor_sequences = np.zeros((N, lags, n_sensors))
    full_states = np.zeros((N, H * W))
    
    for i in range(N):
        sensor_sequences[i] = sensor_series[i:i + lags]
        full_states[i] = data[i + lags - 1].ravel()
    
    return sensor_sequences, full_states


class SHREDDataset(Dataset):
    """
    Dataset for SHRED training.
    
    Handles scaling and provides:
    - sensor_sequences: LSTM input (scaled)
    - full_states: reconstruction target (scaled)
    - current_sensors: sensor values in state scale (for HF residual)
    """
    
    def __init__(self, sensor_sequences, full_states, sensor_indices,
                 scaler_sensor=None, scaler_state=None, fit_scaler=False):
        self.N, self.lags, self.n_sensors = sensor_sequences.shape
        self.sensor_indices = sensor_indices
        
        sensors_flat = sensor_sequences.reshape(-1, self.n_sensors)
        
        self.scaler_sensor = scaler_sensor if scaler_sensor else StandardScaler()
        self.scaler_state = scaler_state if scaler_state else StandardScaler()
        
        if fit_scaler:
            sensors_scaled = self.scaler_sensor.fit_transform(sensors_flat)
            self.states_scaled = self.scaler_state.fit_transform(full_states)
        else:
            sensors_scaled = self.scaler_sensor.transform(sensors_flat)
            self.states_scaled = self.scaler_state.transform(full_states)
        
        self.sensors_scaled = sensors_scaled.reshape(self.N, self.lags, self.n_sensors)
        self.current_sensors_state_scale = self.states_scaled[:, sensor_indices]
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sensors_scaled[idx], dtype=torch.float32),
            torch.tensor(self.states_scaled[idx], dtype=torch.float32),
            torch.tensor(self.current_sensors_state_scale[idx], dtype=torch.float32)
        )
