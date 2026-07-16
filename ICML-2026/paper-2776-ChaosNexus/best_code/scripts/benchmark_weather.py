#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import torch
from scipy.stats import sem, t
from tqdm import tqdm
import warnings

# Suppress potential RuntimeWarnings from numpy/scipy when computing statistics on small arrays
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================================
# SECTION 1: Configuration and Hyperparameters
# =====================================================================================

BASE_DIR = "./weather_results"
DATA_SUBDIR = "test_example"  # Subfolder where prediction/label files are located
EVAL_HORIZONS = [24, 48, 72, 96, 120] # Time steps/horizons for evaluation

# Dimension names (corresponding to the 5 channels in the weather data)
CHANNEL_NAMES = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]

# =====================================================================================
# SECTION 2: Core Metric Calculation (with 95% Confidence Interval)
# =====================================================================================

def _format_ci(values: list | np.ndarray, confidence: float = 0.95) -> str:
    """
    Calculates and formats the mean and the specified confidence interval (Mean ± CI).

    Args:
        values: A list or numpy array of metric values (e.g., MSE per sample).
        confidence: The confidence level (default 0.95 for 95% CI).
    
    Returns:
        A formatted string: "Mean.xxxx ± CI.xxxx" or "N/A" if insufficient data.
    """
    data = np.array(values)
    # Filter out non-finite values (NaN, Inf)
    data = data[np.isfinite(data)]
    n = len(data)
    
    # Need at least 2 samples to compute standard error and CI
    if n < 2: return "N/A"
    
    mean_val = np.mean(data)
    std_err = sem(data)  # Standard error of the mean
    
    # Calculate the margin of error (h) using the t-distribution
    # t.ppf((1 + confidence) / 2., n - 1) is the t-score
    h = std_err * t.ppf((1 + confidence) / 2., n - 1)
    
    return f"{mean_val:.4f} \u00B1 {h:.4f}" # Use \u00B1 for ±

def compute_metrics_per_sample(y_true, y_pred):
    """
    Computes standard time-averaged metrics (MSE, MAE, sMAPE) for each sample.
    The time dimension is averaged, preserving the Batch dimension.

    Input shape: [Batch, Time] (Univariate/Single Channel)
    
    Returns:
        A dictionary containing lists of metric values, one value per sample/batch item.
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    results = {}
    # 1. MSE (Mean Squared Error), averaged over the Time dimension
    results["mse"] = np.mean(errors ** 2, axis=1).tolist()
    # 2. MAE (Mean Absolute Error), averaged over the Time dimension
    results["mae"] = np.mean(abs_errors, axis=1).tolist()
    
    # 3. sMAPE (Symmetric Mean Absolute Percentage Error)
    # Formula: 100 * mean_t ( 2 * |y_true - y_pred| / (|y_true| + |y_pred|) )
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8 # Add epsilon for stability
    pointwise_smape = 2.0 * abs_errors / denominator
    results["smape"] = (100.0 * np.mean(pointwise_smape, axis=1)).tolist()
        
    return results

def compute_metrics_per_channel_with_ci(y_true, y_pred, channel_names_list):
    """
    Iterates through each channel/dimension, calculates metrics for all samples,
    and returns the mean ± 95% CI for each channel.

    Input shape: [Batch, Time, Dims]
    
    Returns:
        A list of dictionaries, where each dictionary represents a channel's 
        metrics (mean ± CI) for the current time horizon.
    """
    B, T, D = y_true.shape
    results_list = []
    
    for d in range(D):
        true_d = y_true[:, :, d]
        pred_d = y_pred[:, :, d]
        
        # Calculate metric list per sample for the current channel
        sample_metrics = compute_metrics_per_sample(true_d, pred_d)
        
        # Get channel name
        dim_name = channel_names_list[d] if d < len(channel_names_list) else f"Dim_{d}"
        
        # Format metrics as Mean ± CI
        row = {
            "channel": dim_name,
            "mse": _format_ci(sample_metrics["mse"]),
            "mae": _format_ci(sample_metrics["mae"]),
            "smape": _format_ci(sample_metrics["smape"])
        }
        results_list.append(row)
        
    return results_list

# =====================================================================================
# SECTION 3: Data Loading and Alignment 
# =====================================================================================

def load_and_align_data(model_folder_path):
    """
    Loads prediction and label data from files and aligns their dimensions.

    Logic:
    1. Load Predictions (y_pred) and Labels (y_label).
    2. Extract the ground truth Tensor from the potentially complex label structure.
    3. Convert Tensors to Numpy arrays.
    4. Transpose dimensions to ensure the standard shape is (Batch, Time, Channels).
    5. Align the time dimension: Truncate Labels (y_true) to match the length of Predictions (y_pred).

    Returns:
        A tuple (y_true, y_pred) with shape [B, T_pred, C], or (None, None) on failure.
    """
    data_dir = os.path.join(model_folder_path, DATA_SUBDIR)
    labels_path = os.path.join(data_dir, "labels_test_small.pt")
    preds_path = os.path.join(data_dir, "predictions_test_small.pt")

    if not (os.path.exists(labels_path) and os.path.exists(preds_path)):
        # print(f"Skipping {model_folder_path}: Files not found.")
        return None, None

    try:
        # 1. Load data
        y_pred_raw = torch.load(preds_path, map_location='cpu')
        y_label_raw = torch.load(labels_path, map_location='cpu')

        # 2. Extract Ground Truth Tensor (Robust extraction logic)
        y_true = None
        if isinstance(y_label_raw, torch.Tensor):
            y_true = y_label_raw
        elif isinstance(y_label_raw, dict):
            # Check common keys for ground truth/target values
            for key in ['gt', 'future_values', 'target', 'future', 'y']:
                if key in y_label_raw:
                    y_true = y_label_raw[key]
                    break
        elif isinstance(y_label_raw, (list, tuple)) and len(y_label_raw) >= 2:
            # Assume ground truth is the second element (e.g., [context, future])
            y_true = y_label_raw[1]
        
        if y_true is None:
            print(f"Warning: Failed to parse ground truth from {labels_path}")
            return None, None

        # Convert to Numpy
        if isinstance(y_pred_raw, torch.Tensor): y_pred = y_pred_raw.detach().numpy()
        else: y_pred = np.array(y_pred_raw)
            
        if isinstance(y_true, torch.Tensor): y_true = y_true.detach().numpy()
        else: y_true = np.array(y_true)

        # 3. Dimension Transposition (B, C, T) -> (B, T, C)
        # Transpose if the channel dimension (C) is smaller than the time dimension (T)
        # assuming Dims (C=5) < Time (T=120 or 640)
        if y_true.ndim == 3 and y_true.shape[1] < y_true.shape[2]: 
            y_true = np.transpose(y_true, (0, 2, 1))
        
        if y_pred.ndim == 3 and y_pred.shape[1] < y_pred.shape[2]:
            y_pred = np.transpose(y_pred, (0, 2, 1))
            
        # Basic check for channel consistency
        if y_true.shape[-1] != y_pred.shape[-1]:
             print(f"Error: Channel dimensions mismatch in {model_folder_path}: Label={y_true.shape[-1]}, Pred={y_pred.shape[-1]}")
             return None, None

        # 4. Time Alignment (Truncate Labels to match Prediction length)
        # Pred shape: (B, T_pred, C)
        # Label shape: (B, T_total, C)
        t_pred = y_pred.shape[1]
        t_label = y_true.shape[1]

        if t_label >= t_pred:
            # Use the last T_pred time steps of the label as ground truth
            y_true = y_true[:, -t_pred:, :]
        else:
            print(f"Error: Label length ({t_label}) < Prediction length ({t_pred}) in {model_folder_path}")
            return None, None
        
        # Final shape check
        if y_true.shape != y_pred.shape:
            print(f"Error: Final shapes mismatch in {model_folder_path}: Label={y_true.shape}, Pred={y_pred.shape}")
            return None, None

        return y_true, y_pred

    except Exception as e:
        print(f"Error processing {model_folder_path}: {e}")
        return None, None

# =====================================================================================
# SECTION 4: Main Execution Program
# =====================================================================================

def main():
    """
    Main function to iterate through all model result folders, load data, 
    compute evaluation metrics (MSE, MAE, sMAPE) with 95% CI for specified 
    time horizons, and save the results to a CSV file.
    """
    # Discover model directories
    model_dirs = [f.path for f in os.scandir(BASE_DIR) if f.is_dir()]
    print(f"Found {len(model_dirs)} model directories under {BASE_DIR}.")

    for folder in tqdm(model_dirs, desc="Processing Models"):
        
        model_name = os.path.basename(folder)
        
        # 1. Load and align data
        y_true, y_pred = load_and_align_data(folder)
        if y_true is None:
            continue

        # 2. Determine channel names based on data dimension
        num_dims = y_true.shape[2]
        current_channels = CHANNEL_NAMES
        if num_dims != len(CHANNEL_NAMES):
            print(f"Warning: Dimension count ({num_dims}) mismatches predefined names ({len(CHANNEL_NAMES)}).")
            current_channels = [f"Dim_{i}" for i in range(num_dims)]

        # 3. Compute metrics for each evaluation horizon
        results_df_list = []
        for h in EVAL_HORIZONS:
            # Check if prediction length is sufficient for the horizon
            if h > y_true.shape[1]:
                # print(f"Warning: Horizon {h} is longer than prediction length {y_true.shape[1]}. Skipping.")
                continue
            
            # Slice: Use the first h steps (Time dimension)
            y_true_h = y_true[:, :h, :]
            y_pred_h = y_pred[:, :h, :]
            
            # Compute channel-wise metrics with CI
            metrics = compute_metrics_per_channel_with_ci(y_true_h, y_pred_h, current_channels)
            
            # Organize results into a DataFrame
            df_h = pd.DataFrame(metrics)
            df_h.insert(0, "horizon", h)
            df_h.insert(0, "model", model_name)
            results_df_list.append(df_h)

        # 4. Save results
        if results_df_list:
            final_df = pd.concat(results_df_list, ignore_index=True)
            # Standardize column order
            cols = ["model", "horizon", "channel", "mse", "mae", "smape"]
            final_df = final_df[cols]
            
            save_path = os.path.join(folder, "metrics_evaluation_CI.csv")
            final_df.to_csv(save_path, index=False)
            
    print("\n>>> All evaluations complete and results saved.")

if __name__ == "__main__":
    main()