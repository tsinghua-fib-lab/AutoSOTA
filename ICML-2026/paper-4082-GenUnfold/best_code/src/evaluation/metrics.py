# src/evaluation/metrics.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from numpy.linalg import eigh
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import r2_score # scikit-learn for R^2
from scipy.stats import gaussian_kde
import logging
from typing import List, Tuple, Dict, Any, Union, Optional

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Import analysis functions to evaluate derived properties
from ..analysis.mechanical_properties import calculate_unfolding_energy, calculate_max_force, find_force_peaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def compute_fid(real, generated):
    real, generated = _flatten(real), _flatten(generated)
    # calculate mean and covariance statistics
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = generated.mean(axis=0), np.cov(generated,  rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def compute_kid(
    true: ArrayLike,
    generated: ArrayLike,
    subset_size: int = 10,
    n_subsets: int = 100,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> Tuple[float, float]:
    """Kernel Inception Distance via unbiased polynomial‑kernel MMD (robust to 1‑D)."""
    r = _flatten(true)
    g = _flatten(generated)
    feat_dim = r.shape[1]
    if gamma is None:
        gamma = 1.0 / feat_dim

    subset = min(subset_size, r.shape[0], g.shape[0])
    if subset < subset_size:
        raise ValueError(
            f"subset_size={subset_size} exceeds available samples "
            f"(real={r.shape[0]}, fake={g.shape[0]})."
        )

    mmd2 = []
    rng = np.random.default_rng(0)
    for _ in range(n_subsets):
        idx_r = rng.choice(r.shape[0], subset, replace=False)
        idx_g = rng.choice(g.shape[0], subset, replace=False)
        xr, xg = r[idx_r], g[idx_g]

        k_rr = _poly_kernel(xr, xr, degree, gamma, coef0)
        k_gg = _poly_kernel(xg, xg, degree, gamma, coef0)
        k_rg = _poly_kernel(xr, xg, degree, gamma, coef0)
        mmd2.append(k_rr.mean() + k_gg.mean() - 2.0 * k_rg.mean())

    mmd2 = np.asarray(mmd2, dtype=np.float64)
    return float(mmd2.mean()), float(mmd2.std(ddof=1))


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _flatten(arr: ArrayLike) -> np.ndarray:
    """Convert to **(N, D)** float64 on CPU with view when possible."""
    if torch is not None and isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)

    # (N,) → (N,1)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.reshape(arr.shape[0], -1).astype(np.float64, copy=False)


def _poly_kernel(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    gamma: float,
    coef0: float,
) -> np.ndarray:
    """Polynomial kernel 〈x,y〉^d with scaling γ and bias coef0."""
    return (gamma * x @ y.T + coef0) ** degree


def calculate_r2(true_curves: np.ndarray, generated_curves: np.ndarray) -> float:
    """
    Calculates the R^2 (coefficient of determination) between true and generated curves.
    Treats each curve as a high-dimensional sample.

    Args:
        true_curves (np.ndarray): Array of true F-E curves. Shape (num_samples, curve_length, channels).
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).

    Returns:
        float: The calculated R^2 score. Returns np.nan if shapes don't match or inputs are invalid.
    """
    if true_curves.shape != generated_curves.shape:
        logging.error(f"Shape mismatch for R^2 calculation: {true_curves.shape} vs {generated_curves.shape}")
        return np.nan
    if true_curves.ndim < 2 or generated_curves.ndim < 2:
        logging.error("Input arrays must have at least 2 dimensions (samples, length, [channels]).")
        return np.nan

    # Reshape curves to (num_samples, feature_dimension) for r2_score
    num_samples = true_curves.shape[0]
    feature_dim = np.prod(true_curves.shape[1:])
    true_flat = true_curves.reshape(num_samples, feature_dim).mean(axis=0)
    gen_flat = generated_curves.reshape(num_samples, feature_dim).mean(axis=0)

    try:
        r2 = r2_score(gen_flat, true_flat)
        return r2
    except Exception as e:
        logging.error(f"Error calculating R^2 score: {e}")
        return np.nan


def calculate_relative_l2_error(true_curves: np.ndarray, generated_curves: np.ndarray) -> float:
    """
    Calculates the average Relative L2 Error (RMSE / ||true_curve||_2) per curve in the batch.

    Args:
        true_curves (np.ndarray): Array of true F-E curves. Shape (num_samples, curve_length, channels).
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).

    Returns:
        float: The average relative L2 error. Returns np.nan if shapes don't match or inputs are invalid.
    """
    if true_curves.shape != generated_curves.shape:
        logging.error(f"Shape mismatch for Relative L2 Error calculation: {true_curves.shape} vs {generated_curves.shape}")
        return np.nan
    if true_curves.ndim < 2 or generated_curves.ndim < 2:
        logging.error("Input arrays must have at least 2 dimensions (samples, length, [channels]).")
        return np.nan

    # Flatten the last dimensions to calculate L2 norm per sample
    true_flat = true_curves.reshape(true_curves.shape[0], -1)
    gen_flat = generated_curves.reshape(generated_curves.shape[0], -1)

    # Calculate L2 norm of the difference (RMSE for each sample)
    l2_diff = np.linalg.norm(true_flat - gen_flat, ord=2, axis=1)

    # Calculate L2 norm of the true curve for normalization
    true_norm = np.linalg.norm(true_flat, ord=2, axis=1)

    # Avoid division by zero for true curves with zero norm
    non_zero_norm_mask = true_norm > 1e-6 # Use a small threshold
    if not np.any(non_zero_norm_mask):
        logging.warning("All true curves have zero norm. Cannot calculate relative L2 error.")
        return np.nan

    # Calculate relative L2 error for curves with non-zero norm
    relative_l2_errors = np.zeros_like(l2_diff)
    relative_l2_errors[non_zero_norm_mask] = l2_diff[non_zero_norm_mask] / true_norm[non_zero_norm_mask]

    # Return the average relative L2 error over the batch
    return np.mean(relative_l2_errors[non_zero_norm_mask])



# =============================================================================
# Discriminative Score
# =============================================================================
class Discriminator(nn.Module):
    """A simple 1D CNN for time series classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class LSTMDiscriminator(nn.Module):
    """A simple LSTM for time series classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        """
        Args:
            input_dim (int): The number of features per time step (channels).
            hidden_dim (int): The number of features in the hidden state of the LSTM.
            num_layers (int): The number of recurrent layers.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input shape is (batch, seq_len, features)
            bidirectional=True  # Can be set to True for better performance
        )

        # The output of the LSTM is the hidden state of the last layer.
        # We need to project this into a single value for classification.
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input time series tensor. Shape should be (batch_size, input_dim, seq_len).

        Returns:
            A tensor of shape (batch_size, 1) with the classification scores.
        """
        # The LSTM layer expects the input shape to be (batch, seq_len, features).
        # Your input x is (batch, input_dim, seq_len), so we need to permute it.
        x_permuted = x.permute(0, 2, 1)  # (batch_size, seq_len, input_dim)

        # Pass the data through the LSTM
        # The output 'lstm_out' contains the hidden state for each time step.
        # 'h_n' and 'c_n' are the final hidden and cell states.
        lstm_out, (h_n, c_n) = self.lstm(x_permuted)

        # We use the hidden state of the final time step of the last layer for classification.
        # h_n shape is (num_layers, batch_size, hidden_dim).
        # We get the last layer's hidden state (h_n[-1]) and use it for the final linear layer.
        final_hidden_state = h_n[-1]  # shape: (batch_size, hidden_dim)

        # Pass the final hidden state to the classification layers
        output = self.final_layer(final_hidden_state)

        return output


def compute_discriminative_score(
        true: ArrayLike,
        generated: ArrayLike,
        epochs: int = 100,
        batch_size: int = 64,
        discriminator: str = "LSTM",
) -> float:
    """
    Trains a classifier to distinguish between real and generated data.
    Returns the accuracy on a test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    true_data = torch.tensor(_flatten(true), dtype=torch.float32).unsqueeze(1)
    generated_data = torch.tensor(_flatten(generated), dtype=torch.float32).unsqueeze(1)



    # Correctly shape the time series data (batch, channels, length)
    if len(true_data.shape) == 2:
        true_data = true_data.unsqueeze(1)
    if len(generated_data.shape) == 2:
        generated_data = generated_data.unsqueeze(1)

    all_data = torch.cat([true_data, generated_data], dim=0)
    labels = torch.cat([torch.ones(len(true_data)), torch.zeros(len(generated_data))], dim=0).unsqueeze(1)

    dataset = TensorDataset(all_data, labels)

    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, and optimizer
    input_dim = true_data.shape[1]
    if discriminator == 'LSTM':
        model = LSTMDiscriminator(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device)
    elif discriminator == 'CNN':
        model = Discriminator(input_dim, hidden_dim=64).to(device)
    else:
        raise ValueError

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in tqdm(range(epochs)):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    discriminative_score = np.abs(0.5 - accuracy)
    return  discriminative_score



def compute_kde_diff(ture: ArrayLike, gen: ArrayLike) -> float:
    """Compute Jensen-Shannon Divergence (JSD) between two arrays."""
    ture = np.array(ture).reshape(-1)
    gen = np.array(gen).reshape(-1)

    # Compute kde
    kde_ture = gaussian_kde(ture)
    kde_gen = gaussian_kde(gen)

    # Define a common scope of evaluation points
    x_range = np.linspace(0, ture.max() * 1.1, 500)

    # Evaluate the density values of two KDEs
    density_ture = kde_ture.evaluate(x_range)
    density_gen = kde_gen.evaluate(x_range)

    # Note: The jensenshannon function needs to input a probability distribution, so the density value needs to be normalized.
    # Normalization step: Use trapezoidal rule to calculate the area under the curve, and then divide it
    # This is a necessary step to ensure that the integral is approximate to 1
    area_ture = np.trapz(density_ture, x_range)
    area_gen = np.trapz(density_gen, x_range)

    normalized_density_ture = density_ture / area_ture
    normalized_density_gen = density_gen / area_gen

    # compute JSD
    js_distance = jensenshannon(normalized_density_ture, normalized_density_gen)

    return js_distance



def evaluate_across_length(
    true_curves: np.ndarray,
    generated_curves: np.ndarray,
    lengths: np.ndarray,
    *,
    bins: list | np.ndarray | None = None,
    metric_fns: dict[str, callable] | None = None,
    include_overall: bool = True,
) -> pd.DataFrame:
    """
    Evaluate generation quality across protein length bins.

    Parameters
    ----------
    true_curves : np.ndarray
        Shape (N, T) or (N, T, C). Ground-truth force–extension curves or features.
    generated_curves : np.ndarray
        Same shape as true_curves. Model outputs / generated curves.
    lengths : np.ndarray
        Shape (N,). Protein sequence lengths (integers).
    bins : list | np.ndarray | None, optional
        Bin edges for length bucketing. Default: fixed bins [0, 50, 100, 150, 200, 250].
        You can also pass np.quantile(...) edges if you want quantile bins.
    metric_fns : dict[str, callable] | None, optional
        Mapping {metric_name: fn(y_true, y_pred) -> float}. If None, attempts to
        auto-import common metrics from your local `metrics.py`.
    include_overall : bool, optional
        If True, appends an 'overall' row across all lengths.

    Returns
    -------
    pd.DataFrame
        Columns: ['bin_left', 'bin_right', 'count', <metric columns>]
    """
    # Basic checks
    true_curves = np.asarray(true_curves)
    generated_curves = np.asarray(generated_curves)
    lengths = np.asarray(lengths)

    assert true_curves.shape == generated_curves.shape, \
        f"Shape mismatch: true {true_curves.shape} vs gen {generated_curves.shape}"
    #assert lengths.ndim == 1 and lengths.shape[0] == true_curves.shape[0], \
        #f"lengths must be (N,), got {lengths.shape} for N={true_curves.shape[0]}"

    # Default fixed bins up to max length 250
    if bins is None:
        bins = np.arange(25, 275, 25)
    else:
        bins = np.asarray(bins)
        assert np.all(np.diff(bins) > 0), "bins must be strictly increasing"

    # Collect metrics
    if metric_fns is None or len(metric_fns) == 0:
        metric_fns = {'rmse': calculate_relative_l2_error, 'fid': compute_fid, 'properties': evaluate_mechanical_properties}
        if len(metric_fns) == 0:
            # Provide a minimal fallback RMSE if nothing is available
            def _rmse(y_true, y_pred):
                y_true = np.asarray(y_true)
                y_pred = np.asarray(y_pred)
                return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            metric_fns = {"rmse": _rmse}

    records = []
    N = lengths.shape[0]

    # Evaluate per bin
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        mask = (lengths > left) & (lengths <= right)  # (left, right]
        idx = np.where(mask)[0]
        if idx.size == 0:
            # Gracefully handle empty bins
            rec = {"bin_left": int(left), "bin_right": int(right), "count": 0}
            for mname in metric_fns:
                rec[mname] = np.nan
            records.append(rec)
            continue

        y_true_bin = true_curves[idx]
        y_pred_bin = generated_curves[idx]

        rec = {"bin_left": int(left), "bin_right": int(right), "count": int(idx.size)}
        for mname, mfn in metric_fns.items():
            try:
                score = mfn(y_true_bin, y_pred_bin)
                # convert tensors to float if your metric returns torch tensors
                if hasattr(score, "item"):
                    score = score.item()
                if mname == 'properties':
                    rec['f_jsd'] = score['max_force']['Jensen-Shannon Divergence']
                    rec['E_jsd'] = score['unfolding_energy']['Jensen-Shannon Divergence']
                    rec['true_force'] = score['max_force']['true_vals']
                    rec['pred_force'] = score['max_force']['gen_vals']
                    rec['true_energy'] = score['unfolding_energy']['true_vals'] / 512 * lengths[idx]
                    rec['pred_energy'] = score['unfolding_energy']['gen_vals'] / 512 * lengths[idx]
                else:
                    rec[mname] = float(score)
            except Exception as e:
                rec[mname] = np.nan

        records.append(rec)

    # Optional overall row
    if include_overall:
        rec = {"bin_left": 0, "bin_right": int(bins[-1]), "count": int(N)}
        for mname, mfn in metric_fns.items():
            try:
                score = mfn(true_curves, generated_curves)
                if hasattr(score, "item"):
                    score = score.item()
                if mname == 'properties':
                    print(score)
                    rec['f_jsd'] = score['max_force']['Jensen-Shannon Divergence']
                    rec['E_jsd'] = score['unfolding_energy']['Jensen-Shannon Divergence']
                    rec['true_force'] = score['max_force']['true_vals']
                    rec['pred_force'] = score['max_force']['gen_vals']
                    rec['true_energy'] = score['unfolding_energy']['true_vals'] / 512
                    rec['pred_energy'] = score['unfolding_energy']['gen_vals'] / 512
                else:
                    rec[mname] = float(score)
            except Exception:
                rec[mname] = np.nan
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    # Nice ordering: bins by left edge; 'overall' at the end (already appended)
    return df


def evaluate_across_class(
    true_curves: np.ndarray,
    generated_curves: np.ndarray,
    caths: np.ndarray,
    *,
    lengths: np.ndarray = None,
    bins: list | np.ndarray | None = None,
    metric_fns: dict[str, callable] | None = None,
    include_overall: bool = True,
) -> pd.DataFrame:
    """
    Evaluate generation quality across protein length bins.

    Parameters
    ----------
    true_curves : np.ndarray
        Shape (N, T) or (N, T, C). Ground-truth force–extension curves or features.
    generated_curves : np.ndarray
        Same shape as true_curves. Model outputs / generated curves.
    caths : np.ndarray
        Shape (N,). Protein classification (integers).
    bins : list | np.ndarray | None, optional
        Bin edges. Default: fixed bins [1, 2, 3, 4, 5, 6] -> [Mainly Alpha, Mainly Beta, Alpha Beta, Few, Special].
        You can also pass np.quantile(...) edges if you want quantile bins.
    metric_fns : dict[str, callable] | None, optional
        Mapping {metric_name: fn(y_true, y_pred) -> float}. If None, attempts to
        auto-import common metrics from your local `metrics.py`.
    include_overall : bool, optional
        If True, appends an 'overall' row across all lengths.

    Returns
    -------
    pd.DataFrame
        Columns: ['bin_left', 'bin_right', 'count', <metric columns>]
    """
    # Basic checks
    true_curves = np.asarray(true_curves)
    generated_curves = np.asarray(generated_curves)
    caths = np.asarray(caths)
    if lengths is not None:
        lengths = np.asarray(lengths)
    else:
        lengths = np.ones(len(generated_curves), dtype=int)

    assert true_curves.shape == generated_curves.shape, \
        f"Shape mismatch: true {true_curves.shape} vs gen {generated_curves.shape}"

    # Default fixed bins up to max length 250
    if bins is None:
        bins = [1, 2, 3, 4]
    else:
        bins = np.asarray(bins)
        assert np.all(np.diff(bins) > 0), "bins must be strictly increasing"

    # Collect metrics
    if metric_fns is None or len(metric_fns) == 0:
        metric_fns = {'rmse': calculate_relative_l2_error, 'fid': compute_fid, 'properties': evaluate_mechanical_properties}
        if len(metric_fns) == 0:
            # Provide a minimal fallback RMSE if nothing is available
            def _rmse(y_true, y_pred):
                y_true = np.asarray(y_true)
                y_pred = np.asarray(y_pred)
                return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            metric_fns = {"rmse": _rmse}

    records = []
    N = caths.shape[0]

    # Evaluate per bin
    for i in range(len(bins)):
        left, right = bins[i], bins[i]
        mask = caths == str(bins[i])  # (left, right]
        idx = np.where(mask)[0]
        if idx.size == 0:
            # Gracefully handle empty bins
            rec = {"bin_left": int(left), "bin_right": int(right), "count": 0}
            for mname in metric_fns:
                rec[mname] = np.nan
            records.append(rec)
            continue

        y_true_bin = true_curves[idx]
        y_pred_bin = generated_curves[idx]

        rec = {"bin_left": int(left), "bin_right": int(right), "count": int(idx.size)}
        for mname, mfn in metric_fns.items():
            try:
                score = mfn(y_true_bin, y_pred_bin)
                # convert tensors to float if your metric returns torch tensors
                if hasattr(score, "item"):
                    score = score.item()
                if mname == 'properties':
                    rec['f_jsd'] = score['max_force']['Jensen-Shannon Divergence']
                    rec['E_jsd'] = score['unfolding_energy']['Jensen-Shannon Divergence']
                    rec['true_force'] = score['max_force']['true_vals']
                    rec['pred_force'] = score['max_force']['gen_vals']
                    rec['true_energy'] = score['unfolding_energy']['true_vals'] / 512 * lengths[idx]
                    rec['pred_energy'] = score['unfolding_energy']['gen_vals'] / 512 * lengths[idx]
                else:
                    rec[mname] = float(score)
            except Exception as e:
                rec[mname] = np.nan


        records.append(rec)

    # Optional overall row
    if include_overall:
        rec = {"bin_left": 0, "bin_right": int(bins[-1]), "count": int(N)}
        for mname, mfn in metric_fns.items():
            try:
                score = mfn(true_curves, generated_curves)
                if hasattr(score, "item"):
                    score = score.item()
                if mname == 'properties':
                    print(score)
                    rec['f_jsd'] = score['max_force']['Jensen-Shannon Divergence']
                    rec['E_jsd'] = score['unfolding_energy']['Jensen-Shannon Divergence']
                    rec['true_force'] = score['max_force']['true_vals']
                    rec['pred_force'] = score['max_force']['gen_vals']
                    rec['true_energy'] = score['unfolding_energy']['true_vals'] / 512
                    rec['pred_energy'] = score['unfolding_energy']['gen_vals'] / 512
                else:
                    rec[mname] = float(score)
            except Exception:
                rec[mname] = np.nan
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    # Nice ordering: bins by left edge; 'overall' at the end (already appended)
    return df



def evaluate_mechanical_properties(
    true_curves: np.ndarray,
    generated_curves: np.ndarray,
    property_extraction_params: Dict[str, Any] = None # Parameters for analysis functions
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the accuracy of inferred mechanical properties by comparing
    properties extracted from generated curves to those extracted from true curves.

    Args:
        true_curves (np.ndarray): Array of true F-E curves. Shape (num_samples, curve_length, channels).
        generated_curves (np.ndarray): Array of generated F-E curves. Shape (num_samples, curve_length, channels).
        property_extraction_params (Dict[str, Any], optional): Dictionary of parameters
                                                               for analysis functions like find_force_peaks.
                                                               Defaults to None.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are property names
                                     ('unfolding_energy', 'max_force', 'num_peaks', 'avg_unfolding_force', etc.)
                                     and values are dictionaries containing evaluation metrics
                                     (e.g., {'r2': value, 'mean_abs_error': value}).
                                     Returns empty dict if shapes don't match or extraction fails.
    """
    if true_curves.shape != generated_curves.shape:
        logging.error(f"Shape mismatch for property evaluation: {true_curves.shape} vs {generated_curves.shape}")
        return {}
    num_samples = true_curves.shape[0]
    true_properties: Dict[str, List[float]] = {}
    gen_properties: Dict[str, List[float]] = {}

    # Default extraction parameters if none provided
    default_extraction_params = {
        'find_peaks': {'height': 0, 'distance': 50, 'prominence': 0.02} # Add your typical values
        # Add default params for WLC fitting if you evaluate those
    }
    if property_extraction_params is None:
        property_extraction_params = default_extraction_params

    #logging.info(f"Extracting and evaluating mechanical properties for {num_samples} samples...")

    for i in range(num_samples):
        true_curve_1d = true_curves[i, :, 0] # Assuming 1 channel (force)
        gen_curve_1d = generated_curves[i, :, 0]

        # --- Extract properties from True Curve ---
        try:
            true_energy = calculate_unfolding_energy(true_curve_1d, extension_step=1.0) # Use a dummy step if physical ext is unknown
            true_max_force = calculate_max_force(true_curve_1d)


            # Store true properties
            if 'unfolding_energy' not in true_properties: true_properties['unfolding_energy'] = []
            if 'max_force' not in true_properties: true_properties['max_force'] = []

            true_properties['unfolding_energy'].append(true_energy)
            true_properties['max_force'].append(true_max_force)


        except Exception as e:
            logging.warning(f"Error extracting properties from true curve {i}: {e}. Skipping this sample for true properties.")
            continue # Skip sample if true property extraction fails (issue with data)


        # --- Extract properties from Generated Curve ---
        try:
            gen_energy = calculate_unfolding_energy(gen_curve_1d, extension_step=1.0) # Use the same dummy step
            gen_max_force = np.max(gen_curve_1d)


            # Store generated properties
            if 'unfolding_energy' not in gen_properties: gen_properties['unfolding_energy'] = []
            if 'max_force' not in gen_properties: gen_properties['max_force'] = []


            gen_properties['unfolding_energy'].append(gen_energy)
            gen_properties['max_force'].append(gen_max_force)


        except Exception as e:
             logging.warning(f"Error extracting properties from generated curve {i}: {e}. Skipping this sample for generated properties.")
             # Add placeholder NaNs to keep lists aligned if some true properties were extracted
             for prop_name in true_properties.keys(): # Iterate based on properties successfully extracted from true curves
                  if prop_name not in gen_properties: gen_properties[prop_name] = [] # Initialize if first failure
                  gen_properties[prop_name].append(np.nan) # Add NaN for failed sample


    # --- Evaluate Metrics for Each Property ---
    evaluation_metrics: Dict[str, Dict[str, float]] = {}

    # Ensure that we only evaluate properties for which we have both true and generated values for at least some samples
    #common_properties = set(true_properties.keys()) & set(gen_properties.keys())
    common_properties = ['max_force', 'unfolding_energy']

    for prop_name in common_properties:
        true_vals = np.array(true_properties[prop_name])
        gen_vals = np.array(gen_properties[prop_name])

        # Remove NaNs to only compare valid pairs
        valid_mask = (~np.isnan(true_vals)) & (~np.isnan(gen_vals))
        if not np.any(valid_mask):
             logging.warning(f"No valid pairs for property '{prop_name}'. Skipping evaluation for this property.")
             continue

        true_valid = true_vals[valid_mask]
        gen_valid = gen_vals[valid_mask]

        if len(true_valid) < 2:
             logging.warning(f"Need at least 2 valid samples to calculate R^2 for property '{prop_name}'. Skipping R^2.")
             r2 = np.nan
        else:
             try:
                 r2 = r2_score(true_valid, gen_valid)
             except Exception as e:
                 logging.warning(f"Error calculating R^2 for property '{prop_name}': {e}. Setting to NaN.")
                 r2 = np.nan

        jsd = compute_kde_diff(true_valid, gen_valid)

        # Calculate Mean Absolute Error (MAE)
        #mae = np.mean(np.abs(true_valid - gen_valid))

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((true_valid - gen_valid)**2))

        # Calculate Mean Relative Error (MRE) - be cautious with division by zero/small numbers
        # MRE = np.mean(np.abs(true_valid - gen_valid) / np.abs(true_valid)) # Avoid division by zero

        # Calculate Fréchet Inception Distance (FID)
        #fid = compute_fid(true_valid, gen_valid)


        evaluation_metrics[prop_name] = {
            'r2': r2,
            #'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'Jensen-Shannon Divergence': jsd,
            #'fid': fid,
            #'num_valid_samples': len(true_valid)
            'true_vals': true_vals,
            'gen_vals': gen_vals,
        }

        #logging.info(f"Property '{prop_name}' metrics: R^2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} (N={len(true_valid)})")


    logging.info("Mechanical property evaluation complete.")

    return evaluation_metrics


# Example Usage
if __name__ == "__main__":
    print("--- Testing metrics.py ---")

    # Create dummy true and generated curves
    num_samples = 130
    fe_len = 130
    channels = 1


    true_curves, generated_curves = (np.load('../../scripts/checkpoints/DiffusionModelTrainer/true_curves.npy'),
                         np.load('../../scripts/checkpoints/DiffusionModelTrainer/generated_curves.npy'))



    # --- Test Curve Shape Metrics ---
    print("\n--- Testing Curve Shape Metrics ---")
    r2_score_curves = calculate_r2(true_curves, generated_curves)
    print(f"Overall R^2 for curve shapes: {r2_score_curves:.4f}")

    rel_l2_error = calculate_relative_l2_error(true_curves, generated_curves)
    print(f"Average Relative L2 Error: {rel_l2_error:.4f}")

    kid_mean, kid_var = compute_kid(true_curves, generated_curves)
    print(f"kid_mean: {kid_mean:.4f}, kid_var: {kid_var:.4f}")

    fid = compute_fid(true_curves, generated_curves)
    print(f"fid: {fid:.4f}")


    # --- Test Mechanical Property Evaluation ---
    print("\n--- Testing Mechanical Property Evaluation ---")
    # Need parameters for peak finding for 'num_peaks', 'avg_unfolding_force'
    peak_params = {'height': 0, 'distance': 10, 'prominence': 0.05}
    property_evaluation_metrics = evaluate_mechanical_properties(
        true_curves,
        generated_curves,
        property_extraction_params={'find_peaks': peak_params}
    )

    print("\nMechanical Property Evaluation Metrics:")
    for prop_name, metrics in property_evaluation_metrics.items():
        print(f"Property: {prop_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
