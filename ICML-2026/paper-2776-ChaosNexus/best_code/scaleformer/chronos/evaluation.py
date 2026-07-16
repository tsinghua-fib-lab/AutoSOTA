import os
import time
import json
from collections import defaultdict
from typing import Callable, Tuple, List

import numpy as np
import torch
from scipy.signal import welch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, sem, t
from tqdm.auto import tqdm

from dysts.metrics import compute_metrics, estimate_kl_divergence 
from dysts.analysis import gp_dim 
import dysts.flows as flows

from gluonts.itertools import batcher
from scaleformer.chronos.dataset import ChronosDataset
from scaleformer.chronos.pipeline import ChronosPipeline
from scaleformer.utils import safe_standardize

def _format_ci(values: list | np.ndarray, confidence: float = 0.95) -> str:
    data = np.array(values)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 2: return "N/A"
    mean_val = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2., n - 1)
    return f"{mean_val:.4f} ± {h:.4f}"

def _format_rmse_ci(diffs: list | np.ndarray, confidence: float = 0.95) -> str:
    diffs = np.array(diffs)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n < 2: return "N/A"
    squared_errors = diffs ** 2
    mse_mean = np.mean(squared_errors)
    mse_stderr = sem(squared_errors)
    rmse_mean = np.sqrt(mse_mean)
    rmse_stderr = 0.0 if rmse_mean < 1e-9 else mse_stderr / (2 * rmse_mean)
    h = rmse_stderr * t.ppf((1 + confidence) / 2., n - 1)
    return f"{rmse_mean:.4f} ± {h:.4f}"

def _compute_standard_metrics_per_sample(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metric_list: list[str]
) -> dict[str, list[float]]:
    n_samples = y_true.shape[0]
    results = defaultdict(list)
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    if "mse" in metric_list or "MSE" in metric_list:
        results["mse"] = np.mean(errors ** 2, axis=(1, 2)).tolist()

    if "mae" in metric_list or "MAE" in metric_list:
        results["mae"] = np.mean(abs_errors, axis=(1, 2)).tolist()
        
    if "smape" in metric_list or "sMAPE" in metric_list:
        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
        pointwise_smape = 2.0 * abs_errors / denominator
        results["smape"] = (100.0 * np.mean(pointwise_smape, axis=(1, 2))).tolist()

    if "spearman" in metric_list or "coefficient_of_variation" in metric_list:
        spearman_vals = []
        for i in range(n_samples):
            flat_true = y_true[i].flatten()
            flat_pred = y_pred[i].flatten()
            if np.std(flat_true) < 1e-9 or np.std(flat_pred) < 1e-9:
                spearman_vals.append(np.nan)
            else:
                corr, _ = spearmanr(flat_true, flat_pred)
                spearman_vals.append(corr)
        results["spearman"] = spearman_vals

    return results

def _calculate_psd_metrics_raw(
    trues_batch: np.ndarray, 
    preds_batch: np.ndarray, 
    metric_type: str,
    fs: float = 1.0, 
    nperseg: int = 128
) -> list[float]:
    n_samples, horizon, n_dims = trues_batch.shape
    if horizon < nperseg: return []
    results = []
    epsilon = np.finfo(float).eps
    welch_params = dict(fs=fs, nperseg=nperseg)

    for i in range(n_samples):
        sample_vals = []
        for d in range(n_dims):
            try:
                _, P_true = welch(trues_batch[i, :, d], **welch_params)
                _, P_pred = welch(preds_batch[i, :, d], **welch_params)

                if metric_type == "ME-APE":
                    power_threshold = np.max(P_true) * 0.01
                    valid_mask = P_true > power_threshold
                    if not np.any(valid_mask): continue
                    ape = np.mean(np.abs((P_pred[valid_mask] - P_true[valid_mask]) / (P_true[valid_mask] + epsilon))) * 100
                    sample_vals.append(ape)
                elif metric_type == "ME-LRw":
                    sum_P_true = np.sum(P_true)
                    weights = P_true / (sum_P_true + epsilon)
                    log_ratio = np.abs(np.log((P_pred + epsilon) / (P_true + epsilon)))
                    lrw = np.sum(weights * log_ratio)
                    sample_vals.append(lrw)
            except: continue
        if sample_vals: results.append(np.mean(sample_vals))
        else: results.append(np.nan)
    return results

def max_lyapunov_exponent_rosenstein_multivariate(
    data: np.ndarray, lag: int | None = None, min_tsep: int | None = None,
    tau: float = 1, trajectory_len: int = 64, fit: str = "polyfit", fit_offset: int = 0,
) -> float:
    data = np.asarray(data, dtype="float32")
    n, d = data.shape
    max_tsep_factor = 0.25
    if lag is None or min_tsep is None:
        f = np.fft.rfft(data[:, 0], n * 2 - 1)
    if min_tsep is None:
        mf = np.fft.rfftfreq(n * 2 - 1) * np.abs(f)
        mf = np.mean(mf[1:]) / np.sum(np.abs(f[1:]))
        min_tsep = int(np.ceil(1.0 / mf))
        if min_tsep > max_tsep_factor * n: min_tsep = int(max_tsep_factor * n)

    orbit = data
    m = len(orbit)
    dists = cdist(orbit, orbit, metric="euclidean")
    mask = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :]) < min_tsep
    dists[mask] = float("inf")

    ntraj = m - trajectory_len + 1
    if ntraj <= 0: return -np.inf

    nb_idx = np.argmin(dists[:ntraj, :ntraj], axis=1)
    div_traj = np.zeros(trajectory_len, dtype=float)
    for k in range(trajectory_len):
        indices = (np.arange(ntraj) + k, nb_idx + k)
        div_traj_k = dists[indices]
        nonzero = np.where(div_traj_k != 0)
        div_traj[k] = -np.inf if len(nonzero[0]) == 0 else np.mean(np.log(div_traj_k[nonzero]))

    ks = np.arange(trajectory_len)
    finite = np.where(np.isfinite(div_traj))
    ks = ks[finite]
    div_traj = div_traj[finite]
    if len(ks) < 2: return -np.inf

    poly = np.polyfit(ks[fit_offset:], div_traj[fit_offset:], 1)
    return poly[0] / tau

def _calculate_le_metrics_raw(
    trues_batch: np.ndarray, preds_batch: np.ndarray, tau: float, trajectory_len: int
) -> tuple[list[float], list[float]]:
    n_samples, horizon, _ = trues_batch.shape
    le_trues, le_preds = [], []
    if (horizon - trajectory_len + 1) <= 0: return [], []

    for i in range(n_samples):
        try:
            le_t = max_lyapunov_exponent_rosenstein_multivariate(trues_batch[i], tau=tau, trajectory_len=trajectory_len)
            le_trues.append(le_t)
        except: le_trues.append(np.nan)
        try:
            le_p = max_lyapunov_exponent_rosenstein_multivariate(preds_batch[i], tau=tau, trajectory_len=trajectory_len)
            le_preds.append(le_p)
        except: le_preds.append(np.nan)
    return le_trues, le_preds

def _calculate_gpdim_raw(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    diffs = []
    for i in range(y_true.shape[0]):
        try:
            dt = gp_dim(y_true[i])
            dp = gp_dim(y_pred[i])
            diffs.append(dt - dp) 
        except: diffs.append(np.nan)
    return diffs

def _calculate_kld_raw(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    klds = []
    for i in tqdm(range(y_true.shape[0]), desc="Calc KLD", leave=False, ncols=80):
        try:
            kl = estimate_kl_divergence(y_true[i], y_pred[i])
            if kl is not None and np.isfinite(kl): klds.append(kl)
            else: klds.append(np.nan)
        except: klds.append(np.nan)
    return klds

def _get_system_dt(system_name: str) -> float:
    STEPS_PER_PERIOD = 128.0 
    try:
        system_name_clean = system_name.split("_pp")[0]
        if "_" in system_name_clean:
            parts = system_name_clean.split("_")
            period = max(getattr(flows, parts[0])().period, getattr(flows, parts[1])().period)
        else:
            period = getattr(flows, system_name_clean)().period
        return period / STEPS_PER_PERIOD
    except:
        return 1.0

def calculate_and_save_per_dimension_metrics(
    labels: np.ndarray, 
    predictions: np.ndarray, 
    system_name: str, 
    output_dir: str = "results_per_dim", 
    filename: str = "metrics_log.txt"
):
    if labels.shape != predictions.shape:
        print(f"\n[Metrics Logging] Skipping '{system_name}': Shape mismatch.")
        return
    try:
        timesteps_to_calculate = [24, 48, 72, 96, 120]
        prediction_length = labels.shape[1]
        num_dimensions = labels.shape[2]
        valid_timesteps = [t for t in timesteps_to_calculate if t <= prediction_length]
        
        if not valid_timesteps: return

        output_lines = [f"{system_name}\n"]
        for d in range(num_dimensions):
            for step in valid_timesteps:
                step_index = step - 1
                labels_step = labels[:, step_index, d]
                preds_step = predictions[:, step_index, d]
                
                mse = np.mean((labels_step - preds_step) ** 2)
                mae = np.mean(np.abs(labels_step - preds_step))
                
                denom = np.abs(labels_step) + np.abs(preds_step)
                smape_terms = np.zeros_like(denom, dtype=float)
                mask = denom != 0
                smape_terms[mask] = 2 * np.abs(preds_step[mask] - labels_step[mask]) / denom[mask]
                smape = 100 * np.mean(smape_terms)

                line = f"[Dimension {d} @ Step {step:03d}] MSE: {mse:.4f}, MAE: {mae:.4f}, sMAPE: {smape:.4f}\n"
                output_lines.append(line)
        output_lines.append('\n')

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), 'a') as f:
            f.writelines(output_lines)
            
    except Exception as e:
        print(f"\n[Metrics Logging] Error for '{system_name}': {e}")

def evaluate_chronos_forecast(
    pipeline: ChronosPipeline,
    systems: dict[str, ChronosDataset],
    batch_size: int,
    prediction_length: int,
    system_dims: dict[str, int],
    metric_names: list[str] | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
    parallel_sample_reduction_fn: Callable | None = None,
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
    prediction_kwargs: dict | None = None,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[int, dict[str, dict[str, float]]],
]:
    system_predictions = {}
    system_contexts = {}
    system_labels = {}
    system_metrics = defaultdict(dict)
    prediction_kwargs = prediction_kwargs or {}
    
    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))
        
    max_horizon_len = 0
    if eval_subintervals:
        max_horizon_len = max((end - start) for start, end in eval_subintervals)

    if parallel_sample_reduction_fn is None:
        parallel_sample_reduction_fn = lambda x: x

    pbar = tqdm(systems, desc="Forecasting...")
    for system in pbar:
        dataset = systems[system]
        num_sys = len(dataset.datasets)
        dim = system_dims[system]
        predictions, labels, contexts, future_values = [], [], [], []

        for batch in batcher(dataset, batch_size=batch_size):
            past_values, future_values_batch = zip(
                *[(data["past_values"], data["future_values"]) for data in batch]
            )

            past_batch = torch.cat(past_values, dim=0)
            context = past_batch.cpu().numpy()

            predict_args = {
                "context": past_batch,
                "prediction_length": prediction_length,
                **prediction_kwargs,
            }
            
            preds = pipeline.predict(**predict_args).transpose(0, 1).cpu().numpy()
            num_samples = preds.shape[0]
            
            future_batch = torch.cat(future_values_batch, dim=0).cpu().numpy()

            if preds.shape[-1] > future_batch.shape[-1]:
                preds = preds[..., : future_batch.shape[-1]]

            if redo_normalization:
                future_batch = safe_standardize(future_batch, context=context)
                preds = safe_standardize(preds, context=context[None, :, :])
                context = safe_standardize(context)

            labels.append(future_batch)
            predictions.append(preds)
            contexts.append(context)

        predictions = (
            np.concatenate(predictions, axis=1)
            .reshape(num_samples, num_sys, dim, -1, prediction_length)
            .transpose(0, 1, 3, 4, 2)
            .reshape(num_samples, -1, prediction_length, dim)
        )
        predictions = parallel_sample_reduction_fn(predictions)
        
        labels = (
            np.concatenate(labels, axis=0)
            .reshape(num_sys, dim, -1, prediction_length)
            .transpose(0, 2, 3, 1)
            .reshape(-1, prediction_length, dim)
        )
        
        contexts = (
            np.concatenate(contexts, axis=0)
            .reshape(num_sys, dim, -1, contexts[0].shape[-1])
            .transpose(0, 2, 3, 1)
            .reshape(-1, contexts[0].shape[-1], dim)
        )

        if metric_names is not None:
            system_tau = _get_system_dt(system)
            
            for start, end in eval_subintervals:
                horizon_key = end - start
                if system not in system_metrics[horizon_key]:
                    system_metrics[horizon_key][system] = {}

                trues_sub = labels[:, start:end, :]
                preds_sub = predictions[:, start:end, :]

                std_results = _compute_standard_metrics_per_sample(
                    trues_sub, preds_sub, metric_names
                )
                for m_name, m_vals in std_results.items():
                    system_metrics[horizon_key][system][m_name] = _format_ci(m_vals)

                nperseg = min(128, horizon_key)
                if "ME-APE" in metric_names:
                    raw_ape = _calculate_psd_metrics_raw(trues_sub, preds_sub, "ME-APE", nperseg=nperseg)
                    system_metrics[horizon_key][system]["ME-APE"] = _format_ci(raw_ape)
                
                if "ME-LRw" in metric_names:
                    raw_lrw = _calculate_psd_metrics_raw(trues_sub, preds_sub, "ME-LRw", nperseg=nperseg)
                    system_metrics[horizon_key][system]["ME-LRw"] = _format_ci(raw_lrw)

                if "max_lyap_gt" in metric_names or "max_lyap_pred" in metric_names:
                    traj_len = 32 if horizon_key <= 128 else 128
                    if horizon_key >= traj_len:
                        raw_le_true, raw_le_pred = _calculate_le_metrics_raw(
                            trues_sub, preds_sub, tau=system_tau, trajectory_len=traj_len
                        )
                        if "max_lyap_gt" in metric_names:
                            system_metrics[horizon_key][system]["max_lyap_gt"] = _format_ci(raw_le_true)
                        if "max_lyap_pred" in metric_names:
                            system_metrics[horizon_key][system]["max_lyap_pred"] = _format_ci(raw_le_pred)

                if "gd_rmse" in metric_names:
                    raw_gpdim_diffs = _calculate_gpdim_raw(trues_sub, preds_sub)
                    system_metrics[horizon_key][system]["gd_rmse"] = _format_rmse_ci(raw_gpdim_diffs)

                if "kld" in metric_names:
                    if horizon_key == max_horizon_len:
                        raw_kld = _calculate_kld_raw(trues_sub, preds_sub)
                        system_metrics[horizon_key][system]["kld"] = _format_ci(raw_kld)

        if return_predictions:
            system_predictions[system] = predictions.transpose(0, 2, 1)
        if return_contexts:
            system_contexts[system] = contexts.transpose(0, 2, 1)
        if return_labels:
            system_labels[system] = labels.transpose(0, 2, 1)

        pbar.set_postfix({"system": system, "num systems": num_sys})

    return (
        system_predictions if return_predictions else None,
        system_contexts if return_contexts else None,
        system_labels if return_labels else None,
        system_metrics,
    )