import os
import time
from datetime import datetime
from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
import torch
from dysts.metrics import compute_metrics
from gluonts.itertools import batcher
from scaleformer.scaleformer.dataset import TimeSeriesDataset
from scaleformer.scaleformer.pipeline import PatchTSTPipeline
from scaleformer.utils import safe_standardize
from tqdm import tqdm
from scipy.signal import welch
import dysts.flows as flows
from scipy.spatial.distance import cdist

from dysts.metrics import estimate_kl_divergence
from dysts.analysis import gp_dim
import json
import scaleformer.scaleformer.scaleformer as model_script

from scipy.stats import spearmanr, sem, t
from scipy.signal.windows import hann

def _format_ci(values: list | np.ndarray, confidence: float = 0.95) -> str:
    data = np.array(values)
    data = data[np.isfinite(data)]
    n = len(data)
    
    if n < 2:
        return "N/A"
    
    mean_val = np.mean(data)
    std_err = sem(data)
    
    h = std_err * t.ppf((1 + confidence) / 2., n - 1)
    
    return f"{mean_val:.4f} ± {h:.4f}"

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
        mse_per_sample = np.mean(errors ** 2, axis=(1, 2))
        results["mse"] = mse_per_sample.tolist()

    if "mae" in metric_list or "MAE" in metric_list:
        mae_per_sample = np.mean(abs_errors, axis=(1, 2))
        results["mae"] = mae_per_sample.tolist()
        
    if "smape" in metric_list or "sMAPE" in metric_list:
        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
        pointwise_smape = 2.0 * abs_errors / denominator
        smape_per_sample = 100.0 * np.mean(pointwise_smape, axis=(1, 2))
        results["smape"] = smape_per_sample.tolist()

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

def max_lyapunov_exponent_rosenstein_multivariate(
    data: np.ndarray,
    lag: int | None = None,
    min_tsep: int | None = None,
    tau: float = 1,
    trajectory_len: int = 64,
    fit: str = "RANSAC",
    fit_offset: int = 0,
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
        if min_tsep > max_tsep_factor * n:
            min_tsep = int(max_tsep_factor * n)

    orbit = data
    m = len(orbit)
    dists = cdist(orbit, orbit, metric="euclidean")

    mask = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :]) < min_tsep
    dists[mask] = float("inf")

    ntraj = m - trajectory_len + 1
    min_traj = min_tsep * 2 + 2

    if ntraj <= 0:
        raise ValueError(
            f"Not enough data points. Need {-ntraj + 1} additional data points."
        )
    if ntraj < min_traj:
        raise ValueError(
            f"Not enough data points. Need {min_traj} trajectories, but only {ntraj} could be created."
        )

    nb_idx = np.argmin(dists[:ntraj, :ntraj], axis=1)

    div_traj = np.zeros(trajectory_len, dtype=float)
    for k in range(trajectory_len):
        indices = (np.arange(ntraj) + k, nb_idx + k)
        div_traj_k = dists[indices]
        nonzero = np.where(div_traj_k != 0)
        div_traj[k] = (
            -np.inf if len(nonzero[0]) == 0 else np.mean(np.log(div_traj_k[nonzero]))
        )

    ks = np.arange(trajectory_len)
    finite = np.where(np.isfinite(div_traj))
    ks = ks[finite]
    div_traj = div_traj[finite]

    if len(ks) < 1:
        return -np.inf
    
    if fit == "RANSAC":
        try:
            from sklearn.linear_model import RANSACRegressor
            model = RANSACRegressor(random_state=0)
            model.fit(ks[fit_offset:, None], div_traj[fit_offset:, None])
            le = model.estimator_.coef_[0][0] / tau
            return le
        except ImportError:
            print("RANSAC fit failed, falling back to polyfit. Please install scikit-learn for RANSAC.")

    poly = np.polyfit(ks[fit_offset:], div_traj[fit_offset:], 1)
    le = poly[0] / tau
    return le

def _get_system_dt(system_name: str) -> float:
    STEPS_PER_PERIOD = 128.0

    try:
        system_name_without_pp = system_name.split("_pp")[0]
        
        is_skew = "_" in system_name_without_pp
        
        if is_skew:
            driver_name, response_name = system_name_without_pp.split("_")
            driver_system = getattr(flows, driver_name)()
            response_system = getattr(flows, response_name)()
            period = max(driver_system.period, response_system.period)
        else:
            sys = getattr(flows, system_name_without_pp)()
            period = sys.period
        
        if STEPS_PER_PERIOD <= 0 or period <= 0:
             raise ValueError(f"无效的 period ({period}) 或 steps ({STEPS_PER_PERIOD})")
             
        avg_dt = period / STEPS_PER_PERIOD
        
        return avg_dt
        
    except Exception as e:
        print(f"无法从 dysts.flows 获取 '{system_name}' 的 dt: {e}. 回退到 tau=1.0")
        return 1.0

def _calculate_le_metrics_raw(
    trues_batch: np.ndarray, 
    preds_batch: np.ndarray, 
    tau: float,
    trajectory_len: int
) -> tuple[list[float], list[float]]:
    n_samples, horizon, n_dims = trues_batch.shape
    le_trues = []
    le_preds = []
    
    if (horizon - trajectory_len + 1) <= 0:
        return [], []

    for i in range(n_samples):
        try:
            le_true = max_lyapunov_exponent_rosenstein_multivariate(
                trues_batch[i], tau=tau, trajectory_len=trajectory_len, fit="polyfit"
            )
            le_trues.append(le_true)
        except Exception:
            le_trues.append(np.nan)

        try:
            le_pred = max_lyapunov_exponent_rosenstein_multivariate(
                preds_batch[i], tau=tau, trajectory_len=trajectory_len, fit="polyfit"
            )
            le_preds.append(le_pred)
        except Exception:
            le_preds.append(np.nan)
            
    return le_trues, le_preds

def _calculate_psd_metrics_raw(trues_batch, preds_batch, metric_type, fs=1.0, nperseg=None):
    n_samples, horizon, n_space = trues_batch.shape
    results = []
    epsilon = 1e-8

    if horizon >= 16 and n_space >= 16:
        win_t = hann(horizon)
        win_x = hann(n_space)
        window_2d = np.outer(win_t, win_x)
    else:
        window_2d = np.ones((horizon, n_space))

    scale_factor = 1.0 / (np.sum(window_2d**2) + epsilon)

    for i in range(n_samples):
        data_true = trues_batch[i, :, :]
        data_pred = preds_batch[i, :, :]

        fft_true = np.fft.fft2(data_true * window_2d)
        fft_pred = np.fft.fft2(data_pred * window_2d)

        P_true = np.abs(np.fft.fftshift(fft_true))**2 * scale_factor
        P_pred = np.abs(np.fft.fftshift(fft_pred))**2 * scale_factor

        if metric_type == "ME-LRw":
            total_energy = np.sum(P_true) + epsilon
            w = P_true / total_energy

            log_diff = np.abs(np.log((P_pred + epsilon) / (P_true + epsilon)))
            
            lrw = np.sum(w * log_diff)
            results.append(lrw)
        
        else:
            results.append(np.nan)

    return results

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

def _calculate_gpdim_raw(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    diffs = []
    for i in range(y_true.shape[0]):
        try:
            dim_true = gp_dim(y_true[i])
            dim_pred = gp_dim(y_pred[i])
            diffs.append(dim_true - dim_pred) 
        except:
            diffs.append(np.nan)
    return diffs

def _calculate_kld_raw(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    klds = []
    for i in tqdm(range(y_true.shape[0]), desc="Calc KLD", leave=False, ncols=80):
        try:
            kl = estimate_kl_divergence(y_true[i], y_pred[i])
            if kl is not None and np.isfinite(kl):
                klds.append(kl)
            else:
                klds.append(np.nan)
        except:
            klds.append(np.nan)
    return klds

def evaluate_mlm_model(
    pipeline: PatchTSTPipeline,
    systems: dict[str, TimeSeriesDataset],
    batch_size: int,
    metric_names: list[str] | None = None,
    undo_normalization: bool = False,
    return_processed_past_values: bool = False,
    return_masks: bool = False,
    return_completions: bool = False,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[int, dict[str, dict[str, float]]],
]:
    assert pipeline.mode == "pretrain", "Model must be in pretrain mode"
    system_completions = {}
    system_processed_past_values = {}
    system_timestep_masks = {}
    system_metrics = defaultdict(dict)
    context_length = pipeline.model.config.context_length
    for system in tqdm(systems, desc="Evaluating MLM pretrain model"):
        dataset = systems[system]
        all_completions = []
        all_processed_past_values = []
        all_timestep_masks = []
        for i, batch in enumerate(batcher(dataset, batch_size=batch_size)):
            past_values = [data["past_values"] for data in batch]
            past_batch = torch.stack(past_values, dim=0).to(pipeline.device)
            completions_output = pipeline.model.generate_completions(
                past_batch, past_observed_mask=None
            )
            completions = (
                completions_output.completions.reshape(
                    past_batch.shape[0], past_batch.shape[-1], -1
                )
                .detach()
                .cpu()
                .numpy()
                .transpose(0, 2, 1)
            )
            patch_size = completions_output.completions.shape[-1]
            if completions_output.mask is None:
                raise ValueError("Mask is None")
            patch_mask = completions_output.mask.detach().cpu().numpy()
            timestep_mask = np.repeat(patch_mask, repeats=patch_size, axis=2)
            if completions_output.patched_past_values is None:
                raise ValueError("Patched past values are None")
            processed_past_values = (
                completions_output.patched_past_values.reshape(
                    past_batch.shape[0], past_batch.shape[-1], -1
                )
                .detach()
                .cpu()
                .numpy()
                .transpose(0, 2, 1)
            )
            if undo_normalization:
                if completions_output.loc is None or completions_output.scale is None:
                    raise ValueError("Loc or scale is None")
                loc = completions_output.loc.detach().cpu().numpy()
                scale = completions_output.scale.detach().cpu().numpy()
                completions = completions * scale + loc
                processed_past_values = processed_past_values * scale + loc
            if metric_names is not None:
                eval_metrics = compute_metrics(
                    completions,
                    processed_past_values,
                    include=metric_names,
                )
                for metric, value in eval_metrics.items():
                    system_metrics[system][metric] += (
                        value - system_metrics[system][metric]
                    ) / (i + 1)
            if return_completions:
                all_completions.append(completions)
            if return_processed_past_values:
                all_processed_past_values.append(processed_past_values)
            if return_masks:
                all_timestep_masks.append(timestep_mask)
        if return_completions:
            full_completion = np.concatenate(all_completions, axis=0)
            system_completions[system] = full_completion.transpose(0, 2, 1)
        if return_processed_past_values:
            full_processed_past_values = np.concatenate(
                all_processed_past_values, axis=0
            )
            system_processed_past_values[system] = full_processed_past_values.transpose(
                0, 2, 1
            )
        if return_masks:
            full_timestep_masks = np.concatenate(all_timestep_masks, axis=0)
            system_timestep_masks[system] = full_timestep_masks
    return (
        system_completions if return_completions else None,
        system_processed_past_values if return_processed_past_values else None,
        system_timestep_masks if return_masks else None,
        {context_length: system_metrics},
    )


def _calculate_metric_on_samples(
    y_true: np.ndarray, y_pred: np.ndarray, horizon: int, metric_fn: Callable
) -> Tuple[float, float]:
    n_samples = y_true.shape[0]
    if y_true.shape[1] < horizon:
        return np.nan, np.nan
    metric_values = [
        metric_fn(y_true[i, :horizon], y_pred[i, :horizon])
        for i in range(n_samples)
    ]
    return np.mean(metric_values), np.std(metric_values)

def _calculate_gpdim_with_confidence(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    assert y_true.shape == y_pred.shape and len(y_true.shape) == 3
    dim_diffs = []
    for i in range(y_true.shape[0]):
        try:
            dim_true = gp_dim(y_true[i])
            dim_pred = gp_dim(y_pred[i])
            dim_diffs.append(dim_true - dim_pred)
        except Exception:
            continue
    if not dim_diffs:
        return np.nan, np.nan
    diffs_arr = np.array(dim_diffs)
    rmse = np.sqrt(np.mean(np.square(diffs_arr)))
    std_dev = np.std(diffs_arr)
    return rmse, std_dev

def _calculate_kld_with_confidence(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    batch_size = y_true.shape[0]
    klds = []
    for i in tqdm(range(batch_size), desc="  Calculating KLD per sample", leave=False, ncols=80):
        try:
            kl = estimate_kl_divergence(y_true[i], y_pred[i])
            if kl is not None and np.isfinite(kl):
                klds.append(kl)
        except Exception:
            continue
    return np.mean(klds) if klds else np.nan, np.std(klds) if klds else np.nan

def calculate_and_print_extended_metrics(labels: np.ndarray, predictions: np.ndarray):
    print("\n--- Extended Metrics (mean ± std) ---")
    metric_definitions = {
        "mse_t1":         (lambda true, pred: np.mean((true - pred)**2), 1),
        "mae_t1":         (lambda true, pred: np.mean(np.abs(true - pred)), 1),
        "relative_l2_t1": (lambda true, pred: np.linalg.norm(true - pred) / (np.linalg.norm(true) + 1e-8), 1),
        "smape_t128":     (lambda true, pred: 100 * np.mean(np.abs(pred - true) / (np.abs(true) + np.abs(pred) + 1e-8)), 128),
        "smape_t512":     (lambda true, pred: 100 * np.mean(np.abs(pred - true) / (np.abs(true) + np.abs(pred) + 1e-8)), 512),
    }
    for name, (metric_fn, horizon) in metric_definitions.items():
        mean, std = _calculate_metric_on_samples(labels, predictions, horizon, metric_fn)
        if not np.isnan(mean):
            print(f"  - {name:<15}: {mean:.4f} ± {std:.4f}")
        else:
            print(f"  - {name:<15}: Not Computable (prediction length < {horizon})")
    horizon_512 = 512
    if labels.shape[1] >= horizon_512:
        labels_512 = labels[:, :horizon_512, :]
        preds_512 = predictions[:, :horizon_512, :]
        gpdim_mean, gpdim_std = _calculate_gpdim_with_confidence(labels_512, preds_512)
        if not np.isnan(gpdim_mean):
            print(f"  - {'gpdim_rmse_t512':<15}: {gpdim_mean:.4f} ± {gpdim_std:.4f}")
        else:
            print(f"  - {'gpdim_rmse_t512':<15}: Calculation Failed")
    else:
        print(f"  - {'gpdim_rmse_t512':<15}: Not Computable (prediction length < {horizon_512})")
    if labels.shape[1] >= horizon_512:
        labels_512 = labels[:, :horizon_512, :]
        preds_512 = predictions[:, :horizon_512, :]
        kld_mean, kld_std = _calculate_kld_with_confidence(labels_512, preds_512)
        if not np.isnan(kld_mean):
            print(f"  - {'kld_t512':<15}: {kld_mean:.4f} ± {kld_std:.4f}")
        else:
             print(f"  - {'kld_t512':<15}: Calculation Failed")
    else:
        print(f"  - {'kld_t512':<15}: Not Computable (prediction length < {horizon_512})")
    print("---------------------------------------")


def evaluate_forecasting_model(
    pipeline: PatchTSTPipeline,
    systems: dict[str, TimeSeriesDataset],
    batch_size: int,
    prediction_length: int,
    metric_names: list[str] | None = None,
    parallel_sample_reduction_fn: Callable | None = None,
    channel_sampler: Callable | None = None,
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
    prediction_kwargs: dict | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[int, dict[str, dict[str, str]]],
]:
    assert pipeline.mode == "predict", "Model must be in predict mode"
    system_predictions = {}
    system_contexts = {}
    system_labels = {}
    system_metrics = defaultdict(dict)
    prediction_kwargs = prediction_kwargs or {}
    
    expert_vector_root_dir = "expert_vectors_tsne_data"
    os.makedirs(expert_vector_root_dir, exist_ok=True)
    
    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))
        
    max_horizon_len = 0
    if eval_subintervals:
        max_horizon_len = max((end - start) for start, end in eval_subintervals)

    if parallel_sample_reduction_fn is None:
        parallel_sample_reduction_fn = lambda x: x

    with torch.cuda.device(pipeline.device):
        for system in tqdm(systems, desc="Forecasting..."):
            dataset = systems[system]
            predictions, labels, contexts, future_values = [], [], [], []

            system_tau = _get_system_dt(system)

            for batch_idx, batch in enumerate(batcher(dataset, batch_size=batch_size)):
                past_values, future_values = zip(
                    *[(data["past_values"], data["future_values"]) for data in batch]
                )
                past_batch = torch.stack(past_values, dim=0).to(pipeline.device)
                
                preds = (
                    pipeline.predict(
                        past_batch, prediction_length=prediction_length, **prediction_kwargs
                    )
                    .transpose(0, 1)
                    .cpu()
                    .numpy()
                )
                
                context = past_batch.cpu().numpy()
                future_batch = torch.stack(future_values, dim=0).cpu().numpy()

                if preds.shape[2] > future_batch.shape[1]:
                    preds = preds[..., : future_batch.shape[1], :]
                
                if channel_sampler is not None:
                    future_batch = channel_sampler(
                        torch.from_numpy(future_batch), resample_inds=False
                    ).numpy()
                    context = channel_sampler(
                        torch.from_numpy(context), resample_inds=False
                    ).numpy()

                if redo_normalization:
                    preds = safe_standardize(preds, context=context[None, :, :], axis=2)
                    future_batch = safe_standardize(future_batch, context=context, axis=1)
                    context = safe_standardize(context, axis=1)

                labels.append(future_batch)
                predictions.append(preds)
                contexts.append(context)

            predictions = np.concatenate(predictions, axis=1)
            predictions = parallel_sample_reduction_fn(predictions)
            labels = np.concatenate(labels, axis=0)
            contexts = np.concatenate(contexts, axis=0)

            if metric_names is not None:
                standard_keys = ["mse", "mae", "smape", "spearman", "coefficient_of_variation"]
                
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

                    nperseg = min(128, end - start)
                    if "ME-APE" in metric_names:
                        raw_ape = _calculate_psd_metrics_raw(trues_sub, preds_sub, "ME-APE", nperseg=nperseg)
                        system_metrics[horizon_key][system]["ME-APE"] = _format_ci(raw_ape)
                        
                    if "ME-LRw" in metric_names:
                        raw_lrw = _calculate_psd_metrics_raw(trues_sub, preds_sub, "ME-LRw", nperseg=nperseg)
                        system_metrics[horizon_key][system]["ME-LRw"] = _format_ci(raw_lrw)

                    if "max_lyap_gt" in metric_names or "max_lyap_pred" in metric_names:
                        traj_len = 32 if (end - start) <= 128 else 128
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
                        else:
                            pass

            if return_predictions:
                system_predictions[system] = predictions.transpose(0, 2, 1)
            if return_contexts:
                system_contexts[system] = contexts.transpose(0, 2, 1)
            if return_labels:
                system_labels[system] = labels.transpose(0, 2, 1)
                
    model_script.CURRENT_TEST_SYSTEM = None

    return (
        system_predictions if return_predictions else None,
        system_contexts if return_contexts else None,
        system_labels if return_labels else None,
        system_metrics,
    )