"""
SGWT High/Low Frequency Cutoff Auto-Tuner.

This module implements an algorithm to automatically determine the optimal
`n_high_scales` parameter for the SpectralInputDecoupler.

Algorithm (low-hyperparameter):
1. Fine-grained SGWT decomposition (e.g., M=12 scales).
2. Per-scale diagnostics on reconstructed node signals W_j:
    - Spatial correlation (mean |off-diagonal|).
    - Distribution discrepancy (mean KS statistic across random node pairs).
    - Energy (mean squared amplitude).
3. Cut-point selection without hand-tuned thresholds:
    - Energy quantile: first scale where cumulative energy reaches 90%.
    - Curvature: argmin of the second difference of smoothed correlation.
    - Peak correlation: argmax of smoothed correlation.
    - Final n_high_scales = median of the above candidates (clamped to [0, M]).
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch

# 引用项目内现有的 SGWT 组件
# 假设本文件位?ST CP/conformal_model/scale/ 目录?
from conformal_model.scale.arch.spectral.sgwt import (
    SpectralGraphWaveletTransform, 
    GraphWaveletKernelFactory
)

# 复用 sgwt_probe 中的 KS 统计量计算逻辑
# 如果导入困难，也可以直接?_ks_statistic_2samp 函数复制到此?
def _ks_statistic_2samp(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no p-value)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.sort(x)
    y = np.sort(y)
    n = x.size
    m = y.size
    z = np.sort(np.concatenate([x, y], axis=0))
    fx = np.searchsorted(x, z, side="right") / float(n)
    fy = np.searchsorted(y, z, side="right") / float(m)
    return float(np.max(np.abs(fx - fy)))


def compute_spatial_correlation_metric(data: np.ndarray) -> float:
    """
    计算空间相关性指标：相关矩阵非对角元素的平均绝对值?
    
    Args:
        data: (S, N) numpy array, S=samples, N=nodes.
        
    Returns:
        float: Mean absolute off-diagonal correlation.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (S, N), got shape {data.shape}")
    if data.shape[0] < 2 or data.shape[1] < 2:
        return 0.0

    # 居中?    X_centered = data - data.mean(axis=0, keepdims=True)
    
    # 计算协方? (N, N)
    # 注意：如?S 很小，协方差估计可能不稳定，建议 S >= 2*N
    cov = X_centered.T @ X_centered / float(data.shape[0] - 1)
    
    # 计算标准差用于归一?
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    
    # 避免除以?
    outer_std[outer_std == 0] = 1.0
    
    # 相关系数矩阵
    corr = cov / outer_std
    
    # 提取非对角线元素
    mask = ~np.eye(corr.shape[0], dtype=bool)
    off_diag_corrs = corr[mask]
    
    # 返回绝对值的均?
    # 如果是纯白噪声，此值应接近 1/sqrt(S)
    if off_diag_corrs.size == 0:
        return 0.0
    return float(np.mean(np.abs(off_diag_corrs)))


def compute_distribution_discrepancy_metric(
    data: np.ndarray, num_pairs: int = 200, seed: int = 42
) -> float:
    """
    计算分布不一致性指标：随机节点对之间的平均 KS 统计量?
    
    Args:
        data: (S, N) numpy array.
        num_pairs: number of random node pairs to check.
        
    Returns:
        float: Mean KS statistic.
    """
    rng = np.random.default_rng(seed)
    N = data.shape[1]
    if N < 2 or data.shape[0] < 2:
        return 0.0
    
    ks_values = []
    for _ in range(num_pairs):
        # 随机采样两个不同的节?
        u, v = rng.choice(N, size=2, replace=False)
        d = _ks_statistic_2samp(data[:, u], data[:, v])
        ks_values.append(d)
        
    return float(np.nanmean(ks_values))


def auto_tune_high_freq_cutoff(
    adj_matrix: np.ndarray,
    signal_data: torch.Tensor | np.ndarray,
    max_scales: int = 12,
    kernel_type: str = "mexican_hat",
    max_samples: int = 5000,
    corr_threshold: float | None = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    自动寻找最佳的高频/低频分界?(n_high_scales)?
    
    Args:
        adj_matrix: (N, N) 邻接矩阵?
        signal_data: (B, T, N) ?(S, N) 形状的观测数据?
                     如果?(B, T, N)，将被展平为 (B*T, N)?
        max_scales: 扫描使用的最大尺度数 (M)。建议设大一??10-15)以获得细粒度视图?
        kernel_type: SGWT 核类型，应与模型保持一?("meyer" / "mexican_hat")?
        max_samples: 诊断时最多使用的样本数，避免过大计算成本?
        corr_threshold: (Optional) 判定?结构?的相关性阈值?
                        如果不提供，将使用基于梯度的肘点检测法?
        verbose: 是否打印扫描日志?
        
    Returns:
        result dict: {
            "suggested_n_high_scales": int,
            "metrics": {
                "scales": List[float],
                "spatial_corr": List[float],
                "dist_ks": List[float]
            }
        }
    """
    # 1. 数据预处?
    if isinstance(signal_data, torch.Tensor):
        signal_data = signal_data.detach().cpu().numpy()
        
    if signal_data.ndim == 3:
        # (B, T, N) -> (B*T, N)
        B, T, N = signal_data.shape
        signal_data = signal_data.reshape(B * T, N)
    elif signal_data.ndim == 2:
        N = signal_data.shape[1]
    else:
        raise ValueError(f"Unexpected data shape: {signal_data.shape}")
        
    # 为了计算效率，如果数据量太大，进行下采样
    MAX_SAMPLES = int(max_samples)
    if signal_data.shape[0] > MAX_SAMPLES:
        rng = np.random.default_rng(0)
        idx = rng.choice(signal_data.shape[0], size=MAX_SAMPLES, replace=False)
        signal_data = signal_data[idx]
        
    if verbose:
        print(f"[AutoTune] Initializing SGWT with N={N}, M={max_scales} scales...")
        print(f"[AutoTune] Using {signal_data.shape[0]} samples for diagnostics.")

    # 2. 初始?SGWT
    # 核心逻辑参?spec_decoupled_components.py 中的 SpectralInputDecoupler
    sgwt = SpectralGraphWaveletTransform(adj_matrix)
    factory = GraphWaveletKernelFactory(sgwt.lmax)

    # 与模型保持一致的 kernel_type，默?meyer 紧框?
    g_func, h_func, scales = factory.get_kernels(kernel_type=kernel_type, M=max_scales)
    
    # 准备特征值和特征向量
    evals = np.asarray(sgwt.evals, dtype=np.float32)  # (N,)
    evecs = np.asarray(sgwt.evecs, dtype=np.float32)  # (N, N)
    signal_data = np.asarray(signal_data, dtype=np.float32)
    
    # 将信号变换到谱域: X_hat = X @ V (S, N)
    # 注意: evecs ?(N, N), signal ?(S, N)
    # math: (S, N) x (N, N) -> (S, N)
    x_hat = signal_data @ evecs
    
    metric_corr = []
    metric_ks = []
    metric_energy = []
    
    if verbose:
        print(f"{'Scale Idx':<10} {'Scale Val':<10} {'Spat.Corr':<12} {'Dist.KS':<12} {'Energy':<12}")
        print("-" * 50)

    # 3. 逐层扫描 (Scan)
    # scales 列表是从小到?(Scale 0: 小尺?高频 -> Scale M-1: 大尺?低频)
    for i, s_val in enumerate(scales):
        # 计算该尺度的谱响?g(s, lambda)
        g_resp = np.asarray(g_func(float(s_val), evals), dtype=np.float32)  # (N,)
        
        # 谱域滤波: W_hat = X_hat * g_resp
        w_hat = x_hat * g_resp[np.newaxis, :]
        
        # 逆变换回节点? W = W_hat @ V.T
        w_node = w_hat @ evecs.T # (S, N)
        
        # 计算指标
        c_val = compute_spatial_correlation_metric(w_node)
        d_val = compute_distribution_discrepancy_metric(w_node)
        e_val = float(np.mean(w_node ** 2))

        metric_corr.append(c_val)
        metric_ks.append(d_val)
        metric_energy.append(e_val)
        
        if verbose:
            print(f"{i:<10} {s_val:<10.4f} {c_val:<12.4f} {d_val:<12.4f} {e_val:<12.4e}")

    # 4. 自动判定截断?(Decision Logic)
    # 高频（靠前的尺度）预期相关性低；进入低频后相关性抬升?

    corrs = np.array(metric_corr)
    energy = np.array(metric_energy)

    if corrs.size >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=float)
        corrs_smooth = np.convolve(corrs, kernel, mode="same")
    else:
        corrs_smooth = corrs.copy()

    cut_idx = max_scales // 2  # 默认中分

    if corr_threshold is not None:
        thr = np.where(corrs_smooth > corr_threshold)[0]
        if len(thr) > 0:
            cut_idx = int(thr[0])
    else:
        candidates: list[int] = []

        if energy.size > 0:
            energy_sum = float(np.sum(energy))
            norm_energy = energy / (energy_sum + 1e-12)
            energy_cum = np.cumsum(norm_energy)
            energy_cut = int(np.argmax(energy_cum >= 0.9)) if np.any(energy_cum >= 0.9) else max_scales // 2
            candidates.append(energy_cut)
        else:
            energy_cum = np.array([])

        if corrs_smooth.size >= 2:
            # Transition point: steepest increase in correlation.
            dcorr = np.diff(corrs_smooth)
            dcorr_cut = int(np.argmax(dcorr) + 1)
            candidates.append(dcorr_cut)

        if corrs_smooth.size >= 3:
            second_diff = np.diff(corrs_smooth, n=2)
            curvature_cut = int(np.argmin(second_diff) + 1)
            candidates.append(curvature_cut)

        if candidates:
            cut_idx = int(np.median(candidates))

    # 防御性截断，确保范围合法
    cut_idx = int(max(0, min(max_scales, cut_idx)))

    if verbose:
        print(f"[AutoTune] Detected transition at scale index {cut_idx}.")
        print(f"[AutoTune] Suggestion: n_high_scales = {cut_idx}")
        print(f"  - High Freq (Noise/Exchangeable): Scales 0 to {cut_idx-1}")
        print(f"  - Low Freq (Structure/Backbone): Scales {cut_idx} to {max_scales-1}")

    return {
        "suggested_n_high_scales": cut_idx,
        "metrics": {
            "scales": [float(s) for s in scales],
            "spatial_corr": metric_corr,
            "dist_ks": metric_ks,
            "energy": metric_energy,
        },
        "diagnostics": {
            "energy_cum": [] if energy.size == 0 else [float(x) for x in np.cumsum(energy) / (float(np.sum(energy)) + 1e-12)],
            "kernel_type": str(kernel_type)
        }
    }
