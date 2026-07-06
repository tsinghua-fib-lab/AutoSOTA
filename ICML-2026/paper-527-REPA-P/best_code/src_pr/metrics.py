import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_nmse(gt, pred):
    """Calculate Normalized Mean Squared Error (NMSE)."""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def calculate_rmse(gt, pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((gt - pred) ** 2))

def calculate_psnr(gt, pred):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    return psnr(gt, pred, data_range=gt.max() - gt.min())

def calculate_ssim(gt, pred):
    """Calculate Structural Similarity Index (SSIM)."""
    return ssim(gt, pred, data_range=gt.max() - gt.min(), channel_axis=0)