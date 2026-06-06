import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.optim.lr_scheduler
from skimage.metrics import structural_similarity
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO

def set_seed(seed=1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_mat(mat_path: str, key: str) -> torch.Tensor:
    data = sio.loadmat(mat_path)[key]
    return torch.from_numpy(data.astype(np.float32))

def get_mat_key(mat_path: str):
    mat = sio.loadmat(mat_path)
    return next(k for k in mat.keys() if not k.startswith("__"))


def preprocess_inpainting(file_path, obs_ratio, device, key=None):
    """
    Load clean tensor and generate incomplete observation.
    """
    if file_path.endswith('.mat'):
        gt = load_mat(file_path, key)
    else:
        gt = Image.open(file_path).convert("RGB")
        gt = torch.from_numpy(np.array(gt, dtype=np.float32))

    H, W, C = gt.shape

    if C == 3:
        gt = gt.permute(2, 0, 1)
        gt = transforms.Resize((256, 256))(gt)
        gt = gt.permute(1, 2, 0)
    else:
        if (H, W) == (512, 512):
            gt = gt.permute(2, 0, 1)
            gt = transforms.Resize((256, 256))(gt)
            gt = gt.permute(1, 2, 0)
        elif min(H, W) > 256:
            start_h = (H - 256) // 2
            start_w = (W - 256) // 2
            gt = gt[start_h:start_h + 256, start_w:start_w + 256]
        elif H==144 and W==176:
            gt = gt[:, :, :100]
        else:
            M = min(H, W)
            start_h = (H - M) // 2
            start_w = (W - M) // 2
            gt = gt[start_h:start_h + M, start_w:start_w + M]

    gt = gt.to(device).float()
    gt = gt / gt.max()

    H, W, C = gt.shape
    num_pixels = H * W
    num_observed = int(num_pixels * obs_ratio)

    mask = torch.zeros((H, W, C), device=device)

    for c in range(C):
        idx = torch.randperm(num_pixels, device=device)[:num_observed]
        mask[:, :, c].view(-1)[idx] = 1

    observed = gt * mask

    return gt, observed, mask

def add_gaussian_noise(tensor, std):
    noise = torch.randn_like(tensor) * std
    return tensor + noise

def preprocess_denoising(file_path, noise_std, device, key):
    """
    Load clean tensor and generate noisy observation (Gaussian noise).
    """
    gt = load_mat(file_path, key)
    H, W, C = gt.shape

    if (H, W) == (512, 512):
        gt = gt.permute(2, 0, 1)
        gt = transforms.Resize((256, 256))(gt)
        gt = gt.permute(1, 2, 0)

    elif min(H, W) > 256:
        top = (H - 256) // 2
        left = (W - 256) // 2
        gt = gt[top:top + 256, left:left + 256]

    else:
        crop_size = min(H, W)
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
        gt = gt[top:top + crop_size, left:left + crop_size]

    gt = gt.to(device)
    gt = gt / gt.max()
    noisy = add_gaussian_noise(gt, std=noise_std)

    return gt, noisy

def preprocess_super_resolution(image_path, device, scale=4, crop_size=768):
    """
    Generate HR/LR image pair for super-resolution.
    """
    gt = Image.open(image_path).convert("RGB")
    gt = np.array(gt, dtype=np.float32) / 255.0
    H, W, _ = gt.shape

    if min(H, W) >= crop_size:
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
        gt = gt[top:top + crop_size, left:left + crop_size]

    gt_hr = torch.from_numpy(gt).to(device)
    pool = nn.AvgPool2d(kernel_size=scale, stride=scale)
    gt_lr = pool(gt_hr.permute(2, 0, 1).unsqueeze(0))
    gt_lr = gt_lr.squeeze(0).permute(1, 2, 0)

    return gt_hr, gt_lr


def calculate_psnr(recon, gt):
    recon = recon.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse)

def calculate_ssim(recon, gt):
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    if torch.is_tensor(recon):
        recon = recon.detach().cpu().numpy()

    assert gt.shape == recon.shape

    if gt.ndim == 3:
        return np.mean([
            structural_similarity(
                gt[:, :, c],
                recon[:, :, c],
                data_range=1.0
            )
            for c in range(gt.shape[2])
        ])

    elif gt.ndim == 4:
        return np.mean([
            structural_similarity(
                gt[:, :, c, t],
                recon[:, :, c, t],
                data_range=1.0
            )
            for t in range(gt.shape[3])
            for c in range(gt.shape[2])
        ])

    else:
        raise ValueError("Input must be 3D or 4D tensor")


def calculate_nrmse(recon, gt):
    recon = recon.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    rmse = np.sqrt(np.mean((gt - recon) ** 2))
    denom = np.sqrt(np.mean(gt ** 2))

    return rmse / denom

def visualize_pointcloud(x_coord, y_R, y_G, y_B, filename="pointcloud.png",
                             elev=-86, azim=-86, figsize=(6, 6), point_size=5):
    """
    Renders a point cloud with RGB colors and saves it to a file.
    """
    if torch.is_tensor(x_coord):
        x_coord = x_coord.cpu().numpy()
    if torch.is_tensor(y_R):
        y_R = y_R.cpu().numpy()
    if torch.is_tensor(y_G):
        y_G = y_G.cpu().numpy()
    if torch.is_tensor(y_B):
        y_B = y_B.cpu().numpy()

    color = np.stack([y_R, y_G, y_B], axis=1) / 255.0
    color = np.clip(color, 0, 1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(x_coord[:, 0], x_coord[:, 1], x_coord[:, 2], facecolors=color, marker='o', s=point_size)

    ax.view_init(elev=elev, azim=azim)
    ax.dist = 100
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img_data = np.array(img)
    mask = np.any(img_data != 255, axis=-1)
    coords = np.argwhere(mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img_cropped = img.crop((x0, y0, x1, y1))
        img_cropped.save(filename)
    else:
        img.save(filename)

def compute_metrics(gt, pred):
    """
    Computes the Normalized Root Mean Square Error (NRMSE) and R-squared (R^2).
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    gt_val = gt[:, -1].reshape(-1)
    pred_val = pred.reshape(-1)

    rmse = np.sqrt(np.mean((gt_val - pred_val) ** 2))
    denom = np.sqrt(np.mean(gt_val ** 2))
    nrmse = rmse / denom

    ss_res = np.sum((gt_val - pred_val) ** 2)
    ss_tot = np.sum((gt_val - np.mean(gt_val)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    return nrmse, r2
