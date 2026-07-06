"""Fast COD metrics computation using vectorized operations."""
import numpy as np
import cv2
import os
import glob

def compute_mae(preds, gts):
    return np.abs(preds - gts).mean()

def compute_fmeasure(pred, gt, beta2=0.3):
    """Fast F-measure using adaptive threshold."""
    if pred.max() == pred.min():
        return 0.0
    prec, recall = np.zeros(256), np.zeros(256)
    for i, th in enumerate(np.linspace(0, 1, 256)):
        bin_pred = (pred >= th).astype(np.float64)
        tp = (bin_pred * gt).sum()
        fp = (bin_pred * (1 - gt)).sum()
        fn = ((1 - bin_pred) * gt).sum()
        prec[i] = tp / (tp + fp + 1e-8)
        recall[i] = tp / (tp + fn + 1e-8)
    f = (1 + beta2) * prec * recall / (beta2 * prec + recall + 1e-8)
    return f.max()

def compute_emeasure(pred, gt):
    """Fast E-measure."""
    if pred.max() == pred.min():
        return 0.0
    gt_mean = gt.mean()
    if gt_mean == 0:
        return 1.0
    scores = []
    for th in np.linspace(0, 1, 256):
        bin_pred = (pred >= th).astype(np.float64)
        fm = bin_pred.mean()
        gm = gt_mean
        align = (2 * fm * gm) / (fm + gm + 1e-8)
        scores.append(align)
    return np.max(scores)

def compute_smeasure(pred, gt, alpha=0.5):
    """Fast S-measure."""
    if pred.max() == pred.min():
        return 0.0
    y = gt.mean()
    if y == 0:
        Q = 1.0 - pred.mean()
    elif y == 1:
        Q = pred.mean()
    else:
        gt_fg = (gt >= 0.5).astype(np.float64)
        gt_bg = 1.0 - gt_fg
        n_fg = gt_fg.sum()
        n_bg = gt_bg.sum()
        if n_fg == 0 or n_bg == 0:
            Q = _ssim(pred, gt)
        else:
            mu_fg = (pred * gt_fg).sum() / max(n_fg, 1)
            mu_bg = (pred * gt_bg).sum() / max(n_bg, 1)
            Q = y * mu_fg + (1 - y) * (1 - mu_bg)
    Sr = _ssim(pred, gt)
    return alpha * Sr + (1 - alpha) * Q

def _ssim(pred, gt):
    C1, C2 = 0.01, 0.03
    mx, my = pred.mean(), gt.mean()
    sx = np.sqrt(((pred - mx) ** 2).mean())
    sy = np.sqrt(((gt - my) ** 2).mean())
    sxy = ((pred - mx) * (gt - my)).mean()
    return ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx**2 + my**2 + C1) * (sx**2 + sy**2 + C2) + 1e-8)

# Main
pred_dir = "/repo/res/curriseg/COD10K/"
gt_dir = "/datasets/TestDataset/COD10K/GT/"

pred_files = sorted(glob.glob(pred_dir + "*.png"))
gt_files = sorted(glob.glob(gt_dir + "*.png") + glob.glob(gt_dir + "*.tif"))
print("Preds:", len(pred_files), "GTs:", len(gt_files))

results = {"M": 0, "Fbeta": 0, "Ephi": 0, "Salpha": 0}
count = 0
for i, pf in enumerate(pred_files):
    name = os.path.basename(pf)
    base = os.path.splitext(name)[0]
    
    # Find matching GT
    matched = None
    for gf in gt_files:
        gb = os.path.splitext(os.path.basename(gf))[0]
        if gb == base:
            matched = gf
            break
    if matched is None:
        continue
    
    pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    gt = cv2.imread(matched, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    results["M"] += np.abs(pred - gt).mean()
    results["Fbeta"] += compute_fmeasure(pred, gt)
    results["Ephi"] += compute_emeasure(pred, gt)
    results["Salpha"] += compute_smeasure(pred, gt)
    count += 1
    
    if (i + 1) % 200 == 0:
        m = results["M"] / count
        print("  %d/%d | M=%.6f Fbeta=%.4f Ephi=%.4f Salpha=%.4f" % (
            i+1, len(pred_files), m,
            results["Fbeta"]/count, results["Ephi"]/count, results["Salpha"]/count))

for k in results:
    results[k] /= count

print("\n===== FINAL METRICS (COD10K, %d images) =====" % count)
print("M:      %.6f" % results["M"])
print("Fbeta:  %.4f" % results["Fbeta"])
print("Ephi:   %.4f" % results["Ephi"])
print("Salpha: %.4f" % results["Salpha"])
