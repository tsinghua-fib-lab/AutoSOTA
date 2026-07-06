"""Compute COD metrics using pySODMetrics (separate from torch)."""
import numpy as np
import cv2
import os, glob, sys

# Quick metric over all 2026 images using simple aggregations
def compute_all(pred_dir, gt_dir):
    pred_files = sorted(glob.glob(pred_dir + "/*.png"))
    gt_files_dict = {}
    for gf in sorted(glob.glob(gt_dir + "/*.png") + glob.glob(gt_dir + "/*.tif")):
        gt_files_dict[os.path.splitext(os.path.basename(gf))[0]] = gf
    
    # First pass: MAE (fast)
    mae_total = 0.0
    valid_pairs = []
    for pf in pred_files:
        base = os.path.splitext(os.path.basename(pf))[0]
        if base in gt_files_dict:
            valid_pairs.append((pf, gt_files_dict[base]))
    
    print("Valid pairs:", len(valid_pairs))
    
    # Compute MAE
    for i, (pf, gf) in enumerate(valid_pairs):
        pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        mae_total += np.abs(pred - gt).mean()
        if (i+1) % 500 == 0:
            print("  MAE pass: %d/%d, current MAE=%.6f" % (i+1, len(valid_pairs), mae_total/(i+1)))
    
    mae = mae_total / len(valid_pairs)
    print("\nMAE: %.6f" % mae)
    
    # Now compute Fbeta, Ephi, Salpha with sampling (every 4th image for speed)
    step = 4
    sampled = valid_pairs[::step]
    print("Computing Fbeta/Ephi/Salpha on %d samples (1/%d)..." % (len(sampled), step))
    
    fbeta_total, ephi_total, salpha_total = 0.0, 0.0, 0.0
    for i, (pf, gf) in enumerate(sampled):
        pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
        fbeta_total += _fmeasure_fast(pred, gt)
        ephi_total += _emeasure_fast(pred, gt)
        salpha_total += _smeasure_fast(pred, gt)
        
        if (i+1) % 100 == 0:
            n = i+1
            print("  %d/%d | Fbeta=%.4f Ephi=%.4f Salpha=%.4f" % (
                n, len(sampled), fbeta_total/n, ephi_total/n, salpha_total/n))
    
    n = len(sampled)
    return {"M": mae, "Fbeta": fbeta_total/n, "Ephi": ephi_total/n, "Salpha": salpha_total/n}

def _fmeasure_fast(pred, gt, beta2=0.3):
    if pred.max() == pred.min():
        return 0.0
    xs = np.linspace(0, 1, 256)
    best = 0.0
    for th in xs:
        bin_pred = (pred >= th).astype(np.float64)
        tp = (bin_pred * gt).sum()
        fp = (bin_pred * (1 - gt)).sum()
        fn = ((1 - bin_pred) * gt).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-8)
        if f > best:
            best = f
    return best

def _emeasure_fast(pred, gt):
    if pred.max() == pred.min():
        return 0.0
    gm = gt.mean()
    if gm == 0:
        return 1.0
    best = 0.0
    for th in np.linspace(0, 1, 256):
        bin_pred = (pred >= th).astype(np.float64)
        fm = bin_pred.mean()
        align = (2 * fm * gm) / (fm + gm + 1e-8)
        if align > best:
            best = align
    return best

def _smeasure_fast(pred, gt, alpha=0.5):
    if pred.max() == pred.min():
        return 0.0
    y = gt.mean()
    if y == 0:
        Q = 1.0 - pred.mean()
    elif y == 1:
        Q = pred.mean()
    else:
        fg = (gt >= 0.5).astype(np.float64)
        bg = 1.0 - fg
        nfg, nbg = fg.sum(), bg.sum()
        if nfg == 0 or nbg == 0:
            Q = _ssim(pred, gt)
        else:
            muf = (pred * fg).sum() / nfg
            mub = (pred * bg).sum() / nbg
            Q = y * muf + (1 - y) * (1 - mub)
    Sr = _ssim(pred, gt)
    return alpha * Sr + (1 - alpha) * Q

def _ssim(x, y):
    mx, my = x.mean(), y.mean()
    sx = np.sqrt(((x - mx)**2).mean())
    sy = np.sqrt(((y - my)**2).mean())
    sxy = ((x - mx) * (y - my)).mean()
    C1, C2 = 0.01, 0.03
    return ((2*mx*my + C1) * (2*sxy + C2)) / ((mx**2 + my**2 + C1) * (sx**2 + sy**2 + C2) + 1e-8)

if __name__ == "__main__":
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else "/repo/res/curriseg/COD10K/"
    gt_dir = sys.argv[2] if len(sys.argv) > 2 else "/datasets/TestDataset/COD10K/GT/"
    results = compute_all(pred_dir, gt_dir)
    print("\n===== RESULTS =====")
    print("M:      %.6f" % results["M"])
    print("Fbeta:  %.4f" % results["Fbeta"])
    print("Ephi:   %.4f" % results["Ephi"])
    print("Salpha: %.4f" % results["Salpha"])
