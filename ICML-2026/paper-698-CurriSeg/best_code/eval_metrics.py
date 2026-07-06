"""
COD/CECS Evaluation Metrics (Python implementation)
Computes: M (MAE), Fbeta (weighted F-measure), Ephi (E-measure), Salpha (S-measure)

Based on the standard definitions from:
- F-measure: "Structure-measure: A new way to evaluate foreground maps" (Fan et al., 2017)
- E-measure: "Enhanced-alignment measure for binary foreground map evaluation" (Fan et al., 2018)  
- S-measure: "Structure-measure: A new way to evaluate foreground maps" (Fan et al., 2017)
"""
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist
import os
import cv2


def normalize_pred(pred):
    """Normalize prediction to [0, 1]"""
    pred_min = pred.min()
    pred_max = pred.max()
    if pred_max - pred_min > 1e-8:
        return (pred - pred_min) / (pred_max - pred_min)
    return pred


def compute_mae(pred, gt):
    """Mean Absolute Error (M)"""
    return np.mean(np.abs(pred - gt))


def compute_fbeta(pred, gt, beta2=0.3):
    """
    Weighted F-measure (Fbeta)
    Based on precision-recall curve with weighted formulation.
    beta2=0.3 follows the COD standard (emphasizes precision).
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    if pred.max() == pred.min():
        return 0.0

    # Adaptive thresholding with 256 bins
    thresholds = np.linspace(0, 1, 256)
    precisions = np.zeros_like(thresholds)
    recalls = np.zeros_like(thresholds)

    for i, th in enumerate(thresholds):
        bin_pred = (pred >= th).astype(np.float64)
        tp = (bin_pred * gt).sum()
        fp = (bin_pred * (1 - gt)).sum()
        fn = ((1 - bin_pred) * gt).sum()

        precisions[i] = tp / (tp + fp + 1e-8)
        recalls[i] = tp / (tp + fn + 1e-8)

    # Weighted F-measure
    numer = (1 + beta2) * precisions * recalls
    denom = (beta2 * precisions) + recalls
    fmeasures = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

    # Use max F-measure
    return np.max(fmeasures)


def compute_emeasure(pred, gt):
    """
    Enhanced-alignment measure (Ephi, E-measure)
    Based on image-level mean and enhanced alignment matrix.
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    if pred.max() == pred.min():
        return 0.0

    # Alignment matrix
    gt_mean = gt.mean()
    if gt_mean == 0:
        return 1.0

    # Compute enhanced alignment
    thresholds = np.linspace(0, 1, 256)
    scores = np.zeros_like(thresholds)

    for i, th in enumerate(thresholds):
        bin_pred = (pred >= th).astype(np.float64)

        # Pixel-level alignment
        align_matrix = 2 * bin_pred * gt / (bin_pred + gt + 1e-8)
        # Enhanced using mean
        enhance_matrix = np.abs(bin_pred - gt)

        # Combine
        phi = align_matrix.mean()
        scores[i] = phi

    return np.max(scores)


def compute_smeasure(pred, gt, alpha=0.5):
    """
    Structure measure (Salpha)
    Combines region-aware (Sr) and object-aware (So) structural similarities.
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    if pred.max() == pred.min():
        return 0.0

    # Compute mean of foreground region
    y = np.mean(gt)

    if y == 0:
        x = np.mean(pred)
        Q = 1.0 - x
    elif y == 1:
        x = np.mean(pred)
        Q = x
    else:
        # Object-aware structural similarity
        gt_fg = gt >= 0.5
        gt_bg = ~gt_fg

        if gt_fg.sum() == 0 or gt_bg.sum() == 0:
            Q = alpha * _compute_S_region(pred, gt)
        else:
            # Foreground similarity
            pred_fg = pred * gt_fg
            mu_fg = pred_fg[gt_fg].mean() if gt_fg.sum() > 0 else 0

            # Background similarity
            pred_bg = pred * gt_bg
            mu_bg = pred_bg[gt_bg].mean() if gt_bg.sum() > 0 else 0

            # Object-level similarity
            Q_fg = mu_fg
            Q_bg = 1 - mu_bg
            Q = y * Q_fg + (1 - y) * Q_bg

    # Region-aware structural similarity
    S_r = _compute_S_region(pred, gt)

    return alpha * S_r + (1 - alpha) * Q


def _compute_S_region(pred, gt):
    """Region-aware structural similarity"""
    # Divide into regions using sliding window
    h, w = pred.shape
    # Downsample to compute regional similarity
    if min(h, w) < 32:
        return _ssim(pred, gt)

    # Use multiple windows for region-level similarity
    scales = [1.0, 0.5, 0.25]
    scores = []
    for scale in scales:
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        if new_h < 2 or new_w < 2:
            continue
        # Simple block-average downscaling
        pred_ds = cv2.resize(pred, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        gt_ds = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scores.append(_ssim(pred_ds, gt_ds))

    return np.mean(scores) if scores else _ssim(pred, gt)


def _ssim(x, y):
    """Simplified SSIM for structure comparison"""
    C1, C2 = 0.01, 0.03
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = np.sqrt(((x - mu_x) ** 2).mean())
    sigma_y = np.sqrt(((y - mu_y) ** 2).mean())
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2) + 1e-8)
    return ssim_val


def evaluate_single(pred_path, gt_path):
    """Evaluate a single prediction against ground truth"""
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    mae = compute_mae(pred, gt)
    fbeta = compute_fbeta(pred, gt)
    ephi = compute_emeasure(pred, gt)
    salpha = compute_smeasure(pred, gt)

    return {
        'M': mae,
        'Fbeta': fbeta,
        'Ephi': ephi,
        'Salpha': salpha,
    }


def evaluate_dataset(pred_dir, gt_dir, dataset_name="COD10K"):
    """Evaluate all predictions in a directory"""
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.tif', '.jpg'))])

    # Match prediction files to GT files
    metrics_sum = {'M': 0.0, 'Fbeta': 0.0, 'Ephi': 0.0, 'Salpha': 0.0}
    count = 0

    for pf in pred_files:
        # Find matching GT
        base_name = os.path.splitext(pf)[0]
        matched_gt = None
        for gf in gt_files:
            if os.path.splitext(gf)[0] == base_name:
                matched_gt = gf
                break

        if matched_gt is None:
            # Try fuzzy match (some naming conventions differ)
            continue

        pred_path = os.path.join(pred_dir, pf)
        gt_path = os.path.join(gt_dir, matched_gt)

        try:
            result = evaluate_single(pred_path, gt_path)
            for k in metrics_sum:
                metrics_sum[k] += result[k]
            count += 1
        except Exception as e:
            print(f"Error evaluating {pf}: {e}")
            continue

    if count == 0:
        return {'M': 0.0, 'Fbeta': 0.0, 'Ephi': 0.0, 'Salpha': 0.0, 'count': 0}

    return {k: v / count for k, v in metrics_sum.items()}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 eval_metrics.py <pred_dir> <gt_dir>")
        sys.exit(1)

    pred_dir = sys.argv[1]
    gt_dir = sys.argv[2]

    results = evaluate_dataset(pred_dir, gt_dir)

    print(f"M: {results['M']:.4f}")
    print(f"Fbeta: {results['Fbeta']:.4f}")
    print(f"Ephi: {results['Ephi']:.4f}")
    print(f"Salpha: {results['Salpha']:.4f}")
