import sys, os, glob
import numpy as np
import cv2

pred_dir = "/repo/res/curriseg/COD10K/"
gt_dir = "/datasets/TestDataset/COD10K/GT/"

pred_files = sorted(glob.glob(pred_dir + "*.png"))
gt_dict = {}
for gf in sorted(glob.glob(gt_dir + "*.png") + glob.glob(gt_dir + "*.tif")):
    gt_dict[os.path.splitext(os.path.basename(gf))[0]] = gf

pairs = [(pf, gt_dict[os.path.splitext(os.path.basename(pf))[0]]) 
         for pf in pred_files 
         if os.path.splitext(os.path.basename(pf))[0] in gt_dict]
print("Pairs:", len(pairs))

# Use verified implementations
# E-measure and S-measure from py_sod_metrics (old API - works)
from py_sod_metrics import Emeasure as EmeasureOld, Smeasure as SmeasureOld
EM = EmeasureOld()
SM = SmeasureOld()
mae_total = 0.0

# Pre-collect all valid data for F-measure
all_preds = []
all_gts = []

for i, (pf, gf) in enumerate(pairs):
    pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    EM.step(pred=pred, gt=gt)
    SM.step(pred=pred, gt=gt)
    mae_total += np.abs(pred - gt).mean()
    all_preds.append(pred)
    all_gts.append(gt)
    
    if (i + 1) % 500 == 0:
        print("  %d/%d, M=%.6f" % (i + 1, len(pairs), mae_total / (i + 1)))

mae = mae_total / len(pairs)
em = EM.get_results()
sm = SM.get_results()

# Compute F-measure manually with proper algorithm
# Standard weighted F-measure (beta^2 = 0.3)
beta2 = 0.3
precisions = np.zeros(256)
recalls = np.zeros(256)

for t_idx, th in enumerate(np.linspace(0, 1, 256)):
    tp_total, fp_total, fn_total = 0.0, 0.0, 0.0
    for pred, gt in zip(all_preds, all_gts):
        bin_pred = (pred >= th).astype(np.float64)
        tp_total += (bin_pred * gt).sum()
        fp_total += (bin_pred * (1 - gt)).sum()
        fn_total += ((1 - bin_pred) * gt).sum()
    precisions[t_idx] = tp_total / (tp_total + fp_total + 1e-8)
    recalls[t_idx] = tp_total / (tp_total + fn_total + 1e-8)

# Find max F-measure
f_scores = (1 + beta2) * precisions * recalls / (beta2 * precisions + recalls + 1e-8)
# Fix any NaN
f_scores = np.nan_to_num(f_scores, nan=0.0)
fbeta = f_scores.max()

print("\n===== FINAL RESULTS =====")
print("M:       %.6f" % mae)
print("Fbeta:   %.4f" % fbeta)
print("Ephi:    %.4f" % em["em"]["adp"])
print("Salpha:  %.4f" % sm["sm"])
