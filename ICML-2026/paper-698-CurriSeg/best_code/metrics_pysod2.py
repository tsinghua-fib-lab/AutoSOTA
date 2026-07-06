import sys, os, glob
import numpy as np
import cv2

pred_dir = "/repo/res/curriseg/COD10K/"
gt_dir = "/datasets/TestDataset/COD10K/GT/"

pred_files = sorted(glob.glob(pred_dir + "*.png"))
gt_dict = {}
for gf in sorted(glob.glob(gt_dir + "*.png") + glob.glob(gt_dir + "*.tif")):
    gt_dict[os.path.splitext(os.path.basename(gf))[0]] = gf

# Match pairs (keep as list of arrays of different sizes)
pairs = []
for pf in pred_files:
    base = os.path.splitext(os.path.basename(pf))[0]
    if base in gt_dict:
        pairs.append((pf, gt_dict[base]))

print("Pairs:", len(pairs))

# Compute using py_sod_metrics
from py_sod_metrics import Fmeasure, Emeasure, Smeasure, MAE as MAEclass

FM = Fmeasure(beta=0.3)
EM = Emeasure()
SM = Smeasure()
MAEM = MAEclass()
mae_total = 0.0

for i, (pf, gf) in enumerate(pairs):
    pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    FM.step(pred=pred, gt=gt)
    EM.step(pred=pred, gt=gt)
    SM.step(pred=pred, gt=gt)
    MAEM.step(pred=pred, gt=gt)
    mae_total += np.abs(pred - gt).mean()
    
    if (i + 1) % 500 == 0:
        print("  %d/%d" % (i + 1, len(pairs)))

mae = mae_total / len(pairs)
fbeta = FM.get_results()
ephi = EM.get_results()
salpha = SM.get_results()
mae2 = MAEM.get_results()

print("\n===== FINAL RESULTS (COD10K, %d images) =====" % len(pairs))
print("M (manual): %.6f" % mae)
print("M (pysod):  %.6f" % mae2["mae"])
print("Fbeta:      %.4f" % fbeta["fm"]["adp"])
print("Ephi:       %.4f" % ephi["em"]["adp"])
print("Salpha:     %.4f" % salpha["sm"])
