"""Use PySODMetrics for correct metric computation."""
import sys, os
import numpy as np
import cv2
import glob

# pip install numpy>=2 first, then compute metrics, then restore
pred_dir = "/repo/res/curriseg/COD10K/"
gt_dir = "/datasets/TestDataset/COD10K/GT/"

pred_files = sorted(glob.glob(pred_dir + "*.png"))
gt_dict = {}
for gf in sorted(glob.glob(gt_dir + "*.png") + glob.glob(gt_dir + "*.tif")):
    gt_dict[os.path.splitext(os.path.basename(gf))[0]] = gf

# Match pairs
masks = []
gts = []
names = []
for pf in pred_files:
    base = os.path.splitext(os.path.basename(pf))[0]
    if base in gt_dict:
        pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        gt = cv2.imread(gt_dict[base], cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        masks.append(pred)
        gts.append(gt)
        names.append(base)

print("Loaded %d pairs" % len(masks))

masks = np.array(masks)
gts = np.array(gts)
print("Shapes: masks=%s, gts=%s" % (str(masks.shape), str(gts.shape)))

# Compute MAE manually
mae = np.abs(masks - gts).mean()
print("MAE: %.6f" % mae)

# Now use PySODMetrics for Fbeta, Ephi, Salpha
from py_sod_metrics import Fmeasure, Emeasure, Smeasure, MAE

# F-measure
FM = Fmeasure()
for i in range(len(masks)):
    FM.step(pred=masks[i], gt=gts[i])
fbeta = FM.get_results()["fm"]
print("Fbeta (adp): %.4f" % fbeta["adp"])
print("Fbeta (max): %.4f" % fbeta["max"])
print("Fbeta (mean): %.4f" % fbeta["mean"])

# E-measure
EM = Emeasure()
for i in range(len(masks)):
    EM.step(pred=masks[i], gt=gts[i])
ephi = EM.get_results()["em"]
print("Ephi (adp): %.4f" % ephi["adp"])
print("Ephi (max): %.4f" % ephi["max"])
print("Ephi (mean): %.4f" % ephi["mean"])

# S-measure  
SM = Smeasure()
for i in range(len(masks)):
    SM.step(pred=masks[i], gt=gts[i])
salpha = SM.get_results()["sm"]
print("Salpha: %.4f" % salpha)

print("\n===== FINAL =====")
print("M:      %.4f" % mae)
print("Fbeta:  %.4f" % fbeta["adp"])
print("Ephi:   %.4f" % ephi["adp"])
print("Salpha: %.4f" % salpha)
