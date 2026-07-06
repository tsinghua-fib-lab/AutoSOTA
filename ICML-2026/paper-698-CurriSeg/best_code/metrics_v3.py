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

from py_sod_metrics import FmeasureV2, Emeasure, Smeasure
from py_sod_metrics.fmeasurev2 import FmeasureHandler

FM = FmeasureV2()
fm_handler = FmeasureHandler(beta=0.3)
FM.add_handler("fm", fm_handler)
EM = Emeasure()
SM = Smeasure()
mae_total = 0.0

for i, (pf, gf) in enumerate(pairs):
    pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    FM.step(pred=pred, gt=gt)
    EM.step(pred=pred, gt=gt)
    SM.step(pred=pred, gt=gt)
    mae_total += np.abs(pred - gt).mean()
    
    if (i + 1) % 500 == 0:
        print("  %d/%d" % (i + 1, len(pairs)))

mae = mae_total / len(pairs)
fm = FM.get_results()
em = EM.get_results()
sm = SM.get_results()

print("\n===== FINAL RESULTS =====")
print("M:       %.6f" % mae)
print("Fbeta:   %.4f" % fm["fm"]["fm"])
print("Ephi:    %.4f" % em["em"]["adp"])
print("Salpha:  %.4f" % sm["sm"])
