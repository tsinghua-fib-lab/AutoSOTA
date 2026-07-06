"""Compute M, Fbeta, Ephi, Salpha for all datasets."""
import sys, os
sys.path.insert(0, "/repo")
from eval_metrics import evaluate_dataset

BASE_DIR = "/repo/res/curriseg"
DATASETS = ["COD10K", "CAMO", "CHAMELEON"]
GT_BASE = "/datasets/TestDataset"

print("=" * 60)
print("CurriSeg (FEDER+) Evaluation Results")
print("=" * 60)

for ds in DATASETS:
    pred_dir = os.path.join(BASE_DIR, ds)
    gt_dir = os.path.join(GT_BASE, ds, "GT")
    
    if not os.path.isdir(pred_dir):
        print("\n" + ds + ": pred dir not found")
        continue
    
    n_preds = len([f for f in os.listdir(pred_dir) if f.endswith(".png")])
    n_gts = len([f for f in os.listdir(gt_dir) if f.endswith((".png", ".tif"))])
    print("\n" + ds + ": %d preds, %d GTs" % (n_preds, n_gts))
    
    results = evaluate_dataset(pred_dir, gt_dir, ds)
    print("  M:     %.4f" % results["M"])
    print("  Fbeta: %.4f" % results["Fbeta"])
    print("  Ephi:  %.4f" % results["Ephi"])
    print("  Salpha: %.4f" % results["Salpha"])
