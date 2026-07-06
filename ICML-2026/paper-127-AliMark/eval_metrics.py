#!/usr/bin/env python3
"""Extract evaluation metrics from detection results file."""
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
import os, sys, json

block_size = sys.argv[1] if len(sys.argv) > 1 else "8"
det_file = f"_result/detection/block_size_{block_size}/c4_AliMark_facebook_opt-1.3b.json"

if not os.path.exists(det_file):
    print(f"ERROR: Detection file not found: {det_file}")
    sys.exit(1)

df = pd.read_json(det_file, orient="index")
so = []
sw = {}

for _, r in df.iterrows():
    if r.get("original_result") and isinstance(r["original_result"], dict):
        if "detect_result" in r["original_result"]:
            so.append(r["original_result"]["detect_result"]["score"])
    for col in df.columns:
        if col in ["question", "reference", "original_result", "unwatermarked_result"]:
            continue
        if isinstance(r.get(col), dict) and "detect_result" in r[col]:
            score = r[col]["detect_result"]["score"]
            name = col.replace("_result", "")
            sw.setdefault(name, []).append(score)

print(f"n_original={len(so)}")
results = {}
for attack, scores in sorted(sw.items()):
    if len(scores) == 0:
        continue
    y = [0] * len(so) + [1] * len(scores)
    ys = so + scores
    fpr, tpr, _ = roc_curve(y, ys)
    roc = auc(fpr, tpr)
    def si(x):
        if x < fpr[0]: return tpr[0]
        elif x > fpr[-1]: return tpr[-1]
        return float(np.interp(x, fpr, tpr))
    results[attack] = {
        "AUROC": round(roc * 100, 2),
        "TPR_01pct": round(si(0.001) * 100, 2),
        "TPR_05pct": round(si(0.005) * 100, 2),
        "TPR_1pct": round(si(0.01) * 100, 2),
        "TPR_5pct": round(si(0.05) * 100, 2),
        "n": len(scores),
    }
    print(f"Attack: {attack:<45} n={len(scores):<4} AUROC={roc*100:.2f}% TPR@0.1%={si(0.001)*100:.2f}% TPR@1%={si(0.01)*100:.2f}%")

print("\nJSON output:")
print(json.dumps(results, indent=2))
