"""HiDe evaluation on V*Bench for Qwen2.5-VL 3B.
Reproduces: Attr=85.2, Spatial=61.8, Avg=75.9 on V*Bench (191 samples).
Settings: temperature=0.0, max_visual_tokens=16384, min_visual_tokens=256.
"""
# Clean old results to prevent accumulation across runs
import os, glob
for f in glob.glob("Vstar_results*.json"):
    os.remove(f)
print("Cleaned old results files")

import os, sys, random
import numpy as np
import torch.multiprocessing as mp

os.chdir("/repo/Hide/Qwen2.5")
sys.path.insert(0, ".")

mp.set_start_method("spawn", force=True)

# Reproducibility
random.seed(2077)
np.random.seed(2077)

from inference import cycle_epoch_infer
from utiles import load_dataset_Vstar_json

# Paper settings (Section 5.1)
MAX_PIXELS = 16384
SIGMA = [2]  # sharper attention (less smoothing)
THRESHOLD = [0.7]
DATASET = "Vstar.json"
OUTPUT = "Vstar_results.json"

dataset = load_dataset_Vstar_json(DATASET)
random.shuffle(dataset)
print(f"Loaded {len(dataset)} samples")

cycle_epoch_infer(0, 0, dataset, OUTPUT, MAX_PIXELS, SIGMA, THRESHOLD)
print("Inference complete!")

# Compute metrics
from Vstar_Metric import calculate_category_accuracy, read_multi_line_json_objects
df = calculate_category_accuracy(read_multi_line_json_objects(OUTPUT))
print("\n=== V*Bench Results ===")
print(df.T.to_string())
print("\nKey metrics (HiDe, sigma=3, threshold=0.7):")
col = "HiDe_s3_t0.7_accuracy"
attr_val = float(df[df.category=="direct_attributes"][col].iloc[0].rstrip("%"))
spatial_val = float(df[df.category=="relative_position"][col].iloc[0].rstrip("%"))
avg_val = float(df[df.category=="Overall"][col].iloc[0].rstrip("%"))
print(f"  Attr (direct_attributes): {attr_val}%")
print(f"  Spatial (relative_position): {spatial_val}%")
print(f"  Avg (Overall): {avg_val}%")
