"""Profile different SELECT_LAYER values on a balanced subset."""
import os, sys, random, json, time
import numpy as np
import torch.multiprocessing as mp

os.chdir("/repo/Hide/Qwen2.5")
sys.path.insert(0, ".")

mp.set_start_method("spawn", force=True)

random.seed(2077)
np.random.seed(2077)

# We will dynamically change SELECT_LAYER via file editing
# This script expects SELECT_LAYER to already be set before import
import importlib

from utiles import load_dataset_Vstar_json

MAX_PIXELS = 16384
SIGMA = [3]
THRESHOLD = [0.7]
DATASET = "Vstar_profile_subset.json"
OUTPUT = "profile_results.json"

dataset = load_dataset_Vstar_json(DATASET)
random.shuffle(dataset)
print(f"Profile subset: {len(dataset)} samples")

# Need to re-read SELECT_LAYER for reporting
import modeling_qwen2_5_vl_re_infer as m
print(f"Testing SELECT_LAYER = {m.SELECT_LAYER}")

from inference import cycle_epoch_infer
t0 = time.time()
cycle_epoch_infer(0, 0, dataset, OUTPUT, MAX_PIXELS, SIGMA, THRESHOLD)
elapsed = time.time() - t0
print(f"Inference took {elapsed:.1f}s")

# Compute metrics
from Vstar_Metric import calculate_category_accuracy, read_multi_line_json_objects
df = calculate_category_accuracy(read_multi_line_json_objects(OUTPUT))
col = "HiDe_s3_t0.7_accuracy"
attr_val = float(df[df.category=="direct_attributes"][col].iloc[0].rstrip("%"))
spatial_val = float(df[df.category=="relative_position"][col].iloc[0].rstrip("%"))
avg_val = float(df[df.category=="Overall"][col].iloc[0].rstrip("%"))
print(f"RESULTS: layer={m.SELECT_LAYER} Attr={attr_val:.1f} Spatial={spatial_val:.1f} Avg={avg_val:.1f}")
