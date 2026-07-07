"""Profile sigma x threshold combinations on balanced subset."""
import os, sys, random, json, time
import numpy as np
import torch.multiprocessing as mp

os.chdir("/repo/Hide/Qwen2.5")
sys.path.insert(0, ".")

mp.set_start_method("spawn", force=True)
random.seed(2077)
np.random.seed(2077)

from utiles import load_dataset_Vstar_json
from inference import cycle_epoch_infer
from Vstar_Metric import calculate_category_accuracy, read_multi_line_json_objects

MAX_PIXELS = 16384
DATASET = "Vstar_profile_subset.json"

# Test multiple sigma x threshold combinations
sigmas = [2, 3, 4, 5]
thresholds = [0.3, 0.5, 0.7, 0.9]

results_summary = []

for sigma in sigmas:
    for threshold in thresholds:
        # Clean old results
        for f in ["profile_results.json"] + [f"profile_results_rank-{r}.json" for r in range(10)]:
            if os.path.exists(f):
                os.remove(f)
        
        dataset = load_dataset_Vstar_json(DATASET)
        random.shuffle(dataset)
        
        t0 = time.time()
        cycle_epoch_infer(0, 0, dataset, "profile_results.json", MAX_PIXELS, [sigma], [threshold])
        elapsed = time.time() - t0
        
        # Need to wait for file writes
        import glob
        results_files = glob.glob("profile_results*.json")
        
        # Merge if multiple rank files
        if len(results_files) > 1:
            from Vstar_Metric import merge_json_files
            data = merge_json_files(len(results_files), 1, "profile_results.json")
        else:
            data = read_multi_line_json_objects("profile_results.json")
        
        if not data:
            print(f"SKIP sigma={sigma} thresh={threshold}: no results")
            continue
            
        df = calculate_category_accuracy(data)
        col = [c for c in df.columns if c.endswith("_accuracy") and c.startswith("HiDe")][0]
        attr_val = float(df[df.category=="direct_attributes"][col].iloc[0].rstrip("%"))
        spatial_val = float(df[df.category=="relative_position"][col].iloc[0].rstrip("%"))
        avg_val = float(df[df.category=="Overall"][col].iloc[0].rstrip("%"))
        
        print(f"sigma={sigma}, thresh={threshold}: Attr={attr_val:.1f} Spatial={spatial_val:.1f} Avg={avg_val:.1f} ({elapsed:.0f}s)")
        results_summary.append({
            "sigma": sigma, "threshold": threshold,
            "Attr": attr_val, "Spatial": spatial_val, "Avg": avg_val, "time": elapsed
        })

print("\n=== Sorted by Avg ===")
for r in sorted(results_summary, key=lambda x: x["Avg"], reverse=True):
    print(f"  sigma={r[\"sigma\"]}, t={r[\"threshold\"]}: Attr={r[\"Attr\"]:.1f} Spatial={r[\"Spatial\"]:.1f} Avg={r[\"Avg\"]:.1f}")

print("\n=== Sorted by Spatial ===")
for r in sorted(results_summary, key=lambda x: x["Spatial"], reverse=True):
    print(f"  sigma={r[\"sigma\"]}, t={r[\"threshold\"]}: Attr={r[\"Attr\"]:.1f} Spatial={r[\"Spatial\"]:.1f} Avg={r[\"Avg\"]:.1f}")

# Save results
with open("sigma_threshold_profile.json", "w") as f:
    json.dump(results_summary, f, indent=2)
