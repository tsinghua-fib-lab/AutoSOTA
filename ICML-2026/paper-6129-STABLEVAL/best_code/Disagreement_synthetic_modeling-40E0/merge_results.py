import json
import shutil
import pandas as pd
from pathlib import Path

# Paths
dir1 = Path("results/comparison_ablation_labels3and5")
dir2 = Path("results/comparison_ablation_labels7and9")
out = Path("results/comparison_ablation_labels_all")

# Create output directory
if out.exists():
    shutil.rmtree(out)
out.mkdir(exist_ok=True)

# 1. Merge results.json
with open(dir1 / "results.json") as f:
    r1 = json.load(f)
with open(dir2 / "results.json") as f:
    r2 = json.load(f)

merged = {
    "param": r1["param"],
    "values": r1["values"] + r2["values"],
    "configs": r1["configs"] + r2["configs"],
    "metrics": r1["metrics"] + r2["metrics"],
    "full_configs": r1["full_configs"] + r2["full_configs"],
}
with open(out / "results.json", "w") as f:
    json.dump(merged, f, indent=2)
print("Merged results.json")

# 2. Merge metrics.csv
m1 = pd.read_csv(dir1 / "metrics.csv")
m2 = pd.read_csv(dir2 / "metrics.csv")
pd.concat([m1, m2]).to_csv(out / "metrics.csv", index=False)
print("Merged metrics.csv")

# 3. Merge all_metrics.csv
a1 = pd.read_csv(dir1 / "all_metrics.csv")
a2 = pd.read_csv(dir2 / "all_metrics.csv")
pd.concat([a1, a2]).to_csv(out / "all_metrics.csv", index=False)
print("Merged all_metrics.csv")

# 4. Copy raw folders
(out / "raw").mkdir(exist_ok=True)
for folder in (dir1 / "raw").iterdir():
    if folder.is_dir():
        shutil.copytree(folder, out / "raw" / folder.name)
        print(f"Copied raw/{folder.name}")
for folder in (dir2 / "raw").iterdir():
    if folder.is_dir():
        shutil.copytree(folder, out / "raw" / folder.name)
        print(f"Copied raw/{folder.name}")

print(f"\nDone! Merged to {out}")
print(f"Now run: python scripts/generate_plots.py {out}")