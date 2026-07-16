import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = "../../result/monkey_analysis_denoise_data"
files = []
for fname in os.listdir(output_dir):
    if not fname.startswith("result_") or not fname.endswith(".pkl"):
        continue
    core = fname[len("result_"):-len(".pkl")]
    model, dataset = core.split("_", 1)
    files.append((model, dataset, os.path.join(output_dir, fname)))

diffs = {}
for model, dataset, path in files:
    with open(path, "rb") as f:
        res = pickle.load(f)
    test_gt = res["test_pass_datk_gts"]
    train_dist = res["train_pass_dist"]
    test_lr = res["test_pass_lr"]
    mse_dist = mean_squared_error(test_gt, train_dist)
    mse_lr   = mean_squared_error(test_gt, test_lr)
    diff = mse_dist - mse_lr
    diffs.setdefault(dataset, {})[model] = diff

diff_df = pd.DataFrame.from_dict(diffs, orient="index")

plt.figure(figsize=(len(diff_df.columns)*0.5 + 3, len(diff_df.index)*0.5 + 3))
sns.set_style("white")
abs_max = np.abs(diff_df.values).max()
ax = sns.heatmap(
    diff_df,
    cmap="bwr",
    center=0,
    vmin=-abs_max,
    vmax= abs_max,
    linewidths=0.5,
    linecolor="lightgray",
    cbar_kws={"label": "Dist MSE - LR MSE"},
    annot=True,
    fmt=".2f",
    annot_kws={"fontsize": 8}
)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Dataset", fontsize=12)
ax.set_title("Difference in Test MSE (Dist - LR)", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("monkey_mse_difference_heatmap_denoise.png", dpi=300, bbox_inches="tight")
