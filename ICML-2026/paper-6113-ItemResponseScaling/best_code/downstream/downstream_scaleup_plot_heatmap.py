import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

split_method = "hardeasy_split"
with open(f"downstream_data_{split_method}.pkl", "rb") as f:
    results_dict = pickle.load(f)

rows = []
for scenario, model_results in results_dict.items():
    models = sorted(model_results.keys())
    diffs = []
    for m in models:
        res = model_results[m]
        linear_mse = mean_squared_error(res["gt_ctt_test"], res["train_linears"])
        irt_mse    = mean_squared_error(res["gt_ctt_test"], res["test_irts"])
        diffs.append(linear_mse - irt_mse)
    rows.append((scenario, models, diffs))

# Extract unique, ordered list of models (they should be the same for every scenario)
all_models = rows[0][1]
all_models = [m.split("-intermediate-checkpoints")[0] if m.endswith("-intermediate-checkpoints") else m for m in all_models]


# Build a DataFrame
scenario_names = [r[0] for r in rows]
data_matrix    = np.vstack([r[2] for r in rows])  # shape = (n_scenarios, n_models)
diff_df        = pd.DataFrame(data_matrix, index=scenario_names, columns=all_models)

plt.figure(figsize=(len(all_models) * 0.5 + 3, len(scenario_names) * 0.5 + 3))
sns.set_style("white")

# Use a diverging colormap (“bwr”) centered at 0:
#  • negatives → blue tones
#  • zero      → white
#  • positives → red tones
abs_max = np.abs(data_matrix).max()
ax = sns.heatmap(
diff_df,
cmap="bwr",
center=0,
vmin=-abs_max,
vmax=abs_max,
linewidths=0.5,
linecolor="lightgray",
cbar_kws={"label": "Linear MSE − IRT MSE"},
annot=True,
fmt=".2f",
annot_kws={"fontsize": 8}
)

ax.set_xlabel("Model")
ax.set_ylabel("Dataset")
ax.set_title("Difference in Test MSE (Linear − IRT)")

# Rotate x-labels for readability
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("downstream_mse_difference_heatmap.png", dpi=300, bbox_inches="tight")
