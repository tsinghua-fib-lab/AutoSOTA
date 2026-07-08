import matplotlib.pyplot as plt
import numpy as np
import json

# parameters
title = '<title>'
data_file = '<*.json file obtained from multi_remote_evaluate>'
result_fig = '<output file>'
labels = ['SDR']
markers = ['o'] 
sample_every = 0.2

with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

assert len(data) == len(labels), "data file must contain the same number of groups as labels"
points = [v for k, v in data.items()]

for label, point, marker in zip(labels, points, markers):
    point_np = np.array(point)
    max_y_index = np.argmax(point_np[:, 1])
    point_np = point_np[:max_y_index + 1]

    x = point_np[:, 0]
    y = point_np[:, 1]
    if y.max() > 1:
        y /= 100

    x_min, x_max = x.min(), x.max()
    x_target = np.linspace(x_min, x_max, int(1 / sample_every) + 1)
    indices = [np.abs(x - xt).argmin() for xt in x_target]
    indices = sorted(set(indices))
    selected_points = point_np[indices]

    plt.plot(selected_points[:, 0], selected_points[:, 1], marker=marker, label=label)

plt.xlabel("Total Cost in USD")
plt.ylabel("Performance")
plt.legend(loc="lower right")
plt.title(title)
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig(result_fig, bbox_inches="tight")
