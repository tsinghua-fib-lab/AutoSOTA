import os
import json
import matplotlib.pyplot as plt
import numpy as np

steps = ["10","20","40","80","120","160","200","230","270","310","350","390"]
steps_int = [int(s) for s in steps]
base_dir = "results_iteration"
cache_file = "cached_results.json"
use_cache = True  # 若设为 False 将强制重新加载原始文件并覆盖缓存

configs = {
    "DFT": "numina-cot-raw-qwen-2.5-math-1.5b-1epoch-scale-0-samples-100000-lr-5e-5",
    "SFT": "numina-cot-raw-qwen-2.5-math-1.5b-1epoch-samples-100000-lr-5e-5",
}

datasets = ["math_oai", "minerva_math", "olympiadbench", "aime24", "amc23"]

dataset_titles = {
    "math_oai": "Math 500",
    "minerva_math": "Minerva Math",
    "olympiadbench": "Olympiad Bench",
    "aime24": "AIME 2024",
    "amc23": "AMC 2023",
    "average": "Average Acc"
}

plt.style.use("ggplot")
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
axs = axs.flatten()

# 加载或生成数据
all_results = {}

if use_cache and os.path.exists(cache_file):
    print("从缓存读取数据...")
    with open(cache_file, "r") as f:
        all_results = json.load(f)
else:
    print("缓存不存在或禁用缓存，开始重新加载原始数据...")
    all_results = {key: {ds: [] for ds in datasets} for key in configs}

    for dataset in datasets:
        for label, dir_path in configs.items():
            for step in steps:
                json_path = os.path.join(
                    base_dir,
                    f"{dir_path}-step-{step}@16",
                    f"{dataset}_metrics.json"
                )
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        acc = data["mean_acc"]
                except FileNotFoundError:
                    print(f"[警告] 找不到文件: {json_path}")
                    acc = 0.0
                all_results[label][dataset].append(acc)

    with open(cache_file, "w") as f:
        json.dump(all_results, f, indent=2)
        print(f"已保存缓存到 {cache_file}")

# 开始绘图
average_results = {key: np.zeros(len(steps)) for key in configs}
valid_counts = np.zeros(len(steps))

for i, dataset in enumerate(datasets):
    ax = axs[i]
    for label in configs:
        acc_list = all_results[label][dataset]
        ax.plot(steps_int, acc_list, marker='o', label=label, linewidth=2)
        average_results[label] += np.array(acc_list)
        if label == list(configs.keys())[0]:  # 只统计一次有效值
            valid_counts += np.array([1 if a > 0 else 0 for a in acc_list])

    ax.set_title(dataset_titles[dataset], fontsize=16)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Average@16", fontsize=12)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True)

# 平均图
ax = axs[-1]
for label in configs:
    avg_acc = average_results[label] / np.maximum(valid_counts, 1)  # 避免除0
    ax.plot(steps_int, avg_acc, marker='o', label=label, linewidth=2)

ax.set_title(dataset_titles["average"], fontsize=16)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Average@16", fontsize=12)
ax.legend(loc='lower right', frameon=True)
ax.grid(True)

plt.tight_layout()
plt.savefig("mean_acc_comparison_all_datasets.jpg", dpi=300)
plt.savefig("mean_acc_comparison_all_datasets.pdf", dpi=300)
plt.show()
