import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.manifold import TSNE

PATH = r"../output/dem_vis/ours"
prompt_list = sorted(os.listdir(PATH))
print(prompt_list)
N, C = np.load(os.path.join(PATH, prompt_list[0])).shape

color_dict = {0: "#B000B0", 1: "#0000B0", 2: "#66BB6A", 3: "#E64A35"}
label_dict = {0: "Extreme", 1: "Severe", 2: "Moderate", 3: "Mild"}
#
#
groups = []
for idx, prompt in enumerate(prompt_list):
    f = np.load(os.path.join(PATH, prompt))
    groups.append(f)
X = np.vstack(groups)
y = np.repeat(np.arange(len(prompt_list)), N).reshape(-1, N)

#
# color_dict = {0: "#E64A35", 1: "#0000B0", 2: "#66BB6A"}
# label_dict = {0: "FPN", 1: "LC", 2: "LR"}

# groups = []
# labels = []
# for idx, prompt in enumerate(prompt_list):
#     f = np.load(os.path.join(PATH, prompt))
#
#     N, _ = f.shape
#     groups.append(f)
#     labels.append([idx] * N)

# X = np.concatenate(groups)
# y = np.concatenate(labels)
# N = X.shape[0]

# X_new = np.vstack([np.vstack([X[0:1, :40]] * 1), np.vstack([X[-2:-1, :40]] * 1)])
# plt.imshow(X_new, cmap='viridis')
# plt.colorbar()
# plt.title('Viridis Colormap')
# plt.savefig("F:/ACMMM25/cmp/vis_prompt1.pdf", bbox_inches='tight', dpi=300)
# plt.show()
# exit()
# print(X.shape)

# # 使用t-SNE进行降维到2维
tsne = TSNE(n_components=2, random_state=42, early_exaggeration=80, perplexity=12)
X_tsne = tsne.fit_transform(X)
# np.save("F:/CVPR2026_IRIE-DIFF/exp/abla_res/pro_vis/tsne/joint", X_tsne)


# X_tsne = np.load(r"F:\ACMMM25\cmp\prompt_vis\tsne\42\ours.npy")
# X_tsne = np.load(r"F:/CVPR2026_IRIE-DIFF/exp/abla_res/pro_vis/tsne\joint.npy")

# 绘制可视化图
plt.figure(figsize=(12, 8))

# plt.rcParams['axes.facecolor'] = '#F3F3F3'
font_legend = font_manager.FontProperties(family='Times new roman', size=28)

plt.rcParams['font.size'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
# 使用不同的颜色表示不同组
X_tsne = X_tsne.reshape(-1, N, 2)
for i in range(len(prompt_list)):
    plt.scatter(X_tsne[i, :, 0], X_tsne[i, :, 1], c=color_dict[i], s=50, label=label_dict[i])
# for i in np.unique(y):
#     X_tsne_print = X_tsne[y == i]
#     plt.scatter(X_tsne_print[:, 0], X_tsne_print[:, 1], c=color_dict[i], s=50, label=label_dict[i])

plt.legend(loc="lower left", handletextpad=0.1, prop=font_legend)
# plt.colorbar()  # 显示颜色条
# plt.title('t-SNE Visualization of Image Feature Vectors')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
plt.show()
