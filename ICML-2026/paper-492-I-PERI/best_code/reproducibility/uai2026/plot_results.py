import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
res_CDE = np.loadtxt('pyCIPHOD/reproducibility/uai2026/results_CDE.txt')
res_NDE = np.loadtxt('pyCIPHOD/reproducibility/uai2026/results_NDE.txt')
num_ts_list = [n for n in range(2,16)]
p_edge_list = [p for p in range(15,0,-1)]
norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)
titlesize = 30
fontsize = 20

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im1 = axes[0].imshow(res_CDE, cmap='viridis', norm=norm)
axes[0].set_title('Controlled Direct Effect', fontsize=titlesize)

im2 = axes[1].imshow(res_NDE, cmap='viridis', norm=norm)
axes[1].set_title('Natural Direct Effect', fontsize=titlesize)

# Apply ticks to both subplots
for ax in axes:
    ax.set_xticks(np.arange(len(num_ts_list)))
    ax.set_xticklabels(num_ts_list)
    ax.set_xlabel("Number of nodes in the SCG", fontsize=fontsize)
    ax.set_yticks(np.arange(len(p_edge_list)))
    ax.set_yticklabels(p_edge_list)
    ax.set_ylabel("Probability of edge existence in the FT-ADMG (%)", fontsize=fontsize)

# Single shared colorbar for both subplots
cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Proportion of identified direct effects", fontsize=fontsize)

plt.show()