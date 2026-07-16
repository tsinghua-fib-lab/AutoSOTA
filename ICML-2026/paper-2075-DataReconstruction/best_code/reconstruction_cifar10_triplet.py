import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import common_utils
import analysis
import analysis_utils
from analysis import find_nearest_neighbour, scale, sort_by_metric, find_best_ssim_scores_batch

sweep = common_utils.common.load_dict_to_obj("./reconstructions/cifar10_vehicles_animals/sweep_d50.txt")

# read model, data, and whatever needed
args, Xtrn, Ytrn, ds_mean, W, model = analysis_utils.sweep_get_data_model(sweep, put_in_sweep=True, run_train_test=True)

# original
x1_paths = [
    "./reconstructions/cifar10_vehicles_animals/loo_nosplit_x.pth"
    ]
# sample-splitting
x2_paths = [
    "./reconstructions/cifar10_vehicles_animals/loo_split_x.pth"
    ]
# device = torch.device("cuda:0")
device = torch.device("cpu")


x1_list = [torch.load(path).to(device) for path in x1_paths]
X1 = torch.cat(x1_list, dim=0)
x2_list = [torch.load(path).to(device) for path in x2_paths]
X2 = torch.cat(x2_list, dim=0)

X1 = X1.to(device)
X2 = X2.to(device)
Xtrn = Xtrn.to(device)
ds_mean = ds_mean.to(device)

# Find Nearest Neighbour
X1_scaled, Xtrn_scaled = scale(X1, Xtrn, ds_mean)
X2_scaled, Xtrn_scaled = scale(X2, Xtrn, ds_mean)

xx1 = find_nearest_neighbour(X1_scaled, Xtrn_scaled, search='ncc', vote='min', use_bb=False, nn_threshold=1000)
xx2 = find_nearest_neighbour(X2_scaled, Xtrn_scaled, search='ncc', vote='min', use_bb=False, nn_threshold=1000)
# # Sort
xx1, yy1, ssims1, sort_idxs1 = sort_by_metric(xx1, Xtrn_scaled, sort='ssim')
xx2, yy2, ssims2, sort_idxs2 = sort_by_metric(xx2, Xtrn_scaled, sort='ssim')
ssims1 = ssims1[sort_idxs1]
ssims2 = ssims2[sort_idxs2]
print(len(xx1))
print(len(xx2))
# Merge two analyses
def intersect_align_and_shuffle(
    xx1, yy1, ssim1, idx1,
    xx2, yy2, ssim2, idx2,
    shuffle=True
):
    idx1 = idx1.cpu()
    idx2 = idx2.cpu()

    common_idx = sorted(set(idx1.tolist()) & set(idx2.tolist()))

    pos1 = {int(idx): i for i, idx in enumerate(idx1.tolist())}
    pos2 = {int(idx): i for i, idx in enumerate(idx2.tolist())}

    xx1_new, xx2_new = [], []
    yy_new = []
    ssim1_new, ssim2_new = [], []

    for idx in common_idx:
        i1 = pos1[idx]
        i2 = pos2[idx]

        xx1_new.append(xx1[i1])
        xx2_new.append(xx2[i2])
        yy_new.append(yy1[i1])
        ssim1_new.append(ssim1[i1])
        ssim2_new.append(ssim2[i2])

    xx1_new = torch.stack(xx1_new)
    xx2_new = torch.stack(xx2_new)
    yy_new  = torch.stack(yy_new)

    ssim1_new = torch.stack(ssim1_new)
    ssim2_new = torch.stack(ssim2_new)
    # mask = (ssim1_new > 0.49) | (ssim2_new > 0.49)

    # xx1_new = xx1_new[mask]
    # xx2_new = xx2_new[mask]
    # yy_new = yy_new[mask]
    # ssim1_new = ssim1_new[mask]
    # ssim2_new = ssim2_new[mask]
    if shuffle:
        perm = torch.argsort(ssim2_new - ssim1_new, descending=True)
        xx1_new   = xx1_new[perm]
        xx2_new   = xx2_new[perm]
        yy_new    = yy_new[perm]
        ssim1_new = ssim1_new[perm]
        ssim2_new = ssim2_new[perm]

    return xx1_new, xx2_new, yy_new, ssim1_new, ssim2_new
top_k = 500
xx1_new, xx2_new, yy_new, ssim1_new, ssim2_new = intersect_align_and_shuffle(xx1[:top_k], yy1[:top_k], ssims1[:top_k], sort_idxs1[:top_k],
                           xx2[:top_k], yy2[:top_k], ssims2[:top_k], sort_idxs2[:top_k],
                           shuffle=True)
print(ssim1_new[:20],torch.topk(ssim1_new, 20)[0].mean().item(),torch.mean(ssim1_new).item())
print(ssim2_new[:20],torch.topk(ssim2_new, 20)[0].mean().item(),torch.mean(ssim2_new).item())
print(sum(ssim2_new > ssim1_new).item(), "out of", len(ssim1_new), "have better SSIM in 2nd reconstruction")
# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(ssim1_new.cpu().numpy(), ssim2_new.cpu().numpy(), alpha=0.5)
plt.plot([0,1], [0,1], 'r--')  # Diagonal line
plt.xlabel('SSIM (No Split)',fontsize=16)
plt.ylabel('SSIM (Split)',fontsize=16)
# plt.title('SSIM Comparison')
plt.grid(True)
plt.savefig('./cifar10_ssim_comparison.png', dpi=100)



# color_by_labels = Ytrn[sort_idxs]
color_by_labels = None
figpath='./cifar10_analysis.png'
fig_lines_per_page = len(xx1_new) // 15
analysis.plot_table_cifar(xx1_new, xx2_new, yy_new, ssim=ssim1_new, ssim2=ssim2_new, fig_elms_in_line=15, fig_lines_per_page=fig_lines_per_page*3, figpath=figpath, show=False, dpi=100,color_by_labels=color_by_labels)
