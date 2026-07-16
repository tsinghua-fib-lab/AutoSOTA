import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import common_utils
import analysis
import analysis_utils
from analysis import find_nearest_neighbour, scale, sort_by_metric


# read sweep parameters
sweep = common_utils.common.load_dict_to_obj("./reconstructions/mnist_odd_even/sweep_d50.txt")
# read model, data, and whatever needed
args, Xtrn, Ytrn, ds_mean, W, model = analysis_utils.sweep_get_data_model(sweep, put_in_sweep=True, run_train_test=True)

# original
x1_paths = [
    './reconstructions/mnist_odd_even/loo_nosplit_x.pth'
    ]
# sample-splitting
x2_paths = [
    './reconstructions/mnist_odd_even/loo_split_x.pth'
    ]

device = torch.device("cuda:0")

x1_list = [torch.load(path).to(device) for path in x1_paths]
X1 = torch.cat(x1_list, dim=0)
x2_list = [torch.load(path).to(device) for path in x2_paths]
X2 = torch.cat(x2_list, dim=0)

X1 = X1.to(device)
X2 = X2.to(device)
Xtrn = Xtrn.to(device)


xx1 = find_nearest_neighbour(X1, Xtrn, search='l2', vote='min', use_bb=False, nn_threshold=1000)
xx2 = find_nearest_neighbour(X2, Xtrn, search='l2', vote='min', use_bb=False, nn_threshold=1000)
# # Sort
xx1, yy1, ssims1, sort_idxs1 = sort_by_metric(xx1, Xtrn, sort='l2')
xx2, yy2, ssims2, sort_idxs2 = sort_by_metric(xx2, Xtrn, sort='l2')
# scale
xx1, Xtrn_scaled = scale(xx1, Xtrn, ds_mean)
xx2, Xtrn_scaled = scale(xx2, Xtrn, ds_mean)

ssims1 = ssims1[sort_idxs1]
ssims2 = ssims2[sort_idxs2]
# print(ssims1[:20])
# print(ssims2[:20])
# Merge two analyses
def intersect_align_and_shuffle(
    xx1, yy1, metric1, idx1,
    xx2, yy2, metric2, idx2,
    shuffle=True
):
    idx1 = idx1.cpu()
    idx2 = idx2.cpu()
    common_idx = sorted(set(idx1.tolist()) & set(idx2.tolist()))

    pos1 = {int(idx): i for i, idx in enumerate(idx1.tolist())}
    pos2 = {int(idx): i for i, idx in enumerate(idx2.tolist())}

    xx1_new, xx2_new = [], []
    yy_new = []
    metric1_new, metric2_new = [], []

    for idx in common_idx:
        i1 = pos1[idx]
        i2 = pos2[idx]

        xx1_new.append(xx1[i1])
        xx2_new.append(xx2[i2])
        yy_new.append(yy1[i1])
        metric1_new.append(metric1[i1])
        metric2_new.append(metric2[i2])

    # stack
    xx1_new = torch.stack(xx1_new)
    xx2_new = torch.stack(xx2_new)
    yy_new  = torch.stack(yy_new)
    metric1_new = torch.stack(metric1_new)
    metric2_new = torch.stack(metric2_new)

    # ---- optional filter (SSIM threshold) ----
    # mask = (metric1_new < 4.8) | (metric2_new < 4.8)

    # xx1_new = xx1_new[mask]
    # xx2_new = xx2_new[mask]
    # yy_new  = yy_new[mask]
    # metric1_new = metric1_new[mask]
    # metric2_new = metric2_new[mask]

    # ---- optional shuffle ----
    if shuffle:
        perm = torch.argsort(metric1_new - metric2_new, descending=True)
        xx1_new   = xx1_new[perm]
        xx2_new   = xx2_new[perm]
        yy_new    = yy_new[perm]
        metric1_new = metric1_new[perm]
        metric2_new = metric2_new[perm]

    return xx1_new, xx2_new, yy_new, metric1_new, metric2_new

top_k = 500
xx1_new, xx2_new, yy_new, metric1_new, metric2_new = intersect_align_and_shuffle(
        xx1[:top_k], yy1[:top_k], ssims1[:top_k], sort_idxs1[:top_k],
        xx2[:top_k], yy2[:top_k], ssims2[:top_k], sort_idxs2[:top_k],
        shuffle=True
    )

print(metric1_new[:20])
print(metric2_new[:20])
print(metric1_new[:20],torch.topk(metric1_new, 20, largest=False)[0].mean().item(),torch.mean(metric1_new).item())
print(metric2_new[:20],torch.topk(metric2_new, 20, largest=False)[0].mean().item(),torch.mean(metric2_new).item())

print(sum(metric2_new < metric1_new).item(), "out of", len(metric1_new), "have better L2 Distance in 2nd reconstruction")
# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(metric1_new.cpu().numpy(), metric2_new.cpu().numpy(), alpha=0.5)
plt.plot([0,10], [0,10], 'r--') 
plt.xlabel('L2 Distance (No Split)',fontsize=16)
plt.ylabel('L2 Distance (Split)',fontsize=16)
# plt.title('SSIM Comparison')
plt.grid(True)
plt.savefig('./mnist_l2_comparison.png', dpi=100)


color_by_labels = None
figpath='./mnist_analysis.png'
fig_lines_per_page = len(xx1_new) // 15
analysis.plot_table_mnist(xx1_new, xx2_new, yy_new, ssim=metric1_new, metric2=metric2_new, fig_elms_in_line=15, fig_lines_per_page=fig_lines_per_page*3, figpath=figpath, show=False, dpi=100,color_by_labels=color_by_labels)


