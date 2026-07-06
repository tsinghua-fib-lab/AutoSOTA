import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
np.random.seed(1)
torch.manual_seed(1)
n=10000
d = 2
device='cpu'
X = torch.randn(n, d, device=device) - 1
Y = torch.randn(n, d, device=device) + 2


fig, axes = plt.subplots(1, 4, figsize=(6 * 4, 6 * 1))


#
# axes[0].scatter(X[:,0],X[:,1],color='tab:blue',alpha=0.4)
# axes[0].scatter(Y[:,0],Y[:,1],color='tab:red',alpha=0.4)
#
# ax2.scatter(X[:,0],X[:,1],color='tab:blue',alpha=0.4)
# ax2.scatter(Y[:,0],Y[:,1],color='tab:red',alpha=0.4)
ks = [2,5,10, 20, 50, 100, 200, 500, 1000]

means = []
stds =[]
num_points=[]
meansws=[]
stdsws=[]
for k in ks:
    MAEs = np.loadtxt(f'result_varying_ks/MAEs_k{k}.csv', delimiter=',')
    trues = np.loadtxt(f'result_varying_ks/trueSW_k{k}.csv', delimiter=',')
    SWk = np.loadtxt(f'result_varying_ks/SW_k{k}.csv', delimiter=',')
    num_point = np.loadtxt('result_varying_ks/num_points_k{}.csv'.format(k), delimiter=',')
    errors = MAEs/trues
    errorsws= np.abs(SWk-trues)/trues
    means.append(np.mean(errors))
    stds.append(np.std(errors))
    meansws.append(np.mean(errorsws))
    stdsws.append(np.std(errorsws))
    num_points.append(np.mean(num_point))

axes[0].errorbar(ks, meansws, yerr=stdsws, marker='o', capsize=5, color='tab:red',label=r'SW (Random Sampling)')
axes[0].errorbar(ks, means, yerr=stds, marker='D',linestyle='dashed', capsize=5, color='tab:blue',label=r'Stream-SW')
axes[0].set_ylabel(r'Relative Error',fontsize=20)
axes[0].set_xlabel('k',fontsize=20)

axes[0].set_title('Varying $k$, $n=10000$',fontsize=20)
axes[0].grid()

axins = inset_axes(axes[0], width="40%", height="40%", loc="center left")
axins.errorbar(ks, meansws, yerr=stdsws, marker='o', capsize=5, color='tab:red',label=r'SW (Random Sampling)')
axins.errorbar(ks, means, yerr=stds, marker='D',linestyle='dashed', capsize=5, color='tab:blue',label=r'Stream-SW')
# Define zoom region based on mean_DDR_SW
x_range = (1, 11)  # Adjust as needed
y_range = (np.min(means[:3]) - 0.01, np.max(means[:3]) + 0.01)
axins.grid()
axins.set_xlim(x_range)
axins.set_ylim(y_range)
axins.xaxis.set_visible(False)
axins.yaxis.set_visible(False)
# Connect inset with main plot
mark_inset(axes[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")
axes[0].legend(fontsize=14)

n=10000
axes[1].plot(ks, (3*np.array(ks)+2*np.log2(n/(2/3*np.array(ks)))).astype(np.int32)+1, color='tab:red', marker='o',label=r'SW (Random Sampling)')
axes[1].plot(ks, num_points, color='tab:blue',linestyle='dashed', marker='D',label=r'Stream-SW')
axes[1].set_ylabel(r'Number of Samples',fontsize=20)
axes[1].set_title('Varying $k$, $n=10000$',fontsize=20)
axes[1].set_xlabel('k',fontsize=20)
# axes[1].axhline(10000,color='black',label='SW')
axes[1].legend(fontsize=14)
axes[1].grid()
ns = [500,2000,5000,10000,20000,50000]
means = []
stds =[]
num_points=[]
meansws=[]
stdsws=[]
for n in ns:
    MAEs = np.loadtxt('result_varying_ns/MAEs_n{}.csv'.format(n), delimiter=',')
    trues = np.loadtxt('result_varying_ns/trueSW_n{}.csv'.format(n), delimiter=',')
    num_point = np.loadtxt('result_varying_ns/num_points_n{}.csv'.format(n), delimiter=',')
    SWk = np.loadtxt('result_varying_ns/SW_k_n{}.csv'.format(n), delimiter=',')
    errors = MAEs/trues
    errorsws = np.abs(SWk - trues) / trues
    means.append(np.mean(errors))
    stds.append(np.std(errors))
    meansws.append(np.mean(errorsws))
    stdsws.append(np.std(errorsws))
    num_points.append(np.mean(num_point))
axes[2].errorbar(ns, meansws, yerr=stdsws, marker='o', capsize=5, color='tab:red',label=r'SW (Random Sampling)')
axes[2].errorbar(ns, means, yerr=stds, marker='D',linestyle='dashed', capsize=5, color='tab:blue',label=r'StreamSW')
axes[2].set_ylabel(r'Relative Error',fontsize=20)
axes[2].set_title('Varying $n$, $k=100$',fontsize=20)
axes[2].set_xlabel('n',fontsize=20)

axes[2].grid()
axins = inset_axes(axes[2], width="40%", height="40%", loc="upper left"
                                                           "")
axins.xaxis.set_visible(False)
axins.yaxis.set_visible(False)
axins.errorbar(ns, meansws, yerr=stdsws, marker='o', capsize=5, color='tab:red',label=r'SW (Random Sampling)')
axins.errorbar(ns, means, yerr=stds, marker='D',linestyle='dashed', capsize=5, color='tab:blue',label=r'StreamSW')
# Define zoom region based on mean_DDR_SW
x_range = (499, 5001)  # Adjust as needed
y_range = (np.min(means[:3]) - 0.01, np.max(means[:3]) + 0.01)
axins.grid()
axins.set_xlim(x_range)
axins.set_ylim(y_range)
mark_inset(axes[2], axins, loc1=2, loc2=4, fc="none", ec="0.5")
axes[2].legend(fontsize=14)

k=100
axes[3].plot(ns, (3*k+2*np.log2(np.array(ns)/(2/3*k))).astype(np.int32)+1,  marker='o',color='tab:red',label=r'SW (Random Sampling)')
axes[3].plot(ns, num_points,  marker='D',color='tab:blue',linestyle='dashed',label=r'Stream-SW')
axes[3].set_ylabel(r'Number of Samples',fontsize=20)
axes[3].set_xlabel('n',fontsize=20)
axes[3].set_title('Varying $n$, $k=100$',fontsize=20)
# axes[3].plot(ns, ns, color='black',label='SW')
axes[3].legend(fontsize=14)
axes[3].grid()


# plt.suptitle('Gaussians Comparison')
plt.tight_layout()
plt.show()

