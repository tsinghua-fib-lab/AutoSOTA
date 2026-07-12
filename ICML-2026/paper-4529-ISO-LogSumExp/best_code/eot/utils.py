import torch
from torch.distributions import Normal, Categorical, MixtureSameFamily
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pickle
from collections import defaultdict


def is_pow_of_10(k):
    return k > 0 and math.log10(k) == int(math.log10(k))


def format_number(x):
    if x >= 1e6:
        return f'{x * 1e-6:.0f}M'
    elif x >= 1e3:
        return f'{x * 1e-3:.0f}k'
    else:
        return f'{x:.0f}'


def get_distributions():
    # Source distribution
    mean_mu = 1.
    std_mu = 1. / math.sqrt(8 * math.pi)
    mu = Normal(mean_mu, std_mu)

    # Target distribution
    means_nu = torch.tensor([0., 2.])
    stds_nu = torch.sqrt(torch.tensor([0.02, 1. / (2 * math.pi)]))
    comp = Normal(means_nu, stds_nu)

    mix = Categorical(torch.tensor([0.5, 0.5]))
    nu = MixtureSameFamily(mix, comp)
    return mu, nu


def get_kernel_mat(x, y, sigma_sq):
    C = torch.cdist(x.unsqueeze(1), y.unsqueeze(1)) ** 2
    K = torch.exp(-C / sigma_sq)
    return K


def plot_curves_with_std(curves, save_path, logx=True, logy=True, ylabel=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 8))

    for lbl, curve in curves.items():
        iters, curve_mean, curve_std = curve

        plt.plot(iters, curve_mean, label=lbl)
        plt.fill_between(
            iters,
            curve_mean - curve_std,
            curve_mean + curve_std,
            alpha=0.2
        )

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Objective' if ylabel is None else ylabel, fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True, which="both", alpha=0.7)
    plt.ylim(ymin, ymax)

    tirck_formatter = FuncFormatter(lambda x, pos: format_number(x))
    plt.gca().xaxis.set_major_formatter(tirck_formatter)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_curves(curves, save_path):
    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10.colors  # 10 distinct colors
    linestyles = ["-", "--", "-.", ":"]
    label_to_color = {lbl: colors[i % len(colors)] for i, lbl in enumerate(curves)}
    label_to_style = {lbl: linestyles[i % len(linestyles)] for i, lbl in enumerate(curves)}

    for lbl, realizations in curves.items():
        for i, (xs, ys) in enumerate(realizations):
            # show label only once in legend
            label = lbl if i == 0 else ""
            ls = label_to_style[lbl]
            lw = 1.5 if ls == '-' else 2
            plt.plot(xs, ys, color=label_to_color[lbl], linestyle=ls, label=label, linewidth=lw)

    plt.yscale("log")
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Optimality gap', fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True, which="both", alpha=0.7)

    tirck_formatter = FuncFormatter(lambda x, pos: format_number(x))
    plt.gca().xaxis.set_major_formatter(tirck_formatter)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_max_val(seeds, lr=5.):
    curves = []
    for seed in seeds:
        fname = f"trajectories/ref_seed{seed}_lr{lr}.pickle"
        with open(fname, "rb") as file:
            obj_values, iters = pickle.load(file)
            curves.append(obj_values)

    max_val = np.array(curves).max()
    save_path = "reference_opt_val.txt"
    with open("reference_opt_val.txt", "w") as f:
        f.write(f"{max_val:.17g}")
    print(f"Saved reference optimal value to {save_path}")

    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)

    results = {'SGD on semi-discrete problem': (iters, mean_curve, std_curve)}
    save_path = f'plots/ref_sol_lr{lr}.png'
    plot_curves_with_std(results, save_path, logx=True, logy=False)
    print(f"Saved convergence plot of SGD on semi-discrete problem to {save_path}")


def plot_dual(param_grid):
    with open("reference_opt_val.txt") as f:
        opt_val = float(f.read())

    trajectories = defaultdict(list)
    results = dict()

    for sigma_sq, lr, seed in param_grid:
        fname = f"trajectories/dual_seed{seed}_lr{lr}_sig{sigma_sq}.pickle"
        with open(fname, "rb") as file:
            obj_values, iters = pickle.load(file)
            trajectories[rf"SGD on dual problem, $\sigma^2={sigma_sq}$"].append(obj_values)

    for lbl, curves in trajectories.items():
        mean_curve = opt_val - np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        results[lbl] = (iters, mean_curve, std_curve)

    save_path = f'plots/dual.png'
    plot_curves_with_std(results, save_path, logx=False, logy=False, ylabel="Optimality gap")
    print(f"Saved convergence plot of kernel SGD on dual problem to {save_path}")


def plot_divergence(param_grid):
    with open("reference_opt_val.txt") as f:
        opt_val = float(f.read())

    results = defaultdict(list)

    for sigma_sq, lr, seed in param_grid:
        fname = f"trajectories/dual_seed{seed}_lr{lr}_sig{sigma_sq}.pickle"
        with open(fname, "rb") as file:
            obj_values, iters = pickle.load(file)
            errors = opt_val - np.array(obj_values)
            results[rf"$\sigma^2={sigma_sq}$"].append((iters, errors))

    save_path = f'plots/divergence.png'
    plot_curves(results, save_path)
    print(f"Saved the plot with divergent curves to {save_path}")



def plot_dual_vs_semidual(param_grid_dual, param_grid_semi):
    with open("reference_opt_val.txt") as f:
        opt_val = float(f.read())

    trajectories = defaultdict(list)
    results = dict()

    for sigma_sq, lr, seed in param_grid_dual:
        fname = f"trajectories/dual_seed{seed}_lr{lr}_sig{sigma_sq}.pickle"
        with open(fname, "rb") as file:
            obj_values, iters = pickle.load(file)
            trajectories[rf"SGD on dual problem, $\sigma^2={sigma_sq}$"].append(obj_values)

    for sigma_sq, lr, seed, rho in param_grid_semi:
        fname = f"trajectories/semidual_lr{lr}_sig{sigma_sq}_rho{rho}_seed{seed}.pickle"
        with open(fname, "rb") as file:
            obj_values, iters_ = pickle.load(file)
            trajectories[rf"Proposed approach, $\rho={rho}$"].append(obj_values)

    for lbl, curves in trajectories.items():
        mean_curve = opt_val - np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        iter_numbers = iters if lbl.startswith("SGD") else iters_
        results[lbl] = (iter_numbers, mean_curve, std_curve)

    save_path = f'plots/dual_vs_semidual.png'
    plot_curves_with_std(results, save_path, logx=False, logy=True, ylabel="Optimality gap", ymax=2.)
    print(f"Saved convergence plot of both approaches to {save_path}")


def plot_distributions():
    mu, nu = get_distributions()

    x = torch.linspace(-1, 3, 1000)
    single_gaussian = torch.exp(mu.log_prob(x))
    mixture = torch.exp(nu.log_prob(x))

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(x, single_gaussian, label=r'Density of $\mu$', color='blue')
    plt.plot(x, mixture, label=r'Density of $\nu$', color='red')

    plt.xlabel('x', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)

    ax.xaxis.set_major_locator(MaxNLocator(5))  # ~5 ticks on x-axis
    ax.yaxis.set_major_locator(MaxNLocator(5))  # ~5 ticks on y-axis
    plt.grid(True)

    save_path = 'plots/distributions.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved distributions' density plot to {save_path}")
