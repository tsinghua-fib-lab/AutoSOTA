import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use serif fonts
    "font.serif": ["Computer Modern"],  # Use the LaTeX default font
})
#Scale the plot size
plt.rcParams['figure.figsize'] = [7, 5]
# Scale font size for latex
plt.rcParams.update({'font.size': 14})

folder = '2025-04-09__16_29_25'
color = 'r'

folder_path = os.path.join(os.getcwd(), '../experiments/result', 'synthetic_p')
df = pd.read_csv(os.path.join(folder_path, folder, "syn_rct_gamma1.csv"))
df_obj = pd.read_csv(os.path.join(folder_path, folder, "obj_all.csv"))
df_constr = pd.read_csv(os.path.join(folder_path, folder, "constr_all.csv"))

# folder_path = os.path.join(os.getcwd(), 'result', 'test')
# df = pd.read_csv(os.path.join(folder_path, folder, "motivating_ex.csv"))

def plot(x, y, xlabel, ylabel, xlim, ylim):
    plt.figure()
    plt.plot(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid()

# Figure mean objective
plot(df['beta'], df['mean obj'], r'$\beta$', r'mean $P(L = 1)$', [0, 0.4], [0, 1.0])
beta = np.linspace(0.05, 0.4, 100)
# x_var = beta*2/0.55
# p_l_a = 0.55/2 * x_var**2 + (1-x_var) * 0.8
# plt.plot(beta, p_l_a, label='p(l|a)', color='black')
# plt.legend(['Proposed method', 'Oracle method'])
plt.tight_layout()
plt.savefig(os.path.join(folder_path, folder, 'mean_obj.png'))

# Coverage y criteria
plot(df['beta'], df['mean constr'], r'$\beta$', r'mean $P(L=1|A=1)$', [0, 0.4], [0, 0.4])
plt.plot([0, 1],[0, 1], color='k', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(folder_path, folder, 'mean_criteria.png'))

# Coverage y criteria
# plot(df['beta'], df['Z'], r'$\beta$', r'mean $P(L=1|A=1)$', [0, 0.4], [0, 0.4])
# plt.plot([0, 1],[0, 1], color='k', linestyle='--')
# plt.tight_layout()
# plt.savefig(os.path.join(folder_path, folder, 'mean_criteria_z.png'))

plot(df['mean constr'], df['mean obj'], r'$P(L=1|A=1)$', r'$P(L = 1)$', [0, 0.4], [0, 1])
plt.tight_layout()
plt.savefig(os.path.join(folder_path, folder, 'obj_vs_criteria.png'))
# Compute mean and confidence interval
def plot_confidence(df_plot, text, xlim, ylim):
    plt.figure()
    mean_values = df_plot.mean(axis=0)
    std_values = df_plot.std(axis=0)
    n = df_plot.shape[0] - 1  # Number of draws
    t_value = stats.t.ppf(0.95, df=n - 1)
    ci = t_value * (std_values / np.sqrt(n))  # 90% Confidence Interval
    pi = t_value * (std_values * np.sqrt(1 + 1/n))  # 90% Prediction Interval

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['beta'], mean_values, label="Mean", color='blue')
    plt.fill_between(df['beta'], mean_values - pi, mean_values + pi, color='blue', alpha=0.2, label="90% CI")

    plt.xlabel(r'$\beta$')
    plt.ylabel(text)
    plt.legend()
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid()

def plot_quantiles(df_plot, text, xlim, ylim):
    plt.figure()
    mean_values = df_plot.mean(axis=0)
    quantiles = df_plot.quantile([0.1, 0.25, 0.75, 0.9], axis=0)

    # Extract 5th and 95th percentiles
    lower_bound = quantiles.loc[0.1]
    upper_bound = quantiles.loc[0.9]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['beta'], mean_values, label="Mean", color='black')
    # plt.plot(df['beta'], quantiles.loc[0.1], label="10/90 quantile", color='blue', linestyle='-')
    # plt.plot(df['beta'], quantiles.loc[0.25], label="25/75 quantile", color='blue', linestyle='--')
    # plt.plot(df['beta'], quantiles.loc[0.75], color='blue', linestyle='--')
    # plt.plot(df['beta'], quantiles.loc[0.9], color='blue', linestyle='-')
    plt.fill_between(df['beta'], lower_bound, upper_bound, color='blue', alpha=0.2, label="10-90 quantile")
    plt.fill_between(df['beta'], quantiles.loc[0.25], quantiles.loc[0.75], color='blue', alpha=0.4, label="25-75 quantile")

    plt.xlabel(r'$\beta$')
    plt.ylabel(text)
    plt.legend()
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid()


# plot_confidence(df_obj, "Objective", [0, 0.4], [0, 1])
# plt.show()
#
# plot_confidence(df_constr, "Constraint", [0, 0.4], [0, 0.4])
# plt.plot([0, 0.4],[0, 0.4], color='k', linestyle='--')
# plt.show()
#
plot_quantiles(df_constr, "Constraint", [0, 0.4], [0, 0.4])
plt.plot([0, 0.4],[0, 0.4], color='k', linestyle='--')
plt.show()
