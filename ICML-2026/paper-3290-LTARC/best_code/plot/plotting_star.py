import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use serif fonts
    "font.serif": ["Computer Modern"],  # Use the LaTeX default font
})

#Scale the plot size
plt.rcParams['figure.figsize'] = [7, 5]
# Scale font size for latex
plt.rcParams.update({'font.size': 14})

folder = '2025-02-13__14_34_20'
color = 'r'

folder_path = os.path.join(os.getcwd(), '../result', 'star')
df = pd.read_csv(os.path.join(folder_path, folder, "star_ex.csv"))

def plot(x, y, xlabel, ylabel, xlim, ylim):
    plt.figure()
    plt.plot(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid()

# Figure mean objective
plot(df['beta'], df['mean obj'], r'$\beta$', r'$P(L = 1)$', [0, 0.5], [0, 1])
plt.tight_layout()
plt.savefig(os.path.join(folder_path, folder, 'mean_obj.png'))

# Coverage y criteria
plot(df['beta'], df['mean constr'], r'$\beta$', r' $P(L=1|A=1)$', [0, 0.5], [0, 0.5])
plt.plot([0, 1],[0, 1], color='k', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(folder_path, folder, 'mean_criteria.png'))

plot(df['mean constr'], df['mean obj'], r'$P(L=1|A=1)$', r'$P(L = 1)$', [0, 0.5], [0, 1])
plt.tight_layout()
plt.savefig(os.path.join(folder_path, folder, 'obj_vs_criteria.png'))
