import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import pickle
import pandas as pd
import argparse
import os
import numpy as np
from functools import reduce
import re


def make_masks_fig(num_param):
    plt.figure()
    summary_all = num_param.groupby(['mask_type', 'model', 'top_x_percent']).agg({
        'size': ['mean', 'std', 'count'],
        'size_percentage': ['mean', 'std']
    }).reset_index()
    summary_all.columns = ['mask_type', 'model', 'top_x_percent', 'mean_size', 'std_size', 'count', 'mean_size_percentage', 'std_size_percentage']

    summary_all.mask_type = summary_all.mask_type.where(summary_all.mask_type != 'compliment', 'difference')

    # Plot the combined data
    sns.set_style("white", {'axes.grid' : True})
    sns.set_context("notebook", rc={"grid.linewidth": 0.5, "lines.linewidth": 1.5})
    # Plotting the mean size
    sns.lineplot(
        data=summary_all,
        x='top_x_percent',
        y='mean_size_percentage',
        hue='mask_type',
        style='mask_type',
        linewidth = 4,
        markers=False,
        dashes=True,
        palette='Set1'
    )

    plt.title('')
    plt.xlabel('Top k% of Changed Weights')
    plt.ylabel('Zero Out Ratio (%)')
    plt.legend(loc=2, prop={'size': 16})

    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./figs/mask_size_plot_combined.pdf', bbox_inches='tight', dpi=300, format='pdf')

def main():
    num_param = pd.read_csv('/home/sheng136/DeconDTN/code/misc/csv/mask_sizes.csv')
    make_masks_fig(num_param)

if __name__ == '__main__':
    main()