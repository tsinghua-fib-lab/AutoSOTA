import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--d', nargs="+", default=6, type=int)
parser.add_argument('--bias_type', default="dummy_results", type=str)
parser.add_argument('--u_type', default="disc", type=str)
parser.add_argument('--n_val', default=2000, type=int)
parser.add_argument('--n_rct', default=50000, type=int)
args = parser.parse_args()

ds = args.d if isinstance(args.d, list) else [args.d]
target_vars = {"SE_S": "S", "SE_A": "A", "SE_Y1": "Y"}

pval = 0.01

sb_type2_flag = "sel_bias_type2" in args.bias_type or "selbias2" in args.bias_type

for d in ds:
    save_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f"/results_U_{args.u_type}_ntrain-{args.n_rct}_nval-{args.n_val}/{args.bias_type}/d{d}")
    df = pd.read_csv(os.path.join(save_dir, 'results.csv')) 

    # cp = sns.color_palette("tab10")
    cp = ("#0f7ba2", "#dd5129", "#43b284")

    fig, axs = plt.subplots(1, 3, figsize=(6,1.5)) 
    
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')

    for idx, (key, plot_name) in enumerate(target_vars.items()):
        # axs[idx].set_xlabel(rf"$\rho (b1, {plot_name})$", fontsize=10)
        axs[idx].axhline(y=-np.log10(pval), color='lightgray', linestyle='--', label=f'p = {pval}')
        axs[idx].axvline(x=0, color='lightgray', linestyle='--', label=f'p = {pval}')
        
        hat_rho = df[key + "_r"]
        hat_p = df[key + "_p"]
        log_p_val = np.clip(-np.log10(hat_p), a_min=None, a_max=5)
        axs[idx].scatter(hat_rho, log_p_val, color=cp[idx], s=1, alpha=0.5)

        if sb_type2_flag and "SE_S" in key:
            percent = int(100 * len(df.query(f"({key}_r > 0 & {key}_p < {pval})")) / len(df))
            df_select = df.query(f"({key}_r > 0)")
        elif sb_type2_flag and "SE_A" in key: 
            percent = int(100 * len(df.query(f"({key}_r < 0 & {key}_p < {pval})")) / len(df))
            df_select = df.query(f"({key}_r < 0)")
        else: 
            percent = int(100 * len(df.query(f"{key}_r > 0 & {key}_p < {pval}")) / len(df))
            df_select = df.query(f"({key}_r > 0)")
        average_signal = df_select[f"{key}_r"].mean()
        std_dev        = df_select[f"{key}_r"].std()
        print(f"signal for {key}_r: {average_signal} +/- {std_dev}")
        # print(f"signal for {key}_r: {df[f"{key}_r"].mean()} +/- {df[f"{key}_r"].std()}")

        axs[idx].text(0.8, 0.5, f"%{percent}", fontsize=10, ha='center', va='center', transform=axs[idx].transAxes)

        #  for plotting all four figures with the same x and y limits
        x0, x1 = axs[idx].get_xlim()
        y0, y1 = axs[idx].get_ylim()
        
        x_min, x_max = min(x_min, x0), max(x_max, x1)
        y_min, y_max = min(y_min, y0), max(y_max, y1)

    # axs[0].set_ylabel('-log10(p-value)', fontsize=10)

    for i,ax in enumerate(axs.flat):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        lower_bound = 0
        upper_bound = x_max
        if sb_type2_flag and i == 0: 
            # lower_bound = x_min
            # upper_bound = 0
            lower_bound = 0
            upper_bound = x_max
        elif sb_type2_flag and i == 1: 
            lower_bound = x_min
            upper_bound = 0
            # lower_bound = 0
            # upper_bound = x_max
        ax.fill_between(
            x=[lower_bound, upper_bound],  # From x=0 to the current right limit
            y1=-np.log10(pval),    # Lower boundary
            y2=y_max,      # Current upper limit
            color='dimgray',     # Light gray color
            alpha=0.15,             # Transparency (adjust as needed)
            zorder=0              # Put it behind other plot elements
        )
        

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot_v2.svg'))

    #############################################################################################

