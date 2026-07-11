from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os
import fsspec
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pathlib
import re

def order_indices_by_priority(df, priority_order):

    priority_order = [idx for idx in priority_order if idx in df.index]
    remaining = [idx for idx in df.index if idx not in priority_order]
    
    new_index_order = priority_order + remaining
    df = df.reindex(new_index_order)
    return df

def load_pickle(path):  
    # Load the dictionary from the pickle file
    if "/home" not in path:
        path = os.path.expanduser(f'~/{path}')
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def load_results_from_pickle(folder,
                            time_threshold=None, #"%y_%m_%d_%H_%M_%S" "25_07_08_14_59_36"
                            keep_run_type=None,
                            is_output=False,
                            is_synthetic=True,
                            recursive=False,             # for recursive search
                            ignore_folder_prefix=None,  # None, str. ignore folders starting with this prefix
                            pct_sig_calc=True,
                            rename_dict = {
                                    'hide_and_seek': 'Hide&Seek',
                                    'hide_and_seek_ens': 'Hide&Seek_ens',
                                    'l2x': 'L2X',
                                    'lime': 'LIME',
                                    'invase': 'INVASE',
                                    'shap_xgboost':'SHAP',
                                    'lasso':"LASSO",
                                    'random_forest':"RForest",
                                    'realx':'REAL-x'
                                },
                            numeric_cols = ['lmbda','accuracy', 'roc_auc', 'pct_sig', 'roc_auc','pr_auc','TPR_mean','FDR_mean','F1', 'f1']
                                ):

    if recursive:
        path2 = f'~/Data/{folder}/**/*'
    else:
        path2 = f'~/Data/{folder}/*'

    # Create a filesystem instance (local filesystem in this case)
    fs = fsspec.filesystem('file')
    # List all files in a directory (recursive=False by default)
    ALL_FILES2 = fs.glob(path2)
    
    results_files2 = [file for file in ALL_FILES2 if "results" in file]
    
    all_series = []
    for file2 in results_files2:

        # 2. Check if any parent folder starts with the ignore prefix
        if ignore_folder_prefix:
            # pathlib.Path(file).parts breaks the path into a tuple of folders/files
            # e.g., ('~', 'Data', 'hide_and_seek', 'zz_invas', 'results.pkl')
            parts = pathlib.Path(file2).parts
            
            # We use [:-1] to only check the directory names, not the file name itself
            if any(part.startswith(ignore_folder_prefix) for part in parts[:-1]):
                continue

        temp_dict = load_pickle(file2)
        filtered_dict = {k: v for k, v in temp_dict.items() if k != "Output"}

        if time_threshold is not None:
            threshold = datetime.strptime(time_threshold, "%y_%m_%d_%H_%M_%S")
            if ('time_run' not in filtered_dict.keys()) or (pd.to_datetime(filtered_dict["time_run"], format="%Y-%m-%d_%H-%M-%S", errors="coerce") < threshold):
                continue
        if keep_run_type is not None:
            if filtered_dict['run_type'] != keep_run_type:
                continue
        if is_output:
            output = temp_dict['Output']

            metrics = ['roc_auc_score', 'average_precision_score', 'accuracy_score']
            models = ['val', 'dis']
            stats = ['mean', 'std']
            
            new_dict = {}
            
            for i, metric in enumerate(metrics):
                for j, model in enumerate(models):
                    new_dict[f'{metric}'] = output[i, j]
                    # new_dict[f'{metric}_{model}_std'] = output[i, j+2]
        
        match = re.search(r"_(Syn\d+[SQ]?)_", file2)
        
        if is_synthetic:
            if match:
                syn = match.group(1)
        
        if is_output:
            series1 = pd.Series(new_dict)
            if is_synthetic:
                series2['syn'] = syn
            series2 = pd.Series(filtered_dict)
            # Concatenate along the rows
            combined_series = pd.concat([series2, series1],axis=0)
            # combined_series.name = series1.name
            all_series.append(combined_series)
        else:
            series2 = pd.Series(filtered_dict)
            # series2.name = syn
            if is_synthetic:
                series2['syn'] = syn
            all_series.append(series2)
        
    # return all_series
    results = pd.concat(all_series, axis=1)
    results = results[sorted(results.columns)]

    priority_order = [
    "syn", "TPR_mean", "FDR_mean", "TPR_std", "FDR_std",
    "roc_auc_score_val", "roc_auc_score_dis",
    "average_precision_score_val", "average_precision_score_dis",
    "accuracy_score_val", "accuracy_score_dis"
    ]
    results = order_indices_by_priority(results, priority_order)
    
    results = results.T
    results['run_id'] = results.apply(lambda x: str(x['time_run']) + '_' + str(x['run_type']),axis=1)
    # results['F1'] = results.apply(compute_rowwise_metrics, axis=1)
    results['model'] = results['model_type'].replace(rename_dict)
    results['seed'] = results['seed'].astype(int)
    
    if ('pct_sig' not in results.columns) and pct_sig_calc:
        results['pct_sig'] = results['binary_mask'].apply(lambda x: np.mean(x))

    for col in numeric_cols:
        if col in results.columns:
            results[col] = results[col].astype(float)
    return results

def performance_metric(score, g_truth): #not used

        n = len(score)
        Temp_TPR = np.zeros([n,])
        Temp_FDR = np.zeros([n,])
        
        for i in range(n):
    
            # TPR    
            TPR_Nom = np.sum(score[i,:] * g_truth[i,:])
            TPR_Den = np.sum(g_truth[i,:])
            Temp_TPR[i] = 100 * float(TPR_Nom)/float(TPR_Den+1e-8)
        
            # FDR
            FDR_Nom = np.sum(score[i,:] * (1-g_truth[i,:]))
            FDR_Den = np.sum(score[i,:])
            Temp_FDR[i] = 100 * float(FDR_Nom)/float(FDR_Den+1e-8)
    
        return np.mean(Temp_TPR), np.mean(Temp_FDR), np.std(Temp_TPR), np.std(Temp_FDR)


def compute_rowwise_metrics(row, just_f1=True):
    g = np.array(row['g_test'])        # shape (n, p)
    pred = np.array(row['binary_mask'])
    
    # True Positives, False Positives, False Negatives per row
    TP = np.sum((pred == 1) & (g == 1), axis=1)
    FP = np.sum((pred == 1) & (g == 0), axis=1)
    FN = np.sum((pred == 0) & (g == 1), axis=1)
    
    # Compute metrics per row
    TPR = TP / (TP + FN + 1e-10)
    FDR = FP / (TP + FP + 1e-10)
    F1  = 2 * TP / (2 * TP + FP + FN + 1e-10)
    
    # Take mean across all rows in this experiment
    if just_f1 == True:
        return pd.Series({
                'F1' : np.mean(F1)
            })
    else:
        return pd.Series({
                'TPR': np.mean(TPR),
                'FDR': np.mean(FDR),
                'F1' : np.mean(F1)
            })

def plot_mask_distributions(df, n_features=None, font_scale=1.5,
                            save_path=None):

    """
    Plots histograms and KDEs of mask values for each feature across different syn levels.
    Args:        
        df (pd.DataFrame): DataFrame containing 'mask' and 'syn' columns.
        'mask' column should contain arrays of shape (n_samples, n_features)
        n_features (int, optional): Number of features to plot. If None, it will be inferred from the first mask array.
        font_scale (float): Scaling factor for font sizes in the plot.
    """

    if n_features is None:
        n_features = df.iloc[0]['mask'].shape[1]
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern", "DejaVu Serif"],
        'axes.labelsize': 12 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
            "pdf.fonttype": 42,
    "ps.fonttype": 42,
    })
    sns.set_style("whitegrid")

    syn_levels = sorted(df['syn'].unique())
    n_syns = len(syn_levels)
    palette = sns.color_palette("Set2", n_colors=10)

    fig, axes = plt.subplots(n_features, n_syns, figsize=(20, 25), 
                             sharex=True, sharey=False)

    for col_idx, syn_val in enumerate(syn_levels):
        syn_data = df[df['syn'] == syn_val]
        masks = np.vstack(syn_data['mask'].values)
        
        for row_idx in range(n_features):
            ax = axes[row_idx, col_idx]
            feat_data = masks[:, row_idx]
            
            # 1. Histograms
            sns.histplot(feat_data, bins=40, ax=ax, stat="density", 
                         color=palette[col_idx % 8], 
                         alpha=0.8, edgecolor='black', linewidth=0.1)

            # 2. KDE
            sns.kdeplot(feat_data, ax=ax, color='black', 
                        linewidth=1.5, alpha=1.0, cut=0, bw_adjust=0.4)

            # 3. Axis and Label Formatting
            if row_idx == 0:
                # Increased size to 18 * font_scale for "Syn X"
                ax.set_title(f'Syn {int(syn_val)}', fontweight='bold', size=18 * font_scale)
            
            if col_idx == 0:
                # Added \mathbf for LaTeX bold and increased size to 20 * font_scale
                # Offset shifted to -0.6 to accommodate larger font
                ax.annotate(f'$\\mathbf{{X_{{{row_idx + 1}}}}}$', 
                            xy=(-0.6, 0.5), 
                            xycoords='axes fraction',
                            ha='right', va='center', 
                            fontsize=20 * font_scale) 
                ax.set_ylabel('') 
            else:
                ax.set_ylabel('')
            
            ax.set_xlim(-0.05, 1.05)

    # Adjusting left margin to make room for the larger X_i labels
    plt.subplots_adjust(left=0.15)
    plt.tight_layout()

    if save_path is not None:
        save_path = os.path.expanduser(save_path)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

def plot_metric_vs_lambda(df, 
                          metric_col='roc_auc', 
                          lambda_col='lmbda', 
                          model_col='model', 
                          pct_sig_col='pct_sig',
                          groupby_cols=['syn'],
                          query=None,
                          cols=None,
                          ax=None,             
                          annotate_sig=True,
                          is_twin=False,
                          figsize=(10, 6),
                          legend_loc='best',
                          x_ticks=None):      
    
    # 1. Axis Setup
    if ax is None:
        fig, main_ax = plt.subplots(figsize=figsize)
        plot_ax = main_ax
    elif is_twin:
        main_ax = ax
        plot_ax = ax.twinx() 
    else:
        main_ax = ax
        plot_ax = ax

    if cols is not None:
        df = df[cols]
    if groupby_cols is not None:
        groupby_cols = [model_col, lambda_col] + groupby_cols
    else:
        groupby_cols = [model_col, lambda_col]

    if query is not None:
        df = df.query(query).groupby(groupby_cols).mean()
    else:
        df = df.groupby(groupby_cols).mean()

    # 2. Prevent color duplication on twin axes
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_offset = len(main_ax.lines) if ax is not None else 0

    plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    })

    for i, (model, g) in enumerate(df.reset_index()[[model_col,lambda_col,metric_col,pct_sig_col]].groupby(model_col)):
        g = g.sort_values(lambda_col)
        
        line_color = color_cycle[(i + color_offset) % len(color_cycle)]
        marker = 'v' if is_twin else 'o'
        plot_ax.plot(g[lambda_col], g[metric_col], marker=marker, color=line_color, label=f"{metric_col.upper()}")

        if annotate_sig:
            for _, row in g.iterrows():
                plot_ax.annotate(
                    f"{row[pct_sig_col]*100:.0f}%",
                    (row[lambda_col], row[metric_col]),
                    textcoords="offset points",
                    xytext=(0, -8),
                    ha='center',
                    va='top',
                    fontsize=9
                )

    # 3. Labeling
    if not is_twin:
        plot_ax.set_xlabel(r'$\lambda$')
        if x_ticks is not None:
            plot_ax.set_xticks(np.arange(x_ticks[0],x_ticks[1],x_ticks[2]))
            plot_ax.xaxis.set_minor_locator(MultipleLocator(x_ticks[2]/2))
        plot_ax.grid(True, which='both', linestyle='--', alpha=0.9)
        
    plot_ax.set_ylabel(metric_col.replace('_', ' ').upper())
    
    # --- FIXED COMBINED LEGEND LOGIC ---
    
    # Add an invisible line to the axis so Matplotlib tracks the annotation label natively
    proxy_label = f'{pct_sig_col}'
    current_handles, current_labels = plot_ax.get_legend_handles_labels()
    
    if annotate_sig and proxy_label not in current_labels:
        # plot([], []) draws nothing, but registers the label in the axis memory
        # Added marker='o', color='black', and a smaller markersize
        plot_ax.plot([], [], linestyle='none', marker='o', color='black', markersize=1, label=proxy_label)

    # Now we safely pull all handles/labels, knowing the proxy is officially registered
    if is_twin:
        handles1, labels1 = main_ax.get_legend_handles_labels()
        handles2, labels2 = plot_ax.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
    else:
        handles, labels = plot_ax.get_legend_handles_labels()
    
    # Draw ONE unified legend on the main axis
    main_ax.legend(handles=handles, labels=labels, loc=legend_loc)

    return main_ax
