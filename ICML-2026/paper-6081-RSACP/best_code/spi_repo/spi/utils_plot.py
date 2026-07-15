import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import glob
from transporter.utils import get_wc_bounds
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

palette4fig = {'SC': 'red', 'SC (synth)': 'orange', 'SPI': 'dodgerblue', 'SPI [subset]': 'limegreen'}
desc4paper_dict = {'n_cal': r'$m$', 'alpha': r'$\alpha$',
                   'Conditional Coverage': 'Coverage',
                   'Conditional Length': 'Size',
                   'Length': 'Size',
                   'n_cal_cond': r'$m$',
                   'age_range': 'Age range'}
method2legend = {
                 'SPI': 'SPI', 'SC (synth)': 'OnlySynth', 'SPI [subset]': 'SPI-Subset', 'SC': 'OnlyReal'
                 }
method2marker = {'SPI': 'o', 'SC (synth)': 'D', 'SPI [subset]': '*', 'SC': '^'}
hue_order = ['SC', 'SC (synth)', 'SPI', 'SPI [subset]']

class2desc_dict = {
                    '321': 'Admiral', 
                    '392': 'Rock beauty', 
                    '157': 'Papillon', 
                    '326': 'Lycaenid butterfly', 
                    '993': 'Gyromitra', 
                    '991': 'Coral fungus', 
                    '994': 'Stinkhorn', 
                    '389': 'Barracouta', 
                    '395': 'Garfish', 
                    '0': 'Tinca tinca',
                    '13': 'Junco, snowbird',
                    '15': 'American robin',
                    '16':  'Bulbul',
                    '17': 'Jay',
                    '18': 'Magpie',
                    '20': 'Water ouzel',
                    '337': 'Beaver',
                    '207': 'Golden retriever',
                    '208': 'Labrador retriever',
                    '217':'English springer',
                    '222': 'Kuvasz',
                    '250': 'Siberian husky',
                    '270': 'White wolf',
                    '626': 'Lighter, Light',
                    '852': 'Tennis ball',
                    '862': 'Torch',
                    '444': 'Bicycle',
                    '336': 'Marmot',
                    '676': 'Muzzle',
                    '880': 'Unicycle',
                    }


def desc4paper(x):
    if x in desc4paper_dict.keys():
        return desc4paper_dict[x]
    else:
        return x


def get_desc_imagenet_class(desc):
    if pd.isna(desc):
        return desc
    if str(int(desc)) in class2desc_dict.keys():
        return class2desc_dict[str(int(desc))]
    else:
        return desc


def get_bounds(x, x_values, conditional=False, n_cal=50, n_cal_maj=5000, beta=0.1, alpha=0.2, n_classes=None):
    if x == 'n_cal' or x == 'n_cal_cond':
        curr_bounds = {}
        if conditional and x != 'n_cal_cond':
            for n_cal in x_values:
                n_cal_cond = n_cal // n_classes
                wc_lower, wc_upper = get_wc_bounds(n_cal_maj, n_cal_cond, beta, alpha_list=[alpha])
                curr_bounds[n_cal] = (wc_lower[alpha], wc_upper[alpha])
        else:
            for n_cal in x_values:
                wc_lower, wc_upper = get_wc_bounds(n_cal_maj, n_cal, beta, alpha_list=[alpha])
                curr_bounds[n_cal] = (wc_lower[alpha], wc_upper[alpha])
        return curr_bounds
    elif x == 'k':
        curr_bounds = {}
        for k in x_values:
            n_cal_maj_ = n_cal_maj * k
            wc_lower, wc_upper = get_wc_bounds(n_cal_maj_, n_cal, beta, alpha_list=[alpha])
            curr_bounds[k] = (wc_lower[alpha], wc_upper[alpha])
        return curr_bounds
    else:
        wc_lower, wc_upper = get_wc_bounds(n_cal_maj, n_cal, beta, alpha_list=[alpha])
    return wc_lower[alpha], wc_upper[alpha]


def get_x_order(results, x, subset_exp):
    x_order = None
    if x == 'Class':
        x_order_n = ['250', '337', '15', '626', '389']
        if subset_exp:
            x_order_n = ['321', '626', '337', '862', '15']
        elif 'flux' in results['dataset'].values[0].lower():
            x_order_n = ['250', '208', '15', '18', '444']
        x_order = [get_desc_imagenet_class(x_) for x_ in x_order_n]
    elif x == 'age_range':
        x_order = ['0-20', '20-40', '40-60', '60+']
    return x_order


def plot4paper(save_path, results, x=None, hue='Method', y='Coverage',
               methods2plot=None,
               alpha=0.2, bounds=None, showmeans=True, more_bounds=None,
               split_size=False, min_y_upper=29.5, max_y_lower=15, subset_exp=False):
    x_order = get_x_order(results, x, subset_exp)
    font_size = 24
    font_size_ticks = font_size - 7
    plt.rc('legend', fontsize=font_size)
    if methods2plot is None:
        methods2plot = set(results['Method'].values)
    results = results.copy().reset_index()
    if x is None:
        results = results[[hue, y]]
    else:
        results = results[[x, hue, y]]
    results_ = results.loc[results['Method'].isin(methods2plot)]
    results_ = results_.replace([np.inf, -np.inf], np.nan)
    results_ = results_.dropna(inplace=False)
    if results_.empty:
        print('Results are empty after dropping None and inf values --- skipping plot generation...')
        return
    if split_size:
        fig = plt.figure(figsize=([7,4]))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.2, 4], hspace=0.0)
        # Second column, top row
        ax1 = fig.add_subplot(gs[0])
        # Second column, bottom row
        ax = fig.add_subplot(gs[2])
    else:
        if x is None:
            fig = plt.figure(figsize=([3, 4]))
        elif (x == 'Class' and len(x_order) < 10):
            fig = plt.figure(figsize=([7, 4]))
        elif  x == 'Class':
            fig = plt.figure(figsize=([15, 5]))
        elif x == 'age_range':
            fig = plt.figure(figsize=([6, 4]))
        else:
            fig = plt.figure(figsize=([7, 4]))
        ax = fig.add_subplot(111)
    hue_order_curr = [m for m in hue_order if m in methods2plot]
    for m in methods2plot:
        if not m in hue_order_curr:
            hue_order_curr.append(m)
    default_palette = sns.color_palette('PuBuGn', len(hue_order_curr))
    curr_palette = {}
    for i, m in enumerate(hue_order_curr):
        if m in palette4fig.keys():
            curr_palette[m] = palette4fig[m]
        else:
            curr_palette[m] = default_palette[i]
    if x is not None:
        try:
            unique_xs = None
            if x == 'Class' or x == 'age_range':
                unique_xs = x_order
            else:
                unique_xs = sorted(results_[x].unique(), key=lambda v: float(v), reverse=False)
            if split_size:
                sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax1, hue_order=hue_order_curr, order=unique_xs)
            sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax, hue_order=hue_order_curr, order=unique_xs)
        except Exception as e:
            if split_size:
                sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax1, hue_order=hue_order_curr)
            sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax, hue_order=hue_order_curr)
        if showmeans:
            hue_levels = hue_order_curr
            category_levels = list(results_[x].unique()) if unique_xs is None else unique_xs # Order of the x-axis categories
            offsets = np.linspace(-0.26, 0.26, len(hue_levels))  # Adjust for dodge
            if len(results_[hue].unique()) == 1:
                offsets = [0,] * len(hue_levels)
            for i, category in enumerate(category_levels):
                for j, hue_ in enumerate(hue_levels):
                    subset = results_[(results_[x] == category) & (results_[hue] == hue_)]
                    if not subset.empty:
                        mean_value = subset[y].mean()  # Compute mean
                        x_position = i + offsets[j]  # Adjust x position for dodge
                        ax.scatter(x_position, mean_value, color=curr_palette[hue_], edgecolor='black', s=100, zorder=3, marker=method2marker[hue_])
                        if split_size:
                            ax1.scatter(x_position, mean_value, color=curr_palette[hue_], edgecolor='black', s=100, zorder=3, marker=method2marker[hue_])
    else:
        if split_size:
            sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax1, hue_order=hue_order_curr)
        sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax, hue_order=hue_order_curr)
        if showmeans:
            hue_levels = hue_order_curr
            offsets = np.linspace(-0.26, 0.26, len(hue_levels))  # Adjust for dodge
            if len(results_[hue].unique()) == 1:
                offsets = [0,] * len(hue_levels)
            for j, hue_ in enumerate(hue_levels):
                subset = results_[results_[hue] == hue_]
                if not subset.empty:
                    mean_value = subset[y].mean()  # Compute mean
                    x_position = offsets[j]  # Adjust x position for dodge
                    ax.scatter(x_position, mean_value, color=curr_palette[hue_], edgecolor='black', s=100, zorder=3, marker=method2marker[hue_])
                    if split_size:
                        ax1.scatter(x_position, mean_value, color=curr_palette[hue_], edgecolor='black', s=100, zorder=3, marker=method2marker[hue_])
    if y == 'Coverage' or y == 'Conditional Coverage':
        if x != 'alpha':
            ax.axhline(1-alpha, linestyle='dashed', color='black', label=r'$1-\alpha$')
        else:
            xticks = np.array(ax.get_xticks())  # Category positions
            for i, x_ in enumerate(xticks):
                plt.hlines(1-float(unique_xs[i]), x_-0.45, x_+0.45, colors='black', linestyles='dashed')
            plt.plot([], [], color='black', linestyle='dashed', label=r'$1-\alpha$')
        if bounds is not None:
            if isinstance(bounds, dict):
                # plot lower and upper bounds
                xticks = np.array(ax.get_xticks())  # Category positions
                for i, x_ in enumerate(xticks):
                    if x == 'k' and i > 0:
                        break
                    plt.hlines(bounds[float(unique_xs[i])][0], x_-0.45, x_+0.45, colors='red', linestyles='dashed')
                    plt.hlines(bounds[float(unique_xs[i])][1], x_-0.45, x_+0.45, colors='darkgreen', linestyles='dashed')
                plt.plot([], [], color='red', linestyle='dashed', label='Lower-bound')
                plt.plot([], [], color='darkgreen', linestyle='dashed', label='Upper-bound')
            else:
                if isinstance(bounds, tuple):
                    l, u = bounds
                    ax.axhline(l, linestyle='dashed', color='red', label='Lower-bound')
                    ax.axhline(u, linestyle='dashed', color='darkgreen', label='Upper-bound')
                else:
                    ax.axhline(1-bounds, linestyle='dashed', color='red', label=r'$1-\alpha_{\mathrm{SG}}$')
        if more_bounds is not None:
            if isinstance(more_bounds, dict):
                # plot lower and upper bounds
                xticks = np.array(ax.get_xticks())  # Category positions
                for i, x_ in enumerate(xticks):
                    if unique_xs[i] == '-':
                        continue
                    plt.hlines(more_bounds[float(unique_xs[i])][0], x_-0.45, x_+0.45, colors='red', linestyles='dotted')
                    plt.hlines(more_bounds[float(unique_xs[i])][1], x_-0.45, x_+0.45, colors='darkgreen', linestyles='dotted')
                plt.plot([], [], color='red', linestyle='dotted', label='Lower-bound [s]')
                plt.plot([], [], color='darkgreen', linestyle='dotted', label='Upper-bound [s]')
            else:
                if isinstance(more_bounds, tuple):
                    l, u = more_bounds
                    ax.axhline(l, linestyle='dotted', color='red', label='Lower-bound [s]')
                    ax.axhline(u, linestyle='dotted', color='darkgreen', label='Upper-bound [s]')
    if x == 'Class':
        plt.xticks(rotation=-30)
    plt.tick_params(axis='both', which='major', labelsize=font_size_ticks)
    ax.set_xlabel(desc4paper(x), fontsize=font_size)
    ax.set_ylabel(desc4paper(y), fontsize=font_size)
    if split_size:
        ax1.tick_params(axis='both', which='major', labelsize=font_size_ticks)
        ymin, ymax = ax.get_ylim()
        ax1.set_ylim([min_y_upper, 30.5])
        ax1.set_yticks([30])
        y_ticks = [i * 5 + 5 for i in range(max_y_lower//5)]
        ax.set_yticks(y_ticks)
        ax.set_ylim([ymin, max_y_lower])
        ax1.set_ylabel("", fontsize=font_size)
        ax1.get_legend().remove()
        ax1.get_xaxis().set_visible(False)
    desc = '_split_y' if split_size else ''
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    labels = [method2legend[x] if x in method2legend.keys() else x for x in labels]
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04, 0.5))
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + '/' + y + desc + '.pdf', bbox_inches='tight')
    ax.get_legend().remove()
    plt.savefig(save_path + '/' + y + desc + '_no_legend.pdf', bbox_inches='tight')



def filter_and_plot(save_path, results, x, methods2plot=None, conditional=False, to_table=False, only_results_bounds=False, subset_exp=False,
                    alpha=0.05, beta=0.4, **kwargs):
    curr_results = results.copy()
    for k, v in kwargs.items():
        if k == 'n_classes' or k == 'n_classes_maj' or v == None:
            continue
        if isinstance(v, list):
            curr_results = curr_results[curr_results[k].astype(str).isin(v)]
        else:
            curr_results = curr_results[curr_results[k].astype(str) == str(v)]
    curr_results = curr_results[curr_results['alpha'].astype(str) == str(alpha)]
    curr_results = curr_results[curr_results['beta'].astype(str) == str(beta)]

    if 'Class' in curr_results.columns:
        curr_results['Class'] = curr_results['Class'].apply(get_desc_imagenet_class)
    
    if to_table:
        method_order = ['SC', 'SC (synth)', 'SPI', 'SPI [subset]']
        if methods2plot is not None:
            curr_results = curr_results.loc[curr_results['Method'].isin(methods2plot)]
        grouped = curr_results.groupby([x, 'Method'])[['Conditional Coverage', 'Conditional Length']]
        mean_df = grouped.mean().rename(columns=lambda x: f"{x}_mean")
        std_df = grouped.sem().rename(columns=lambda x: f"{x}_se")

        # Merge mean and std
        agg_df = pd.merge(mean_df, std_df, left_index=True, right_index=True).reset_index()

        # Format the values with ± in parentheses
        agg_df['Conditional Coverage'] = agg_df.apply(
            lambda row: f"{row['Conditional Coverage_mean']*100:.1f} (± {row['Conditional Coverage_se']*100:.1f})",
            axis=1
        )
        agg_df['Conditional Length'] = agg_df.apply(
            lambda row: f"{row['Conditional Length_mean']:.1f} (± {row['Conditional Length_se']:.1f})",
            axis=1
        )

        # Pivot for coverage and length (now string values)
        cov_df = agg_df.pivot(index=x, columns='Method', values='Conditional Coverage')
        len_df = agg_df.pivot(index=x, columns='Method', values='Conditional Length')

        # Reorder columns
        cov_df = cov_df[[m for m in method_order if m in cov_df.columns]]
        len_df = len_df[[m for m in method_order if m in len_df.columns]]

        # Concatenate and export
        final_df = pd.concat([cov_df, len_df], axis=1).reset_index()

        latex_table = final_df.to_latex(
            index=False,
            column_format='l' + 'c' * (final_df.shape[1] - 1),
            escape=False  # keep method names as-is
        )

        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, 'table.tex')
        with open(file_path, 'w+') as f:
            f.write(latex_table)
        return

    if x == 'k':
        k_values = set(curr_results['k'].values)
        k_values_int = [int(k_) for k_ in k_values]
        min_k = min(k_values_int)
        # Filter for plotting
        curr_results = curr_results[
            (curr_results['Method'] == 'SPI [subset]') | 
            ((curr_results['Method'].isin(['SC', 'SPI'])) & (curr_results['k'] == str(min_k)))
        ]
    x_values = curr_results[x].unique() if x is not None else None
    if conditional:  # conditional
        if x != 'n_cal' and x != 'n_cal_cond':
            n_cal_cond = curr_results['n_cal_cond'].values[0]
        else:
            n_cal_cond = None
        n_cal_maj_cond = curr_results['n_cal_maj_cond'].values[0]
        n_cal_maj_cond_ = n_cal_maj_cond
        if subset_exp and x != 'k':
            n_cal_maj_cond_ = int(kwargs['k']) * n_cal_maj_cond_
        curr_bounds = get_bounds(x, x_values, conditional=conditional, n_cal=int(n_cal_cond), n_cal_maj=int(n_cal_maj_cond_), alpha=alpha, beta=beta,
                                 n_classes=curr_results['n_classes'].values[0])
    else:
        n_cal_cond = curr_results['n_cal'].values[0]
        n_cal_maj_cond = curr_results['n_cal_maj'].values[0]
        curr_bounds = get_bounds(x, x_values, conditional=conditional, alpha=alpha, beta=beta, n_cal=n_cal_cond, n_cal_maj=n_cal_maj_cond)
    if subset_exp:
        x_ = x if x != 'k' else None
        more_bounds = get_bounds(x_, x_values, n_cal=int(n_cal_cond), n_cal_maj=int(n_cal_maj_cond * 100), alpha=alpha, beta=beta)  # assume total 100 subsets
        if x == 'k':
            tmp = more_bounds
            more_bounds = {min_k: tmp}
        tmp = more_bounds
        more_bounds = curr_bounds
        curr_bounds = tmp
    else:
        more_bounds = None
    if only_results_bounds:
        return curr_results, curr_bounds
    y = 'Coverage' if not conditional else 'Conditional Coverage'
    plot4paper(save_path, curr_results, y=y, x=x, methods2plot=methods2plot, alpha=alpha, bounds=curr_bounds, showmeans=True, more_bounds=more_bounds)
    plt.close()
    y = 'Length' if not conditional else 'Conditional Length'
    split_size = ((alpha == 0.02 or alpha == 0.05) or x == 'n_cal' or x == 'n_cal_cond') and 'meps' not in save_path.lower() and x is not None
    min_y_upper, max_y_lower = 29.5, 15
    plot4paper(save_path, curr_results, y=y, x=x, methods2plot=methods2plot, alpha=alpha, bounds=curr_bounds, showmeans=True, more_bounds=more_bounds, split_size=split_size,
               min_y_upper=min_y_upper, max_y_lower=max_y_lower)
    plt.close()


def plot4paper_w_marginal(save_path, results_marg, results, hue='Method', methods2plot=None, x=None, y='Coverage', y_marg='Coverage',
                          alpha=0.1, bounds=None, showmeans=True, sharey=True, split_size=False, min_y_upper=29.5, max_y_lower=15, subset_exp=False):
    x_order = get_x_order(results, x, subset_exp)
    font_size = 24
    font_size_ticks = font_size - 7
    plt.rc('legend', fontsize=font_size)
    if methods2plot is None:
        methods2plot = set(results['Method'].values)
    results = results.copy()
    results_marg = results_marg.copy()
    results = results[[x, hue, y]]
    results_marg = results_marg[[hue, y_marg]]

    results_ = results.loc[results['Method'].isin(methods2plot)]
    results_ = results_.replace([np.inf, -np.inf], np.nan)
    results_ = results_.dropna(inplace=False)

    results_marg_ = results_marg.loc[results_marg['Method'].isin(methods2plot)]
    results_marg_ = results_marg_.replace([np.inf, -np.inf], np.nan)
    results_marg_ = results_marg_.dropna(inplace=False)
    if results_marg_.empty or results_.empty:
        print('Results are empty after dropping None and inf values --- skipping plot generation...')
        return
    if (x == 'Class' and len(x_order) > 5) or (x != 'Class' and len(results_[x].unique()) > 5):
        figsize=[20,8]
    else:
        figsize=[7,4]

    if not split_size:
        fig, (ax1, ax) = plt.subplots(1,2, sharey=sharey, gridspec_kw={'width_ratios': [1,5]}, figsize=(figsize))
    else:
        fig = plt.figure(figsize=(figsize))
        # Set width ratio 1:5 and height ratio 2:1 for rows
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 5], height_ratios=[1, 0.2, 4], hspace=0.0)

        # First column: span both rows
        ax1 = fig.add_subplot(gs[:, 0])  # full height of narrow column
        # Second column, top row
        ax2 = fig.add_subplot(gs[0, 1])
        # Second column, bottom row
        ax = fig.add_subplot(gs[2, 1])
    hue_order_curr = [m for m in hue_order if m in methods2plot]
    for m in methods2plot:
        if not m in hue_order_curr:
            hue_order_curr.append(m)
    default_palette = sns.color_palette('PuBuGn', len(hue_order_curr))
    curr_palette = {}
    for i, m in enumerate(hue_order_curr):
        if m in palette4fig.keys():
            curr_palette[m] = palette4fig[m]
        else:
            curr_palette[m] = default_palette[i]
    if x is not None:
        try:
            unique_xs = None
            if x == 'Class':
                unique_xs = x_order
            else:
                unique_xs = sorted(results_[x].unique(), key=lambda v: float(v), reverse=False)
            if split_size:
                sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax2, hue_order=hue_order_curr, order=unique_xs)
            sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax, hue_order=hue_order_curr, order=unique_xs)
        except:
            if split_size:
                sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax, hue_order=hue_order_curr)
            sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax, hue_order=hue_order_curr)
        if showmeans:
            hue_levels = hue_order_curr
            category_levels = list(results_[x].unique()) if unique_xs is None else unique_xs # Order of the x-axis categories
            if len(hue_levels) > 3:
                offsets = np.linspace(-0.32, 0.32, len(hue_levels))  # Adjust for dodge
            else:
                offsets = np.linspace(-0.26, 0.26, len(hue_levels))  # Adjust for dodge
            if len(results_[hue].unique()) == 1:
                offsets = [0,] * len(hue_levels)
            for i, category in enumerate(category_levels):
                for j, hue_ in enumerate(hue_levels):
                    subset = results_[(results_[x] == category) & (results_[hue] == hue_)]
                    if not subset.empty:
                        mean_value = subset[y].mean()  # Compute mean
                        x_position = i + offsets[j]  # Adjust x position for dodge
                        ax.scatter(x_position, mean_value, color=curr_palette[hue_], edgecolor='black', s=100, zorder=3, marker=method2marker[hue_])
                        if split_size:
                            ax2.scatter(x_position, mean_value, color=curr_palette[hue_], edgecolor='black', s=100, zorder=3, marker=method2marker[hue_])
    else:
        sns.boxplot(results_, x=x, hue=hue, y=y, palette=curr_palette, ax=ax, hue_order=hue_order_curr, order=x_order)
    if split_size:
        plt.subplots_adjust(hspace=0.1)
    # marginal
    sns.boxplot(results_marg_, x=None, hue=hue, y=y_marg, palette=curr_palette, ax=ax1, hue_order=hue_order_curr)
    if showmeans:
        hue_levels = hue_order_curr
        category_levels = [0]
        offsets = np.linspace(-0.26, 0.26, len(hue_levels))  # Adjust for dodge
        if len(results_marg_[hue].unique()) == 1:
            offsets = [0,] * len(hue_levels)
        # offsets = np.linspace(-0.3, 0.3, len(hue_levels))  # Adjust for dodge
        for i, category in enumerate(category_levels):
            for j, hue_ in enumerate(hue_levels):
                subset = results_marg_[(results_marg_[hue] == hue_)]
                if not subset.empty:
                    mean_value = subset[y_marg].mean()  # Compute mean
                    x_position = i + offsets[j]  # Adjust x position for dodge
                    ax1.scatter(x_position, mean_value, color=curr_palette[hue_], edgecolor='black', s=100, zorder=3, marker=method2marker[hue_])
    if y == 'Coverage' or y == 'Conditional Coverage':
        ax.axhline(1-alpha, linestyle='dashed', color='black', label=r'$1-\alpha$')
        ax1.axhline(1-alpha, linestyle='dashed', color='black', label=r'$1-\alpha$')
        if bounds is not None:
            if isinstance(bounds, dict):
                # plot lower and upper bounds
                xticks = np.array(ax.get_xticks())  # Category positions
                for i, x_ in enumerate(xticks):
                    if x == 'k' and i > 0:
                        break
                    plt.hlines(bounds[float(unique_xs[i])][0], x_-0.45, x_+0.45, colors='red', linestyles='dashed')
                    plt.hlines(bounds[float(unique_xs[i])][1], x_-0.45, x_+0.45, colors='darkgreen', linestyles='dashed')
                plt.plot([], [], color='red', linestyle='dashed', label='Lower-bound')
                plt.plot([], [], color='darkgreen', linestyle='dashed', label='Upper-bound')
            else:
                if isinstance(bounds, tuple):
                    l, u = bounds
                    ax.axhline(l, linestyle='dashed', color='red', label='Lower-bound')
                    ax.axhline(u, linestyle='dashed', color='darkgreen', label='Upper-bound')
                    ax1.axhline(l, linestyle='dashed', color='red', label='Lower-bound')
                    ax1.axhline(u, linestyle='dashed', color='darkgreen', label='Upper-bound')

    plt.xticks(rotation=-30)
    ax1.set_xticks(ticks=[0], labels=['Marginal'])
    plt.tick_params(axis='both', which='major', labelsize=font_size_ticks)
    ax1.tick_params(axis='both', which='major', labelsize=font_size_ticks)
    ax.set_xlabel(desc4paper(x), fontsize=font_size)
    ax.set_ylabel(desc4paper(y), fontsize=font_size)
    ax1.set_ylabel(desc4paper(y), fontsize=font_size)
    if not sharey:
        ax.set_ylabel("", fontsize=font_size)
    if split_size:
        ax2.tick_params(axis='both', which='major', labelsize=font_size_ticks)
        ymin, ymax = ax.get_ylim()
        # ax2.set_ylim([min_y_upper, ymax])
        ax2.set_ylim([min_y_upper, 30.5])
        ax2.set_yticks([30])
        y_ticks = [i * 5 + 5 for i in range(max_y_lower//5)]
        ax.set_yticks(y_ticks)
        ax.set_ylim([ymin, max_y_lower])
        ax2.set_ylabel("", fontsize=font_size)
        ax2.get_legend().remove()
        ax2.get_xaxis().set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax1.get_legend().remove()
    fig.tight_layout()
    labels = [method2legend[x] if x in method2legend.keys() else x for x in labels]
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04, 0.5))
    os.makedirs(save_path, exist_ok=True)
    desc = '_split_y' if split_size else ''
    plt.savefig(save_path + '/' + y + desc +'.pdf', bbox_inches='tight')
    ax.get_legend().remove()
    plt.savefig(save_path + '/' + y + desc + '_no_legend.pdf', bbox_inches='tight')


def load_results(results_dir):
    results = pd.DataFrame({})
    files = glob.glob(os.path.join(results_dir, '*/results/results.pkl'))
    for f in files:
        curr_results = pd.read_pickle(f)
        results = pd.concat([results, curr_results], ignore_index=True)
    return results


def load_and_plot(save_path, results_dir, x, methods2plot=None, conditional=False, to_table=False, only_results_bounds=False, subset_exp=False, alpha=0.05, beta=0.4, **kwargs):
    results = load_results(results_dir)
    return filter_and_plot(save_path, results, x, methods2plot=methods2plot, conditional=conditional, to_table=to_table,
                           only_results_bounds=only_results_bounds, subset_exp=subset_exp,
                           alpha=alpha, beta=beta, **kwargs)
