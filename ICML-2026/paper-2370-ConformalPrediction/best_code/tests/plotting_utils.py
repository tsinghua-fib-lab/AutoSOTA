import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.ticker as mtick

def longest_true_sequence(arr):
    # Add False at each end to make sure we always get the length of every True sequence
    # Even when they are at the start or end of the array
    arr = np.concatenate(([False], arr, [False]))
    # Find the indices where the array goes from False to True or True to False
    idx = np.flatnonzero(arr[1:] != arr[:-1])
    # Get lengths of True sequences and get the maximum length
    max_length = np.max(idx[1::2] - idx[::2])
    return max_length

def moving_average(x, window=50):
    x = np.array(x)
    ma = np.zeros(x.shape)
    for i in range(x.shape[0]):
        ma[i] = np.mean(x[max(0, i-window+1):i+1])
    return ma

def set_plot_style():
    """
    Set global plot style for Matplotlib.
    """
    plt.rcParams.update({
        'figure.dpi': 300,            # 屏幕预览时的清晰度 (太高可能会让Notebook变卡，100-150合适)
        'savefig.dpi': 300,
        
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], # 优先用 Arial，没有则回退
        'font.size': 14,              # 全局字体加大
        'axes.labelsize': 16,         # x,y轴标签字体 (Coverage, Time)
        'xtick.labelsize': 14,        # 刻度字体
        'ytick.labelsize': 14,
        
        'lines.linewidth': 2.5,       # 线条粗细 (关键！让图看起来饱满)
        'axes.linewidth': 1.5,        # 坐标轴边框粗细
        
        'axes.grid': True,            # 开启网格
    
        'legend.fontsize': 14,
        'legend.frameon': False,      # 图例不要边框
        
        'figure.figsize': (12, 8),    # 画布大小
        'xtick.direction': 'in',      # 刻度朝内 (可选，看个人喜好)
        'ytick.direction': 'in',
    })
    return None

def plot_everything(coverages_list, sizes_list, sets_list, y, alpha, window_start, window_end, window_loc, coverage_inset, size_inset, set_inset, miscoverage_scatterplot, savename, model_name, datetimes=None, colors=None, labels=None):
    fig, axs = plt.subplots(nrows=2, ncols=len(coverages_list), figsize=(6 * len(coverages_list), 8), sharex=True, sharey=False)
    for coverages, sizes, color, label in zip(coverages_list, sizes_list, colors, labels):
        plot_time_series(axs[0, 0], [coverages], window_start, window_end, window_loc, False, y, color, coverage_inset, False, datetimes, hline=1-alpha, label=label )
        plot_time_series(axs[0, 1], [sizes], window_start, window_end, window_loc, False, y, color, size_inset, False, datetimes, label=label )
    
    for ax, sets, color, label in zip(axs[1,:], sets_list, colors, labels):
        plot_time_series(ax, [sets], window_start, window_end, window_loc, True, y, color, set_inset, miscoverage_scatterplot, datetimes, label=label)
        
    # plot_time_series(axs[0,:], coverages_list, window_start, window_end, window_loc, False, y, "#138085", coverage_inset, False, datetimes, hline=1-alpha )
    # plot_time_series(axs[1,:], sets_list, window_start, window_end, window_loc, True, y, "#EEB362", set_inset, miscoverage_scatterplot, datetimes)
    axs[0,0].set_ylabel('Coverage', fontsize=16)
    axs[0,1].set_ylabel('Size', fontsize=16)
    axs[1,0].set_ylabel('Sets', fontsize=16)
    # axs[0,0].set_title(titles_list[0], fontsize=20)
    # axs[0,1].set_title(titles_list[1], fontsize=20)

    # # Get the max and min values of each axis in axs[0,:] by calling get_ylim
    # ymin = min([ax.get_ylim()[0] for ax in axs[0,:]])
    # ymax = max([ax.get_ylim()[1] for ax in axs[0,:]])

    for ax in axs[0,:]:
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        ax.set_ylim([ymin,ymax])
    
    # axs[0,0].set_yticks([0.5, 0.75, 1.0])
    ymin = axs[0,0].get_ylim()[0]
    ymax = axs[0,0].get_ylim()[1]
    axs[0,0].set_ylim([ymin - 0.2, 1.05])
    axs[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axs[0,0].yaxis.set_tick_params(labelsize=13)
    # axs[0,1].set_yticklabels([])

    # Get the max and min values of each axis in axs[1,:] by calling get_ylim
    ymin = min([ax.get_ylim()[0] for ax in axs[1,:]])
    ymax = max([ax.get_ylim()[1] for ax in axs[1,:]])
    for ax in axs[1,:]:
        ax.set_ylim([ymin, ymax + 0.1*np.abs(ymax)])
    axs[1,1].set_yticklabels([])

    axs[1,0].yaxis.set_tick_params(labelsize=13)
    axs[1,1].yaxis.set_tick_params(labelsize=13)
    axs[1,0].xaxis.set_tick_params(labelsize=13)
    axs[1,1].xaxis.set_tick_params(labelsize=13)

    fig.autofmt_xdate()

    plt.subplots_adjust(left=0.1, bottom=0.15)
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time", fontsize=20, labelpad=30)
    plt.grid(False)
    plt.subplots_adjust(top=0.85)
    lines, labels = axs[0, 0].get_legend_handles_labels()
    
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 0.93), ncol=4)
    os.makedirs('./plots/1v1/' + model_name, exist_ok=True)
    plt.savefig('./plots/1v1/' + model_name + "/" + savename + '.pdf', bbox_inches='tight')

    # Add a plot of miscoverage error
    fig, ax = plt.subplots(figsize=(8, 6))
    for coverages, color, label in zip(coverages_list, colors, labels):
        cumulative_coverages = np.cumsum(coverages) / np.arange(1, len(coverages) + 1)
        miscoverages_error = np.abs(1 - alpha - cumulative_coverages)
        # miscoverages_ma = moving_average(miscoverages, window=100)
        plot_time_series(ax, [miscoverages_error], window_start, window_end, window_loc, False, y, color, False, False, datetimes, label=label)
    ax.set_ylabel('Miscoverage Error', fontsize=16)
    ax.set_yscale('log')
    # ymin = ax.get_ylim()[0]
    # ymax = ax.get_ylim()[1]
    # ax.set_ylim([ymin,ymax])
    ax.yaxis.set_tick_params(labelsize=13)
    fig.autofmt_xdate()
    ax.set_xlabel("Time", fontsize=16)
    plt.subplots_adjust(top=0.85)
    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 0.95), ncol=4)
    plt.savefig('./plots/1v1/' + model_name + "/" + savename + "_miscoverage_error.pdf", bbox_inches='tight')

def plot_time_series(axs, time_series_list, window_start, window_end, window_loc, sets, y, color, inset, miscoverage_scatterplot, datetimes, hline=None, **kwargs):
    # Create a figure and a grid of subplots
    all_axins = []
    # Get the minimum and maximum values for the axes and axins
    # Create a list of time series with only finite values. The time series are all numpy arrays
    if not sets:
        ts_list_finite = [ np.where(np.isfinite(time_series), time_series, np.nan) for time_series in time_series_list ]
    else:
        ts_list_finite = [ [np.where(np.isfinite(time_series[0]) & np.isfinite(time_series[1]), time_series[0], np.nan), np.where(np.isfinite(time_series[0]) & np.isfinite(time_series[1]), time_series[1], np.nan)] for time_series in time_series_list ]
    if not sets:
        minval_ax = np.nanmin([ np.nanmin(time_series) for time_series in ts_list_finite ])
        maxval_ax = np.nanmax([ np.nanmax(time_series) for time_series in ts_list_finite ])
        minval_axins = np.nanmin([ np.nanmin(time_series[window_start:window_end]) for time_series in ts_list_finite ])
        maxval_axins = np.nanmax([ np.nanmax(time_series[window_start:window_end]) for time_series in ts_list_finite ])
    else:
        minval_ax = np.nanmin([ np.nanmin(time_series[0]) for time_series in ts_list_finite ])
        maxval_ax = np.nanmax([ np.nanmax(time_series[1]) for time_series in ts_list_finite ])
        minval_axins = np.nanmin([ np.nanmin(time_series[0][window_start:window_end]) for time_series in ts_list_finite ])
        maxval_axins = np.nanmax([ np.nanmax(time_series[1][window_start:window_end]) for time_series in ts_list_finite ])
    
    if not isinstance(axs, np.ndarray): axs = np.array([axs]) # 兼容函数内部对 axs[i] 的调用
    for i, time_series in enumerate(time_series_list):
        ax = axs[i]

        # Use seaborn to plot the time series on the ax
        if not sets:
            # sns.lineplot(x=datetimes, y=time_series, ax=ax, color=color)
            ax.plot(datetimes, time_series, color=color, **kwargs)
        else:
            cvds = (time_series[0] <= y) & (time_series[1] >= y)
            ax.fill_between(datetimes, np.clip(time_series[0], minval_ax, maxval_ax), np.clip(time_series[1], minval_ax, maxval_ax), color=color)
            ax.plot(datetimes, y, color='black', alpha=0.3, linewidth=1)
            if miscoverage_scatterplot:
                ax.scatter(datetimes[~cvds], y[~cvds], color='purple', alpha=0.7, linewidth=1, s=20)
        if hline is not None:
            ax.axhline(hline, color='black', linestyle='--')
        # sns.despine(ax=ax)  # Despine the top and right axes

        if inset:
            # Define the inset ax in the lower right corner
            if window_loc == 'lower right':
                axins = ax.inset_axes([0.6,0.05,0.4,0.4])
            elif window_loc == 'upper right':
                axins = ax.inset_axes([0.6,0.6,0.4,0.4])
            elif window_loc == 'upper left':
                axins = ax.inset_axes([0.05,0.6,0.4,0.4])

            # Give the inset a different background color
            axins.set_facecolor('whitesmoke')

            # On the inset ax, plot the same time series but only the window of interest
            if not sets:
                # sns.lineplot(x=datetimes[window_start:window_end], y=time_series[window_start:window_end], ax=axins, color=color)
                axins.plot(datetimes[window_start:window_end], time_series[window_start:window_end], color=color)
            else:
                cvds = (time_series[0][window_start:window_end] <= y[window_start:window_end]) & (time_series[1][window_start:window_end] >= y[window_start:window_end])
                axins.fill_between(datetimes[window_start:window_end], np.clip(time_series[0][window_start:window_end], minval_axins, maxval_axins), np.clip(time_series[1][window_start:window_end], minval_axins, maxval_axins), color=color)
                axins.plot(datetimes[window_start:window_end], y[window_start:window_end], color='black', alpha=0.3, linewidth=1)

            if hline is not None:
                axins.axhline(hline, color='#888888', linestyle='--', linewidth=1)

            box_color = "#dcd9d9"
            for axis in ['top','bottom','left','right']:
                axins.spines[axis].set_linewidth(2)
                axins.spines[axis].set_color(box_color)

            # Draw a box of the region of the inset axes in the parent axes and
            # connecting lines between the box and the inset axes area
            mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec=box_color, lw=2)

            # Apply auto ticks on the inset
            axins.xaxis.set_visible(True)
            axins.yaxis.set_visible(True)

            axins.set_xticklabels([])
            axins.set_xticks([])
            axins.set_yticklabels([])
            axins.set_yticks([])

            all_axins += [axins]

        # Set ymin and ymax for insets
        for axin in all_axins:
            axin.set_ylim(minval_axins-0.1*np.abs(minval_axins), maxval_axins + 0.1*np.abs(maxval_axins))
    

# def plot_overview(axes, statistics, alpha=0.05, start=100, **kwargs):
#     """
#     Plot overview of local coverage, local width, and miscoverage convergence.
    
#     Args:
#         axes: List of three Matplotlib axes to plot on.
#         statistics: Dictionary containing 'mean_local_coverages', 'mean_local_radii',
#                     and 'mean_cumulative_coverages'.
#         alpha: Significance level for target coverage line.
#         start: Index to start plotting from.
#         **kwargs: Additional keyword arguments for the plot functions.
#     """
#     plot_local_coverage(axes[0], statistics["mean_local_coverages"], alpha=alpha, start=start, **kwargs)
#     plot_local_width(axes[1], 2 * statistics["mean_local_radii"], start=start, **kwargs)
#     miscoverages = np.abs(1 - alpha - statistics["mean_cumulative_coverages"])
#     plot_miscoverage_convergence(axes[2], miscoverages, start=start, **kwargs)
#     plot_local_deviation(axes[3], 2 * statistics["mean_local_radii_deviations"], start=start, **kwargs)
    
#     axes[0].set_xlabel(None)
#     axes[1].set_xlabel(None)
    
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.88) 

#     return None 

# def plot_local_coverage(axis, local_coverages, alpha=0.05, start=100, **kwargs):
#     """
#     Plot local coverage over time.
    
#     Args:
#         axis: Matplotlib axis to plot on.
#         local_coverages: Array of local coverage values.
#         alpha: Significance level for target coverage line.
#         start: Index to start plotting from.
#         **kwargs: Additional keyword arguments for the plot function.
#     """
#     times = np.arange(start + 1, len(local_coverages) + 1)
#     axis.plot(times, local_coverages[start:], **kwargs)
#     axis.axhline(1 - alpha, color='black', linestyle='--', label='Target Coverage')
#     # axis.text(len(local_coverages) * 1.04, 1 - alpha + 0.02, f'{(1 - alpha):.2f}',
#     #         color='black', fontsize=12, va='top', ha='right')
#     axis.set_xlabel("Time")
#     axis.set_ylabel("Local Coverage")
#     axis.grid("on")
    
#     return None 

# def plot_local_width(axis, local_widths, start=100, **kwargs):
#     """
#     Plot local prediction interval width over time.
    
#     Args:
#         axis: Matplotlib axis to plot on.
#         local_radii: Array of local prediction interval widths.
#         start: Index to start plotting from.
#         **kwargs: Additional keyword arguments for the plot function.
#     """
#     times = np.arange(start + 1, len(local_widths) + 1)
#     axis.plot(times, local_widths[start:], **kwargs)
#     axis.set_xlabel("Time")
#     axis.set_ylabel("Width")
#     axis.grid("on")
    
#     return None 

# def plot_miscoverage_convergence(axis, miscoverages, start=100, **kwargs):
#     """
#     Plot miscoverage convergence over time.
    
#     Args:
#         axis: Matplotlib axis to plot on.
#         miscoverages: Array of miscoverage values.
#         alpha: Significance level for target miscoverage line.
#         start: Index to start plotting from.
#         **kwargs: Additional keyword arguments for the plot function.
#     """
#     times = np.arange(start + 1, len(miscoverages) + 1)
#     axis.plot(times, miscoverages[start:], **kwargs)
#     axis.set_yscale('log')

#     axis.set_xlabel("Time")
#     axis.set_ylabel("Miscoverage Error")
#     axis.grid("on")
    
#     return None

# def plot_local_deviation(axis, local_deviations, start=100, **kwargs):
#     """
#     Plot local prediction interval width deviations over time.
    
#     Args:
#         axis: Matplotlib axis to plot on.
#         local_deviations: Array of local prediction interval width deviations.
#         start: Index to start plotting from.
#         **kwargs: Additional keyword arguments for the plot function.
#     """
#     times = np.arange(start + 1, len(local_deviations) + 1)
#     axis.plot(times, local_deviations[start:], **kwargs)
#     axis.set_xlabel("Time")
#     axis.set_ylabel("Local Deviation of Width")
#     axis.grid("on")
    
#     return None