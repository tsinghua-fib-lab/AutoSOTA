import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from plotting_utils import longest_true_sequence, moving_average,  set_plot_style, plot_everything
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import pickle as pkl
# from matplotlib.dates import date2num
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# from matplotlib.patches import Rectangle
# import pdb
# import matplotlib.patheffects as pe
# import matplotlib.ticker as mtick

def dataframe_to_latex(df):
    # Convert dataframe to have each metric as a row
    df.replace({'ar': 'AR', 'theta': 'Theta', 'transformer': 'Transformer', 'prophet': 'Prophet'}, inplace=True)
    # df.replace({sys.float_info.max : "Inf", "inf" : "Inf"}, inplace=True)
    df_melted = df.melt(id_vars=['Model type', 'Method'])
    df_melted.columns = ['Model type', 'Method', 'Metric', 'Value']

    # Pivot the dataframe to have multi-index columns
    df_pivot = df_melted.pivot_table(index='Metric', columns=['Model type', 'Method'], values='Value')

    # df_pivot.replace({sys.float_info.max : "Inf", np.inf : "Inf", float('inf'): "Inf"}, inplace=True)
    
    # Reindex the metrics in the specific order
    metric_order = ['Marginal coverage', 'Longest err sequence', 'Average set size', 'Median set size', '75% quantile set size', '90% quantile set size', '95% quantile set size']
    df_pivot = df_pivot.reindex(metric_order)

    # Start LaTeX table
    latex_table = "\\begin{tabular}{l" + "l"*len(df_pivot.columns) + "}\n\\toprule\n"

    # Create headers for Model type and Method
    model_types = df_pivot.columns.get_level_values(0).unique()
    current_col = 1
    for model in model_types:
        n_methods = sum(df_pivot.columns.get_level_values(0) == model)
        latex_table += f"& \\multicolumn{{{n_methods}}}{{c}}{{{model}}}"
        current_col += n_methods
    latex_table += " \\\\\n"

    # Create headers for each method
    latex_table += "& " + " & ".join(df_pivot.columns.get_level_values(1)) + " \\\\\n"

    latex_table += "\\midrule\n"

    # Iterate through rows and add them to LaTeX table
    for i, row in df_pivot.iterrows():
        formatted_values = ["{:.3g}".format(val) if pd.notnull(val) else "" for val in row.values]
        latex_table += i + " & " + " & ".join(formatted_values) + " \\\\\n"

    latex_table = latex_table.replace("%", "\\%")

    latex_table += "\\bottomrule\n\\end{tabular}\n"
    return latex_table


if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Plot time series data.')
    parser.add_argument('--filename', help='Path to pickle file containing time series data.')
    parser.add_argument('--key1', help='First key for time series data extraction.')
    parser.add_argument('--lr1', help='Learning rate associated with first key.', type=float)
    parser.add_argument('--key2', help='Second key for time series data extraction.')
    parser.add_argument('--lr2', help='Learning rate associated with second key.', type=float)
    parser.add_argument('--window_length', help='Length of inset window.', default=60, type=int)
    parser.add_argument('--window_start', help='Start of inset window.', default=None, type=int)
    parser.add_argument('--window_loc', help='Location of inset window.', default='upper right', type=str)
    parser.add_argument('--coverage_average_length', help='Length of moving average window for coverage.', default=50, type=int)
    parser.add_argument('--coverage_average_burnin', help='How long to wait before displaying moving average of coverages', default=0, type=int)
    parser.add_argument('--coverage_inset', dest='coverage_inset', default=False, action='store_true')
    parser.add_argument('--size_inset', dest='size_inset', default=False, action='store_true')
    parser.add_argument('--set_inset', dest='set_inset', default=False, action='store_true')
    parser.add_argument('--miscoverage_scatterplot', dest='miscoverage_scatterplot', default=False, action='store_true')

    args = parser.parse_args()

    method_title_map = {
        'SFOGD': 'ScaleFreeOGD' + f' (lr={args.lr2})',
        'KT': 'KrichevskyTrofimov',
        'UP': 'UniversalPortfolio',
        'DtACI': 'DtACI',
    }
    method_color_map ={
        'SFOGD': '#FBBC05',
        'KT': '#4285F4',
        'UP': '#EA4335',
        'DtACI': '#34A853',
    }
    # Parse command line arguments

    args.window_start = args.window_start if args.window_start is not None else -args.window_length
    datasetname = args.filename.split('/')[-1].split('.')[0]
    coverage_average_burnin = args.coverage_average_burnin
    # Translate zero learning rates to actual None
    args.lr1 = None if args.lr1 == 0 else args.lr1
    args.lr2 = None if args.lr2 == 0 else args.lr2
    
    # Load the data from the pickle file
    with open(args.filename, 'rb') as f:
        all_data = pkl.load(f)
    model_names = list(all_data.keys())
    # Make a table with the marginal coverage, the longest sequence of miscoverage events, and the average, median, 75% quantile, 90% quantile, and 95% quantile of the set size
    # The table should have methods on the columns and the rows should be the different quantiles
    df_list_for_table = []
    for model_name in model_names:
        data = all_data[model_name]
        # Extract time series data using provided keys and learning rates
        quantiles_given = data['quantiles_given']
        T_burnin = data['T_burnin']
        sets1 = [np.stack(data[args.key1][args.lr1]['sets'])[T_burnin+1:,0], np.stack(data[args.key1][args.lr1]['sets'])[T_burnin+1:,1]]
        sets2 = [np.stack(data[args.key2][args.lr2]['sets'])[T_burnin+1:,0], np.stack(data[args.key2][args.lr2]['sets'])[T_burnin+1:,1]]
        forecasts = data['forecasts']
        asymmetric = data['asymmetric']
        # if forecasts is a list, clip off the last element of each. otherwise, clip off the last element of the array
        forecasts = [f[T_burnin+1:] for f in forecasts] if isinstance(forecasts, list) else forecasts[T_burnin+1:]
        alpha = data['alpha']
        scores = data['scores'][T_burnin+1]
        y = data['data']['y'].to_numpy().astype(float)[T_burnin+1:]

        if data['real_data']:
            covered1 = moving_average(((y >= sets1[0]) & (y <= sets1[1])).astype(float), args.coverage_average_length)[args.coverage_average_burnin:]
            covered2 = moving_average(((y >= sets2[0]) & (y <= sets2[1])).astype(float), args.coverage_average_length)[args.coverage_average_burnin:]
        else:
            covered1 = moving_average(data[args.key1][args.lr1]['coverages'][T_burnin+1:], args.coverage_average_length)[args.coverage_average_burnin:]
            covered2 = moving_average(data[args.key2][args.lr2]['coverages'][T_burnin+1:], args.coverage_average_length)[args.coverage_average_burnin:]
            
        # Clip the sets and y by coverage_average_burnin
        sets1 = [s[args.coverage_average_burnin:] for s in sets1]
        sets2 = [s[args.coverage_average_burnin:] for s in sets2]
        y = y[args.coverage_average_burnin:]

        # Calculate the set sizes
        sizes1 = sets1[1] - sets1[0]
        sizes2 = sets2[1] - sets2[0]

        # Fill in table.
        # Give all infinite sets the value 'Inf'
        # Recompute marginal coverage and longest err sequence for synthetic data
        if data['real_data']:
            marginal_coverage1 = ((y >= sets1[0]) & (y <= sets1[1])).mean()
            marginal_coverage2 = ((y >= sets2[0]) & (y <= sets2[1])).mean()
            longest_err_seq1 = longest_true_sequence((y < sets1[0]) | (y > sets1[1]))
            longest_err_seq2 = longest_true_sequence((y < sets2[0]) | (y > sets2[1]))
        else:
            marginal_coverage1 = data[args.key1][args.lr1]['coverages'][T_burnin+1+args.coverage_average_burnin:].mean()
            marginal_coverage2 = data[args.key2][args.lr2]['coverages'][T_burnin+1+args.coverage_average_burnin:].mean()
            longest_err_seq1 = np.nan
            longest_err_seq2 = np.nan
        df_list_for_table += [pd.DataFrame({
            'Model type': model_name,
            'Method': method_title_map[args.key1],
            'Marginal coverage': marginal_coverage1,
            # The next line gets the longest sequence of `True' in the boolean array ((y < sets1[0]) | (y > sets1[1]))
            'Longest err sequence': longest_err_seq1,
            'Average set size': np.mean(sizes1) if not np.any(np.isinf(sizes1)) else np.inf,
            'Median set size': np.median(np.nan_to_num(sizes1, nan=np.inf)),
            '75% quantile set size': np.quantile(np.nan_to_num(sizes1, nan=np.inf), 0.75),
            '90% quantile set size': np.quantile(np.nan_to_num(sizes1, nan=np.inf), 0.9),
            '95% quantile set size': np.quantile(np.nan_to_num(sizes1, nan=np.inf), 0.95)
        }, index=[0])]

        df_list_for_table += [pd.DataFrame({
            'Model type': model_name,
            'Method': method_title_map[args.key2],
            'Marginal coverage': marginal_coverage2,
            # The next line gets the longest sequence of `True' in the boolean array ((y < sets2[0]) | (y > sets2[1]))
            'Longest err sequence': longest_err_seq2,
            'Average set size': np.mean(sizes2) if not np.any(np.isinf(sizes2)) else np.inf,
            'Median set size': np.median(np.nan_to_num(sizes2, nan=np.inf)),
            '75% quantile set size': np.quantile(np.nan_to_num(sizes2, nan=np.inf), 0.75),
            '90% quantile set size': np.quantile(np.nan_to_num(sizes2, nan=np.inf), 0.9),
            '95% quantile set size': np.quantile(np.nan_to_num(sizes2, nan=np.inf), 0.95)
        }, index=[0])]

        # Create pandas Series from the arrays with a simple numeric index
        time_series1 = pd.Series(covered1)
        time_series2 = pd.Series(covered2)

        window_start = args.window_start
        window_end = args.window_start + args.window_length

        savename = datasetname + '_' + model_name + '_' + args.key1 + '_lr' + str(args.lr1) + '_' + args.key2 + '_lr' + str(args.lr2) + '_window' + str(args.window_length) + '_start' + str(args.window_start) + str(args.coverage_inset) + str(args.set_inset)

        # Call the plot_time_series function to plot the data
        set_plot_style()
        plot_everything([time_series1, time_series2], [sizes1, sizes2], [sets1, sets2], y, alpha, window_start, window_end, args.window_loc, args.coverage_inset, args.size_inset, args.set_inset, args.miscoverage_scatterplot, savename, model_name, datetimes=data['data'].index[T_burnin+1:][args.coverage_average_burnin:], colors=[method_color_map[args.key1], method_color_map[args.key2]], labels=[method_title_map[args.key1], method_title_map[args.key2]])
    # Create a dataframe from the list of dataframes
    df = pd.concat(df_list_for_table)

    # Create a string representing a latex table with a multi-column index, where the upper column is the model type, the lower columns are the methods, and the rows are the metrics
    latex_table = dataframe_to_latex(df)

    # Write the latex table to a file
    with open('./plots/1v1/' + datasetname + "_" + '_'.join(savename.split('_')[2:-2]) + '.tex', 'w') as f:
        f.write(latex_table)
