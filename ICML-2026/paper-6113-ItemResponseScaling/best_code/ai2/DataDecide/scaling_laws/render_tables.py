from pathlib import Path
import pandas as pd
import argparse
from utils.table import fix_table_rendering, render_result_table
from utils.constants import DATA_DIR, TABLE_DIR
from remote.hf import pull_predictions_from_hf

def save_to_tex(out_str, filename):
    output_path = Path(TABLE_DIR) / filename
    with open(output_path, 'w') as f:
        f.write(out_str)

def _render_mix_table(results):
    mix_table = render_result_table(results, 'mix', only_use_default_scaling_law=True)

    mix_table.loc['Average', 'OLMES Avg. default'] = str(mix_table[mix_table.index != 'Average']['OLMES Avg. default'].str.rstrip('%').astype(float).mean()) + '%'

    # Fix index names
    mix_table = mix_table.rename(columns={
        'Abs Error default': 'Error',
        'Abs Error -750M': 'Error w/o 750M', 
        'Abs Error -750M -530M': 'Error w/o 750M, 530M',
        'Rel Error default': '% Error',
        'Rel Error -750M': '% Error w/o 750M',
        'Rel Error -750M -530M': '% Error w/o 750M, 530M'
    })

    CAPTION = "Prediction error for scaling law fit on \\textsf{primary\\_metric} for all 25 data recipes, averaged across all tasks, along with their \\textsf{primary\\_metric} performance on OLMES. Figure \\ref{fig:mmlu_example_all_metrics} shows an example of each data mix fit across metrics."
    mix_table = mix_table.rename_axis('Data Recipe').to_latex(
        float_format=lambda x: x,
        label="tab:mix-errors",
        escape=False,
        column_format='p{0.3\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}'
    )
    mix_table = mix_table.replace('table', 'table*')\
        .replace('\\begin{table*}', '\\begin{table*}\n\\small\n\\centering')\
        .replace('\\end{table*}', '\\caption{' + CAPTION + '}\n\\end{table*}')
    mix_table = fix_table_rendering(mix_table)

    print(mix_table)

    save_to_tex(mix_table, 'mix_error.tex')


def _render_task_table(results):
    from utils.constants.constants_recepies import MIX_TWO_CLASS_RESULTS

    task_table = render_result_table(results, 'task', only_use_default_scaling_law=True)

    # Add Ian's two class prediction results
    task_table['Decision Acc.'] = task_table.index.map(MIX_TWO_CLASS_RESULTS)
    task_table.loc['Average', 'OLMES Avg. default'] = task_table[task_table.index != 'Average']['OLMES Avg. default'].str.rstrip('%').astype(float).mean()
    task_table.loc['Average', 'Decision Acc.'] = task_table['Decision Acc.'].mean()
    task_table['Decision Acc.'] = (task_table['Decision Acc.'] * 100).round(2).astype(str) + '%'
    task_table['OLMES Avg. default'] = task_table['OLMES Avg. default'].map(lambda x: f"{str(x)[:5]}%")

    # Move olmes_10_macro_avg to bottom of index and remove Average
    task_table = task_table.reindex(
        [idx for idx in task_table.index if idx not in ['olmes_10_macro_avg', 'Average']] + ['olmes_10_macro_avg']
    )

    # Fix index names
    task_table = task_table.rename(columns={
        'Abs Error default': 'Error',
        'Abs Error -750M': 'Error w/o 750M', 
        'Abs Error -750M -530M': 'Error w/o 750M, 530M',
        'Rel Error default': '% Error',
        'Rel Error -750M': '% Error w/o 750M',
        'Rel Error -750M -530M': '% Error w/o 750M, 530M'
    })

    CAPTION = "Per-task absolute prediction error (Error) and relative prediction error (% Error) for the two-step scaling law \\citep{bhagia2024establishingtaskscalinglaws} on \\textsf{primary\\_metric}, along with the average performance over \\textsf{primary\\_metric} and the 2-class best prediction accuracy reported in Fig. \\ref{fig:best_vs_primary_metric_accuracy}."
    task_table = task_table.rename_axis('Task').to_latex(
        float_format=lambda x: x,
        label="tab:task-errors", 
        escape=False,
        column_format='p{0.15\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}p{0.06\\textwidth}'
    )
    task_table = task_table\
        .replace('table', 'table*')\
        .replace('\\begin{table*}', '\\begin{table*}\n\\small\n\\centering')\
        .replace('\\end{table*}', '\\caption{' + CAPTION + '}\n\\end{table*}')\
        .replace('nan%', '--')\
        .replace('OLMES Avg. default', 'Primary Metric Avg.')
    task_table = fix_table_rendering(task_table)
    task_table = task_table.replace('\nOLMES Avg.', '\n\\midrule\nOLMES Avg.')

    print(task_table)

    save_to_tex(task_table, 'fit_error.tex')


def _render_perf_table(results):
    setup_table = render_result_table(results, index='setup', agg_col='metric', include_decision_acc=True)

    setup_table = setup_table\
        .drop(columns=['stacked_y_primary_metric'])\
        .rename_axis('Setup')

    CAPTION = "Average prediction error for different scaling law setups in \\citet{bhagia2024establishingtaskscalinglaws} across tasks on \\textsf{primary\\_metric}. As we are only calculating scaling law fit with 4 model sizes on the same token ratio, we find that predicting Task Loss directly from compute (FLOPs) in step 1 results in the lowest prediction errors for the \\projSuite{} recipes."
    setup_table = setup_table.to_latex(
        float_format=lambda x: x,
        label="tab:setup-errors", 
        escape=False,
        column_format='p{0.4\\textwidth}p{0.08\\textwidth}p{0.08\\textwidth}p{0.11\\textwidth}'
    )
    setup_table = setup_table\
        .replace('table', 'table*')\
        .replace('abs_error_stacked_primary_metric', 'Abs. Error')\
        .replace('rel_error_stacked_primary_metric', 'Rel. Error')\
        .replace('decision_acc_primary_metric', 'Decision Acc.')\
        .replace('\\begin{table*}', '\\begin{table*}\n\\small\n\\centering')\
        .replace('\\end{table*}', '\\caption{' + CAPTION + '}\n\\end{table*}')

    setup_table = fix_table_rendering(setup_table, scaling_law_table=True)

    # \midrule
    # \textbf{\citet{bhagia2024establishingtaskscalinglaws} using 150M, 300M, 530M, 750M}  & ~ & ~\\
    # \midrule
    # \textbf{\citet{bhagia2024establishingtaskscalinglaws} using 150M, 300M, 530M}  & ~ & ~\\
    # \midrule
    # \textbf{\citet{bhagia2024establishingtaskscalinglaws} using 150M, 300M}  & ~ & ~\\

    print(setup_table)

    save_to_tex(setup_table, 'setup_error.tex')


def _render_metric_table(results):
    print('Absolute error for each task/metric pair (3_param-default):')
    pivot = results[results['setup'] == '3_param-default'].pivot_table(
        index='task', 
        columns='metric', 
        values='decision_acc', 
        aggfunc='mean'
    )
    average_row = pivot.mean(axis=0)
    pivot_table_with_avg = pd.concat([pivot, pd.DataFrame([average_row], index=['Average'])])
    metric_decision_acc = (pivot_table_with_avg).round(2).map(lambda x: f"{x:.2f}%").T

    metric_decision_acc = metric_decision_acc.drop(columns=['olmes_10_macro_avg'])

    # Rename columns to clean titles
    from utils.constants.constants_recepies import TASK_NAME_LATEX
    metric_decision_acc = metric_decision_acc.rename(columns=lambda x: TASK_NAME_LATEX.get(x, x))

    CAPTION = "Decision accuracy for scaling law predictions across metrics, using the two-step prediction method."
    task_table = metric_decision_acc.rename_axis('Metric').to_latex(
        float_format=lambda x: x,
        label="tab:metric-errors", 
        escape=False,
        column_format='p{0.15\\textwidth}' + 'p{0.04\\textwidth}'*len(metric_decision_acc.columns)
    )
    task_table = task_table\
        .replace('table', 'table*')\
        .replace('\\begin{table*}', '\\begin{table*}\n\\tiny\n\\centering')\
        .replace('\\end{table*}', '\\caption{' + CAPTION + '}\n\\end{table*}')\
        .replace('CommonsenseQA', 'CSQA')\
        .replace('OpenBookQA', 'OBQA')\
        .replace('ARC-Challenge', 'ARC-C')\
        .replace('ARC-Easy', 'ARC-E')\
        .replace('_', '\\_')
    task_table = fix_table_rendering(task_table)

    print(task_table)

    save_to_tex(task_table, 'task_error.tex')


def render_all_tables(results_path):
    if str(results_path).endswith('.csv'):
        results = pd.read_csv(results_path)
    else:
        results = pd.read_parquet(results_path)

    _render_mix_table(results)
    _render_perf_table(results)
    _render_task_table(results)
    _render_metric_table(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-path', type=str, default='ladder_predictions.csv',
                      help='Path to results CSV (default: ladder_predictions.csv)')
    parser.add_argument('--hf-path', type=str,
                      help='HuggingFace dataset to load results from')
    args = parser.parse_args()

    if args.hf_path:
        local_path = pull_predictions_from_hf(args.hf_path, split_name='scaling_law_fit')
    else:
        local_path = Path(DATA_DIR) / args.result_path
    
    render_all_tables(local_path)