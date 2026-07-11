import os
import yaml
import json
import pandas as pd

# Fraction of points NOT kept by each method
compression_ratios = [0.75, 0.875, 0.9375]

# Model to process
model = 'Qwen/Qwen2.5-7B-Instruct'

# Directory containing results
results_dir = 'results/'

# Specify the desired column order
desired_order = ['qasper_e', 'multifieldqa_en_e', 'hotpotqa_e', '2wikimqa_e', 'gov_report_e', 'multi_news_e', 
                 'trec_e', 'triviaqa_e', 'samsum_e', 'passage_count_e', 'passage_retrieval_en_e', 'lcc_e', 'repobench-p_e', 'average']

# Rename columns to shorter display names
column_mapping = {
    'press_name': 'Method',
    'qasper_e': 'qasper',
    'multifieldqa_en_e': 'multifield',
    'hotpotqa_e': 'hotpot',
    '2wikimqa_e': '2wiki',
    'gov_report_e': 'gov',
    'multi_news_e': 'multinews',
    'trec_e': 'trec',
    'triviaqa_e': 'trivia',
    'samsum_e': 'samsum',
    'passage_count_e': 'p.count',
    'passage_retrieval_en_e': 'p.ret',
    'lcc_e': 'lcc',
    'repobench-p_e': 'repo-p',
    'average': 'average'
}

# Map press_name values to display names
def map_press_name(name):
    press_name_mapping = {
        'no_press': 'Exact',
        'balance_kv': 'BalanceKV',
        'snapkv': 'SnapKV',
        'streaming_llm': 'StreamingLLM',
        'pyramidkv': 'PyramidKV',
        'uniform': 'Uniform'
    }
    
    # Check if it's in the mapping
    if name in press_name_mapping:
        return press_name_mapping[name]
    
    # Check if it starts with compress_kv_
    if name.startswith('compress_kv_'):
        return f'CompressKV'
    
    return name


def collect_results_for_ratio(compression_ratio):
    data = []

    for subfolder in os.listdir(results_dir):
        subfolder_path = os.path.join(results_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        config_path = os.path.join(subfolder_path, 'config.yaml')
        if not os.path.exists(config_path):
            continue

        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        if config.get('model') != model:
            continue

        if config.get('press_name') != 'no_press' and config.get('compression_ratio') != compression_ratio:
            continue

        metrics_path = os.path.join(subfolder_path, 'metrics.json')
        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, 'r') as metrics_file:
            metrics = json.load(metrics_file)

        data.append(
            {
                'press_name': config.get('press_name'),
                'data_dir': config.get('data_dir'),
                'accuracy': metrics.get('all', 0),
            }
        )

    return pd.DataFrame(data)


def build_ratio_table(compression_ratio):
    df = collect_results_for_ratio(compression_ratio)
    if df.empty:
        return None

    pivot_df = df.pivot(index='press_name', columns='data_dir', values='accuracy')
    pivot_df['average'] = pivot_df.mean(axis=1)

    available_cols = [col for col in desired_order if col in pivot_df.columns]
    pivot_df = pivot_df[available_cols]
    pivot_df = pivot_df.sort_values('average', ascending=True).reset_index()

    if 'no_press' in pivot_df['press_name'].values:
        no_press_row = pivot_df[pivot_df['press_name'] == 'no_press']
        pivot_df = pivot_df[pivot_df['press_name'] != 'no_press']
        pivot_df = pd.concat([no_press_row, pivot_df], ignore_index=True)

    pivot_df = pivot_df.rename(columns=column_mapping)

    mapped_methods = pivot_df['Method'].map(map_press_name)
    pivot_df = pivot_df[mapped_methods != pivot_df['Method']].copy()
    pivot_df['Method'] = mapped_methods[mapped_methods.index]

    compresskv_indices = pivot_df[pivot_df['Method'].str.startswith('CompressKV')].index.tolist()
    rows_to_drop = compresskv_indices[:-1] if len(compresskv_indices) > 1 else []
    pivot_df = pivot_df.drop(rows_to_drop)

    pivot_df_for_max = pivot_df[pivot_df['Method'] != 'Exact']
    max_avg = pivot_df_for_max['average'].max() if not pivot_df_for_max.empty else None

    def format_average(row):
        if max_avg is not None and row['Method'] != 'Exact' and row['average'] == max_avg:
            return f'\\textbf{{{row["average"]:.2f}}}'
        return f'{row["average"]:.2f}'

    pivot_df['average'] = pivot_df.apply(format_average, axis=1)
    pivot_df['Method'] = pivot_df['Method'].apply(lambda x: f'\\textbf{{{x}}}')

    return pivot_df


def ratio_label(compression_ratio):
    compression_percent = (compression_ratio * 100)
    return f'{compression_percent}\\% Compression'


def format_numeric_cell(value):
    if pd.isna(value):
        return '-'
    return f'{value:.2f}'


ratio_tables = []
for compression_ratio in compression_ratios:
    ratio_table = build_ratio_table(compression_ratio)
    if ratio_table is not None and not ratio_table.empty:
        ratio_tables.append((compression_ratio, ratio_table))

if not ratio_tables:
    print('No matching results found for the configured compression_ratios.')
    raise SystemExit(0)

display_columns = ['Method']
for source_col in desired_order:
    display_col = column_mapping[source_col]
    if any(display_col in table.columns for _, table in ratio_tables):
        display_columns.append(display_col)

# Reuse the method ordering from the first ratio table for all subsequent blocks.
reference_method_order = ratio_tables[0][1]['Method'].tolist()
for idx in range(1, len(ratio_tables)):
    compression_ratio, ratio_table = ratio_tables[idx]
    ratio_table = ratio_table.set_index('Method')
    ratio_table = ratio_table.reindex(reference_method_order)
    ratio_table = ratio_table.dropna(how='all')
    ratio_table = ratio_table.reset_index()
    ratio_tables[idx] = (compression_ratio, ratio_table)

bold_headers = [f'\\textbf{{{col}}}' for col in display_columns]
column_format = '@{\\hspace{2.5pt}}c' + '@{\\hspace{4pt}}c' * (len(display_columns) - 1) + '@{\\hspace{2.5pt}}'

latex_lines = [
    f'\\begin{{tabular}}{{{column_format}}}',
    '\\toprule',
    ' & '.join(bold_headers) + ' \\\\',
    '\\midrule',
]

for idx, (compression_ratio, ratio_table) in enumerate(ratio_tables):
    ratio_table = ratio_table.reindex(columns=display_columns)

    latex_lines.append(
        f'\\multicolumn{{{len(display_columns)}}}{{c}}{{\\textbf{{{ratio_label(compression_ratio)}}}}} \\\\'
    )
    latex_lines.append('\\midrule')

    for _, row in ratio_table.iterrows():
        row_values = []
        for col in display_columns:
            val = row[col]
            if isinstance(val, str):
                row_values.append(val)
            else:
                row_values.append(format_numeric_cell(val))
        latex_lines.append(' & '.join(row_values) + ' \\\\[.5mm]')

    if idx < len(ratio_tables) - 1:
        latex_lines.append('\\midrule')

latex_lines.append('\\bottomrule')
latex_lines.append('\\end{tabular}')

print('\n'.join(latex_lines))