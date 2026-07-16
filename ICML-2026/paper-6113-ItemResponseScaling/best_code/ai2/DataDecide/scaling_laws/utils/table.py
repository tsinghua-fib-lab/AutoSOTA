from IPython.display import display, HTML
from utils.constants.constants_recepies import DATA_NAME_LATEX, SETUP_NAME_LATEX, TASK_NAME_LATEX
import pandas as pd

def highlight(x, as_prct=True, inverse=False):
    if inverse:
        if x > 0: style = 'color: red'
        elif x < 0: style = 'color: green'
        else: style = ''
    else:
        if x < 0: style = 'color: red'
        elif x > 0: style = 'color: green'
        else: style = ''
    if as_prct:
        text = f'<span style="{style};">{x*100:+.1f}%</span>'
    else:
        text = f'<span style="{style};">{x:+.2f}</span>'
    return text


def display_task_variants(results, key, transpose=False, as_prct=True, inverse=False, ascending=False):
    results = results.copy()
    results = results.transpose()

    # Split tasks "arc_easy:para" => ("arc_easy", "para")
    results['Task'] = results.index.str.split(':').str[0]
    results['Variant'] = results.index.str.split(':').str[1]

    # Reshape to a (Task, Variant) df
    results = results.pivot(index='Task', columns='Variant', values=key)
    results.columns = results.columns.fillna('default rc')
    results = results.infer_objects(copy=False)
    # results = results.fillna('--')
    results = results.fillna(float('-inf'))
    results = results.sort_values(by='default rc', ascending=ascending)

    # Calculate difference from "default rc" setup, display as percentage
    def format_row(x, baseline):
        diff = f' ({highlight(float(x) - baseline, as_prct=as_prct, inverse=inverse)})'
        if as_prct:
            return f"{x*100:.1f}%{diff}"
        else:
            return f"{x:.2f}{diff}"
    for col in results.columns[1:]:
        results[col] = results.apply(
            lambda row: format_row(row[col], row['default rc'])
            if row[col] != float('-inf') else '--',
            axis=1
        )
    results['default rc'] = results['default rc'].apply(
        lambda x: f"{x*100:.1f}%" if as_prct else f"{x:.2f}"
    )

    display_table(results, transpose=transpose)


def display_table(df, transpose=False, monospace=False):
    # Show as HTLM
    if transpose: df = df.transpose()
    html = df.to_html(escape=False)
    styled_html = (
        '<div style="overflow-x: auto; white-space: nowrap;">' +
        html +
        '</div>'
    )
    if monospace: styled_html = styled_html.replace('<table', '<table style="font-family: monospace"')
    styled_html = styled_html.replace('<td', '<td style="white-space: nowrap;"') # no wrap on table cells!
    display(HTML(styled_html))

def render_result_table(results, index, agg_col='setup', only_use_default_scaling_law=False, raw_values=False, include_decision_acc=False):
    """ Convert a df of results to LaTeX """
    if index != 'metric':
        filtered_results = results[results['metric'] == 'primary_metric']
    else:
        filtered_results = results

    # Select numeric columns for aggregation
    agg_cols = ['abs_error_stacked', 'rel_error_stacked', 'stacked_y']
    if include_decision_acc: 
        agg_cols += ['decision_acc']

    # Compute averages for each unique value in 'setup'
    average_by_setup = filtered_results.groupby([index, agg_col])[agg_cols].mean().reset_index()

    # Pivot the table to have 'setup' as columns for comparison
    pivoted_results = average_by_setup.pivot(index=index, columns=agg_col, values=agg_cols).reset_index()

    # Flatten the multi-index columns
    pivoted_results.columns = ['_'.join(col).strip('_') for col in pivoted_results.columns.values]

    # Format percentage columns
    for col in pivoted_results.columns:
        if 'abs_error_stacked' in col or 'rel_error_stacked' in col:
            pivoted_results[col] = (pivoted_results[col] * 100).round(2)

    # Calculate averages for each column and append as a new row
    avg_row = {col: pivoted_results[col].mean() if 'abs_error_stacked' in col or 'rel_error_stacked' in col else 'Average' for col in pivoted_results.columns}
    pivoted_results = pd.concat([pivoted_results, pd.DataFrame([avg_row])], ignore_index=True)

    # Display results
    pivoted_results = pivoted_results.set_index(index)

    # Multiply OLMES Avg col by 100
    for col in pivoted_results.columns:
        if 'stacked_y_' in col and index != 'metric':
            pivoted_results[col] = pivoted_results[col].apply(lambda x: x * 100 if isinstance(x, float) else x)
    
    if not raw_values:
        pivoted_results = pivoted_results.map(lambda x: (f"{x:.2f}" if isinstance(x, float) else x) + "%")

    if only_use_default_scaling_law:
        pivoted_results = pivoted_results[[col for col in pivoted_results.columns if col.endswith("3_param-default") or col.endswith("3_param-no_750M") or col.endswith("3_param-no_750M_no_530M")]]
        pivoted_results = pivoted_results.sort_values('abs_error_stacked_3_param-default')
        pivoted_results.rename(columns=lambda col: col.replace("stacked_y_", "OLMES Avg. ").replace("abs_error_stacked_", "Abs Error ").replace("rel_error_stacked_", "Rel Error ").replace("3_param-", "").replace("no_", "-").replace("_", " "), inplace=True)
        pivoted_results = pd.concat([pivoted_results.loc[pivoted_results.index != "Average"], pivoted_results.loc[["Average"]]])

    if index == 'setup':
        # Create a mapping from index values to their order in SETUP_NAME_LATEX
        setup_order = {k: i for i, k in enumerate(SETUP_NAME_LATEX.keys())}

        # Remove any rows from pivoted_results that aren't in SETUP_NAME_LATEX
        pivoted_results = pivoted_results[pivoted_results.index.isin(list(SETUP_NAME_LATEX.keys()))]
        
        # Sort the index based on the order in SETUP_NAME_LATEX
        pivoted_results = pivoted_results.reindex(sorted(pivoted_results.index, key=lambda x: setup_order.get(x, float('inf')) if x != 'Average' else float('inf')))

    cols_to_drop = [col for col in pivoted_results.columns if "OLMES Avg." in col and 'default' not in col]
    pivoted_results = pivoted_results.drop(columns=cols_to_drop)

    if index == 'metric':
        cols_to_drop = [col for col in pivoted_results.columns if "Rel Error" in col]
        pivoted_results = pivoted_results.drop(columns=cols_to_drop)
    
    return pivoted_results


def fix_table_rendering(table, scaling_law_table=False):
    """ Fix formatting issues with pandas table formatter """
    lines = table.split('\n')
    # Process each line of the table
    for i, line in enumerate(lines):
        # if 'Task' in table: continue
        if not line or line.startswith(' ') or line.startswith('\\'): continue
        if 'Recipe' in line: continue
        
        # Map the data mix name to the latex name
        for key, value in DATA_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)
        
        # Map the data mix name to the latex name
        for key, value in SETUP_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)
        
        # Map the data mix name to the latex name
        for key, value in TASK_NAME_LATEX.items():
            if line.split(' ')[0] == key: 
                lines[i] = lines[i].replace(key, value)

    # Add midrule before Average row
    for i, line in enumerate(lines):
        if line.strip().startswith('Average'):
            lines.insert(i, '\\midrule')
            break
    
    # Rejoin and print
    table_str = '\n'.join(lines).replace('%', '\%')

    if scaling_law_table:
        table_str = table_str.replace('\n    ', '\n\\hspace{1em}\\hspace{1em}').replace('\n  ', '\n\\hspace{1em}')
        table_str = table_str.replace('\\midrule\n\\midrule', '\\midrule')

    return table_str
