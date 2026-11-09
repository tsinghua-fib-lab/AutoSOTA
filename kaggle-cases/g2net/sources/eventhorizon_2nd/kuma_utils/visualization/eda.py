import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from pprint import pprint, pformat

import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns

from ..preprocessing import analyze_column
from ..utils import is_env_notebook

try:
    import japanize_matplotlib
except:
    print('japanize_matplotlib not found.')


def explore_data(
        train_df: pd.DataFrame, test_df: pd.DataFrame=None,
        exclude_columns: List[str]=[], normalize: bool=True, 
        plot: bool=True, scale: Union[int, float]=1.0, save_to: Union[str, Path]=None,
        verbose: bool=True,
    ) -> None:

    if verbose:
        print(f'data shape: {train_df.shape}, {test_df.shape if test_df is not None else None}\n')

    ''' Scan columns '''
    train_columns = train_df.columns.tolist()
    if test_df is not None:
        test_columns = test_df.columns.tolist()
        include_columns = set(train_columns) & set(test_columns)
        include_columns = list(include_columns - set(exclude_columns))
    else:
        include_columns = list(set(train_columns) - set(exclude_columns))
    n_columns = len(include_columns)
    if verbose:
        print(f'Included columns: \n{pformat(include_columns, compact=True)} ({n_columns})\n')
    if plot:
        fig_length = int(np.ceil(np.sqrt(n_columns)))
        fig = plt.figure(figsize=(fig_length*6*scale, fig_length*3*scale), tight_layout=True)
    
    ''' Check columns '''
    for icol, col in enumerate(include_columns):
        print(f'{icol:-{len(str(n_columns))+1}} {col}: ')
        vals = train_df[col]
        if test_df is not None:
            test_vals = test_df[col]

        column_type = analyze_column(vals)
        if column_type == 'numeric':
            nan_count = vals.isnull().sum()
            print(f'\n train NaN: {nan_count} ({nan_count/len(vals):.3f})')
            if test_df is not None:
                test_nan_count = test_vals.isnull().sum()
                print(f' test NaN: {test_nan_count} ({test_nan_count/len(test_vals):.3f}\n')
            
            if plot:
                ax1 = fig.add_subplot(fig_length, fig_length, icol+1)
                sns.distplot(vals.dropna(), ax=ax1, label='train')
                if test_df is not None:
                    sns.distplot(test_vals.dropna(), ax=ax1, label='test')
                ax1.legend()

        elif column_type == 'categorical':
            val_cnt = vals.value_counts(dropna=False)
            if test_df is not None:
                val_cnt = pd.concat(
                    [val_cnt, test_vals.value_counts(dropna=False)], 
                    axis=1).fillna(0)
                val_cnt.columns = ['train', 'test']
                if normalize:
                    val_cnt['train'] = val_cnt['train'] / val_cnt['train'].sum()
                    val_cnt['test'] = val_cnt['test'] / val_cnt['test'].sum()
                else:
                    val_cnt = val_cnt.astype(int)
            else:
                val_cnt = pd.DataFrame(val_cnt)
                val_cnt.columns = ['train']
                if normalize:
                    val_cnt['train'] = val_cnt['train'] / val_cnt['train'].sum()
                else:
                    val_cnt = val_cnt.astype(int)
            print(f'\n n_classes={len(val_cnt)}')
            print(f'{val_cnt}\n')

            if plot:
                ax1 = fig.add_subplot(fig_length, fig_length, icol+1)
                ax1.set_title(col)
                train_uni = set(vals.unique().tolist())
                if test_df is not None:
                    test_uni = set(test_vals.unique().tolist())
                    common_uni = train_uni & test_uni
                    train_uni = train_uni - common_uni
                    test_uni = test_uni - common_uni
                    venn2(subsets=(
                        len(train_uni), len(test_uni), len(common_uni)),
                        set_labels=('train', 'test'), ax=ax1
                    )
                else:
                    venn2(subsets=(
                        0, 0, len(train_uni)),
                        set_labels=('train', 'train'), ax=ax1
                    )

    if plot:
        if save_to:
            fig.savefig(save_to)
        plt.show()
