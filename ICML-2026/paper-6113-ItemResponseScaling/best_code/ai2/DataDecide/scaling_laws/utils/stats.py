import itertools
import pandas as pd
from tqdm import tqdm

from utils.dataloader import get_slice
from utils.constants.constants_models import get_compute, DATA_DECIDE_MODEL_NAMES, DATA_DECIDE_SIZES
from utils.constants import REVERSED_METRICS
from utils.scaling_laws import compute_2_class


def get_perf_size(df, size, task, metric):
    """ Get performance of all models at a specific size """
    _slice: pd.DataFrame = get_slice(df, task=task)
    _slice = _slice[((_slice['size'] == size)) & (_slice['model'].isin(DATA_DECIDE_MODEL_NAMES))]
    if isinstance(task, str):
        _slice = _slice[_slice['task'] == task]
    elif isinstance(task, list):
        _slice = _slice[_slice['task'].isin(task)]

    # Only aggregate numerical columns
    numerical_cols = _slice.select_dtypes(include='number').columns.tolist()
    non_numerical_cols = _slice.select_dtypes(exclude='number').columns.tolist()
    _slice = _slice.groupby('model', as_index=False).agg({col: 'mean' for col in numerical_cols} | {col: 'first' for col in non_numerical_cols})
    _slice['task_name'] = 'aggregate'

    _slice = _slice.reset_index().sort_values('step')[['model', 'mix', 'step', 'size', metric]]
    _slice['compute'] = _slice['size'].apply(lambda x: get_compute(x) if '-' in x else x)
    _slice = _slice.sort_values(metric, ignore_index=True)
    return _slice


def get_perf_size_simple(df, size, task, metric):
    """ Get performance of all models at a specific size """
    _slice: pd.DataFrame = get_slice(df, task=task)
    
    # Get highest step for each model
    _slice = _slice[((_slice['size'] == size)) & (_slice['task'] == task)]
    _slice = _slice.loc[_slice.groupby('model')['step'].idxmax()]
    
    _slice['compute'] = _slice['model'].apply(lambda x: get_compute(x.split('-')[-2]) if '-' in x else x)
    _slice = _slice.sort_values(metric, ignore_index=True)
    
    return _slice


def compute_decision_accuracy(df, results, target_size):
    target_rankings = {}
    for task in results['task'].unique():
        for metric in results['metric'].unique():
            target_rankings[(task, metric)] = list(get_perf_size_simple(df, target_size, task, metric)['group'])

    results_grouped = results.groupby(['task', 'metric', 'setup'])

    decision_accs = []
    for (task, metric, setup), group in tqdm(results_grouped):
        if len(group) == 0:
            continue
            
        predicted_ranking = list(group.sort_values('stacked_pred')['mix'])
        target_ranking = target_rankings[(task, metric)]
        
        if metric in REVERSED_METRICS and metric not in REVERSED_METRICS:
            predicted_ranking = list(reversed(predicted_ranking))
        
        dec_acc = compute_2_class(predicted_ranking, target_ranking)

        dec_acc *= 100
        
        decision_accs.append({
            'task': task,
            'metric': metric,
            'setup': setup, 
            'decision_acc': dec_acc
        })

    # Create decision accuracy dataframe
    decision_acc_df = pd.DataFrame(decision_accs)
    results.drop(columns=[col for col in results.columns if col.startswith('decision_acc')], inplace=True) # delete existing decision acc cols
    results = results.merge(
        decision_acc_df,
        on=['task', 'metric', 'setup'],
        how='left'
    )

    return results


def construct_2class_table(df, selected_tasks, small_metric, target_metric, model_sizes=DATA_DECIDE_SIZES):
    """
    Compute 2-class accuracy. We are predicting primary_metric at 1B using the metric at a smaller scale
    """
    if not isinstance(small_metric, list): small_metric = [small_metric]

    combinations = list(itertools.product(small_metric, model_sizes, selected_tasks))
    two_class = pd.DataFrame(columns=['metric', 'size', 'task', 'accuracy'])

    for metric, size, task in tqdm(combinations, desc='Computing two class accuracy', disable=(len(combinations) < 50)):
        _slice = get_slice(df, task=task)
        # _slice = _slice[((_slice['size'] == size)) & (_slice['task'] == task) & (_slice['model'].isin(DATA_DECIDE_MODEL_NAMES))] # get data for small scale
        _slice = _slice[((_slice['size'] == size)) & (_slice['model'].isin(DATA_DECIDE_MODEL_NAMES))] # get data for small scale
        steps = [sorted(_slice['step'].unique())[-1]]
        for step in steps:
            # get data at the small scale
            small_scale = get_perf_size(df, size, task, metric)['mix']

            # predict at the target scale (1B) 
            target_scale = get_perf_size(df, '1B', task, target_metric)['mix']

            # display(_slice)
            # # display(target_scale)
            # if size == '150M':
            #     raise RuntimeError()
            
            if metric in REVERSED_METRICS and target_metric not in REVERSED_METRICS: small_scale = reversed(small_scale)
            try:
                accuracy = compute_2_class(small_scale, target_scale)
            except Exception as e:
                print((metric, size, task), e)
                accuracy = float('-inf')

            # Get tokens/compute of small scale
            step_slice = _slice[_slice['step'] == float(step)]
            step_slice = step_slice.reset_index(drop=True)
            # tokens = step_slice['tokens'][0]
            try:
                compute = get_compute(step_slice['size'][0])
            except Exception as e:
                print((metric, size, task), e)
                compute = float('-inf')

            new_entry = pd.DataFrame({
                'metric': [metric],
                'size': [size], 
                'step': [step], 
                'task': [str(task)],
                'accuracy': [accuracy],
                # 'tokens': [tokens],
                'compute': [compute]
            })
            new_entry = new_entry.dropna(axis=1, how='all')            
            two_class = two_class.dropna(axis=1, how='all')            
            two_class = pd.concat([two_class, new_entry], ignore_index=True)

    # Create two dataframes - one for best accuracies and one for corresponding metrics
    best_acc_df = two_class.loc[two_class.groupby(['task', 'size', 'step'])['accuracy'].idxmax()][['task', 'size', 'step', 'accuracy', 'compute']].reset_index(drop=True)
    best_metric_df = two_class.loc[two_class.groupby(['task', 'size', 'step'])['accuracy'].idxmax()][['task', 'size', 'step', 'metric', 'compute']].reset_index(drop=True)

    # Create pivot tables with size in specified order
    acc_pivot = best_acc_df.pivot(index='task', columns=['size', 'compute'], values='accuracy')[model_sizes]
    metric_pivot = best_metric_df.pivot(index='task', columns=['size', 'compute'], values='metric')[model_sizes]

    # Add average row
    acc_pivot.loc['average'] = acc_pivot.mean()

    return two_class, acc_pivot, metric_pivot
