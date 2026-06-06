from typing import Dict
import numpy as np
import torch
from probts.model.forecaster import Forecaster
import importlib
import json
import pandas as pd
import pickle
import os

def update_metrics(new_metrics: Dict, stage: str, key: str = '', target_dict = {}):
    prefix = stage if key == '' else f'{stage}_{key}'
    for metric_name, metric_value in new_metrics.items():
        metric_key = f'{prefix}_{metric_name}'
        if metric_key not in target_dict:
            target_dict[metric_key] = []
            
        if isinstance(metric_value, list):
            target_dict[metric_key] = target_dict[metric_key] + metric_value
        else:
            target_dict[metric_key].append(metric_value)
        
    return target_dict

def calculate_average(metrics_dict: Dict, hor=''):
    metrics = {}
    if hor != '':
        hor = hor + '/'

    for key, value in metrics_dict.items():
        metrics[hor+key] = np.mean(value)
    return metrics


def calculate_weighted_average(metrics_dict: Dict, batch_size: list, hor=''):
    metrics = {}
    for key, value in metrics_dict.items():
        metrics[hor+key] = np.sum(value * np.array(batch_size)) / np.sum(batch_size)
    return metrics

def save_point_error(target, predict, input_dict, hor_str):
    if hor_str not in input_dict:
        input_dict[hor_str] = {'MAE': [], 'target': [], 'forecast': []}
    
    abs_error = np.abs(target - predict)

    input_dict[hor_str]['MAE'].append(abs_error)
    input_dict[hor_str]['target'].append(target)
    input_dict[hor_str]['forecast'].append(predict)
    return input_dict


def load_checkpoint(Model, checkpoint_path, scaler=None, learning_rate=None, no_training=False, **kwargs):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    # Extract the arguments for the forecaster
    forecaster_args = checkpoint['hyper_parameters']['forecaster']

    if isinstance(forecaster_args, Forecaster):
        forecaster = forecaster_args
    else:
        module_path, class_name = forecaster_args['class_path'].rsplit('.', 1)
        forecaster_class = getattr(importlib.import_module(module_path), class_name)
        
        # Add any missing required arguments
        forecaster_args = forecaster_args['init_args']
        forecaster_args.update(kwargs)
        
        # Create the forecaster
        forecaster = forecaster_class(**forecaster_args)
    
    forecaster.no_training = no_training
    
    if learning_rate is None:
        learning_rate = checkpoint['hyper_parameters'].get('learning_rate', 1e-3)
    
    # Create the model instance
    model = Model(
        forecaster=forecaster,
        scaler=scaler,
        num_samples=checkpoint['hyper_parameters'].get('num_samples', 100),
        learning_rate=learning_rate,
        quantiles_num=checkpoint['hyper_parameters'].get('quantiles_num', 10),
        load_from_ckpt=checkpoint['hyper_parameters'].get('load_from_ckpt', None),
        **kwargs  # Pass additional arguments here
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_hor_str(prediction_length, dataloader_idx):
    if dataloader_idx is not None:
        hor_str = str(prediction_length[dataloader_idx])
    elif type(prediction_length) == list:
        hor_str = str(prediction_length[0])
    else:
        hor_str = str(prediction_length)
    return hor_str


def save_exp_summary(pl_module, inference=False):
    exp_summary = {}
    
    model_summary = pl_module.model_summary_callback._summary(pl_module.trainer, pl_module.model)
    exp_summary['total_parameters'] = model_summary.total_parameters
    exp_summary['trainable_parameters'] = model_summary.trainable_parameters
    exp_summary['model_size'] = model_summary.model_size
    
    memory_summary = pl_module.memory_callback.memory_summary
    exp_summary['memory_summary'] = memory_summary
    
    time_summary = pl_module.time_callback.time_summary
    exp_summary['time_summary'] = time_summary
    for batch_key, batch_time in time_summary.items():
        if len(batch_time) > 0:
            exp_summary[f'mean_{batch_key}'] = sum(batch_time) / len(batch_time)
    
    exp_summary['sampling_weight_scheme'] = pl_module.model.sampling_weight_scheme
    
    if inference:
        summary_save_path = f"{pl_module.save_dict}/inference_summary.json"
    else:
        summary_save_path = f"{pl_module.save_dict}/summary.json"

    with open(summary_save_path, 'w') as f:
        json.dump(exp_summary, f, indent=4)
    print(f"Summary saved to {summary_save_path}")
    
    
def save_csv(model_args,save_dict, model, context_length):
    if len(model.avg_hor_metrics) > 0:
        horizon_list = []
        for horizon in model.avg_hor_metrics:
            horizon_dict = model.avg_hor_metrics[str(horizon)]
            horizon_dict['horizon'] = horizon
            horizon_list.append(horizon_dict)
            
        df = pd.DataFrame(horizon_list)
        
    else:
        df = pd.DataFrame([model.avg_metrics])

    df['model_args'] = model_args
    df = df[['model_args'] + df.columns[:-1].tolist()]
    if not model.forecaster.no_training:
        test_result_file = 'horizons_results'
    else:
        test_result_file = f'testctx_{context_length}_horizons_results'
        
    df.to_csv(f'{save_dict}/{test_result_file}.csv', index='idx')
    print('horizons result saved to ', f'{save_dict}/{test_result_file}.csv')