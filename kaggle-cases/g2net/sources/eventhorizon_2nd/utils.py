from pprint import pformat
import types
import os
import requests


def print_config(cfg, logger=None):

    def _print(text):
        if logger is None:
            print(text)
        else:
            logger(text)
    
    items = [
        'name', 
        'cv', 'num_epochs', 'batch_size', 'seed',
        'dataset', 'dataset_params', 'num_classes', 'transforms', 'splitter',
        'model', 'model_params', 'weight_path', 'optimizer', 'optimizer_params',
        'scheduler', 'scheduler_params', 'batch_scheduler', 'scheduler_target',
        'criterion', 'eval_metric', 'monitor_metrics',
        'amp', 'parallel', 'hook', 'callbacks', 'deterministic', 
        'clip_grad', 'max_grad_norm',
        'pseudo_labels'
    ]
    _print('===== CONFIG =====')
    for key in items:
        try:
            val = getattr(cfg, key)
            if isinstance(val, (type, types.FunctionType)):
                val = val.__name__ + '(*)'
            if isinstance(val, (dict, list)):
                val = '\n'+pformat(val, compact=True, indent=2)
            _print(f'{key}: {val}')
        except:
            _print(f'{key}: ERROR')
    _print(f'===== CONFIGEND =====')


def notify_me(text):
    # Sample: LINE notify API
    # line_notify_token = '{Your token}'
    # line_notify_api = 'https://notify-api.line.me/api/notify'
    # headers = {'Authorization': f'Bearer {line_notify_token}'}
    # data = {'message': '\n' + text}
    # requests.post(line_notify_api, headers=headers, data=data)
    pass
