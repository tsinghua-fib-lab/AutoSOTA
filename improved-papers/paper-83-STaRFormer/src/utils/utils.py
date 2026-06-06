import random
import numpy as np
import torch

__all__ = [
    'seed_everything',
    'seed_worker',
    'recusively_convert_dict_for_tensorboard'
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def recusively_convert_dict_for_tensorboard(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, list):
            dictionary[key] = str(value)

        if not isinstance(value, (int, str, bool, float, torch.Tensor, dict)):
            print(TypeError(f'{type(value)}'))


        if isinstance(value, dict):
            recusively_convert_dict_for_tensorboard(value)
    
    return dictionary