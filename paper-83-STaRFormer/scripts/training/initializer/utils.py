import os.path as osp
import torch.nn.functional as F
import torch.nn as nn

from omegaconf import DictConfig
from datetime import datetime
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

def initialize_activation_function(config: DictConfig):
    if config.model.sequence_model.activation == 'elu':
        activation = F.elu
    elif config.model.sequence_model.activation == 'relu':
        activation = F.relu
    elif config.model.sequence_model.activation == 'selu':
        activation = F.selu
    elif config.model.sequence_model.activation == 'gelu':
        activation = F.gelu
    elif config.model.sequence_model.activation == 'silu':
        activation = F.silu
    elif config.model.sequence_model.activation == 'tanh':
        activation = F.tanh
    elif config.model.sequence_model.activation == 'sigmoid':
        activation = F.sigmoid
    else:
        raise RuntimeError(f'{config.model.sequence_model.activation} not found!')

    return activation

def initialize_activation_function_from_str(activation: str):
    if activation == 'elu':
        activation = nn.ELU()
    elif activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'selu':
        activation = nn.SELU()
    elif activation == 'gelu':
        activation = nn.GELU()
    elif activation == 'silu':
        activation = nn.SiLU()
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        raise RuntimeError(f'{activation} not found!')

    return activation

def initialize_exp_logger(config: DictConfig, **kwargs):
    now = datetime.now()
    instance = now.strftime("%Y-%m-%d_%H:%M:%S")

    assert kwargs.get('dir_path', None) is not None
    assert kwargs.get('dataset', None) is not None
    assert kwargs.get('training_method', None) is not None
    
    if config.logger.name == 'tensorboard':
        exp_logger = TensorBoardLogger(
            save_dir=osp.join(kwargs['dir_path'], 'runs', 'tensorboard', kwargs['dataset'], kwargs['training_method'], 'harbor'),
            name=instance
        )

    elif config.logger.name == 'wandb':
        exp_logger = WandbLogger(
            name=instance, #config.logger.exp_name, 
            save_dir=osp.join(kwargs['dir_path'], 'runs', 'wandb', kwargs['dataset'], kwargs['training_method'], 'harbor'), 
            project=config.logger.project, 
            entity=config.logger.entity
        )

    return exp_logger, instance