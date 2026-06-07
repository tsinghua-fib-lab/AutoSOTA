import abc
import logging
from enum import Enum

import torch
import torch.nn as nn

from .nets import MLP

logger = logging.getLogger(__name__)


def get_net(config: dict, *args, **kwargs) -> nn.Module:
    if config['type'] == 'mlp':
        return MLP(config, *args, **kwargs)
    else:
        raise NotImplementedError
    

class Predictor(abc.ABC):
    class TrainingMode(Enum):
        TRAIN = 'train'
        EVAL = 'eval'

    def __init__(self, config: dict):
        self.config = config
        self.type = config['type']
        # config['net']['head'] = self.type

        self.mode = Predictor.TrainingMode.TRAIN
        self.net: nn.Module = get_net(config['net'])
        self.loss_fn = nn.BCEWithLogitsLoss()

    def prepare_for_inference(self, *args, **kwargs):
        ...

    def load_pretrained(self, pretrained_net_path: str):
        self.net.load_state_dict(torch.load(pretrained_net_path, weights_only=True))
        self.to_eval_mode()
        logger.info(f'Loaded pretrained net from {pretrained_net_path}')

    def _is_train_mode(self):
        return self.mode == Predictor.TrainingMode.TRAIN
    
    def to_train_mode(self):
        self.mode = Predictor.TrainingMode.TRAIN
        self.net.train()

    def to_eval_mode(self):
        self.mode = Predictor.TrainingMode.EVAL
        self.net.eval()

    def to_mode(self, mode):
        if mode == Predictor.TrainingMode.TRAIN:
            self.to_train_mode()
        elif mode == Predictor.TrainingMode.EVAL:
            self.to_eval_mode()
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
    def get_mode(self):
        return self.mode

    def to_device(self, device):
        self.device = device
        self.net.to(device)

    @abc.abstractmethod
    def calculate_loss(self, targets, inputs, return_preds=False):
        raise NotImplementedError
    
    @abc.abstractmethod
    def predict(self, inputs):
        raise NotImplementedError
