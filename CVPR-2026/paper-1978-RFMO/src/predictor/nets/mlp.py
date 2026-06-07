import json
import time

import numpy as np
import torch
import torch.nn as nn

from .heads import (
    AutoregressiveHead,
    MultinomialHead,
    GibbsSamplingHead
)


class MLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_features = config['n_features']
        self.n_classes = config['n_classes']
        self.hidden_size_per_layer = config['hidden_size_per_layer']
        self.dropout_rate = config['dropout_rate']
        
        dim_per_layer = [self.n_features] + self.hidden_size_per_layer
        self.backbone = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(dim_per_layer[i], dim_per_layer[i+1]),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ) for i in range(len(dim_per_layer)-1)]
        )
        
        self.head_type = config['head']

        if self.head_type == 'br':
            self.head = nn.Linear(dim_per_layer[-1], self.n_classes)

        elif self.head_type == 'ar':
            head_hidden_size = config.get('head_hidden_size', None)
            self.topo_order: list = list(range(self.n_classes))
            self.parents: dict[int, list] = {i: list(range(i)) for i in range(self.n_classes)}
            self.head = AutoregressiveHead(dim_per_layer[-1], head_hidden_size, self.topo_order, self.parents)

        elif self.head_type == 'mn':
            self.head = MultinomialHead(dim_per_layer[-1], self.n_classes)
        elif self.head_type == 'gibbs':
            head_hidden_size = config.get('head_hidden_size', None)
            self.head = GibbsSamplingHead(dim_per_layer[-1], self.n_classes, head_hidden_size)
        elif self.head_type == 'ccs':
            self.head = ConvexCalibratedSurrogatesHead(dim_per_layer[-1], self.n_classes)
        elif self.head_type == 'lp':
            self.head = LabelPowersetHead(dim_per_layer[-1], self.n_classes)
        else:
            raise ValueError(f'Invalid head: {config["head"]}')

    def forward_backbone(self, x) -> torch.Tensor:
        return self.backbone(x)

    def forward_head(self, x, y=None) -> torch.Tensor:
        if self.head_type == 'br':
            return self.head(x)
        return self.head(x, y)

    def forward(self, x, y=None, return_last_layer_features=False):
        # start_backbone = time.time()
        last_layer_features = self.forward_backbone(x)
        # end_backbone = time.time()
        # print(f"forward_backbone time: {end_backbone-start_backbone:.8f}")
        outputs = self.forward_head(last_layer_features, y)
        # print(f"forward_head time: {time.time()-end_backbone:.8f}")
        # import sys; sys.exit()
        
        if return_last_layer_features:
            return outputs, last_layer_features
        return outputs
    
    def sample(self, x: torch.Tensor, num_samples: int, temperature: float, order=None, burn_in: int | None=None) -> torch.Tensor:
        assert self.head_type in ['ar', 'gibbs'],  'Sampling is only supported for AutoregressiveHead or GibbsSamplingHead'
        # start_backbone = time.time()
        x = self.forward_backbone(x)
        # end_backbone = time.time()
        # print(f"forward_backbone time: {end_backbone-start_backbone:.8f}")
        if self.head_type == 'gibbs':
            return self.head.sample(x, num_samples, temperature, burn_in)
        elif self.head_type == 'ar':
            ret = self.head.sample(x, num_samples, temperature)
            # print(f"sample time: {time.time()-end_backbone:.8f}")
            return ret
        else:
            raise NotImplementedError('Sampling is only supported for DAG type of ar or oaar')
        
    def greedy_predict(self, x: torch.Tensor) -> torch.Tensor:
        assert self.head_type == 'ar', 'Greedy prediction is only supported for AutoregressiveHead'

        x = self.forward_backbone(x)
        return self.head.greedy_predict(x)

    def evaluate_log_likelihood(self, inputs, targets, order, cut_off=-1) -> torch.Tensor:
        assert self.head_type == 'ar', 'Evaluating log likelihood is only supported for AutoregressiveHead'

        x = self.forward_backbone(inputs)
        return self.head.evaluate_log_likelihood(x, targets, order, cut_off)
