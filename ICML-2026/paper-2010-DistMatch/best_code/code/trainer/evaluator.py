from typing import Callable

import torch
from torch import nn
from torchmetrics import MetricCollection
from tqdm import tqdm

from utils.utils import get_device


class Evaluator:

    def __init__(self, move_batch_to_device: Callable, map_to_model_in: Callable,
                 val_metrics: MetricCollection, map_to_metrics_in: Callable, gpu_id):
        self._move_batch_to_device = move_batch_to_device
        self._map_to_model_in = map_to_model_in
        self._val_metrics = val_metrics.clone()
        self._map_to_metrics_in = map_to_metrics_in
        self.device = get_device(gpu_id)
        self._val_metrics.to(device=self.device)

    def eval(self, dataloader, trained_model: nn.Module, eval_text="on given dataloader"):
        pbar = tqdm(dataloader, desc=f'Evaluate model {eval_text}')

        model_out_log = []
        trained_model.to(device=self.device)
        trained_model.train(False)
        for batch_idx, batch_data in enumerate(pbar):
            batch_data = self._move_batch_to_device(batch_data=batch_data, device=self.device)
            model_in = self._map_to_model_in(batch_data)  # Gen model Input
            with torch.no_grad():
                model_out = trained_model(val=True, **model_in)
                metric_in = self._map_to_metrics_in(model_out, batch_data)
                m_val = self._val_metrics(**metric_in)
                model_out_log.append(model_out)

        # compute mean train_metrics over dataset
        metric_vals = self._val_metrics.compute()
        return model_out_log, metric_vals
