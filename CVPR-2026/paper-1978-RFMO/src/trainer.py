import os
import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
try:
    import wandb
except ImportError:
    wandb = None

from metrics import (
    AverageMeter,
    MetricTracker,
    calc_mlc_metrics,
)
from predictor import Predictor

class Trainer():
    def __init__(
            self,
            config: dict,
            predictor: Predictor,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            test_loader: Optional[torch.utils.data.DataLoader]=None,
            use_log: bool=True,
            use_wandb: bool=False,
        ):
        self.config: dict = config
        self.predictor: Predictor = predictor
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.val_loader: torch.utils.data.DataLoader = val_loader
        self.test_loader: Optional[torch.utils.data.DataLoader] = test_loader
        self.num_classes: int = train_loader.dataset.num_classes

        self.device = self.config['device']
        self.predictor.to_device(self.device)

        self.start_epoch = 1  # TODO: support resume
        self.n_epochs = self.config['n_epochs']
        self.val_period = self.config['val_period']
        self.early_stop_patience = self.config['early_stop_patience']
        self._reset_metrics()

        self.is_special = self.predictor.net.__class__.__name__ in ['GCNResnet']

        # Optimization
        # parameters = []
        # for name, param in self.predictor.net.named_parameters():
        #     if param.requires_grad:
        #         if 'masked' in name:
        #             parameters.append({'params': param, 'lr': self.config['optimizer']['args']['lr'] * 10})
        #         else:
        #             parameters.append({'params': param})
        # self.optimizer = getattr(torch.optim, self.config['optimizer']['type'])(parameters, **self.config['optimizer']['args'])
        if self.is_special:
            self.optimizer = getattr(torch.optim, self.config['optimizer']['type'])(self.predictor.net.get_config_optim(lr=self.config['optimizer']['args']['lr'], lrp=0.1), **self.config['optimizer']['args'])
        else:
            self.optimizer = getattr(torch.optim, self.config['optimizer']['type'])(self.predictor.net.parameters(), **self.config['optimizer']['args'])

        lr_scheduler_cls = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['type'])
        if self.config['lr_scheduler']['type'] == 'CosineAnnealingLR':
            self.lr_scheduler = lr_scheduler_cls(self.optimizer, T_max=self.n_epochs)
        elif self.config['lr_scheduler']['type'] == 'CosineAnnealingWarmRestarts':
            self.lr_scheduler = lr_scheduler_cls(self.optimizer, T_0=50, eta_min=2e-4, T_mult=2)
        elif self.config['lr_scheduler']['type'] == 'ExponentialLR':
            self.lr_scheduler = lr_scheduler_cls(self.optimizer, gamma=0.99)
        elif self.config['lr_scheduler']['type'] == 'LambdaLR':
            self.lr_scheduler = lr_scheduler_cls(self.optimizer, lr_lambda=lambda epoch: 1)  # constant lr
        elif self.config['lr_scheduler']['type'] == 'StepLR':
            self.lr_scheduler = lr_scheduler_cls(self.optimizer, step_size=10, gamma=0.9)
        else:
            raise NotImplementedError
        
        self.logger = logging.getLogger(__name__)
        self.use_log = use_log
        self.use_wandb = use_wandb

        self.save_dir = self.config['save_dir']
        self.pivot_metric = self.config['pivot_metric']
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_model_pth = os.path.join(self.save_dir, 'best_model.pth')
        self.pretrained_model_pth = self.config.get('pretrained_model_path')

        self.log_info(
            f'Train set size: {len(self.train_loader.dataset)}' + \
            f', Val set size: {len(self.val_loader.dataset)}' + \
            (f', Test set size: {len(self.test_loader.dataset)}' if self.test_loader else '')
        )
        self.log_info(f'Model size: {sum(p.numel() for p in self.predictor.net.parameters() if p.requires_grad)}')

    def log_info(self, msg: str):
        if self.use_log:
            self.logger.info(msg)

    def _train_epoch(self, epoch):
        self.predictor.to_train_mode()
        self._reset_metrics_epoch()

        if self.is_special:
            decay = 0.1 if sum(epoch == np.array([30, 80])) > 0 else 1.0
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # self.log_info(f"batch: {batch_idx}")
            batch_loss = self.predictor.calculate_loss(targets.to(self.device), inputs.to(self.device))

            self.optimizer.zero_grad()
            batch_loss.backward()
            if self.is_special:
                torch.nn.utils.clip_grad_norm_(self.predictor.net.parameters(), max_norm=10)
            self.optimizer.step()

            self.total_loss.update(batch_loss.item(), inputs.size(0))

        self.log_info('*'*5 + f' Train Results (Epoch {epoch}): ' + '*'*5)
        self.log_info(f'loss: {self.total_loss.average}')

        results = {'train/loss': self.total_loss.average}
        if self.use_wandb:
            wandb.log(results, step=epoch)

        return results

    def _val_epoch(self, epoch):
        self.predictor.to_eval_mode()
        self._reset_metrics_epoch()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                batch_loss, preds = self.predictor.calculate_loss(targets.to(self.device), inputs.to(self.device), return_preds=True)

                self.total_loss.update(batch_loss.item(), inputs.size(0))

                stats = {
                    'preds': preds.detach().cpu().numpy(),
                    'targets': targets.cpu().numpy(),
                }
                self._update_stats(**stats)

        metrics = self._calc_metrics()
        metrics['neg_loss'] = -self.total_loss.average

        pivot_metric = metrics[self.pivot_metric]
        self.val_metric_tracker.update(pivot_metric)
        torch.save(self.predictor.net.state_dict(), os.path.join(self.save_dir, f'epoch_{epoch}_model.pth'))
        if self.val_metric_tracker.is_ready() and self.val_metric_tracker.value() > self.best_val_metric:
            self.best_val_metric = self.val_metric_tracker.value()
            self.best_val_epoch = epoch
            torch.save(self.predictor.net.state_dict(), self.best_model_pth)
            self.log_info(f'Best model saved at {self.best_model_pth} (epoch {epoch}) with {self.pivot_metric}={self.best_val_metric}')

        self.log_info('*'*5 + f' Validation Results (Epoch {epoch}): ' + '*'*5)
        for k, v in metrics.items():
            self.log_info(f'{k}: {v}')

        results = {
            'val/loss': self.total_loss.average,
            'val/hamming_accuracy': metrics['hamming_accuracy'],
            'val/subset_accuracy': metrics['subset_accuracy'],
            'val/instance_f1': metrics['instance_f1'],
            'val/micro_f1': metrics['micro_f1'],
            'val/macro_f1': metrics['macro_f1'],
            f'val/best_{self.pivot_metric}': self.best_val_metric,
        }
        if self.use_wandb:
            wandb.log(results, step=epoch)
        
        return results

    def train(self):
        assert self.pretrained_model_pth is None, f'Conflict: execute train() when given pretrained_model_pth={self.pretrained_model_pth}'

        self._reset_metrics()
        for epoch in range(self.start_epoch, self.n_epochs+1):
            train_results = self._train_epoch(epoch)

            if self.val_loader is not None and (epoch % self.val_period == 0 or epoch == self.n_epochs):
                val_results = self._val_epoch(epoch)
                if self.val_metric_tracker.is_ready() and self.best_val_epoch + self.early_stop_patience < epoch:
                    self.log_info(f'Early stop at epoch {epoch}')
                    break

            self.lr_scheduler.step()

    def test(self):
        if self.pretrained_model_pth is not None:
            self.predictor.load_pretrained(self.pretrained_model_pth)
        else:
            self.predictor.load_pretrained(self.best_model_pth)
        
        do_order_selection = self.config.get('do_order_selection', False)
        if self.config.get('do_order_selection', False):
            best_order = self.predictor.order_selection(self.val_loader)

        self.test_predictions = []
        self._reset_metrics_epoch()
        import time
        start_time = time.time()
        mat_or_fft_time = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):

                # batch_loss, preds = self.predictor.calculate_loss(targets, inputs, return_preds=True)
                    
                # self.total_loss.update(batch_loss.item(), inputs.size(0))

                if do_order_selection:
                    preds = self.predictor.predict(inputs, order=best_order)
                else:
                    preds, _mat_or_fft_time = self.predictor.predict(inputs.to(self.device), return_time=True)
                    mat_or_fft_time += _mat_or_fft_time

                stats = {
                    'preds': preds.detach().cpu().numpy(),
                    'targets': targets.cpu().numpy(),
                }
                self._update_stats(**stats)

                # if batch_idx == 10:
                #     import sys;sys.exit()

                self.test_predictions.append(preds.detach().cpu().numpy())
        print(f"Total inference time: {time.time() - start_time} seconds")
        self.test_predictions = np.concatenate(self.test_predictions)

        metrics = self._calc_metrics()

        self.log_info(f'Matmul/FFT time: {mat_or_fft_time} seconds')
        
        self.log_info('*'*5 + f' Test Results (Best Epoch {self.best_val_epoch}): ' + '*'*5)
        for k, v in metrics.items():
            self.log_info(f'{k}: {v}')

        results = {
            # 'test/loss': self.total_loss.average,
            'test/hamming_accuracy': metrics['hamming_accuracy'],
            'test/subset_accuracy': metrics['subset_accuracy'],
            'test/instance_f1': metrics['instance_f1'],
            'test/micro_f1': metrics['micro_f1'],
            'test/macro_f1': metrics['macro_f1'],
        }

        if self.use_wandb:
            wandb.log(results)

        return results

    def _reset_metrics(self):
        self.best_val_metric = float('-inf')
        self.best_val_epoch = 0
        self.val_metric_tracker = MetricTracker(window_size=1)

    def _reset_metrics_epoch(self):
        self.total_loss = AverageMeter()
        self.preds = []
        self.targets = []
        self.probs = []
        self.metrics = {}

    def _update_stats(
            self,
            preds: np.ndarray,
            targets: np.ndarray,
            probs: Optional[np.ndarray]=None,
        ):
        self.preds.append(preds)
        self.targets.append(targets)
        if probs is not None:
            self.probs.append(probs)

    def get_metrics(self):
        return self.metrics

    def _calc_metrics(self):
        self.metrics = calc_mlc_metrics(self.preds, self.targets, self.num_classes)
        return self.metrics
