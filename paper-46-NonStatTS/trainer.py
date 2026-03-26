import os
import time
from typing import Optional, Tuple, Mapping, Union, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Optimizer

import models.optimizer as optim
from models.build import load_best_model
from models.forecast import forecast
from datasets.loader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from utils.misc import mkdir
from utils.meters import AverageMeter, ProgressMeter
from config import get_norm_method, get_norm_module_cfg
from utils.misc import prepare_inputs


class Trainer:
    def __init__(
            self,
            cfg,
            model,
            metric_names: Tuple[str],
            loss_names: Tuple[str],
            optimizer: Optional[Union[Optimizer, List[Optimizer]]] = None,
            norm_module: Optional[torch.nn.Module] = None
    ):
        self.cfg = cfg
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.optimizer = optimizer
        # create optimizer
        if self.optimizer is None:
            self.create_optimizer()

        assert len(metric_names) > 0 and len(loss_names) > 0
        self.metric_names = metric_names
        self.loss_names = loss_names

        self.cur_epoch_station = 0
        self.cur_iter_station = 0
        
        self.cur_epoch = 0
        self.cur_iter = 0

        # Create the train and val (test) loaders.
        self.train_loader = get_train_dataloader(self.cfg)
        self.val_loader = get_val_dataloader(self.cfg)
        self.test_loader = get_test_dataloader(self.cfg)

    def create_optimizer(self):
        self.optimizer = optim.construct_optimizer(self.model, self.cfg)
        if self.norm_method == 'RevIN':
            self.optimizer.add_param_group({
                'params': self.norm_module.parameters(), 
                'lr': self.cfg.SOLVER.BASE_LR,
                'weight_decay': self.cfg.SOLVER.WEIGHT_DECAY
                })
        if self.norm_method == "SAN":
            self.optimizer_stat = optim.construct_optimizer(self.norm_module, self.cfg.SAN)

    def train(self):
        if self.cfg.MODEL.NAME == 'OLS':
            self.model.fit_ols_solutions(self.train_loader)
            self.save_best_model()
            return
            
        if self.norm_method == 'SAN':
            best_metric = self.cfg.TRAIN.BEST_METRIC_INITIAL
            for cur_epoch in range(self.cfg.SAN.SOLVER.START_EPOCH, self.cfg.SAN.SOLVER.MAX_EPOCH):
                self.train_epoch_station()
                
                if self._is_eval_epoch(cur_epoch):
                    tracking_meter = self.eval_epoch_station()
                    is_best = self._check_improvement(tracking_meter.avg, best_metric)
                    if is_best:
                        with open(mkdir(self.cfg.SAN.RESULT_DIR) / "best_result.txt", 'w') as f:
                            f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch_station}")
                        self.save_best_norm_module()
                        best_metric = tracking_meter.avg
                self.cur_epoch_station += 1
            self.norm_module = load_best_model(self.cfg.SAN, self.norm_module)
            self.norm_module.requires_grad_(False).eval()
        
        best_metric = self.cfg.TRAIN.BEST_METRIC_INITIAL
        for cur_epoch in range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.MAX_EPOCH):
            self.train_epoch()

            # Evaluate the model on validation set.
            if self._is_eval_epoch(cur_epoch):
                tracking_meter = self.eval_epoch()
                # check improvement
                is_best = self._check_improvement(tracking_meter.avg, best_metric)
                # Save a checkpoint on improvement.
                if is_best:
                    with open(mkdir(self.cfg.RESULT_DIR) / "best_result.txt", 'w') as f:
                        f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg}\tEpoch: {self.cur_epoch}")
                    self.save_best_model()
                    if self.norm_method in ('RevIN', 'DishTS'):
                        self.save_best_norm_module()
                    best_metric = tracking_meter.avg
            self.cur_epoch += 1

    def _check_improvement(self, cur_metric, best_metric):
        if (self.cfg.TRAIN.BEST_LOWER and cur_metric < best_metric) \
                or (not self.cfg.TRAIN.BEST_LOWER and cur_metric > best_metric):
            return True
        else:
            return False

    def train_epoch(self):
        # set meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Epoch: [{}]".format(self.cur_epoch)
        )

        # switch to train mode
        self.model.train()

        data_size = len(self.train_loader)

        start = time.time()

        for cur_iter, inputs in enumerate(self.train_loader):
            self.cur_iter = cur_iter
            # dictionary for logging values
            log_dict = {}

            # measure data loading time
            data_time.update(time.time() - start)

            # Update the learning rate.
            lr = optim.get_epoch_lr(self.cur_epoch + float(cur_iter) / data_size, self.cfg)
            optim.set_lr(self.optimizer, lr)

            # log to W&B
            log_dict.update({
                "lr/": lr
            })

            outputs = self.train_step(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)
            log_dict.update({
                f"Train/{metric_meter.name}": metric_meter.val for metric_meter in metric_meters
            })
            log_dict.update({
                f"Train/{loss_meter.name}": loss_meter.val for loss_meter in loss_meters
            })

            if (cur_iter + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or (cur_iter + 1) == len(self.train_loader):
                progress.display(cur_iter + 1)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if self.cfg.WANDB.ENABLE:
                wandb.log(log_dict)

        log_dict = {}

        # log to W&B
        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict, commit=False)

    def train_epoch_station(self):
        # set meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Station Pretrain Epoch: [{}]".format(self.cur_epoch_station)
        )
        assert self.norm_method == 'SAN'
        
        # switch to train mode
        self.norm_module.train()

        data_size = len(self.train_loader)

        start = time.time()

        for cur_iter, inputs in enumerate(self.train_loader):
            self.cur_iter_station = cur_iter
            # dictionary for logging values
            log_dict = {}

            # measure data loading time
            data_time.update(time.time() - start)

            # Update the learning rate.
            lr = optim.get_epoch_lr(self.cur_epoch_station + float(cur_iter) / data_size, self.cfg.SAN)
            optim.set_lr(self.optimizer_stat, lr)

            # log to W&B
            log_dict.update({
                "lr/": lr
            })

            outputs = self.train_step_station(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)
            log_dict.update({
                f"Train_SAN/{metric_meter.name}": metric_meter.val for metric_meter in metric_meters
            })
            log_dict.update({
                f"Train_SAN/{loss_meter.name}": loss_meter.val for loss_meter in loss_meters
            })

            if (cur_iter + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or (cur_iter + 1) == len(self.train_loader):
                progress.display(cur_iter + 1)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if self.cfg.WANDB.ENABLE:
                wandb.log(log_dict)

        log_dict = {}

        # log to W&B
        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict, commit=False)

    def _get_metric_meters(self):
        return [AverageMeter(metric_name, ":.3f") for metric_name in self.metric_names]

    def _get_loss_meters(self):
        return [AverageMeter(f"Loss {loss_name}", ":.4e") for loss_name in self.loss_names]

    @staticmethod
    def _update_metric_meters(metric_meters, metrics, batch_size):
        assert len(metric_meters) == len(metrics)
        for metric_meter, metric in zip(metric_meters, metrics):
            metric_meter.update(metric.item(), batch_size)

    @staticmethod
    def _update_loss_meters(loss_meters, losses, batch_size):
        assert len(loss_meters) == len(losses)
        for loss_meter, loss in zip(loss_meters, losses):
            loss_meter.update(loss.item(), batch_size)

    def train_step(self, inputs):
        pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
            
        loss = F.mse_loss(pred, ground_truth)
        metric = F.l1_loss(pred, ground_truth)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        outputs = dict(
            losses=(loss,),
            metrics=(metric,)
        )

        return outputs
    
    def train_step_station(self, inputs):
        assert self.norm_method == 'SAN'
        
        # override for different methods
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        enc_window, statistics_pred = self.norm_module.normalize(enc_window)
        
        ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
            
        loss = self.station_loss(ground_truth, statistics_pred)
    
        #! temporal implementation
        metric = loss
            
        self.optimizer_stat.zero_grad()
        loss.backward()
        self.optimizer_stat.step()
        
        outputs = dict(
            losses=(loss,),
            metrics=(metric,)
        )

        return outputs

    def station_loss(self, y, statistics_pred):
        assert self.norm_method == 'SAN'
        
        bs, len, dim = y.shape
        y = y.reshape(bs, -1, self.cfg.DATA.PERIOD_LEN, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_true = torch.cat([mean, std], dim=-1)
        loss = F.mse_loss(statistics_pred, station_true)
        
        return loss

    def _load_from_checkpoint(self):
        pass

    def _find_batch_size(self, inputs):
        """
        Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
        """
        if isinstance(inputs, (list, tuple)):
            for t in inputs:
                result = self._find_batch_size(t)
                if result is not None:
                    return result
        elif isinstance(inputs, Mapping):
            for key, value in inputs.items():
                result = self._find_batch_size(value)
                if result is not None:
                    return result
        elif isinstance(inputs, torch.Tensor):
            return inputs.shape[0] if len(inputs.shape) >= 1 else None
        elif isinstance(inputs, np.ndarray):
            return inputs.shape[0] if len(inputs.shape) >= 1 else None

    def _is_eval_epoch(self, cur_epoch):
        return (cur_epoch + 1 == self.cfg.SOLVER.MAX_EPOCH) or (cur_epoch + 1) % self.cfg.TRAIN.EVAL_PERIOD == 0

    @torch.no_grad()
    def eval_epoch(self):
        # set meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Validation epoch[{}]".format(self.cur_epoch)
        )
        log_dict = {}

        # switch to eval mode
        self.model.eval()

        start = time.time()
        for cur_iter, inputs in enumerate(self.val_loader):
            # measure data loading time
            data_time.update(time.time() - start)

            outputs = self.eval_step(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)

            if self._is_display_iter(cur_iter):
                progress.display(cur_iter + 1)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

        log_dict.update({
            f"Val/{metric_meter.name}": metric_meter.avg for metric_meter in metric_meters
        })
        log_dict.update({
            f"Val/{loss_meter.name}": loss_meter.avg for loss_meter in loss_meters
        })

        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)

        # track the best model based on the first metric
        tracking_meter = metric_meters[0]

        return tracking_meter

    @torch.no_grad()
    def eval_epoch_station(self):
        assert self.norm_method == 'SAN'
        
        # set meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Station Validation epoch[{}]".format(self.cur_epoch_station)
        )
        log_dict = {}

        # switch to eval mode
        self.norm_module.eval()

        start = time.time()
        for cur_iter, inputs in enumerate(self.val_loader):
            # measure data loading time
            data_time.update(time.time() - start)

            outputs = self.eval_step_station(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)

            if self._is_display_iter(cur_iter):
                progress.display(cur_iter + 1)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

        log_dict.update({
            f"Val_SAN/{metric_meter.name}": metric_meter.avg for metric_meter in metric_meters
        })
        log_dict.update({
            f"Val_SAN/{loss_meter.name}": loss_meter.avg for loss_meter in loss_meters
        })

        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)

        # track the best model based on the first metric
        tracking_meter = metric_meters[0]

        return tracking_meter
    
    @torch.no_grad()
    def eval_step(self, inputs):
        pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        
        loss = F.mse_loss(pred, ground_truth)
        metric = F.l1_loss(pred, ground_truth)
        
        outputs = dict(
            losses=(loss,),
            metrics=(metric,)
        )
        
        return outputs
    
    @torch.no_grad()
    def eval_step_station(self, inputs):
        assert self.norm_method == 'SAN'
        
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        enc_window, statistics_pred = self.norm_module.normalize(enc_window)
        
        ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
            
        loss = self.station_loss(ground_truth, statistics_pred)
        with torch.no_grad():
            metric = self.station_loss(ground_truth, statistics_pred)
        
        outputs = dict(
            losses=(loss,),
            metrics=(metric,)
        )
        
        return outputs

    def _is_display_iter(self, cur_iter):
        return (cur_iter + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or (cur_iter + 1) == len(self.val_loader)

    def save_best_model(self):
        checkpoint = {
            "epoch": self.cur_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg.dump(),
        }
        with open(mkdir(self.cfg.TRAIN.CHECKPOINT_DIR) / 'checkpoint_best.pth', "wb") as f:
            torch.save(checkpoint, f)
    
    def save_best_norm_module(self):
        assert self.cfg.NORM_MODULE.ENABLE
        
        norm_module_cfg = get_norm_module_cfg(self.cfg)
        checkpoint = {
            "epoch": self.cur_epoch_station,
            "model_state": self.norm_module.state_dict(),
            # "optimizer_state": self.optimizer_stat.state_dict(),
            "cfg": self.cfg.dump(),
        }
        with open(mkdir(norm_module_cfg.TRAIN.CHECKPOINT_DIR) / 'checkpoint_best.pth', "wb") as f:
            torch.save(checkpoint, f)

    def load_best_model(self):
        model_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
        if os.path.isfile(model_path):
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")

            state_dict = checkpoint['model_state']
            msg = self.model.load_state_dict(state_dict, strict=True)
            assert set(msg.missing_keys) == set()

            print(f"Loaded pre-trained model from {model_path}")
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

        return self.model


def build_trainer(cfg, model, norm_module=None):
    metric_names, loss_names = cfg.MODEL.METRIC_NAMES, cfg.MODEL.LOSS_NAMES
    trainer = Trainer(cfg, model, metric_names, loss_names, norm_module=norm_module)

    return trainer
