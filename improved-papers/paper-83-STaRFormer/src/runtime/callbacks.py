import matplotlib.pyplot as plt
import wandb
import numpy as np
import math 
from abc import abstractmethod
from typing import Any, Mapping, List, Union, Literal
from lightning import Callback, LightningModule, Trainer
from sklearn.metrics import (
    fbeta_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    average_precision_score
    )
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only

import torch 
from torch import Tensor
import torch.nn.functional as F

from src.utils import DatasetOptions, ModelOptions, TrainingModeOptions, MetricOptions, TaskOptions
from omegaconf import DictConfig

__all__ = [
    'LossLoggingCallback',
    'GeolifeMetricLoggerCallback',
    'PAMMetricLoggerCallback',
    'UEAMetricLoggerCallback',
    'P12P19MetricLoggerCallback',
    'TSRMetricLoggerCallback',
    'AnomalyMetricLoggerCallback',
    'ExponentialMovingAverage'
]


class LossLoggingCallback(Callback):
    """
    LossLoggingCallback

    A Lightning Callback that logs loss-related information during training.

    Attributes:
        _config (DictConfig): The configuration object.
    """
    def __init__(self, config: DictConfig,) -> None:
        super().__init__()
        self._config = config

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_wand(pl_module)
        return super().on_fit_start(trainer, pl_module)

    def setup_wand(self, pl_module: LightningModule):
       if self._config.logger.name == 'wandb':
            #wandb.define_metric("train/loss", summary="min")
            #wandb.define_metric("val/loss", summary="min") 
            pl_module.logger.experiment.define_metric("train/loss", summary="min")
            pl_module.logger.experiment.define_metric("val/loss", summary="min")

    #def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """
        # log loss
        self._log_loss(trainer, pl_module, outputs=outputs, mode="train")

        #return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    #def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called when the validation batch ends."""
        # log loss
        self._log_loss(trainer, pl_module, outputs=outputs, mode="val")
    
    #def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called when the test batch ends."""
        # log loss
        self._log_loss(trainer, pl_module, outputs=outputs, mode="test")
    
    def _log_loss(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], mode: Literal['train', 'val', 'test']=TrainingModeOptions.train) -> None:
        if self._config.task == TaskOptions.classification:
            loss = outputs['loss']
        
            if self._config.model.sequence_model.name == ModelOptions.starformer and self._config.loss.loss_fn == 'drm_loss':
                if self._config.model.sequence_model.masking is not None:
                    if mode == 'test':
                        self.log_dict({f'{mode}/loss_ce': loss}, batch_size=outputs['bs'], sync_dist=True)                    
                    elif mode == 'val': # or mode == 'test':
                        self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_ce': outputs['loss_ce']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_mse': outputs['loss_mse']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_mse_masked': outputs['loss_mse_masked']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_mse_unmasked': outputs['loss_mse_unmasked']}, batch_size=outputs['bs'], sync_dist=True)
                    else:
                        self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)    
                        self.log_dict({f'{mode}/loss_ce': outputs['loss_ce']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_mse': outputs['loss_mse']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_mse_masked': outputs['loss_mse_masked']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_mse_unmasked': outputs['loss_mse_unmasked']}, batch_size=outputs['bs'], sync_dist=False)
                else:
                    if mode in ['val', 'test']:
                        self.log_dict({f'{mode}/loss_ce': loss}, batch_size=outputs['bs'], sync_dist=True)    
                    else:
                        self.log_dict({f'{mode}/loss_ce': loss}, batch_size=outputs['bs'], sync_dist=False)
            
            elif self._config.model.sequence_model.name == ModelOptions.starformer and self._config.loss.loss_fn == 'darem_sscl':
                if self._config.model.sequence_model.masking is not None:
                    if mode == 'test':
                        self.log_dict({f'{mode}/loss_ce': loss}, batch_size=outputs['bs'], sync_dist=True)                    
                    elif mode == 'val': # or mode == 'test':
                        self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_ce': outputs['loss_ce']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive': outputs['loss_contrastive']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive_batch_sim': outputs['loss_contrastive_batch_sim']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive_class_sim': outputs['loss_contrastive_class_sim']}, batch_size=outputs['bs'], sync_dist=True)
                        #self.log_dict({f'{mode}/loss_margin': outputs['loss_margin']}, batch_size=outputs['bs'], sync_dist=True)
                    else:
                        self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)    
                        self.log_dict({f'{mode}/loss_ce': outputs['loss_ce']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive': outputs['loss_contrastive']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive_batch_sim': outputs['loss_contrastive_batch_sim']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive_class_sim': outputs['loss_contrastive_class_sim']}, batch_size=outputs['bs'], sync_dist=False)
                        #self.log_dict({f'{mode}/loss_margin': outputs['loss_margin']}, batch_size=outputs['bs'], sync_dist=False)
                else:
                    if mode in ['val', 'test']:
                        self.log_dict({f'{mode}/loss_ce': loss}, batch_size=outputs['bs'], sync_dist=True)    
                    else:
                        self.log_dict({f'{mode}/loss_ce': loss}, batch_size=outputs['bs'], sync_dist=False)                
            else:           
                if mode in ['val', 'test']:
                    self.log_dict({f'{mode}/loss_ce': loss}, batch_size=outputs['bs'], sync_dist=True)    
                else:
                    self.log_dict({f'{mode}/loss_ce': loss}, batch_size=outputs['bs'], sync_dist=False)
        
        elif self._config.task == TaskOptions.regression:
            loss = outputs['loss']

            if self._config.model.sequence_model.name == ModelOptions.starformer and self._config.loss.loss_fn == 'darem_sscl':
                if self._config.model.sequence_model.masking is not None:
                    if mode == 'test':
                        self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=True)                    
                    elif mode == 'val': # or mode == 'test':
                        self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_task': outputs['loss_task']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive': outputs['loss_contrastive']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive_batch_sim': outputs['loss_contrastive_batch_sim']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive_class_sim': outputs['loss_contrastive_class_sim']}, batch_size=outputs['bs'], sync_dist=True)
                        #self.log_dict({f'{mode}/loss_margin': outputs['loss_margin']}, batch_size=outputs['bs'], sync_dist=True)
                    else:
                        self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)    
                        self.log_dict({f'{mode}/loss_task': outputs['loss_task']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive': outputs['loss_contrastive']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive_batch_sim': outputs['loss_contrastive_batch_sim']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive_class_sim': outputs['loss_contrastive_class_sim']}, batch_size=outputs['bs'], sync_dist=False)
                        #self.log_dict({f'{mode}/loss_margin': outputs['loss_margin']}, batch_size=outputs['bs'], sync_dist=False)
                else:
                    if mode in ['val', 'test']:
                        self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=True)    
                    else:
                        self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=False)                
            else:           
                if mode in ['val', 'test']:
                    self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=True)    
                else:
                    self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=False)

        elif self._config.task == TaskOptions.anomaly_detection:
            loss = outputs['loss']

            if self._config.model.sequence_model.name == ModelOptions.starformer and self._config.loss.loss_fn == 'darem_sscl':
                if self._config.model.sequence_model.masking is not None:
                    if mode == 'test':
                        self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=True)                    
                    elif mode == 'val': # or mode == 'test':
                        self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_task': outputs['loss_task']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive': outputs['loss_contrastive']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive_batch_sim': outputs['loss_contrastive_batch_sim']}, batch_size=outputs['bs'], sync_dist=True)
                        self.log_dict({f'{mode}/loss_contrastive_class_sim': outputs['loss_contrastive_class_sim']}, batch_size=outputs['bs'], sync_dist=True)
                        #self.log_dict({f'{mode}/loss_margin': outputs['loss_margin']}, batch_size=outputs['bs'], sync_dist=True)
                    else:
                        self.log_dict({f'{mode}/loss': loss}, batch_size=outputs['bs'], sync_dist=False)    
                        self.log_dict({f'{mode}/loss_task': outputs['loss_task']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive': outputs['loss_contrastive']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive_batch_sim': outputs['loss_contrastive_batch_sim']}, batch_size=outputs['bs'], sync_dist=False)
                        self.log_dict({f'{mode}/loss_contrastive_class_sim': outputs['loss_contrastive_class_sim']}, batch_size=outputs['bs'], sync_dist=False)
                        #self.log_dict({f'{mode}/loss_margin': outputs['loss_margin']}, batch_size=outputs['bs'], sync_dist=False)
                else:
                    if mode in ['val', 'test']:
                        self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=True)    
                    else:
                        self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=False)                
            else:           
                if mode in ['val', 'test']:
                    self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=True)    
                else:
                    self.log_dict({f'{mode}/loss_task': loss}, batch_size=outputs['bs'], sync_dist=False)


class BaseMetricLoggingCallback(Callback):
    """BaseMetricLoggingCallback

    A base-class Lightning Callback for logging training related metrics.
    """
    def __init__(self, 
                 config: DictConfig, 
                 log_cm_train: bool=False, 
                 log_cm_val: bool=False):
        super().__init__()
        """Initialize the base metric logging callback.

        Args:
            config (DictConfig): Configuration object.
            log_cm_train (bool, optional): If True, enable logging of the training
                confusion matrix. Defaults to False.
            log_cm_val (bool, optional): If True, enable logging of the validation
                confusion matrix. Defaults to False.
        """
        self._validation_step_outputs = []
        self._testing_step_outputs = []
        self._config = config
        self._log_cm_train = log_cm_train
        self._log_cm_val = log_cm_val
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_wandb(pl_module)
        return super().on_fit_start(trainer, pl_module)

    def setup_wandb(self, pl_module: LightningModule):
        if self._config.logger.name == 'wandb':
            pl_module.logger.experiment.define_metric("train/acc", summary="max")
            pl_module.logger.experiment.define_metric("train/fbeta", summary="max")
            pl_module.logger.experiment.define_metric("val/acc", summary="max")
            pl_module.logger.experiment.define_metric("val/fbeta", summary="max")
    
    @abstractmethod
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int) -> None:
        raise NotImplementedError

    @abstractmethod 
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        raise NotImplementedError

    @abstractmethod
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        raise NotImplementedError
    
    # logging functionality
    def _log_img(self, trainer: Trainer, mode: Literal['train', 'val', 'test']=TrainingModeOptions.train, title: str=None, img=None) -> None:
        assert img is not None
        trainer.logger.log_image(key=f'{mode}/{title}', images=[img])
    
    def _log_metric(self, 
                    metric_key: Literal['acc', 'cm', 'f05', 'f1', 'precision', 'recall'], 
                    metric_val: float=None, 
                    bs: int=None, 
                    mode: Literal['train', 'val', 'test']=TrainingModeOptions.train,
                    prog_bar: bool=None, 
                    logger: bool=None
                    ) -> None:
        assert metric_key in MetricOptions.get_options()
        assert bs is not None
        assert metric_val is not None

        if prog_bar is None:
            if self._config.dataset in [DatasetOptions.p12, DatasetOptions.p19]:
                prog_bar = True if metric_key in [MetricOptions.auroc, MetricOptions.auprc] else False
            elif self._config.dataset in DatasetOptions.tsr:
                prog_bar = True if metric_key in [MetricOptions.rmse, MetricOptions.mae] else False
            elif self._config.dataset in DatasetOptions.anomaly:
                prog_bar = True if metric_key in [MetricOptions.f1, MetricOptions.precision, MetricOptions.recall] else False
            else:
                prog_bar = True if metric_key in [MetricOptions.accuracy, MetricOptions.f05] else False

        if logger is None:
            logger = True if metric_key in [MetricOptions.accuracy, MetricOptions.f05, MetricOptions.f1, MetricOptions.precision, MetricOptions.recall, MetricOptions.auroc, MetricOptions.auprc, MetricOptions.rmse, MetricOptions.mae] else False

        sync_dist = True if mode in [TrainingModeOptions.val, TrainingModeOptions.test] else False
        
        self.log_dict({f'{mode}/{metric_key}': metric_val}, batch_size=bs, sync_dist=sync_dist, prog_bar=prog_bar, logger=logger) 
    
    def _log_acc(self, 
                 trainer: Trainer, pl_module: LightningModule, outputs: Union[Tensor, Mapping[str, Any], None], 
                 mode: Literal['train', 'val', 'test']=TrainingModeOptions.train, prog_bar: bool=None, logger: bool=None
                 ) -> None:
        if prog_bar is None:
            prog_bar = True 
        if logger is None:
            logger = True

        sync_dist = True if mode in [TrainingModeOptions.val, TrainingModeOptions.test] else False

        if outputs.get('acc', None) is not None:
            acc = outputs['acc'].item() if isinstance(outputs['acc'], Tensor) else outputs['acc']
            self.log_dict({f'{mode}/acc': acc}, batch_size=outputs['bs'], sync_dist=sync_dist, prog_bar=prog_bar, logger=logger)    
            
        if outputs.get('acc_autoregressive', None) is not None:
            acc = outputs['acc_autoregressive'].item() if isinstance(outputs['acc_autoregressive'], Tensor) else outputs['acc_autoregressive']
            self.log_dict({f'{mode}/acc_autoregressive': acc}, batch_size=outputs['bs'], sync_dist=sync_dist, prog_bar=prog_bar, logger=logger)    
    
    def _log_table(self, trainer: Trainer, key: str, columns: list, data: list, mode: Literal['train', 'val', 'test']=TrainingModeOptions.val):
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_table(key=f'{mode}/{key}', columns=columns, data=data)


    def _create_cm_figure(self, cm_matrix, display_labels: List=[0,1], cmap: str='viridis'):
        """ Creates Confusion Matrix Figure. """
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=display_labels)
        disp.plot(cmap=cmap)

        plt.close(disp.figure_)
        return disp.figure_


class GeolifeMetricLoggerCallback(BaseMetricLoggingCallback):
    def __init__(self, config, log_cm_train = False, log_cm_val = False):
        super().__init__(config, log_cm_train, log_cm_val)
    

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mode = 'train'
        preds = outputs['preds'].flatten()
        labels = outputs['labels'].flatten()
        bs = outputs['bs']
        # fbeta calc and log
        fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
        self._log_metric(metric_key=MetricOptions.f05, metric_val=fbeta, bs=bs, mode=mode)

        # confusion matrix calc and log
        if self._log_cm_train:
            cm_matrix = outputs['cm']
            #tn, fp, fn, tp = cm_matrix.ravel()
            display_labels = self._config.datamodule.display_labels if self._config.datamodule.get('display_labels', None) is not None else [0,1]
            cm_fig = self._create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)
            
            if self._config.logger.name == 'tensorboard':
                trainer.logger.experiment.add_figure(f'{mode}/Confusion Matrix', cm_fig, trainer.global_step)
            elif self._config.logger.name == 'wandb':
                trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
        
        self._log_acc(trainer, pl_module, outputs=outputs, mode=mode)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the validation batch ends."""
        self._validation_step_outputs.append(outputs)
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="val")
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mode = 'val'
        preds = []
        labels = []
        cms = []
        bs = self._validation_step_outputs[0]['bs']

        for batch_outputs in self._validation_step_outputs:
            preds.append(batch_outputs['preds'].flatten())
            labels.append(batch_outputs['labels'].flatten())
            cms.append(batch_outputs['cm'])

        # fbeta calc and log    
        labels = torch.concat(labels)
        preds = torch.concat(preds)
        
        fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
        self._log_metric(metric_key=MetricOptions.f05, metric_val=fbeta, bs=bs, mode=mode)

        # confusion matrix calc and log
        if self._log_cm_val:
            cm_matrix = sum(cms)
            #tn, fp, fn, tp = cm_matrix.ravel()
            display_labels = self._config.datamodule.display_labels if self._config.datamodule.get('display_labels', None) is not None else [0,1]
            cm_fig = self._create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)
            if self._config.logger.name == 'tensorboard':
                trainer.logger.experiment.add_figure(f'{mode}/Confusion Matrix', cm_fig, trainer.global_step)
            elif self._config.logger.name == 'wandb':
                trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
        
        self.__log_geolife_per_class_scores_tabel(trainer=trainer, preds=preds, labels=labels, mode=mode)

        self._validation_step_outputs.clear()
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the test batch ends."""
        self._testing_step_outputs.append(outputs)
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="test")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        mode = 'test'

        preds = []
        labels = []
        cms = []
        bs = self._testing_step_outputs[0]['bs']

        for batch_outputs in self._testing_step_outputs:
            preds.append(batch_outputs['preds'].flatten())
            labels.append(batch_outputs['labels'].flatten())
            cms.append(batch_outputs['cm'])
        
        # fbeta calc and log
        labels = torch.concat(labels)
        preds = torch.concat(preds)
        fbeta = fbeta_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro", beta=0.5)
        self._log_metric(metric_key=MetricOptions.f05, metric_val=fbeta, bs=bs, mode=mode)
        
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)
        
        # log res in tabel
        columns = [MetricOptions.accuracy, MetricOptions.f05, MetricOptions.f1, MetricOptions.precision, MetricOptions.recall]
        acc = torch.concat([outputs['acc'].reshape(1,-1) for outputs in self._testing_step_outputs]).mean().item()
        self._log_table(trainer, key="Scores", columns=columns, data=[[acc, fbeta, f1, precision, recall]], mode=mode)

        # confusion matrix calc and log
        cm_matrix = sum(cms)
        display_labels = self._config.datamodule.display_labels if self._config.datamodule.get('display_labels', None) is not None else [0,1]
        cm_fig = self._create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)

        if self._config.logger.name == 'tensorboard':
            trainer.logger.experiment.add_figure(f'{mode}/Confusion Matrix', cm_fig, trainer.global_step)

        elif self._config.logger.name == 'wandb':
            trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
        
        self.__log_geolife_per_class_scores_tabel(trainer=trainer, preds=preds, labels=labels, mode=mode)

        self._testing_step_outputs.clear()
    

    def __log_geolife_per_class_scores_tabel(self, trainer: Trainer, preds: Tensor, labels: Tensor, mode: Literal['train', 'val', 'test']=TrainingModeOptions.val):
        fbeta_per_class = fbeta_score(
            labels.detach().cpu().numpy(), 
            preds.detach().cpu().numpy(), 
            average=None, beta=0.5)

        sorted_by_label = {
            k: {'pred': [], 'truth': []} 
            for k in range(len(list(trainer.datamodule.dataset.idx2label.keys())))
        }
        for p, l in zip(preds, labels):
            sorted_by_label[l.item()]['pred'].append(p.item())
            sorted_by_label[l.item()]['truth'].append(l.item())
        
        acc_per_label = {}
        for k in sorted_by_label.keys():
            if acc_per_label.get(k, None) is None:
                acc_per_label[k] = {'acc': None, 'fbeta': None}
            
            #print(sorted_by_label[k]['truth'], bool(sorted_by_label[k]['truth']))
            if bool(sorted_by_label[k]['truth']):
                acc_per_label[k]['acc'] = (torch.sum(
                            torch.eq(torch.tensor(sorted_by_label[k]['pred']), 
                            torch.tensor(sorted_by_label[k]['truth']))
                        ) / len(torch.tensor(sorted_by_label[k]['truth']))).item()
                
                acc_per_label[k]['fbeta'] = fbeta_per_class[k] if k <= len(fbeta_per_class)-1 else 0.0 
            else:
                acc_per_label[k]['acc'] = 0.0
                acc_per_label[k]['fbeta'] = 0.0
        

        # accuracy
        columns = [trainer.datamodule.dataset.idx2label[k] for k in acc_per_label.keys()]
        acc_data = [acc_per_label[k]['acc'] for k in acc_per_label.keys()]
        fbeta_data = [acc_per_label[k]['fbeta'] for k in acc_per_label.keys()]
        self._log_table(trainer, key="Accuracy", columns=columns, data=[acc_data], mode=mode)
        self._log_table(trainer, key="Fbeta", columns=columns, data=[fbeta_data], mode=mode)


class PAMMetricLoggerCallback(BaseMetricLoggingCallback):
    def __init__(self, config, log_cm_train = False, log_cm_val = False):
        super().__init__(config, log_cm_train, log_cm_val) 
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mode = 'train'
        preds = outputs['preds'].flatten()
        labels = outputs['labels'].flatten()
        bs = outputs['bs']
        # calc an log metrics
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)
        
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="train")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the validation batch ends."""
        self._validation_step_outputs.append(outputs)
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="val")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        mode = 'val'
        preds, labels, cms = [], [], []
        bs = self._validation_step_outputs[0]['bs']

        for batch_outputs in self._validation_step_outputs:
            preds.append(batch_outputs['preds'].flatten())
            labels.append(batch_outputs['labels'].flatten())
            cms.append(batch_outputs['cm'])

        # fbeta calc and log    
        labels = torch.concat(labels)
        preds = torch.concat(preds)
        
        # calc an log metrics
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)

        # confusion matrix
        if self._config.logger.name == 'wandb':
            cm_matrix = sum(cms)
            #tn, fp, fn, tp = cm_matrix.ravel()
            display_labels = self._config.datamodule.display_labels
            cm_fig = self._create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)

            trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
        
        self._validation_step_outputs.clear()
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the test batch ends."""
        self._testing_step_outputs.append(outputs)
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="test")
    
    def on_test_epoch_end(self, trainer, pl_module):
        mode = 'test'
        preds = []
        labels = []
        cms = []
        bs = self._testing_step_outputs[0]['bs']

        for batch_outputs in self._testing_step_outputs:
            preds.append(batch_outputs['preds'].flatten())
            labels.append(batch_outputs['labels'].flatten())
            cms.append(batch_outputs['cm'])
        
        # fbeta calc and log
        labels = torch.concat(labels)
        preds = torch.concat(preds)
        
        # calc an log metrics
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)
        # log res in tabel
        columns = [MetricOptions.accuracy, MetricOptions.f1, MetricOptions.precision, MetricOptions.recall]
        acc = torch.concat([outputs['acc'].reshape(1,-1) for outputs in self._testing_step_outputs]).mean().item()
        self._log_table(trainer, key="Scores", columns=columns, data=[[acc, f1, precision, recall]], mode=mode)

        # confusion matrix
        if self._config.logger.name == 'wandb':
            cm_matrix = sum(cms)
            #tn, fp, fn, tp = cm_matrix.ravel()
            display_labels = self._config.datamodule.display_labels
            cm_fig = self._create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)
            trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])

        self._testing_step_outputs.clear()

    
class UEAMetricLoggerCallback(BaseMetricLoggingCallback):
    def __init__(self, config, log_cm_train = False, log_cm_val = False):
        super().__init__(config, log_cm_train, log_cm_val) 
    
    def setup_wandb(self, pl_module: LightningModule):
        if self._config.logger.name == 'wandb':
            pl_module.logger.experiment.define_metric("train/acc", summary="max")
            pl_module.logger.experiment.define_metric("train/f1", summary="max")
            pl_module.logger.experiment.define_metric("train/precision", summary="max")
            pl_module.logger.experiment.define_metric("train/recall", summary="max")
            pl_module.logger.experiment.define_metric("train/loss_ce", summary="min")
            # val
            pl_module.logger.experiment.define_metric("val/acc", summary="max")
            pl_module.logger.experiment.define_metric("val/f1", summary="max")
            pl_module.logger.experiment.define_metric("val/precision", summary="max")
            pl_module.logger.experiment.define_metric("val/recall", summary="max")
            pl_module.logger.experiment.define_metric("val/loss_ce", summary="min")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mode = 'train'
        preds = outputs['preds'].flatten()
        labels = outputs['labels'].flatten()
        bs = outputs['bs']
        # calc an log metrics
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)
        
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="train")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the validation batch ends."""
        self._validation_step_outputs.append(outputs)
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="val")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        mode = 'val'
        accs, preds, labels, cms = [], [], [], []
        bs = self._validation_step_outputs[0]['bs']

        for batch_outputs in self._validation_step_outputs:
            accs.append(batch_outputs['preds'].flatten())
            preds.append(batch_outputs['preds'].flatten())
            labels.append(batch_outputs['labels'].flatten())
            cms.append(batch_outputs['cm'])

        # fbeta calc and log    
        labels = torch.concat(labels)
        preds = torch.concat(preds)
        
        acc = torch.sum(torch.eq(preds.flatten(), labels.flatten().int())) / len(labels.flatten())
        self._log_metric(metric_key=MetricOptions.accuracy, metric_val=acc.item(), bs=bs, mode=mode)

        # calc an log metrics
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)

        # confusion matrix
        if self._config.logger.name == 'wandb':
            cm_matrix = sum(cms)
            #tn, fp, fn, tp = cm_matrix.ravel()
            display_labels = self._config.datamodule.display_labels
            cm_fig = self._create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)

            trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
        
        self._validation_step_outputs.clear()



def create_probability_array(probabilities: Tensor):
    result = []

    for prob in probabilities:
        complement = 1 - prob.item()
        if prob < 0.5:
            result.append([prob.item(), complement])
        else:
            result.append([complement, prob.item()])

    return np.array(result)
        

class P12P19MetricLoggerCallback(BaseMetricLoggingCallback):
    def __init__(self, config, log_cm_train = False, log_cm_val = False):
        super().__init__(config, log_cm_train, log_cm_val)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mode = 'train'
        preds = outputs['preds'].flatten()
        probas = outputs['probas'].flatten()
        labels = outputs['labels'].flatten()
        bs = outputs['bs']
        # calc an log metrics
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        
        auroc = roc_auc_score(labels.detach().cpu().numpy(), probas.detach().cpu().numpy(), average='macro')
        auprc = average_precision_score(labels.detach().cpu().numpy(), probas.detach().cpu().numpy(), average='macro')
        
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.auroc, metric_val=auroc, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.auprc, metric_val=auprc, bs=bs, mode=mode)
        
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="train")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the validation batch ends."""
        self._validation_step_outputs.append(outputs)
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="val")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        mode = 'val'
        preds, probas, labels, cms = [], [], [], []
        bs = self._validation_step_outputs[0]['bs']

        for batch_outputs in self._validation_step_outputs:
            preds.append(batch_outputs['preds'].flatten())
            probas.append(batch_outputs['probas'].flatten())
            labels.append(batch_outputs['labels'].flatten())
            cms.append(batch_outputs['cm'])

        # fbeta calc and log    
        labels = torch.concat(labels)
        preds = torch.concat(preds)
        probas = torch.concat(probas)
        
        # calc an log metrics
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        
        try:
            auroc = roc_auc_score(labels.detach().cpu().numpy(), probas.detach().cpu().numpy(), average='macro')
        except:
            auroc = 0.0 # fixes error
        auprc = average_precision_score(labels.detach().cpu().numpy(), probas.detach().cpu().numpy(), average='macro')
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.auroc, metric_val=auroc, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.auprc, metric_val=auprc, bs=bs, mode=mode)

        # confusion matrix
        if self._config.logger.name == 'wandb':
            cm_matrix = sum(cms)
            #tn, fp, fn, tp = cm_matrix.ravel()
            display_labels = self._config.datamodule.display_labels
            cm_fig = self._create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)

            trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])
        
        self._validation_step_outputs.clear()
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the test batch ends."""
        self._testing_step_outputs.append(outputs)
        # log acc
        self._log_acc(trainer, pl_module, outputs=outputs, mode="test")
    
    def on_test_epoch_end(self, trainer, pl_module):
        mode = 'test'
        preds = []
        probas = []
        labels = []
        cms = []
        bs = self._testing_step_outputs[0]['bs']

        for batch_outputs in self._testing_step_outputs:
            preds.append(batch_outputs['preds'].flatten())
            probas.append(batch_outputs['probas'].flatten())
            labels.append(batch_outputs['labels'].flatten())
            cms.append(batch_outputs['cm'])
        
        # fbeta calc and log
        labels = torch.concat(labels)
        preds = torch.concat(preds)
        probas = torch.concat(probas)
        
        # calc an log metrics
        f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        
        try:
            auroc = roc_auc_score(labels.detach().cpu().numpy(), probas.detach().cpu().numpy(), average='macro')
        except:
            auroc = 0.0 # fixes error
        auprc = average_precision_score(labels.detach().cpu().numpy(), probas.detach().cpu().numpy(), average='macro')
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=precision, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=recall, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.auroc, metric_val=auroc, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.auprc, metric_val=auprc, bs=bs, mode=mode)
        # log res in tabel
        columns = [MetricOptions.accuracy, MetricOptions.f1, MetricOptions.precision, 
                   MetricOptions.recall, MetricOptions.auroc, MetricOptions.auprc]
        acc = torch.concat([outputs['acc'].reshape(1,-1) for outputs in self._testing_step_outputs]).mean().item()
        self._log_table(trainer, key="Scores", columns=columns, data=[[acc, f1, precision, recall, auroc, auprc]], mode=mode)

        # confusion matrix
        if self._config.logger.name == 'wandb':
            cm_matrix = sum(cms)
            #tn, fp, fn, tp = cm_matrix.ravel()
            display_labels = self._config.datamodule.display_labels
            cm_fig = self._create_cm_figure(cm_matrix=cm_matrix, display_labels=display_labels)
            trainer.logger.log_image(key=f'{mode}/Confusion Matrix', images=[cm_fig])

        self._testing_step_outputs.clear()

from sklearn.metrics import mean_absolute_error, mean_squared_error

class TSRMetricLoggerCallback(BaseMetricLoggingCallback):
    def __init__(self, config, log_cm_train = False, log_cm_val = False):
        super().__init__(config, log_cm_train, log_cm_val)
    
    def setup_wandb(self, pl_module: LightningModule):
        if self._config.logger.name == 'wandb':
            pl_module.logger.experiment.define_metric("train/rmse", summary="min")
            pl_module.logger.experiment.define_metric("train/mae", summary="min")
            pl_module.logger.experiment.define_metric("train/loss_task", summary="min")
            pl_module.logger.experiment.define_metric("val/rmse", summary="min")
            pl_module.logger.experiment.define_metric("val/mae", summary="min")
            pl_module.logger.experiment.define_metric("val/loss_task", summary="min")
    
    def _calc_regr_metrics(self, y_scaler, preds, labels, verbose=False):
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(preds.squeeze().detach().cpu().numpy().reshape(-1,1)).flatten()
            y = y_scaler.inverse_transform(labels.squeeze().detach().cpu().numpy().reshape(-1,1)).flatten()
        else:
            y_pred = preds.squeeze().detach().cpu().numpy().reshape(-1,1).flatten()
            y = labels.squeeze().detach().cpu().numpy().reshape(-1,1).flatten()
        
        if verbose: 
            print(y_pred, y)
            print(math.sqrt(mean_squared_error(y, y_pred)))

        rmse = math.sqrt(mean_squared_error(y, y_pred)) #math.sqrt( ((y_pred - y)**2).sum() / y_pred.shape[0] )
        mae =  mean_absolute_error(y, y_pred) #( np.abs(y_pred - y).sum() / y_pred.shape[0] )
        return rmse, mae
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mode = 'train'
        preds = outputs['preds'].flatten()
        targets = outputs['targets'].flatten()
        bs = outputs['bs']
        # calc an log metrics
        rmse, mae = self._calc_regr_metrics(None, preds, targets, False)
        
        self._log_metric(metric_key=MetricOptions.rmse, metric_val=rmse, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.mae, metric_val=mae, bs=bs, mode=mode)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the validation batch ends."""
        self._validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        mode = 'val'
        preds, labels, targets = [], [], []
        bs = self._validation_step_outputs[0]['bs']
        #scaler = self._validation_step_outputs[0]['scaler']
        for batch_outputs in self._validation_step_outputs:
            preds.append(batch_outputs['preds'].flatten())
            labels.append(batch_outputs['labels'].flatten())
            targets.append(batch_outputs['targets'].flatten())

        targets = torch.concat(targets)
        preds = torch.concat(preds)

        rmse, mae = self._calc_regr_metrics(None, preds, labels=targets, verbose=True)
        
        self._log_metric(metric_key=MetricOptions.rmse, metric_val=rmse, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.mae, metric_val=mae, bs=bs, mode=mode)

        self._validation_step_outputs.clear()
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the test batch ends."""
        self._testing_step_outputs.append(outputs)
    
    def on_test_epoch_end(self, trainer, pl_module):
        mode = 'test'
        preds = []
        labels = []
        bs = self._testing_step_outputs[0]['bs']
        scaler = self._testing_step_outputs[0]['scaler']

        for batch_outputs in self._testing_step_outputs:
            preds.append(batch_outputs['preds'].flatten())
            labels.append(batch_outputs['labels'].flatten())
        
        labels = torch.concat(labels)
        preds = torch.concat(preds)
        rmse, mae = self._calc_regr_metrics(scaler['y'], preds, labels)
        
        self._log_metric(metric_key=MetricOptions.rmse, metric_val=rmse, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.mae, metric_val=mae, bs=bs, mode=mode)

        self._testing_step_outputs.clear()


class AnomalyMetricLoggerCallback(BaseMetricLoggingCallback):
    def __init__(self, config, log_cm_train = False, log_cm_val = False):
        super().__init__(config, log_cm_train, log_cm_val)
    
    def setup_wandb(self, pl_module: LightningModule):
        if self._config.logger.name == 'wandb':
            pl_module.logger.experiment.define_metric("train/acc", summary="max")
            pl_module.logger.experiment.define_metric("train/f1", summary="max")
            pl_module.logger.experiment.define_metric("train/precision", summary="max")
            pl_module.logger.experiment.define_metric("train/recall", summary="max")
            
            pl_module.logger.experiment.define_metric("val/acc", summary="max")
            pl_module.logger.experiment.define_metric("val/f1", summary="max")
            pl_module.logger.experiment.define_metric("val/precision", summary="max")
            pl_module.logger.experiment.define_metric("val/recall", summary="max")
    
    def _calc_metrics(self, preds, labels, timestamps, delay: int=7):
        if preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        
        if labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        new_preds = [
            reconstruct_label(timestamp=ts.detach().cpu(), label=pred.detach().cpu(),
                mask=l.detach().cpu() != -1)
            for (pred, ts, l) in zip(preds, timestamps, labels)
        ]
        new_labels = [
            reconstruct_label(timestamp=ts.detach().cpu(), label=l.detach().cpu(), 
                              mask=l.detach().cpu() != -1)
            for l, ts in zip(labels, timestamps)
        ]

        new_pred_adj = [
            get_range_proba_torch(np, nl, delay=delay)
            for np, nl in zip(new_preds, new_labels)
        ]

        new_pred_adj = torch.concat(new_pred_adj, dim=0)
        new_labels = torch.concat(new_labels, dim=0)
        
        # metrics 
        acc = torch.sum(torch.eq(new_pred_adj, new_labels)) / len(new_labels)
        f1 = f1_score(new_labels.detach().cpu().numpy(), new_pred_adj.detach().cpu().numpy(), average='macro', labels=np.unique(new_labels.detach().cpu().numpy()))
        pre = precision_score(new_labels.detach().cpu().numpy(), new_pred_adj.detach().cpu().numpy(), average='macro', zero_division=np.nan)
        rec = recall_score(new_labels.detach().cpu().numpy(), new_pred_adj.detach().cpu().numpy(), average='macro', zero_division=np.nan)
        
        return acc.item(), f1, pre, rec
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mode = 'train'
        preds = outputs['preds']#.flatten()
        labels = outputs['labels']#.flatten()
        timestamps = outputs['timestamps']
        bs = outputs['bs']

        timestamps = [
            F.pad(ts, pad=(0, labels.shape[-1]-ts.shape[0]), 
                mode='constant', value=-1)
            for ts in timestamps
        ]
        
        # calc an log metrics    
        acc, f1, pre, rec = self._calc_metrics(preds, labels, timestamps, delay=trainer.datamodule.dataset._delay)
        
        self._log_metric(metric_key=MetricOptions.accuracy, metric_val=acc, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=pre, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=rec, bs=bs, mode=mode)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Called when the validation batch ends."""
        self._validation_step_outputs.append(outputs)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        mode = 'val'
        preds, labels, timestamps = [], [], []
        bs = self._validation_step_outputs[0]['bs']

        for batch_outputs in self._validation_step_outputs:
            p = batch_outputs['preds']
            if self._config.dataset == DatasetOptions.yahoo:
                preds.append(batch_outputs['preds'].squeeze(-1))
                labels.append(batch_outputs['labels'].reshape(p.shape).squeeze(-1))
            else:
                preds.append(batch_outputs['preds'])
                labels.append(batch_outputs['labels'].reshape(p.shape))
            timestamps.extend(batch_outputs['timestamps'])
        
        if self._config.dataset == DatasetOptions.kpi:
            labels = torch.concat(labels).squeeze(-1)
            preds = torch.concat(preds).squeeze(-1)

            timestamps = [
                F.pad(ts, pad=(0, trainer.datamodule.dataset._max_seq_len-ts.shape[0]), 
                    mode='constant', value=-1)
                for ts in timestamps
            ]
        elif self._config.dataset == DatasetOptions.yahoo:

            labels = [
                F.pad(label, pad=(0, trainer.datamodule.dataset._max_seq_len-label.shape[1]), 
                    mode='constant', value=-1)
                for label in labels
            ]

            preds = [
                F.pad(pred, pad=(0, trainer.datamodule.dataset._max_seq_len-pred.shape[1]), 
                    mode='constant', value=-1)
                for pred in preds
            ]

            timestamps = [
                F.pad(ts, pad=(0, trainer.datamodule.dataset._max_seq_len-ts.shape[0]), 
                    mode='constant', value=-1)
                for ts in timestamps
            ]
            
            labels = torch.concat(labels).squeeze(-1)
            preds = torch.concat(preds).squeeze(-1)
        else:
            raise ValueError(f'{self._config.dataset} is not a known anomaly dataset!')
        
        acc, f1, pre, rec = self._calc_metrics(preds, labels, timestamps, delay=trainer.datamodule.dataset._delay)
        
        self._log_metric(metric_key=MetricOptions.accuracy, metric_val=acc, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.f1, metric_val=f1, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.precision, metric_val=pre, bs=bs, mode=mode)
        self._log_metric(metric_key=MetricOptions.recall, metric_val=rec, bs=bs, mode=mode)

        self._validation_step_outputs.clear()
    
    # not test set


def reconstruct_label(timestamp: List | np.ndarray, label: Tensor, mask: Tensor=None):
    """
    Adjusted from https://github.com/zhihanyue/ts2vec/blob/main/tasks/anomaly_detection.py
    
    do not consider padded labels:
    """
    
    timestamp = torch.as_tensor(timestamp, dtype=torch.int64, device=timestamp.device)
    #print(label.unique())
    if mask is not None:
        label = label[mask]
        timestamp = timestamp[mask]
    #print(label.unique())
    assert -1 not in label.unique(), f'Padding label cannot be used for evaluation, {label.unique()}'
    index = torch.argsort(timestamp)

    timestamp_sorted = torch.as_tensor(timestamp[index], device=timestamp.device)
    interval = np.min(np.diff(timestamp_sorted))

    label = torch.as_tensor(label, dtype=torch.int64, device=label.device)
    label = torch.as_tensor(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = torch.zeros(((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=torch.int64, device=label.device)
    new_label[idx] = label

    return new_label

def get_range_proba_torch(predict, label, delay=7):
    """
    Adjusted from https://github.com/zhihanyue/ts2vec/blob/main/tasks/anomaly_detection.py
    """
    splits = torch.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = predict.clone()
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


from torch_ema import ExponentialMovingAverage as EMA
from typing import Dict

class ExponentialMovingAverage(Callback):
    """
    Exponential Moving Average Callback that performs a smoothed updates on the model. 

    ::math $\theta_{EMA, t+1} = (1 - \lambda) * \theta_{EMA, t} + \lambda * \theta_t$
    $\theta_{EMA, t}$ EMA model weights and $\theta_t$ the model weights. 

    Using EMA: https://github.com/fadel/pytorch_ema
    """
    def __init__(self, decay, *args, **kwargs):
        """
        Args:
            decay (float): Decay factor in exponentially moving average.
        """
        self.decay = decay
        self.ema = None
        self._ema_state_dict = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.ema is None:
            self.ema = EMA(pl_module.model.parameters(), decay=self.decay)
        if self._ema_state_dict is not None:
            self.ema.load_state_dict(self._ema_state_dict)
            self._ema_state_dict = None

        # load average parameters, to have same starting point as after validation
        self.ema.store()
        self.ema.copy_to()

    def on_train_epoch_start(self, trainer, pl_module):
        self.ema.restore()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
         self.ema.update()
        
    def on_validation_epoch_start(self, trainer, pl_module, outputs=None, batch=None, batch_idx=None, dataloader_idx = 0):
        self.ema.store()
        self.ema.copy_to()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "ema" in state_dict:
            if self.ema is None:
                self._ema_state_dict = state_dict["ema"]
            else:
                self.ema.load_state_dict(state_dict["ema"])

    def state_dict(self) -> Dict[str, Any]:
        return {"ema": self.ema.state_dict()}
    