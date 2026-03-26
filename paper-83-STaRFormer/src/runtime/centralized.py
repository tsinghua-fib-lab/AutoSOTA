import matplotlib.pyplot as plt
import logging
import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf
from typing import Any, Mapping, Union, List, Tuple
from sklearn.metrics import fbeta_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import PackedSequence
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

import wandb


from ..utils import DatasetOptions, ModelOptions, ClassificationMethodOptions, PredictionOptions, MaskingOptions, TrainingModeOptions, TaskOptions
from ..utils.datasets import BaseData
from src.nn import DAReMLoss


__all__ = ["CentralizedModel"]


class CentralizedModel(L.LightningModule):
    """CentralizedModel

    A LightningModule that wraps a backbone PyTorch model and provides
    a centralized training/validation/test workflow with configurable
    task-specific behavior.

    Attributes:
        config (dict-like or OmegaConf): Global configuration for the model,
            dataset, task, logging, loss functions, and model options, etc...
        model (nn.Module): The underlying neural network module to be trained.
    """
    def __init__(self, config: DictConfig, model: nn.Module, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        """Initialize the centralized model wrapper.

        Args:
            config (OmegaConf): Configuration object containing.
            model (nn.Module): The actual neural network to be optimized.
        """
        self.config = config
        self.model = model
        self.save_hyperparameters(ignore=['model', 'criterion', 'config'])
        self.configure_cli_logger()
        
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            # log confgis in wandb
            if self.config.logger.name == 'wandb':
                self.log_config_to_wandb()
                
            self.configure_criterion()
            self.logger.log_hyperparams(self.hparams)

        return super().setup(stage)

    @rank_zero_only
    def log_config_to_wandb(self,):
        self.trainer.logger.experiment.config.update(
            OmegaConf.to_container(self.config, resolve=True)
        )

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Union[Tensor, Mapping[str, Any], None]:
        return self.__shared_step(batch, batch_idx, mode=TrainingModeOptions.train, *args, *kwargs)

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Union[Tensor, Mapping[str, Any], None]:
        return self.__shared_step(batch, batch_idx, mode=TrainingModeOptions.val, *args, *kwargs)

    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Union[Tensor, Mapping[str, Any], None]:
        return self.__shared_step(batch, batch_idx, mode=TrainingModeOptions.test, *args, *kwargs)
    
    def __shared_step(self, batch: Tuple[Tensor | PackedSequence] | BaseData,  batch_idx, mode: str, *args: Any, **kwargs: Any) -> Union[Tensor, Mapping[str, Any], None]:
        """Facilitates training/validation/testing logic for different model backbones.

        Args:
            batch (Tuple[Tensor | PackedSequence] | BaseData): Input batch. Structure depends on the dataset; expected to
                provide attributes like data, label, batch_size, seq_len, etc.
            batch_idx (int): Batch index.
            mode (str): One of TrainingModeOptions.train/val/test.
            *args / **kwargs: Additional arguments. May include pred_type.

        Returns:
            forward pass outputs (dict): A dictionary with keys such as preds, loss, targets, labels,
                acc, cm, etc., depending on the task and model type.
        """
        # define prediction requirement
        if self.config.task == TaskOptions.classification:
            if self.config.dataset.lower() == DatasetOptions.dkt or \
                self.config.dataset.lower() in DatasetOptions.uea_binary or \
                self.config.dataset.lower() in [DatasetOptions.p12, DatasetOptions.p19]:
                # binary
                pred_type = PredictionOptions.binary
            elif self.config.dataset.lower() in [DatasetOptions.geolife, DatasetOptions.pam] or \
                self.config.dataset.lower() in DatasetOptions.uea_multi:
                # multiclass
                pred_type = PredictionOptions.multiclass
            else:
                raise RuntimeError(f"Could not identify correct shared step for {self.config.dataset.lower()}.")
        
        elif self.config.task == TaskOptions.regression:
            if self.config.dataset.lower() in DatasetOptions.tsr:
                pred_type = PredictionOptions.binary # irrelevant

        elif self.config.task == TaskOptions.anomaly_detection:
            # anomaly detection generally a binary classification task
            pred_type = PredictionOptions.binary

        return self.__shared_step_wrapper(batch, batch_idx, mode, pred_type=pred_type, *args, **kwargs)

    def __shared_step_wrapper(self, batch: Tuple[Tensor | PackedSequence] | BaseData, batch_idx, mode: str, *args: Any, **kwargs: Any):
        pred_type=kwargs['pred_type']
        return_dict = {}

        if batch.data.data.dtype == torch.float64 and (self.device == torch.device('mps') or self.device == torch.device('mps:0')):
                batch.data = batch.data.float()
                batch.label = batch.label.type(torch.float32)
            
        if self.device == torch.device('mps:0') or self.device == torch.device('mps') \
                or self.device == torch.device('cuda:0') or self.device == torch.device('cuda'):
            batch.data = batch.data.float().to(self.device)
            batch.label = batch.label.to(self.device)
            if self.config.task == TaskOptions.regression: batch.target = batch.target.to(self.device)

        bs = batch.batch_size
        return_dict['bs'] = bs
        return_dict['batch'] = batch

        if batch.label.device != self.device:
            batch.label = batch.label.to(self.device)
        
        if self.config.model.sequence_model.name == ModelOptions.starformer:
            return_dict_step = self.__shared_step_starformer(batch, batch_idx, mode=mode, pred_type=pred_type)
        elif self.config.model.sequence_model.name == 'fcn':
            return_dict_step = self.__shared_step_fcn(batch, batch_idx)
        elif self.config.model.sequence_model.name == ModelOptions.rnn_based:
            return_dict_step = self.__shared_step_rnn(batch, batch_idx, pred_type=pred_type)
        else:
            return RuntimeError(f'{self.config.model.sequence_model.name} not found!')
        
        return_dict.update(return_dict_step)
        if self.config.task == TaskOptions.classification:
            # get accuracy
            preds = return_dict['preds']
            acc = torch.sum(torch.eq(preds.flatten(), batch.label.flatten().int())) / len(batch.label.flatten())
            
            # confusion matrix
            text_labels = list(self.config.datamodule.text_labels)
            cm_matrix = confusion_matrix(
                y_true=batch.label.flatten().detach().cpu().numpy(),
                y_pred=preds.flatten().detach().cpu().numpy(),
                labels=text_labels
                )

        return_dict['labels'] = batch.label
        if self.config.task == TaskOptions.classification:
            return_dict['acc'] = acc
            return_dict['cm'] = cm_matrix

        if self.config.task == TaskOptions.regression:
            #return_dict['scaler'] = batch.scaler
            return_dict['targets'] = batch.target
        
        if self.config.task == TaskOptions.anomaly_detection:
            return_dict['timestamps'] = batch.timestamps

        return return_dict        

    def __shared_step_rnn(self, batch: Tuple[Tensor | PackedSequence] | BaseData, batch_idx, pred_type: str, *args: Any, **kwargs: Any):
        if self.config.model.sequence_model.name in ModelOptions.rnn_based:
            out_dict = self.model(batch.data, N=batch.seq_len, batch_size=batch.batch_size)
        else:
            #outputs = self.model(batch)
            raise NotImplementedError
        
         # compute loss
        loss = self.__compute_loss(logits=out_dict['logits'], batch=batch, pred_type=pred_type)
        # Prediction
        predictions_dict = self.__compute_predictions(logits=out_dict['logits'], pred_type=pred_type)
        predictions_dict.update({'loss': loss})
        return predictions_dict

    def __shared_step_fcn(self, batch: Tuple[Tensor | PackedSequence] | BaseData, batch_idx, pred_type: str=None, *args: Any, **kwargs: Any):
        out_dict = self.model(batch.data)
        loss_params = {
            'logits': out_dict['logits'], 
            'batch': batch, 
            'pred_type': pred_type,
        }
        if self.config.task == TaskOptions.regression:
            loss_params['targets'] = batch.target # ad regression targets for loss calculation

         # compute loss
        loss = self.__compute_loss(**loss_params)
        # Prediction
        predictions_dict = self.__compute_predictions(logits=out_dict['logits'], pred_type=pred_type)
        predictions_dict.update({'loss': loss})
        return predictions_dict

    def __shared_step_starformer(self, batch: Tuple[Tensor | PackedSequence] | BaseData, batch_idx, mode: str, pred_type: str, *args: Any, **kwargs: Any):
        return_dict = {}
        if self.config.model.sequence_model.masking is not None and mode != TrainingModeOptions.test:
            if self.config.loss.loss_fn == 'darem_sscl':
                inputs = {
                    'x': batch.data, 
                    'N': batch.seq_len, 
                    'batch_size': batch.batch_size, 
                    'padding_mask': None, 
                    'mode': mode, 
                    'dataset': self.config.dataset.lower()
                }
                if self.model.__class__.__name__ == "SequenceTextDualModel":
                    if hasattr(batch, 'demogr_desc'):
                        inputs['text'] = batch.demogr_desc 
                    else:
                        raise RuntimeError(f'No known text input!')

                out_dict = self.model(**inputs)

                loss_params = {
                    'logits': out_dict['logits'], 
                    'unmasked': out_dict['embedding_cls'], 
                    'masked': out_dict['embedding_masked'], 
                    'seq_len': batch.seq_len if hasattr(batch, 'seq_len') else None,
                    'y': batch.label, 
                    'mode': mode,
                    'pred_type': pred_type
                }
                if self.config.task == TaskOptions.regression:
                    loss_params['targets'] = batch.target # ad regression targets for loss calculation

                # pass sequence lengths for specific datasets (for padding)
                if self.config.dataset.lower() in [DatasetOptions.geolife, DatasetOptions.dkt, DatasetOptions.p19, DatasetOptions.p12]: 
                    loss_params['seq_len'] = batch.seq_len
                
                loss, loss_task, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.__compute_loss(
                    **loss_params
                )
            
                return_dict['loss'] = loss
                loss_task_key = 'loss_ce' if self.config.task == TaskOptions.classification else 'loss_task'
                return_dict[loss_task_key] = loss_task
                return_dict['loss_contrastive'] = loss_contrastive
                return_dict['loss_contrastive_batch_sim'] = loss_contrastive_batch_sim
                return_dict['loss_contrastive_class_sim'] = loss_contrastive_class_sim
            else:
                raise RuntimeError
            
        else:
            # When testing, only compute binary or multiclass predcition, no contrastive learning
            inputs = {
                    'x': batch.data, 
                    'N': batch.seq_len, 
                    'batch_size': batch.batch_size, 
                    'padding_mask': None, 
                    'mode': mode, 
                    'dataset': self.config.dataset.lower()
                }
            if self.model.__class__.__name__ == "SequenceTextDualModel":
                if hasattr(batch, 'demogr_desc'):
                    inputs['text'] = batch.demogr_desc 
                else:
                    raise RuntimeError(f'No known text input!')
            out_dict = self.model(**inputs)
            # compute loss
            loss_params = {
                'logits': out_dict['logits'], 
                'batch': batch, 
                'mode': mode, 
                'pred_type': pred_type
            }

            if self.config.task == TaskOptions.regression:
                loss_params['targets'] = batch.target

            loss = self.__compute_loss(**loss_params)
            return_dict['loss'] = loss
        
        # Prediction
        predictions_dict = self.__compute_predictions(logits=out_dict['logits'], pred_type=pred_type)
        return_dict.update(predictions_dict)

        return return_dict

    def __compute_loss(self, 
        logits: Tensor, # target scores, no sigmoid applied (no proba's)
        y: Tensor=None, 
        batch=None, 
        unmasked: Tensor=None, 
        masked: Tensor=None,
        mode: str=TrainingModeOptions.train,
        pred_type: str=PredictionOptions.binary,
        seq_len: Tensor=None,
        targets: Tensor=None,
        ):
        """Computes the loss for a given batch.

        This function is responsible for selecting and computing the
        appropriate loss based on the configured task, prediction type, model 
        and any dataset-specific nuances.

        Args:
            logits (Tensor): Model output logits (before any activation like softmax).
            y (Tensor, optional): Ground-truth labels or targets when applicable.
            batch (Tuple[Tensor | PackedSequence] | BaseData, optional): Original batch object; may be used to access targets or other metadata.
            unmasked (Tensor, optional): Unmasked embeddings or logits (used by some losses).
            masked (Tensor, optional): Masked embeddings or logits (used by some losses).
            mode (str, optional): Current mode (train/val/test). Defaults to TrainingModeOptions.train.
            pred_type (str, optional): Type of prediction target (binary, multiclass, etc.).
            seq_len (Tensor, optional): Sequence length metadata for sequence models.
            targets (Tensor, optional): Regression targets when applicable.

        Returns:
            loss (Tensor): The computed loss value.
        """
        if self.config.task == TaskOptions.classification:
            if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.loss.loss_fn == 'darem_sscl':
                if mode == TrainingModeOptions.test:
                    if y is None:
                        assert batch is not None
                        y = batch.label

                    loss = self.criterion['darem_sscl'](
                        y_logits=logits, unmasked=unmasked, masked=masked, y=y, seq_len=seq_len, per_seq_element=False, mode=mode,
                    ) 
                    return loss
                else:
                    loss, loss_ce, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.criterion['darem_sscl'](
                        y_logits=logits, unmasked=unmasked, masked=masked, y=y, seq_len=seq_len, per_seq_element=False, mode=mode,
                    ) 
                    return loss, loss_ce, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim
            
            elif pred_type == PredictionOptions.binary:
                if logits.dtype == torch.double:
                    labels = batch.label.type(logits.dtype)
                else:
                    labels = batch.label
                
                if batch.label.dtype == torch.long or batch.label.dtype == torch.int64 or \
                 batch.label.dtype == torch.int32:
                    labels = batch.label.type(logits.dtype)

                loss = self.criterion['binary_cross_entropy'](logits.flatten(), labels.flatten())
                return loss

            elif pred_type == PredictionOptions.multiclass:
                loss = self.criterion['cross_entropy'](logits, batch.label.flatten())
                return loss

            else:
                raise NotImplementedError(f'{pred_type}')
        
        elif self.config.task == TaskOptions.regression:
            assert targets is not None
            if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.loss.loss_fn == 'darem_sscl':
                if mode == TrainingModeOptions.test:
                    if y is None:
                        assert batch is not None
                        y = batch.label
                    loss = self.criterion['darem_sscl'](
                        y_logits=logits, unmasked=unmasked, masked=masked, 
                        y=y, targest=targets, seq_len=seq_len, 
                        per_seq_element=False, mode=mode,
                    ) 
                    return loss
                else:
                    loss, loss_task, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.criterion['darem_sscl'](
                        y_logits=logits, unmasked=unmasked, masked=masked, 
                        y=y, targets=targets, seq_len=seq_len, 
                        per_seq_element=False, mode=mode,
                    ) 
                    return loss, loss_task, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim
            else:
                loss_fn = 'mean_squarred_error' if self.criterion.get('mean_squarred_error', None) is not None else 'mean_absolute_error'
                loss_task = self.criterion[loss_fn](logits, targets) 
                return loss_task

        elif self.config.task == TaskOptions.anomaly_detection:
            if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.loss.loss_fn == 'darem_sscl':
                
                if mode == TrainingModeOptions.test:
                    if y is None:
                        assert batch is not None
                        y = batch.label
                    loss = self.criterion['darem_sscl'](
                        y_logits=logits, unmasked=unmasked, masked=masked, y=y, seq_len=seq_len, per_seq_element=True, mode=mode,
                    ) 
                    return loss
                else:
                    loss, loss_ce, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.criterion['darem_sscl'](
                        y_logits=logits, unmasked=unmasked, masked=masked, y=y, seq_len=seq_len, per_seq_element=True, mode=mode,
                    ) 
                    return loss, loss_ce, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim
            
            elif pred_type == PredictionOptions.binary:
                if logits.dtype == torch.double:
                    labels = batch.label.type(logits.dtype)
                else:
                    labels = batch.label
                
                #if batch.label.dtype in [torch.long, torch.int64, torch.int32]:
                if batch.label.dtype == torch.long or batch.label.dtype == torch.int64 or \
                 batch.label.dtype == torch.int32:
                    labels = batch.label.type(logits.dtype)
                
                if -1 in labels.unique():
                    padding_mask = labels.flatten() != -1
                    logits = logits.flatten()*padding_mask
                    labels = labels.flatten()*padding_mask
                else:
                    logits = logits.flatten()
                    labels = labels.flatten()

                loss = self.criterion['binary_cross_entropy'](logits, labels)
                return loss

            elif pred_type == PredictionOptions.multiclass:
                loss = self.criterion['cross_entropy'](logits, batch.label.flatten())
                return loss

            else:
                raise NotImplementedError(f'{pred_type}')


        else:
            raise NotImplementedError(f'{self.config.task}')

    def __compute_predictions(self, 
                              logits: Tensor, 
                              pred_type: str='binary_cls', 
        ):
        """Computes the predictions for a given batch.

        Args:
            logits (Tensor): Model output logits (before any activation like softmax).
            pred_type (str, optional): Type of prediction target (binary, multiclass, etc.).

        Returns:
            predictions (dict): Returns a dictionary with the predictions and the 
            prediction probabilites.
        """
        if self.config.task == TaskOptions.classification:
            if pred_type == PredictionOptions.binary:
                probas = F.sigmoid(logits) 
                preds = (probas > 0.5).int() # converts output probabilities to binary prediction labels, i.e., [0,1]
            elif pred_type == PredictionOptions.multiclass:
                probas = F.softmax(logits, dim=1)
                probas_max, probas_argmax = torch.max(probas, dim=1)
                preds = probas_argmax
            else:
                raise NotImplementedError(f'{pred_type}')
        
        elif self.config.task == TaskOptions.regression:
            probas = None
            preds = logits
            
        elif self.config.task == TaskOptions.anomaly_detection:
            if pred_type == PredictionOptions.binary:
                probas = F.sigmoid(logits) 
                preds = (probas > 0.5).int() # converts output probabilities to binary prediction labels, i.e., [0,1]
                
            elif pred_type == PredictionOptions.multiclass:
                probas = F.softmax(logits, dim=1)
                probas_max, probas_argmax = torch.max(probas, dim=1)
                preds = probas_argmax
            else:
                raise NotImplementedError(f'{pred_type}')
            
        else:
            raise NotImplementedError(f'{self.config.task} not implemented!')
        
        return {'preds': preds, 'probas': probas}
            
    def configure_optimizers(self):
        if self.config.optimizer.name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), 
                lr=self.config.training.learning_rate, 
                momentum=self.config.optimizer.momentum
            )
        elif self.config.optimizer.name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(), 
                lr=self.config.training.learning_rate
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), 
                lr=self.config.training.learning_rate, 
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                eps=self.config.optimizer.eps,
                weight_decay=self.config.optimizer.weight_decay,
            )
        
        scheduler = None   
        if self.config.callbacks.lr_scheduler.apply:
            if self.config.callbacks.lr_scheduler.name == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode=self.config.callbacks.lr_scheduler.mode,
                    factor=self.config.callbacks.lr_scheduler.factor,
                    patience=self.config.callbacks.lr_scheduler.patience,
                    min_lr=self.config.callbacks.lr_scheduler.min_lr,
                )
            else:
                raise ValueError(f'{self.config.callbacks.lr_scheduler.name} not knonw!')
        if scheduler == None:
            return optimizer
        else:
            return [optimizer], [{
                'scheduler': scheduler,
                'monitor': self.config.callbacks.lr_scheduler.monitor,
            }]

    def configure_criterion(self):
        self.criterion = {}

        if TaskOptions.classification:
            _multi = [DatasetOptions.geolife] 
            _multi.extend(DatasetOptions.uea_multi)
            _multi.append(DatasetOptions.pam)

            _binary = DatasetOptions.uea_binary
            _binary.append(DatasetOptions.p19)

            if isinstance(self.config.loss.loss_fn, str):
                loss_fns = [self.config.loss.loss_fn]

            for loss_fn in loss_fns:
                if loss_fn == 'cross_entropy':
                    self.criterion[loss_fn] = nn.CrossEntropyLoss()
                elif loss_fn == 'nll_loss':
                    self.criterion[loss_fn] = nn.NLLLoss()
                elif loss_fn == 'binary_cross_entropy':
                    self.criterion[loss_fn] = nn.BCEWithLogitsLoss()
                elif loss_fn == 'mean_squarred_error':
                    self.criterion[loss_fn] = nn.MSELoss()
                elif loss_fn == 'darem_sscl':
                    loss_kwargs = OmegaConf.to_container(self.config.loss, resolve=True)
                    for k in ['loss_fn', 'method', 'pred_type', 'lamdba_sim', 'lambda_contrastive']:
                        if loss_kwargs.get(k, None) is not None:
                            loss_kwargs.pop(k)
                    self.criterion['darem_sscl'] = DAReMLoss(
                        method=self.config.loss.method,
                        pred_type=self.config.loss.pred_type,
                        loss_kwargs=loss_kwargs,
                        task=self.config.loss.task,
                    )
                else:
                    raise ValueError(f'{self.config.loss.loss_fn} is not available!')

        elif TaskOptions.regression:
            if isinstance(self.config.loss.loss_fn, str):
                loss_fns = [self.config.loss.loss_fn]

            for loss_fn in loss_fns:
                if loss_fn == 'mean_squarred_error':
                    self.criterion[loss_fn] = nn.MSELoss()
                elif loss_fn == 'mean_absolut_error':
                    self.criterion[loss_fn] = nn.L1Loss()
                elif loss_fn == 'darem_sscl':
                    loss_kwargs = OmegaConf.to_container(self.config.loss, resolve=True)
                    for k in ['loss_fn', 'method', 'pred_type', 'task', 'task_loss_fn']:
                        loss_kwargs.pop(k)
                    self.criterion['darem_sscl'] = DAReMLoss(
                        method=self.config.loss.method,
                        pred_type=self.config.loss.pred_type,
                        loss_kwargs=loss_kwargs,
                        task=self.config.loss.task,
                        task_loss_fn=self.config.loss.task_loss_fn
                    )
                else:
                    raise ValueError(f'{self.config.loss.loss_fn} is not available!')

        elif TaskOptions.anomaly_detection:
            _binary = DatasetOptions.anomaly

            if isinstance(self.config.loss.loss_fn, str):
                loss_fns = [self.config.loss.loss_fn]

            for loss_fn in loss_fns:
                if loss_fn == 'cross_entropy':
                    self.criterion[loss_fn] = nn.CrossEntropyLoss()
                elif loss_fn == 'nll_loss':
                    self.criterion[loss_fn] = nn.NLLLoss()
                elif loss_fn == 'binary_cross_entropy':
                    self.criterion[loss_fn] = nn.BCEWithLogitsLoss()
                elif loss_fn == 'mean_squarred_error':
                    self.criterion[loss_fn] = nn.MSELoss()
                elif loss_fn == 'darem_sscl':
                    loss_kwargs = OmegaConf.to_container(self.config.loss, resolve=True)
                    for k in ['loss_fn', 'method', 'pred_type', 'lamdba_sim', 'lambda_contrastive']:
                        if loss_kwargs.get(k, None) is not None:
                            loss_kwargs.pop(k)
                    self.criterion['darem_sscl'] = DAReMLoss(
                        method=self.config.loss.method,
                        pred_type=self.config.loss.pred_type,
                        loss_kwargs=loss_kwargs,
                        task=self.config.loss.task,
                    )
                else:
                    raise ValueError(f'{self.config.loss.loss_fn} is not available!')
    
    def configure_cli_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        self.cli_logger = logger