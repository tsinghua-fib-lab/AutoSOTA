import numpy as np
import torch
from torch import optim
from typing import Dict
import lightning.pytorch as pl
import sys

from probts.data import ProbTSBatchData
from probts.data.data_utils.data_scaler import Scaler
from probts.model.forecaster import Forecaster
from probts.utils.evaluator import Evaluator
from probts.utils.metrics import *
from probts.utils.save_utils import update_metrics, calculate_weighted_average, load_checkpoint, get_hor_str
from probts.utils.utils import init_class_helper

def get_weights(sampling_weight_scheme, max_hor):
    '''
    return: w [max_hor]
    '''
    if sampling_weight_scheme == 'random':
        i_array = np.linspace(1 + 1e-5, max_hor - 1e-3, max_hor)
        w = (1 / max_hor) * (np.log(max_hor) - np.log(i_array))
    elif sampling_weight_scheme == 'const':
        w = np.array([1 / max_hor] * max_hor)
    elif sampling_weight_scheme == 'none':
        return None
    else:
        raise ValueError(f"Invalid sampling scheme {sampling_weight_scheme}.")
    
    return torch.tensor(w)


class ProbTSForecastModule(pl.LightningModule):
    def __init__(
        self,
        forecaster: Forecaster,
        scaler: Scaler = None,
        train_pred_len_list: list = None,
        num_samples: int = 100,
        learning_rate: float = 1e-3,
        quantiles_num: int = 10,
        load_from_ckpt: str = None,
        sampling_weight_scheme: str = 'none',
        optimizer_config = None,
        lr_scheduler_config = None,
        **kwargs
    ):
        super().__init__()
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.load_from_ckpt = load_from_ckpt
        self.train_pred_len_list = train_pred_len_list
        self.forecaster = forecaster
        self.optimizer_config = optimizer_config
        self.scheduler_config = lr_scheduler_config
        
        if self.optimizer_config is not None:
            print("optimizer config: ", self.optimizer_config)
            
        if self.scheduler_config is not None:
            print("lr_scheduler config: ", self.scheduler_config)
        
        self.scaler = scaler
        self.evaluator = Evaluator(quantiles_num=quantiles_num)
        
        # init the parapemetr for sampling
        self.sampling_weight_scheme = sampling_weight_scheme
        print(f'sampling_weight_scheme: {sampling_weight_scheme}')
        self.save_hyperparameters()

    @classmethod
    def load_from_checkpoint(self, checkpoint_path, scaler=None, learning_rate=None, no_training=False, **kwargs):
        model = load_checkpoint(self, checkpoint_path, scaler=scaler, learning_rate=learning_rate, no_training=no_training, **kwargs)
        return model

    def training_forward(self, batch_data):
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        batch_data.future_target_cdf = self.scaler.transform(batch_data.future_target_cdf)
        loss = self.forecaster.loss(batch_data)

        if len(loss.shape) > 1:
            loss_weights = get_weights(self.sampling_weight_scheme, loss.shape[1])
            loss = (loss_weights.detach().to(loss.device).unsqueeze(0).unsqueeze(-1) * loss).sum(dim=1)
            loss = loss.mean()
        
        return loss

    def training_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        loss = self.training_forward(batch_data)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def evaluate(self, batch, stage='',dataloader_idx=None):
        batch_data = ProbTSBatchData(batch, self.device)
        pred_len = batch_data.future_target_cdf.shape[1]
        orin_past_data = batch_data.past_target_cdf[:]
        orin_future_data = batch_data.future_target_cdf[:]

        norm_past_data = self.scaler.transform(batch_data.past_target_cdf)
        norm_future_data = self.scaler.transform(batch_data.future_target_cdf)
        self.batch_size.append(orin_past_data.shape[0])
        
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)[:,:, :pred_len]
        
        # Calculate denorm metrics
        denorm_forecasts = self.scaler.inverse_transform(forecasts)
        metrics = self.evaluator(orin_future_data, denorm_forecasts, past_data=orin_past_data, freq=self.forecaster.freq)
        self.metrics_dict = update_metrics(metrics, stage, target_dict=self.metrics_dict)
        
        # Calculate norm metrics
        norm_metrics = self.evaluator(norm_future_data, forecasts, past_data=norm_past_data, freq=self.forecaster.freq)
        self.metrics_dict = update_metrics(norm_metrics, stage, 'norm', target_dict=self.metrics_dict)
        
        l = orin_future_data.shape[1]
        
        if stage != 'test' and self.sampling_weight_scheme not in ['fix', 'none']:
            loss_weights = get_weights('random', l)
        else:
            loss_weights = None

        hor_metrics = self.evaluator(orin_future_data, denorm_forecasts, past_data=orin_past_data, freq=self.forecaster.freq, loss_weights=loss_weights)
        
        if stage == 'test':
            hor_str = get_hor_str(self.forecaster.prediction_length, dataloader_idx)
            if hor_str not in self.hor_metrics:
                self.hor_metrics[hor_str] = {}

            
            self.hor_metrics[hor_str] = update_metrics(hor_metrics, stage, target_dict=self.hor_metrics[hor_str])

        return hor_metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        metrics = self.evaluate(batch, stage='val',dataloader_idx=dataloader_idx)
        return metrics


    def on_validation_epoch_start(self):
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.batch_size = []

    def on_validation_epoch_end(self):
        avg_metrics = calculate_weighted_average(self.metrics_dict, self.batch_size)
        self.log_dict(avg_metrics, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        metrics = self.evaluate(batch, stage='test',dataloader_idx=dataloader_idx)
        return metrics

    def on_test_epoch_start(self):
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.avg_metrics = {}
        self.avg_hor_metrics = {}
        self.batch_size = []

    def on_test_epoch_end(self):
        if len(self.hor_metrics) > 0:
            for hor_str, metric in self.hor_metrics.items():
                self.avg_hor_metrics[hor_str] = calculate_weighted_average(metric, batch_size=self.batch_size)
                self.avg_metrics.update(calculate_weighted_average(metric, batch_size=self.batch_size, hor=hor_str+'_'))
        else:
            self.avg_metrics = calculate_weighted_average(self.metrics_dict, self.batch_size)
        
        if isinstance(self.forecaster.prediction_length, int) or len(self.forecaster.prediction_length) < 2:
            self.log_dict(self.avg_metrics, logger=True)

    def predict_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)
        return forecasts

    def configure_optimizers(self):
        if self.optimizer_config is None:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = init_class_helper(self.optimizer_config['class_name'])
            params = self.optimizer_config['init_args']
            optimizer = optimizer(self.parameters(), **params)
        
        if self.scheduler_config is not None:
            scheduler = init_class_helper(self.scheduler_config['class_name'])
            params = self.scheduler_config['init_args']
            scheduler = scheduler(optimizer=optimizer, **params)
            
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": None,
            }

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return optimizer
