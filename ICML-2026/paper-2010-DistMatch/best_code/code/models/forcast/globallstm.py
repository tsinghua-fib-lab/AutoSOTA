import logging
from typing import Optional, Tuple, Iterable

import hydra
import torch
import wandb
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, MetricCollection
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from loader.dataset import ChronoSplittedTsDataset
from models.base_model import BaseModel
from models.forcast.forcast_base import ForcastModel, FCPredictionData, FCModelPrediction, ForcastMode, \
    PredictionOutputType, FcSingleModelPrediction
from models.forcast.mdn_lstm import MDNLSTMManyToOne
from models.uncertainty.components.eps_ctx_encode import FcModel
from models.uncertainty.components.mdn import MNDCoef
from trainer.evaluator import Evaluator
from trainer.prep_iterator import ManyToOneIterator, CompleteDataset
from trainer.utils import map_merge_dicts, merge_dicts, batch_to_device_all, map_identity

LOGGER = logging.getLogger(__name__)

class SimpleLSTM(BaseModel):
    def __init__(self, input_size, lstm_conf, dropout, use_mc_dropout, **kwargs):
        super().__init__()
        self._mc_samples = 500 if use_mc_dropout else None
        self._constructor_args = dict(input_size=input_size, lstm_conf=lstm_conf, **kwargs)
        self._lstm = nn.LSTM(input_size=input_size, batch_first=True, **lstm_conf)
        self._dropout_rate = dropout
        if self._dropout_rate > 0:
            self._dropout = nn.Dropout(p=self._dropout_rate)
        self._fc = FcModel(input_dim=lstm_conf['hidden_size'], out_dim=1, hidden=())

    def forward(self, x, h=None, **kwargs):
        lstm_out, (h, c) = self._lstm(x, h)
        mc_y_hat = None
        if self._dropout_rate > 0:
            lstm_out = lstm_out[:, -1, :]
            y_hat = self._fc(self._dropout(lstm_out)).unsqueeze(1)
            if not self.training and self._mc_samples is not None:
                # In Evalution we need the samples
                self._dropout.train(True)
                tmp = []
                for i in range(self._mc_samples):
                    tmp.append(self._fc(self._dropout(lstm_out)).unsqueeze(1))
                mc_y_hat = torch.cat(tmp, dim=2)
                del tmp
                self._dropout.train(False)  # TODO
        else:
            assert self._mc_samples is None
            lstm_out = lstm_out[:, -1, :]
            y_hat = self._fc(lstm_out).unsqueeze(1)
        if mc_y_hat is not None:
            return dict(y_hat=y_hat, mc_y_hat=mc_y_hat)
        else:
            return dict(y_hat=y_hat)

    def get_loss_func(self, **kwargs):
        loss_func = MSELoss()
        def loss(y_hat, y, **kwargs):
            mse_loss = loss_func(input=y_hat, target=y)
            return mse_loss, {'mse': mse_loss}

        return loss

    def _get_constructor_parameters(self) -> dict:
        return self._constructor_args

    def get_train_fingerprint(self) -> dict:
        raise NotImplemented("asdf")


class GlobalLSTM(ForcastModel):

    def __init__(self, no_x_features, model_params, **kwargs) -> None:
        self._use_mc_dropout = model_params['use_mc_dropout']
        self._use_with_mdn = model_params.get('use_with_mdn', False)
        assert not (self._use_mc_dropout and self._use_with_mdn)
        super().__init__(forcast_mode=ForcastMode.PREDICT_INDEPENDENT,
                         supported_outputs=(PredictionOutputType.QUANTILE,) if self._use_mc_dropout or self._use_with_mdn else (PredictionOutputType.POINT, ))
        if not self._use_with_mdn:
            self._lstm_model = SimpleLSTM(input_size=no_x_features+1, lstm_conf=model_params['lstm_conf'],
                                          dropout=model_params['dropout'], use_mc_dropout=self._use_mc_dropout)
        else:
            self._lstm_model = MDNLSTMManyToOne(input_size=no_x_features+1, lstm_conf=model_params['lstm_conf'],
                                                dropout=model_params['dropout'], mdn_conf=model_params['mdn_conf'])
        self._seq_len = model_params['seq_len']
        self._batch_size = model_params['batch_size']
        self._train_split = 0.7
        self._train_with_calib = model_params['train_with_calib']
        self._train_only = kwargs.get('train_only', False)
        self._save_prediction = kwargs.get('save_predictions', False)
        self._pre_trained_model_path = model_params.get('pre_trained_model_path', None)
        self._plot_eval_after_train = model_params['plot_eval_after_train']
        self._pre_trained_predictions_paths = model_params.get('pre_trained_predictions_paths', None)
        self._predictions = dict()
        self._prediction_offset = None


    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        raise NotImplemented("Asdf")

    def train_global(self, datasets, alphas, trainer_config, experiment_config):
        # 1) Check if model available -> Load model / Otherwise train
        if self._pre_trained_model_path is None:
            trainer = hydra.utils.instantiate(
                trainer_config,
                experiment_config=experiment_config,
                model=self._lstm_model,
                get_data_loader=lambda: (
                    self._get_dataloader(datasets, is_val=False),
                    self._get_dataloader(datasets, is_val=True),
                ),
                move_batch_to_device=batch_to_device_all(),
                map_to_model_in=map_identity(),
                loss_func=self._lstm_model.get_loss_func(),
                map_to_loss_in=merge_dicts(),
                val_metrics=MetricCollection(MeanSquaredError()),
                train_metrics=MetricCollection(MeanSquaredError()),
                map_to_metrics_in=map_merge_dicts({"y_hat": "preds", "y": "target"}),
            )
            trainer.train()
            if self._plot_eval_after_train:
                # Log Evalution Prediciton after Training
                predicitons, metrics = self._create_predictions_point(datasets, trainer, evaluate_mode='train')
                for dataset in datasets:
                    if self._train_with_calib:
                        split_point = int((dataset.no_train_steps + dataset.no_calib_steps) * self._train_split)
                    else:
                        split_point = int(dataset.no_train_steps * self._train_split)
                    y = dataset.Y_train[split_point:].cpu()
                    y_hat = predicitons[dataset.ts_id][split_point - self._seq_len:].cpu()
                    data = pd.DataFrame(torch.cat((y, y_hat), dim=1).numpy(), columns=["Y", "Y_hat"])
                    fig, ax = plt.subplots()
                    sns.lineplot(data=data, ax=ax)
                    wandb.log({f"Val-Plots-{dataset.ts_id}": fig})
        else:
            self._lstm_model.load_state(self._pre_trained_model_path)

        if self._pre_trained_predictions_paths is None and not self._train_only:
            evaluator = Evaluator(
                gpu_id=experiment_config.gpu_id,
                move_batch_to_device=batch_to_device_all(),
                map_to_model_in=map_identity(),
                val_metrics=MetricCollection(MeanSquaredError()),
                map_to_metrics_in=map_merge_dicts({"y_hat": "preds", "y": "target"}),
            )
            # 2) Create all predictions for train, calib and test
            self._predictions, _ = self._create_predictions_point(datasets, evaluator, evaluate_mode='all')
            if self._save_prediction:
                torch.save(self._predictions, f"{experiment_config.experiment_dir}/predictions.pt")
        elif not self._train_only:
            self._predictions = torch.load(self._pre_trained_predictions_paths, map_location='cpu')

        self._prediction_offset = self._seq_len

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        start = pred_data.step_offset - self._prediction_offset
        end = start + pred_data.no_fc_steps
        prediction = self._predictions[pred_data.ts_id][start:end].to(device=pred_data.X_step.device)
        if self._use_mc_dropout or self._use_with_mdn:
            # ToDo Hack only use 500 mc samples
            assert prediction.shape[1] > 100
            use_mc_samples = 500
            prediction = prediction[:, :use_mc_samples]
            lower = torch.quantile(prediction, pred_data.alpha / 2, dim=1, keepdim=True)
            upper = torch.quantile(prediction, (1 - pred_data.alpha / 2), dim=1, keepdim=True)
            return FcSingleModelPrediction(quantile=(lower, upper))
        else:
            assert prediction.shape[1] == 1
            return FcSingleModelPrediction(point=prediction)

    def _create_predictions_point(self, datasets, evaluator, evaluate_mode='val'):
        predictions = {}
        metrics = {}
        def get_pred_split(dataset: ChronoSplittedTsDataset):
            if evaluate_mode == 'val':
                return torch.cat((dataset.X_train[-self._seq_len:], dataset.X_calib), dim=0),\
                    torch.cat((dataset.Y_train[-self._seq_len:], dataset.Y_calib), dim=0),
            elif evaluate_mode == 'train':
                return dataset.X_train, dataset.Y_train
            elif evaluate_mode == 'test':
                return torch.cat((dataset.X_calib[-self._seq_len:], dataset.X_test), dim=0), \
                    torch.cat((dataset.Y_calib[-self._seq_len:], dataset.Y_calib), dim=0),
            elif evaluate_mode == 'all':
                return dataset.X_full, dataset.Y_full
            else:
                raise ValueError("Invalid mode!")
        for dataset in datasets:
            iterator = ManyToOneIterator([dataset], seq_len=self._seq_len, split_func=get_pred_split)
            dataloader = DataLoader(CompleteDataset(iterator), batch_size=self._batch_size, shuffle=False, drop_last=False)
            model_out, metric_vals = evaluator.eval(dataloader, self._lstm_model, eval_text=f"mode {evaluate_mode} - TS {dataset.ts_id}")
            if self._use_mc_dropout:
                predictions[dataset.ts_id] = torch.concat([o['mc_y_hat'] for o in model_out], dim=0).squeeze(1).to(device='cpu')
            elif self._use_with_mdn:
                pi = torch.concat([o['mdn_coef'].pi for o in model_out], dim=0)
                mu = torch.concat([o['mdn_coef'].mu for o in model_out], dim=0)
                sigma = torch.concat([o['mdn_coef'].sigma for o in model_out], dim=0)
                samples = self._lstm_model.sample_from_mdn_coef(MNDCoef(pi=pi, mu=mu, sigma=sigma))
                predictions[dataset.ts_id] = samples.squeeze(2).to(device='cpu')
            else:
                predictions[dataset.ts_id] = torch.concat([o['y_hat'] for o in model_out], dim=0).squeeze(1)
            metrics[dataset.ts_id] = metric_vals
        return predictions, metrics

    def _get_dataloader(self, datasets, is_val=False):
        def get_split(dataset: ChronoSplittedTsDataset):
            if self._train_with_calib:
                split_point = int((dataset.no_train_steps + dataset.no_calib_steps) * self._train_split)
                if is_val:
                    return dataset.X_full[split_point:dataset.test_step], dataset.Y_full[split_point:dataset.test_step]
                else:
                    return dataset.X_full[:split_point], dataset.Y_full[:split_point]
            else:
                split_point = int(dataset.no_train_steps * self._train_split)
                if is_val:
                    return dataset.X_train[split_point:], dataset.Y_train[split_point:]
                else:
                    return dataset.X_train[:split_point], dataset.Y_train[:split_point]
        iterator = ManyToOneIterator(datasets, seq_len=self._seq_len, split_func=get_split)
        return DataLoader(CompleteDataset(iterator), batch_size=self._batch_size, shuffle=True)

    @property
    def can_handle_different_alpha(self):
        return True

    @property
    def train_per_time_series(self):
        return False