from typing import Tuple, Optional, List

import pandas as pd
import torch
import wandb
import re

from models.forcast.forcast_base import PredictionOutputType
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts


class PrecomputedCMAL(PIModel):

    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._target_variable = kwargs['target_variable']
        self._result_dict = pd.read_pickle(kwargs['result_file_path'])
        reg_res = re.search("seed_(\d+)_", kwargs['result_file_path'])
        if reg_res is None:
            reg_res = re.search("seed(\d+)_", kwargs['result_file_path'])
        seed = reg_res.group(1)
        current_conf = wandb.config["experiment_data"]
        current_conf["seed"] = seed
        wandb.config.update({"experiment_data": current_conf}, allow_val_change=True)
        self._y_norm_param = kwargs["y_mean"], kwargs["y_std"]
        self._current_basin = None
        self._current_quantiles = None

    def calibrate(self, calib_data: [PICalibData], alphas: List[float], **kwargs) -> [PICalibArtifacts]:
        return None

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        self._current_basin = calib_data.ts_id.split('_')[0]
        assert self._current_basin in self._result_dict
        self._current_quantiles = None
        return calib_artifact

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._current_quantiles = self._result_dict[self._current_basin]['1D']['xr'][self._target_variable]
        self._current_quantiles = self._current_quantiles.quantile([kwargs['alpha'] / 2, 0.5, 1 - (kwargs['alpha'] / 2)], dim='samples')
        self._current_quantiles = (self._current_quantiles - self._y_norm_param[0]) / self._y_norm_param[1]

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        assert self._current_basin == pred_data.ts_id.split('_')[0]
        low = torch.tensor(self._current_quantiles[0, pred_data.step_offset_prediction, 0].values)
        mean = torch.tensor(self._current_quantiles[1, pred_data.step_offset_prediction, 0].values)
        high = torch.tensor(self._current_quantiles[2, pred_data.step_offset_prediction, 0].values)
        prediction_result = PIModelPrediction(pred_interval=(low, high), fc_Y_hat=mean)
        return prediction_result

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return self._result_dict is not None
