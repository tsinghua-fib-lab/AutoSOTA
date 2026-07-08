from typing import Optional, List

import numpy as np
import torch

from models.forcast.forcast_base import FCPredictionData, PredictionOutputType
from models.uncertainty.components.eps_memory import FiFoMemory
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PICalibData, PICalibArtifacts, PIModelPrediction
from utils.calc_torch import calc_residuals


class NexCP(PIModel):
    """
    Ported from NexCP (Barber et al. 2022)
    """
    def __init__(self, **kwargs):
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self.mode = kwargs['mode']
        assert self.mode in ['CP-LS', "nexCP-LS", "nexCP-WSL"]
        self._rho = kwargs['rho']
        # self._roh_LS = kwargs['rho_ls']
        self._last_k = kwargs['max_past']
        self._memory = FiFoMemory(self._last_k, store_step_no=False)
        self._use_adaptiveci = kwargs.get('use_adaptiveci', False)
        if self._use_adaptiveci:
            self._gamma = kwargs['gamma']
        self._alpha_t = None

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        Y_hat = self._forcast_service.predict(
            FCPredictionData(ts_id=calib_data.ts_id, X_past=calib_data.X_pre_calib, Y_past=calib_data.Y_pre_calib,
                             X_step=calib_data.X_calib, step_offset=calib_data.step_offset)).point
        eps = torch.abs(calc_residuals(Y=calib_data.Y_calib, Y_hat=Y_hat))
        self._memory.add_transient(torch.empty_like(eps), eps)
        return calib_artifact

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._alpha_t = kwargs['alpha']  # Reset

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs):
        _, X_step, X_past, Y_past, _, step_abs =\
            pred_data.alpha, pred_data.X_step, pred_data.X_past, pred_data.Y_past, pred_data.eps_past,\
            pred_data.step_offset_overall
        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=step_abs)).point
        # Get Quantile
        past_eps = torch.abs(self._memory.eps_chronological).squeeze(-1).cpu().numpy()
        weights = self._get_weights(min(self._last_k, len(past_eps)))
        sort_idx = np.argsort(past_eps)
        try:
            quantile_idx = np.min(np.where(np.cumsum(weights[sort_idx]) >= 1 - self._alpha_t))
            quantile_value = np.sort(past_eps)[quantile_idx]
        except:
            quantile_value = np.sort(past_eps)[-1]

        # Calc Interval
        q_high = Y_hat + quantile_value
        q_low = Y_hat - quantile_value
        prediction_result = PIModelPrediction(pred_interval=(q_low, q_high), fc_Y_hat=Y_hat)
        return prediction_result

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        # Update memory
        eps = calc_residuals(Y=Y_step, Y_hat=pred_result.fc_Y_hat)
        self._memory.add_transient(cxt=torch.empty_like(eps), eps=eps)
        # If Adaptive:
        if self._use_adaptiveci:
            alpha = pred_data.alpha
            pred_int = pred_result.pred_interval
            err_step = 0 if pred_int[0] <= Y_step <= pred_int[1] else 1
            # Simple Mode
            self._alpha_t = self._alpha_t + self._gamma * (alpha - err_step)
            self._alpha_t = max(0, min(1, self._alpha_t))  # Make sure it is between 0 and 1

    def _get_weights(self, length):
        if self.mode in ('nexCP-LS', 'nexCP-WSL'):
            weights = self._rho ** (np.arange(length - 1, 0, -1))
            weights = np.r_[weights, 1]
            weights = weights / np.sum(weights)  # Sum Weight = 1
        else:
            weights = np.ones(length + 1)
        return weights

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return True
