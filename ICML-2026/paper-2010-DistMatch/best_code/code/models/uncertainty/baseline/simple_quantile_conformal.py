from typing import Tuple, Optional, List

import numpy as np

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts


class DefaultQuantileConformal(PIModel):

    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.QUANTILE,))
        self.eps_calib = None
        self._use_adaptiveci = kwargs.get('use_adaptiveci', False)
        if self._use_adaptiveci:
            self._gamma = kwargs['gamma']
        self._alpha_t = None

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        quantile_low, quantile_high = self._forcast_service.predict(
            FCPredictionData(ts_id=calib_data.ts_id, X_past=calib_data.X_pre_calib, Y_past=calib_data.Y_pre_calib,
                             X_step=calib_data.X_calib, step_offset=calib_data.step_offset, alpha=alpha),
            retrieve_tensor=False).quantile

        self.eps_calib = np.maximum(quantile_low - calib_data.Y_calib.numpy(), calib_data.Y_calib.numpy() - quantile_high)
        return PICalibArtifacts(fc_interval=(quantile_low, quantile_high))

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._alpha_t = kwargs['alpha']  # Reset

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        alpha, X_step, X_past, Y_past = pred_data.alpha, pred_data.X_step, pred_data.X_past, pred_data.Y_past
        # Calculate y_hat and prediction interval for current step
        quantile_low, quantile_high = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step, alpha=alpha,
                             step_offset=pred_data.step_offset_overall)).quantile

        width = np.quantile(np.abs(self.eps_calib), float((1 - self._alpha_t)))
        pred_int = quantile_low - width, quantile_high + width
        return PIModelPrediction(pred_interval=pred_int, fc_interval=(quantile_low, quantile_high))

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        # If Adaptive:
        if self._use_adaptiveci:
            alpha = pred_data.alpha
            pred_int = pred_result.pred_interval
            err_step = 0 if pred_int[0] <= Y_step <= pred_int[1] else 1
            # Simple Mode
            self._alpha_t = self._alpha_t + self._gamma * (alpha - err_step)
            self._alpha_t = max(0, min(1, self._alpha_t))  # Make sure it is between 0 and 1

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return self.eps_calib is not None

