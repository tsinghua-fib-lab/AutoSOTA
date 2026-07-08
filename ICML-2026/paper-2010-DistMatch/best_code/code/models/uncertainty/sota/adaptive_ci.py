from typing import Optional, List

import numpy as np

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import PIModel, PIModelPrediction, PIPredictionStepData, \
    PICalibData, PICalibArtifacts
from utils.calc_np import calc_cqr_score, calc_cqr_Q


class AdaptiveCI(PIModel):
    """
    Ported from AdaptiveCI (Gibbs, Candes (2021))
    """
    def __init__(self, **kwargs) -> None:
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.QUANTILE,))
        self._gamma = kwargs.get('gamma', 0.005)
        self._mode = kwargs['mode']
        assert self._mode in ('simple', 'momentum')
        self._alpha_t = None
        self._calib_scores = None
        self._past_errors = np.array([], dtype=int)

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        return self._calibrate_scores(
            ts_id=calib_data.ts_id, X_past=calib_data.X_pre_calib, Y_past=calib_data.Y_pre_calib,
            X_calib=calib_data.X_calib, Y_calib=calib_data.Y_calib,
            alpha=alpha, step_offset=calib_data.step_offset
        )

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._alpha_t = kwargs['alpha']  # Reset

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        # Get Prediction Data
        alpha, X_step, X_past, Y_past = pred_data.alpha, pred_data.X_step, pred_data.X_past, pred_data.Y_past
        # Predict and calc conformal interval
        quantile_low, quantile_high = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step, alpha=alpha,
                             step_offset=pred_data.step_offset_overall)).quantile
        conform_big_q = calc_cqr_Q(self._calib_scores, self._alpha_t)
        pred_int = quantile_low - conform_big_q, quantile_high + conform_big_q
        return PIModelPrediction(pred_interval=pred_int, fc_interval=(quantile_low, quantile_high))

    def _calibrate_scores(self, ts_id, X_past, Y_past, X_calib, Y_calib, alpha, step_offset):
        quantiles_low, quantiles_high = self._forcast_service.predict(
            FCPredictionData(ts_id=ts_id, X_past=X_past, Y_past=Y_past, X_step=X_calib, alpha=alpha,
                             step_offset=step_offset), retrieve_tensor=False).quantile
        self._calib_scores = calc_cqr_score(quantiles_low, quantiles_high, Y_calib.numpy())
        return PICalibArtifacts(fc_interval=(quantiles_low, quantiles_high))

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        # Adapt running alpha
        alpha = pred_data.alpha
        pred_int = pred_result.pred_interval
        err_step = 0 if pred_int[0].numpy() <= Y_step.numpy() <= pred_int[1].numpy() else 1
        if self._mode == 'simple':
            self._alpha_t = self._alpha_t + self._gamma * (alpha - err_step)
        elif self._mode == 'momentum':
            self._past_errors = np.append(self._past_errors, [err_step])
            tmp = np.arange(len(self._past_errors), 0, -1)
            weights = 0.95**(tmp)
            weights = weights / np.sum(weights)
            self._alpha_t = self._alpha_t + self._gamma * (alpha - np.sum(weights * self._past_errors))
        self._alpha_t = max(0, min(1, self._alpha_t))  # Make sure it is between 0 and 1

    def model_ready(self):
        return self._calib_scores is not None

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None

    @property
    def can_handle_different_alpha(self):
        return True