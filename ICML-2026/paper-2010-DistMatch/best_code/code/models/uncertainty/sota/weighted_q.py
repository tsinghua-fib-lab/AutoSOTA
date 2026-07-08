from pathlib import Path
from typing import Optional, List, Tuple

from torch import nn
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import (
    PIModel,
    PIModelPrediction,
    PIPredictionStepData,
    PICalibData,
    PICalibArtifacts,
)

from utils.calc_np import calc_residuals, calc_default_conformal_Q


class WeightedQ(PIModel):
    def __init__(self, **kwargs):
        super(WeightedQ, self).__init__(
            use_dedicated_calibration=True,
            fc_prediction_out_modes=(PredictionOutputType.POINT,),
        )
        self._model: LogisticRegression = None
        self._nonconformity_score: float = None
        self._calib_X_past = None

    def _calibrate(
        self, calib_data: [PICalibData], alphas, **kwargs
    ) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(
        self,
        calib_data: PICalibData,
        alpha,
        calib_artifact: Optional[PICalibArtifacts],
        mix_calib_data: Optional[List[PICalibData]],
        mix_calib_artifact: Optional[List[PICalibArtifacts]],
    ) -> PICalibArtifacts:
        return self._train_regressor(
            ts_id=calib_data.ts_id,
            X_past=calib_data.X_pre_calib,
            Y_past=calib_data.Y_pre_calib,
            X_reg_train=calib_data.X_calib,
            Y_reg_train=calib_data.Y_calib,
            step_offset=calib_data.step_offset,
            alpha=alpha,
        )

    def _train_regressor(
        self,
        ts_id,
        X_past,
        Y_past,
        X_reg_train,
        Y_reg_train,
        step_offset,
        alpha,
    ):
        Y_hat = self._forcast_service.predict(
            FCPredictionData(
                ts_id=ts_id,
                X_past=X_past,
                Y_past=Y_past,
                X_step=X_reg_train,
                step_offset=step_offset,
            ),
            retrieve_tensor=False,
        ).point

        residuals = calc_residuals(Y_hat, Y_reg_train)
        residuals = residuals.detach().cpu().numpy()
        self._nonconformity_score = (residuals / residuals.std()).mean()

        self._calib_X_past = X_reg_train

        X = np.concatenate((X_reg_train, X_past), axis=0)
        Y = np.zeros(len(X))
        Y[len(X_reg_train) :] = 1

        self._model = LogisticRegression()
        self._model.fit(X, Y)

        fc_interval = self.__predict(alpha, Y_hat, X_reg_train)

        return PICalibArtifacts(fc_Y_hat=Y_hat, fc_interval=fc_interval)

    def _predict_step(
        self, pred_data: PIPredictionStepData, **kwargs
    ) -> PIModelPrediction:
        # Retrieve data
        alpha, X_step, X_past, Y_past = (
            pred_data.alpha,
            pred_data.X_step,
            pred_data.X_past,
            pred_data.Y_past,
        )
        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(
                ts_id=pred_data.ts_id,
                X_past=X_past,
                Y_past=Y_past,
                X_step=X_step,
                step_offset=pred_data.step_offset_overall,
            )
        ).point

        self._calib_X_past = np.concatenate([self._calib_X_past, X_step])

        pred_int = self.__predict(alpha, Y_hat, self._calib_X_past)

        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)

    def __predict(self, alpha, Y_hat, X_step) -> Tuple[np.ndarray, np.ndarray]:
        probs_test = self._model.predict_proba(X_step)[:, 1]
        likelihood_ratios = probs_test / (1 - probs_test)
        eps = likelihood_ratios * self._nonconformity_score
        quantile = calc_default_conformal_Q(eps, alpha)

        pred_low = Y_hat - quantile
        pred_high = Y_hat + quantile

        return pred_low, pred_high

    def model_ready(self):
        return True

    def can_handle_different_alpha(self):
        return True
