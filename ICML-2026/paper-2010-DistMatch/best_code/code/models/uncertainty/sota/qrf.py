from typing import Optional, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import (
    PIModel,
    PIModelPrediction,
    PIPredictionStepData,
    PICalibData,
    PICalibArtifacts,
)
from sklearn_quantile import RandomForestQuantileRegressor

from utils.calc_np import calc_residuals


class QRF(PIModel):
    def __init__(self, **kwargs):
        super(QRF, self).__init__(
            use_dedicated_calibration=True,
            fc_prediction_out_modes=(PredictionOutputType.POINT,),
        )
        self._qrf: RandomForestQuantileRegressor = None
        self._past_window_len = kwargs.get("past_window_len", 100)
        self._beta_calc_bins = kwargs.get("beta_calc_bins", 5)

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
        return self._train_qrf(
            ts_id=calib_data.ts_id,
            X_past=calib_data.X_pre_calib,
            Y_past=calib_data.Y_pre_calib,
            X_reg_train=calib_data.X_calib,
            Y_reg_train=calib_data.Y_calib,
            step_offset=calib_data.step_offset,
            alpha=alpha,
        )

    def _get_quantiles(self, alpha):
        high_quantiles = np.linspace(start=0, stop=alpha, num=self._beta_calc_bins)
        return np.concatenate([high_quantiles, 1 - alpha + high_quantiles])

    def _train_qrf(
        self,
        ts_id,
        X_past,
        Y_past,
        X_reg_train,
        Y_reg_train,
        step_offset,
        alpha,
    ):
        calib_artifacts = PICalibArtifacts()

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

        eps_reg_train = calc_residuals(
            y_hat=Y_hat.squeeze(), y=Y_reg_train.numpy().squeeze()
        )[:, None]
        calib_artifacts.fc_Y_hat = Y_hat
        calib_artifacts.eps = eps_reg_train
        self._calib_eps_last = eps_reg_train
        # Split Calib Residuals in Moving windows of stepwise y predictions

        eps_reg_train = eps_reg_train.squeeze()
        X_reg = sliding_window_view(eps_reg_train, window_shape=self._past_window_len)
        X_reg = X_reg[:-1, ..., None]
        Y_reg = eps_reg_train[self._past_window_len :]

        self._qrf = RandomForestQuantileRegressor(
            n_estimators=10,
            max_depth=2,
            criterion="squared_error",
            q=self._get_quantiles(alpha),
        )

        self._qrf.fit(X_reg.squeeze(-1), Y_reg)

        return calib_artifacts

    def _predict_step(
        self, pred_data: PIPredictionStepData, **kwargs
    ) -> PIModelPrediction:
        # Retrieve data
        alpha, X_step, X_past, Y_past, eps_past = (
            pred_data.alpha,
            pred_data.X_step,
            pred_data.X_past,
            pred_data.Y_past,
            pred_data.eps_past[-self._past_window_len :],
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

        eps_reg = np.concatenate(
            [
                self._calib_eps_last,
                np.array(eps_past).reshape(-1, 1),
            ]
        )[None, ...]
        x_test = eps_reg[:, -self._past_window_len :, ...]

        quantiles = self._qrf.predict(x_test.squeeze(-1))
        quantiles = self._choose_tighthest_quantiles(quantiles)

        q_low = quantiles[0]
        q_high = quantiles[1]

        pred_int = Y_hat + q_low, Y_hat + q_high
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)

    def _choose_tighthest_quantiles(self, quantiles: np.ndarray):
        n_split = len(quantiles) // 2
        low_quantiles, high_quantiles = quantiles[:n_split], quantiles[n_split:]
        width = (high_quantiles - low_quantiles).flatten()
        min_width_id = np.argmin(width)
        return np.array([low_quantiles[min_width_id], high_quantiles[min_width_id]])

    def model_ready(self):
        return True

    def can_handle_different_alpha(self):
        return True
