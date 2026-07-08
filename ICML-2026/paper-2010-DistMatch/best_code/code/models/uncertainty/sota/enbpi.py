import math
from typing import Tuple, Dict, Optional, List

import numpy as np
# from doubt import QuantileRegressionForest
from numpy.lib.stride_tricks import sliding_window_view
from sklearn_quantile import RandomForestQuantileRegressor


from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import PIModel, PIModelPrediction, \
    PIPredictionStepData, PICalibData, PICalibArtifacts
from utils.calc_np import calc_residuals


class EnbPIModel(PIModel):
    """
    Ported from EnbPI (Xu et al. 2022)
    !But without the LOO Bootstraping/Calbiration splitting! (though could be done via correct calib/eval calls)
    """
    def __init__(self, **kwargs):
        super().__init__(use_dedicated_calibration=False, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._adaptive_sigma = kwargs['adaptive_sigma']
        self._past_window_len = kwargs['past_window_len']
        self._beta_calc_bins = kwargs.get('beta_calc_bins', 20)
        self._use_adaptiveci = kwargs.get('use_adaptiveci', False)
        if self._use_adaptiveci:
            self._gamma = kwargs['gamma']
        self._alpha_t = None


    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._alpha_t = kwargs['alpha']  # Reset

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        # Retrieve data
        _, X_step, X_past, Y_past, eps_past = pred_data.alpha, pred_data.X_step, pred_data.X_past, \
                                                  pred_data.Y_past, pred_data.eps_past[-self._past_window_len:]
        if self._adaptive_sigma:
            raise NotImplemented("Not implemented yet!")
        else:
             curr_sigma = 1
        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=pred_data.step_offset_overall)).point
        # Get current beta (and sigma)
        beta = self._get_beta_bins(eps_past, self._alpha_t)
        width_low = curr_sigma * np.percentile(eps_past, math.ceil(100 * beta))
        width_high = curr_sigma * np.percentile(eps_past, math.ceil(100 * (1 - self._alpha_t + beta)))
        pred_int = Y_hat + width_low, Y_hat + width_high
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)

    def _get_beta_bins(self, eps, alpha):
        bins = self._beta_calc_bins
        beta_ls = np.linspace(start=0, stop=alpha, num=bins)
        width = np.zeros(bins)
        for i in range(bins):
            width[i] = np.percentile(eps, math.ceil(100 * (1 - alpha + beta_ls[i]))) - \
                       np.percentile(eps, math.ceil(100 * beta_ls[i]))
        i_star = np.argmin(width)
        return beta_ls[i_star]

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        # If Adaptive:
        if self._use_adaptiveci:
            alpha = pred_data.alpha
            pred_int = pred_result.pred_interval
            err_step = 0 if pred_int[0] <= Y_step <= pred_int[1] else 1
            # Simple Mode
            self._alpha_t = self._alpha_t + self._gamma * (alpha - err_step)
            self._alpha_t = max(0, min(1, self._alpha_t))  # Make sure it is between 0 and 1

    def model_ready(self):
        return True

    def required_past_len(self) -> Tuple[int, int]:
        fc_required_len = super().required_past_len()
        return max(fc_required_len[0], self._past_window_len), max(fc_required_len[1], self._past_window_len)

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None
        assert pred_data.eps_past is not None

    @property
    def can_handle_different_alpha(self):
        return True


class SPICModel(PIModel):
    """
    Ported from SPIC (Xu et al. 2022)
    """
    def __init__(self, **kwargs):
        self._retrain_regressor = kwargs['retrain_regressor']
        super().__init__(use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        self._past_window_len = kwargs['past_window_len']
        self._adaptive_sigma = kwargs['adaptive_sigma']
        self._use_xt_for_regressor = kwargs['use_xt']
        self._beta_calc_bins = kwargs.get('beta_calc_bins', 5)
        self._regressor_param = kwargs.get('regressor_param', dict())
        self._use_adaptiveci = kwargs.get('use_adaptiveci', False)
        if self._use_adaptiveci:
            self._gamma = kwargs['gamma']
        self._alpha_t = None
        self.model = None

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        return self._train_eps_regressor(ts_id=calib_data.ts_id, X_past=calib_data.X_pre_calib, Y_past=calib_data.Y_pre_calib,
                                         X_reg_train=calib_data.X_calib, Y_reg_train=calib_data.Y_calib,
                                         alpha=alpha, step_offset=calib_data.step_offset)

    def _get_beta_quantiles(self, alpha):
        bins = self._beta_calc_bins
        high_quantiles = np.linspace(
            start=0, stop=alpha, num=bins
        )
        return np.concatenate([high_quantiles, 1 - alpha + high_quantiles])

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._alpha_t = kwargs['alpha']  # Reset
        self._train_eps_regressor(None, None, None, None, None, self._alpha_t, None, eps_reg_train=self._calib_eps)

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        # Retrieve data
        alpha, X_step, X_past, Y_past, eps_past = pred_data.alpha, pred_data.X_step, pred_data.X_past, \
                                                  pred_data.Y_past, pred_data.eps_past[-self._past_window_len:]

        if self._retrain_regressor:
            self._train_eps_regressor(None, None, None, None, None, alpha, None,
                                      eps_reg_train=np.concatenate((self._calib_eps, eps_past.numpy()))[-len(self._calib_eps):, :])

        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=pred_data.step_offset_overall)).point
        if self._use_xt_for_regressor:
            X_reg = np.concatenate((np.array(eps_past).reshape(1, -1), X_past[-self._past_window_len:].reshape(1, -1)), axis=1)
        else:
            X_reg = np.array(eps_past).reshape(1, -1)
        #beta = self._get_beta_bins(X_reg, alpha)
        #print(f"Beta: {beta}")
        X_reg = np.nan_to_num(X_reg, nan=0, neginf=0, posinf=0)
        widths = self.model.predict(X_reg)
        width_low = widths[0]
        width_high = widths[1]
        #width_low = self._curr_SigmaX * self.model.predict(X_reg, beta)[1][0][0]
        #width_high = self._curr_SigmaX * self.model.predict(X_reg, (1 - alpha + beta))[1][0][1]
        pred_int = Y_hat + width_low, Y_hat + width_high
        return PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)

    def _train_eps_regressor(self, ts_id, X_past, Y_past, X_reg_train, Y_reg_train, alpha, step_offset, eps_reg_train=None):
        self.model = RandomForestQuantileRegressor(**self._regressor_param, q=[alpha, 1 - alpha])
        calib_artifacts = PICalibArtifacts()
        # Calc epsilons if not given
        if eps_reg_train is None:
            Y_hat = self._forcast_service.predict(
                FCPredictionData(ts_id=ts_id, X_past=X_past, Y_past=Y_past, X_step=X_reg_train,
                                 step_offset=step_offset), retrieve_tensor=False).point
            eps_reg_train = calc_residuals(y_hat=Y_hat, y=Y_reg_train.numpy())
            calib_artifacts.fc_Y_hat = Y_hat
            calib_artifacts.eps = eps_reg_train
            self._calib_eps = eps_reg_train
        # Split Calib Residuals in Moving windows of stepwise y predictions
        eps_reg_train = eps_reg_train.squeeze()
        eps_windowed_x = sliding_window_view(eps_reg_train, window_shape=self._past_window_len)
        eps_windowed_y = eps_reg_train[self._past_window_len:]
        # Train Regressor(s)
        if self._use_xt_for_regressor:
            X_windowed = sliding_window_view(X_reg_train, window_shape=self._past_window_len, axis=0)
            X_windowed = X_windowed.reshape(X_windowed.shape[0], -1)
            X_reg = np.concatenate((eps_windowed_x, X_windowed), axis=1)
        else:
            X_reg = eps_windowed_x
        self.model.fit(X_reg[:-1], eps_windowed_y)
        if self._adaptive_sigma:
            raise NotImplemented("Not implemented yet!")
        else:
            self._curr_SigmaX = 1
        return calib_artifacts

    def _get_beta_bins(self, X_reg, alpha):
        bins = self._beta_calc_bins
        beta_ls = np.linspace(start=0, stop=alpha, num=bins)
        width = np.zeros(bins)
        for i in range(bins):
            width[i] = self.model.predict(X_reg, (1 - alpha + beta_ls[i]))[1][0][0] \
                       - self.model.predict(X_reg, beta_ls[i])[1][0][1]
        i_star = np.argmin(width)
        return beta_ls[i_star]

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        # If Adaptive:
        if self._use_adaptiveci:
            alpha = pred_data.alpha
            pred_int = pred_result.pred_interval
            err_step = 0 if pred_int[0] <= Y_step <= pred_int[1] else 1
            # Simple Mode
            self._alpha_t = self._alpha_t + self._gamma * (alpha - err_step)
            self._alpha_t = max(0, min(1, self._alpha_t))  # Make sure it is between 0 and 1

    def model_ready(self):
        return self.model is not None

    def required_past_len(self) -> Tuple[int, int]:
        fc_required_len = super().required_past_len()
        return max(fc_required_len[0], self._past_window_len), max(fc_required_len[1], self._past_window_len)

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None
        assert pred_data.eps_past is not None

    @property
    def can_handle_different_alpha(self):
        return True