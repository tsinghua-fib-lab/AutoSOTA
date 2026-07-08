from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

from models.forcast.forcast_base import ForcastMode, ForcastModel, PredictionOutputType, FCModelPrediction, \
    FCPredictionData, FcSingleModelPrediction


class DummyForcast(ForcastModel):
    """
    Always return 0
    """
    def __init__(self, **kwargs):
        super().__init__(forcast_mode=ForcastMode.ALL_ON_TRAIN, supported_outputs=(PredictionOutputType.POINT,))
        self.value = 0

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        if precalc_fc_steps is not None:
            return FcSingleModelPrediction(point=np.repeat(self.value, precalc_fc_steps)[..., None]), None
        return None

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        return FcSingleModelPrediction(point=np.repeat(self.value, pred_data.no_fc_steps))

    def can_handle_different_alpha(self):
        return True

    @property
    def train_per_time_series(self):
        return True

    @property
    def uses_past_for_prediction(self):
        return False


class AverageForcast(ForcastModel):
    """
    Model which only predicts the average value of the training as constant forcast
    """
    def __init__(self, **kwargs):
        super().__init__(forcast_mode=ForcastMode.ALL_ON_TRAIN, supported_outputs=(PredictionOutputType.POINT,))
        self.avg_value = None

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        self.avg_value = Y.mean(0)
        if precalc_fc_steps is not None:
            return FcSingleModelPrediction(point=np.repeat(self.avg_value, precalc_fc_steps)[..., None]), None
        return None

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        return FcSingleModelPrediction(point=np.repeat(self.avg_value, pred_data.no_fc_steps))

    def can_handle_different_alpha(self):
        return True

    @property
    def train_per_time_series(self):
        return True

    @property
    def uses_past_for_prediction(self):
        return False


class ForestRegForcast(ForcastModel):
    """
    Model which uses Regression Forest (Should replicate the Xu et al. Implementation)
    Not really a forcast but simulation setting: y_t = f(x_t)
    """
    def __init__(self, **kwargs):
        super().__init__(forcast_mode=ForcastMode.ALL_ON_TRAIN, supported_outputs=(PredictionOutputType.POINT,))
        # Settings/Parameters copied from the Xu et al. paper implementation
        self.model = RandomForestRegressor(n_estimators=10, criterion='mse', bootstrap=False, n_jobs=-1)

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        self.model.fit(X, Y)
        if precalc_fc_steps is not None and kwargs['X_full'][X.shape[0]:].shape[0] < precalc_fc_steps:
            raise ValueError("Model can not predict without prediction data!")
        X_full = kwargs['X_full']
        Y_hat_full = self.model.predict(X_full)
        return FcSingleModelPrediction(point=Y_hat_full[..., None]), 0

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        raise NotImplemented("Not possible for this model!")
        #Y_hat = self.model.predict(pred_data.X_step)
        #return FcSingleModelPrediction(point=Y_hat[..., None])

    def _check_pred_data(self, pred_data: FCPredictionData):
        assert pred_data.X_step is not None

    def can_handle_different_alpha(self):
        return False

    @property
    def train_per_time_series(self):
        return True

    @property
    def uses_past_for_prediction(self):
        return False


class RidgeRegForcast(ForcastModel):
    """
    Model which uses Ridge Regression (Should replicate the Xu et al. Implementation)
    Not really a forcast but simulation setting: y_t = f(x_t)
    """
    def __init__(self, **kwargs):
        super().__init__(forcast_mode=ForcastMode.ALL_ON_TRAIN, supported_outputs=(PredictionOutputType.POINT,))
        min_alpha = 0.0001
        max_alpha = 10
        self.model = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        self.model.fit(X, Y)
        if precalc_fc_steps is not None and kwargs['X_full'][X.shape[0]:].shape[0] < precalc_fc_steps:
            raise ValueError("Model can not predict without prediction data!")
        X_full = kwargs['X_full']
        Y_hat_full = self.model.predict(X_full)
        Y_hat_full = Y_hat_full[..., None]
        return FcSingleModelPrediction(point=Y_hat_full), 0

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        raise NotImplemented("Not possible for this model!")
        #Y_hat = self.model.predict(pred_data.X_step)
        #return FcSingleModelPrediction(point=Y_hat[..., None])

    def _check_pred_data(self, pred_data: FCPredictionData):
        assert pred_data.X_step is not None

    def can_handle_different_alpha(self):
        return False

    @property
    def train_per_time_series(self):
        return True

    @property
    def uses_past_for_prediction(self):
        return False

