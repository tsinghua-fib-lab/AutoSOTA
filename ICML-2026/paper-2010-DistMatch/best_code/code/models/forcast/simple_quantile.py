from typing import Dict, Optional

from sklearn.linear_model import QuantileRegressor

from models.forcast.forcast_base import ForcastMode, FCPredictionData, PredictionOutputType, ForcastModel, \
    FcSingleModelPrediction, FCModelPrediction


class LinearQuantileRegForcast(ForcastModel):
    """
    Model which uses linear Quantile Regression
    """
    def __init__(self, **kwargs):
        super().__init__(forcast_mode=ForcastMode.PREDICT_INDEPENDENT, supported_outputs=(PredictionOutputType.QUANTILE,))
        self.model_high = None
        self.model_low = None

    def _train(self, X, Y, precalc_fc_steps=None, *args, **kwargs) -> Optional[FCModelPrediction]:
        alpha = kwargs['alpha']
        self.model_low = QuantileRegressor(quantile=alpha/2)
        self.model_high = QuantileRegressor(quantile=1 - alpha/2)
        self.model_low.fit(X, Y)
        self.model_high.fit(X, Y)
        if precalc_fc_steps is not None:
            raise ValueError("Model can not predict without prediction data!")
        return None

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> FCModelPrediction:
        low_quantile = self.model_low.predict(pred_data.X_step)
        high_quantile = self.model_high.predict(pred_data.X_step)
        return FcSingleModelPrediction(quantile=(low_quantile, high_quantile))

    def _check_pred_data(self, pred_data: FCPredictionData):
        assert pred_data.X_step is not None
        assert pred_data.alpha is not None

    def can_handle_different_alpha(self):
        return False

    @property
    def train_per_time_series(self):
        return True

    @property
    def uses_past_for_prediction(self):
        return False


class QuantileForestRegForcast(ForcastModel):
    """
    Model which uses a quantile forest
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(forcast_mode=ForcastMode.PREDICT_INDEPENDENT, supported_outputs=(PredictionOutputType.QUANTILE,))
        self.model = None

    def _train(self, X, Y, *args, **kwargs):
        pass

    def _predict(self, pred_data: FCPredictionData, *args, **kwargs) -> Dict:
        pass

    def can_handle_different_alpha(self):
        return False

    @property
    def train_per_time_series(self):
        return True

    @property
    def uses_past_for_prediction(self):
        return False
