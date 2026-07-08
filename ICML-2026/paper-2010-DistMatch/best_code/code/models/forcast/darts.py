from abc import ABC
import logging
from typing import Tuple, Optional, Union

import numpy as np
from darts import TimeSeries
from darts.models import (
    VARIMA,
    Prophet,
    LightGBMModel,
    RandomForest,
    LinearRegressionModel,
    TCNModel,
)
from darts.utils.likelihood_models import QuantileRegression

from models.forcast.forcast_base import (
    ForcastModel,
    FCPredictionData,
    FCModelPrediction,
    PredictionOutputType,
    ForcastMode,
    FcSingleModelPrediction,
)
from darts.models import RNNModel

_MODEL = {
    "prophet": Prophet,
    "varima": VARIMA,
    "lightgbm": LightGBMModel,
    "forest": RandomForest,
    "linear": LinearRegressionModel,
    "tcn": TCNModel,
    "rnn": RNNModel,
}
LOGGER = logging.getLogger(__name__)


class DartsModel(ForcastModel, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _predict(
        self, pred_data: FCPredictionData, *args, **kwargs
    ) -> FCModelPrediction:
        raise NotImplemented("Not possible for this model!")

    def can_handle_different_alpha(self):
        return True

    @property
    def train_per_time_series(self):
        return True

    def _get_constructor_parameters(self):
        return {"model": f"darts-{self.model_id}", "model_params": self._model_param}


class QuantileDartsModel(DartsModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            forcast_mode=ForcastMode.ALL_ON_TRAIN,
            supported_outputs=(PredictionOutputType.QUANTILE,),
        )
        self.model_id = kwargs["model"].replace("darts-", "")
        self._model_param = kwargs.get("model_params", {})
        self._inference_alphas = list(kwargs["alpha"])
        quantiles = None  # self._inference_alphas + [0.5]
        if "likelihood" in self._model_param:
            self.model = _MODEL[self.model_id](quantiles=quantiles, **self._model_param)
        else:
            self.model = _MODEL[self.model_id](
                likelihood=QuantileRegression(quantiles=quantiles), **self._model_param
            )
        assert self.model_id in _MODEL

    def _train(
        self, X, Y, precalc_fc_steps=None, *args, **kwargs
    ) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        Y_dart = TimeSeries.from_values(Y.numpy())
        self.model.fit(
            Y_dart, past_covariates=TimeSeries.from_values(kwargs["X_full"].numpy())
        )
        if precalc_fc_steps is not None:
            prediction = self.model.predict(precalc_fc_steps, num_samples=100)
            prediction_func = lambda alpha: FcSingleModelPrediction(
                quantile=(
                    np.array(prediction.quantile_timeseries(alpha).values()),
                    np.array(prediction.quantile_timeseries(1 - alpha).values()),
                )
            )
            return prediction_func, None
        return None


class SimpleDartsModel(DartsModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            forcast_mode=ForcastMode.ALL_ON_TRAIN,
            supported_outputs=(PredictionOutputType.POINT,),
        )
        self.model_id = kwargs["model"].replace("darts-", "")
        self._model_param = kwargs.get("model_params", {})
        self.model = _MODEL[self.model_id](**self._model_param)
        assert self.model_id in _MODEL

    def _train(
        self, X, Y, precalc_fc_steps=None, *args, **kwargs
    ) -> Optional[Tuple[FCModelPrediction, Optional[int]]]:
        Y_dart = TimeSeries.from_values(Y.numpy())
        past_covariates = TimeSeries.from_values(kwargs["X_full"].numpy())

        LOGGER.info("Darts Model fitting ...")
        self.model.fit(Y_dart, past_covariates=past_covariates)
        if precalc_fc_steps is not None:
            LOGGER.info("Darts Model predicting ...")
            prediction = self.model.predict(precalc_fc_steps)
            return FcSingleModelPrediction(point=np.array(prediction.values())), None
        return None
