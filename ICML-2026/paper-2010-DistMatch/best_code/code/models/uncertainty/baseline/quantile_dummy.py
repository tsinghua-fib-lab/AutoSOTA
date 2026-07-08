from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts


class QuantileDummy(PIModel):
    """
    Dummy model which directly uses quantiles from FC model
    """
    def __init__(self, **kwargs) -> None:
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.QUANTILE,))

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        pass

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        # Get Prediction Data
        alpha, X_step, X_past, Y_past = pred_data.alpha, pred_data.X_step, pred_data.X_past, pred_data.Y_past
        # Predict and calc conformal interval
        quantile_low, quantile_high = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step, alpha=alpha,
                             step_offset=pred_data.step_offset_overall)).quantile
        pred_int = quantile_low, quantile_high
        return PIModelPrediction(pred_interval=pred_int, fc_interval=(quantile_low, quantile_high))

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return True
