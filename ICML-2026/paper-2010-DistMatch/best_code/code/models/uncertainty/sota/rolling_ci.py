from typing import Optional, List

from models.forcast.forcast_base import PredictionOutputType
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PICalibData, PICalibArtifacts


class RollingCI(PIModel):
    """
    Ported from RollingCI (Feldman et al. 2022)
    """

    def __init__(self, **kwargs):
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,),
                         ts_ids=kwargs["ts_ids"])

    def _calibrate(self, calib_data: [PICalibData], alphas: List[float], **kwargs) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        return calib_artifact

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs):
        pass

    def model_ready(self):
        return True
