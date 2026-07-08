import sys
from abc import abstractmethod
from typing import Optional, Tuple, List

import torch
from torch import nn

from models.base_model import BaseModel
from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.components.eps_ctx_encode import FcModel
from models.uncertainty.ml_base import CalibTrainerMixin, BATCH_MODE_ONE_TS
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts
from models.uncertainty.score_service import score
from utils.calc_torch import unfold_window


class EpsPredCtxLess(BaseModel, PIModel, CalibTrainerMixin):

    def __init__(self, **kwargs):
        BaseModel.__init__(self)
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        CalibTrainerMixin.__init__(self, batch_mode=BATCH_MODE_ONE_TS,
                                   with_loss_weight=kwargs['with_loss_weight'],
                                   coverage_loss_weight=kwargs['coverage_loss_weight'])
        self._past_window = kwargs['past_window']
        self._eps_buffer = []

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        Y_hat = []
        fc_state_step = []
        calib_artifacts = []
        for c_data in calib_data:
            c_result = self._forcast_service.predict(
                FCPredictionData(ts_id=c_data.ts_id, X_past=c_data.X_pre_calib, Y_past=c_data.Y_pre_calib,
                                 X_step=c_data.X_calib, step_offset=c_data.step_offset))
            Y_hat.append(c_result.point)
            fc_state_step.append(c_result.state)
            eps = score.get(Y=c_data.Y_calib, Y_hat=Y_hat[-1])  # [calib_size, *]
            calib_artifacts.append(PICalibArtifacts(fc_Y_hat=Y_hat[-1], eps=eps, fc_state_step=fc_state_step[-1]))

        trainer_config = kwargs['trainer_config']
        experiment_config = kwargs['experiment_config']
        self._train_model(calib_data, Y_hat=Y_hat, fc_state_step=fc_state_step, alphas=alphas,
                          experiment_config=experiment_config, trainer_config=trainer_config)
        return calib_artifacts

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        self._eps_buffer = list(calib_artifact.eps[-self._past_window:])
        return calib_artifact

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        alpha, X_step, X_past, Y_past, eps_past = pred_data.alpha, pred_data.X_step, pred_data.X_past,\
                                                  pred_data.Y_past, pred_data.eps_past

        # Calculate y_hat and prediction interval for current step
        Y_hat = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=pred_data.step_offset_overall)).point
        eps_q_low, eps_q_high, _ = self._forward(past_eps=torch.tensor(self._eps_buffer[-self._past_window:]).unsqueeze(-1).T,
                                                 Y_hat=Y_hat, alpha=alpha)
        pred_int = Y_hat + score.resolve(eps_q_low), Y_hat + score.resolve(eps_q_high)
        prediction_result = PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)
        return prediction_result

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        step_eps = list(score.get(Y_hat=pred_result.fc_Y_hat, Y=Y_step))
        self._eps_buffer.extend(step_eps)

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return len(self._eps_buffer) > 0

    def _get_calib_ctx(self, calib_data, Y_hat) -> Tuple[torch.tensor, int]:
        return torch.zeros((calib_data.X_calib.shape[0], 0)), 0   # Context is not used!

    #
    # Network Module methods
    #

    def forward(self, *args, **kwargs):
        Y, Y_hat, alpha, mask = kwargs['Y'].detach(), kwargs['Y_hat'].detach(), kwargs['alpha'], kwargs['mask'].detach()
        eps = score.get(Y=Y, Y_hat=Y_hat).detach()  # [batch_size, *]
        past_eps = unfold_window(eps, window_len=self._past_window).squeeze(-1)
        eps_q_low, eps_q_high, beta = self._forward(past_eps=past_eps, Y_hat=Y_hat, alpha=alpha)
        q_low = Y_hat + score.resolve(eps_q_low.unsqueeze(1))
        q_high = Y_hat + score.resolve(eps_q_high.unsqueeze(1))
        return dict(q_low=q_low, q_high=q_high, low_alpha=beta, high_alpha=(beta - alpha + 1))

    @abstractmethod
    def _forward(self, past_eps, Y_hat, alpha) -> Tuple[float, float, float]:
        """
        :param past_eps:    [batch_size, window_len]
        :param Y_hat:       [batch_size, 1]
        :param alpha:       float
        :return: eps_q_low, eps_q_high, beta
        """
        pass

    def required_past_len(self) -> Tuple[int, int]:
        return self._past_window, sys.maxsize


class EpsPredCtxLessFC(EpsPredCtxLess):

    def __init__(self, **kwargs):
        EpsPredCtxLess.__init__(self, **kwargs)
        self._hidden = kwargs['hidden_layer']
        assert len(self._hidden) > 0  # One Hidden Layer is mandatory
        self._predict_beta = True
        # Init Network
        self._model = FcModel(input_dim=self._past_window + 2, out_dim=1, hidden=self._hidden)
        if self._predict_beta:
            # Predict beta share of alpha
            self._beta_network = nn.Sequential(FcModel(input_dim=self._past_window + 2, out_dim=1, hidden=()),
                                               nn.Sigmoid())

    def _forward(self, past_eps, Y_hat, alpha):
        alpha = torch.full((), fill_value=alpha) if isinstance(alpha, float) else alpha
        if self._predict_beta:
            beta = alpha * self._beta_network(torch.cat((past_eps, Y_hat, alpha), dim=1))
        else:
            beta = torch.div(alpha, 2)
        eps_q_low = self._model(torch.cat((past_eps, Y_hat, beta), dim=1))
        eps_q_high = self._model(torch.cat((past_eps, Y_hat, (beta - alpha + 1)), dim=1))
        return eps_q_low, eps_q_high, beta

    def _get_constructor_parameters(self) -> dict:
        pass


class EpsPredCtxLessLSTM(EpsPredCtxLess):

    def __init__(self, **kwargs):
        EpsPredCtxLess.__init__(self, **kwargs)
        self._predict_beta = True
        # Init Network
        self._model = nn.LSTM(input_size=self._past_window + 2, hidden_size=20, num_layers=2, dropout=0.1,)
        if self._predict_beta:
            # Predict beta share of alpha
            self._beta_network = nn.Sequential(FcModel(input_dim=self._past_window + 2, out_dim=1, hidden=()),
                                               nn.Sigmoid())

    def _forward(self, past_eps, Y_hat, alpha):
        alpha = torch.full((), fill_value=alpha) if isinstance(alpha, float) else alpha
        if self._predict_beta:
            beta = alpha * self._beta_network(torch.cat((past_eps, Y_hat, alpha), dim=1))
        else:
            beta = torch.div(alpha, 2)
        eps_q_low = self._model(torch.cat((past_eps, Y_hat, beta), dim=1))
        eps_q_high = self._model(torch.cat((past_eps, Y_hat, (beta - alpha + 1)), dim=1))
        return eps_q_low, eps_q_high, beta

    def _get_constructor_parameters(self) -> dict:
        pass
