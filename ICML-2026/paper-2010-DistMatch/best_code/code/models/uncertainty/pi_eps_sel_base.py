import sys
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

import numpy as np
import torch

from models.uncertainty.score_service import score
from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.components.eps_ctx_gen import EpsilonContextGen
from models.uncertainty.components.eps_memory import FiFoMemory
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts


class ConformalSelectionMixin:
    @staticmethod
    def _calc_conformal_quantiles(selected_eps, alpha: float, no_beta_bins: int, use_absolute_eps=False):
        """
        :param selected_eps:        [m, no_of_eps] selected epsilons for m observation
        :param alpha:               targeted alpha
        :param no_beta_bins:        number of bins in which one wants to search for optimal beta
        :return: Tuple[3]:
                 q_conformal_low    [m, 1] low conformal quantile
                 q_conformal_high   [m, 1] high conformal quantile
                 beta               [m, 1] selected beta
        """
        # Default conf style
        # conformal_with = torch.quantile(torch.abs(selected_eps), float((1 - alpha)))
        # pred_int = Y_hat - conformal_with, Y_hat + conformal_with
        # EnbPI Style
        if not use_absolute_eps:
            beta = EpsSelectionPIBase._get_beta_bins(selected_eps, alpha, no_beta_bins) if no_beta_bins > 1 else alpha / 2
            q_conformal_low = torch.quantile(selected_eps, beta, dim=1)
            q_conformal_high = torch.quantile(selected_eps, (1 - alpha + beta), dim=1)
        else:
            beta = alpha / 2
            q_conformal = torch.quantile(torch.abs(selected_eps), 1 - alpha, dim=1)
            q_conformal_low, q_conformal_high = -q_conformal, q_conformal
        return q_conformal_low, q_conformal_high, beta

    @staticmethod
    def _get_beta_bins(eps, alpha: float, no_beta_bins: int):
        """
        :param eps:              [m, no_of_eps] - selected epsilons for m observation
        :param alpha:            targeted alpha
        :param no_beta_bins:     number of bins in which one wants to search for optimal beta
        :return:                 [m, 1] selected beta
        """
        bins = no_beta_bins
        beta_ls = np.linspace(start=0, stop=alpha, num=bins)
        width = torch.zeros((eps.shape[0], bins))
        for i in range(bins):
            width[:, i] = torch.quantile(eps.squeeze(-1), (1 - alpha + beta_ls[i])) - torch.quantile(eps.squeeze(-1),
                                                                                                     beta_ls[i])
        i_stars = torch.argmin(width, dim=1)
        most_frequent_i_star = torch.mode(i_stars)
        return beta_ls[most_frequent_i_star[0]]


class EpsSelectionPIBase(PIModel, ConformalSelectionMixin, ABC):
    """
    PI Model which applies the default or enbPI prediction methods of conformal prediction but tries to fined the
    "right" epsilons
    """
    def __init__(self, use_dedicated_calibration: bool, fc_prediction_out_modes: Tuple[PredictionOutputType],
                 ctx_mode: str, past_window: int, no_of_beta_bins: int, ts_ids):
        PIModel.__init__(self, use_dedicated_calibration, fc_prediction_out_modes)
        ConformalSelectionMixin.__init__(self)
        self._ctx_gen = EpsilonContextGen(mode=ctx_mode, ts_ids=ts_ids)
        self._past_window = past_window
        self._no_beta_bins = no_of_beta_bins

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        """
        Calculates the PI prediction for a SINGLE step
        """
        alpha, X_step, X_past, Y_past, eps_past = pred_data.alpha, pred_data.X_step, pred_data.X_past,\
                                                  pred_data.Y_past, pred_data.eps_past
        # Calculate y_hat and prediction interval for current step
        fc_result = self._forcast_service.predict(
            FCPredictionData(ts_id=pred_data.ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=pred_data.step_offset_overall))
        Y_hat = fc_result.point
        fc_state_step = fc_result.state
        current_ctx = self._calc_context(X_past=X_past[-self._past_window:],
                                         Y_past=Y_past[-self._past_window:],
                                         eps_past=eps_past[-self._past_window:] if eps_past is not None else None,
                                         X_step=X_step.squeeze(dim=0),
                                         Y_hat_step=Y_hat.squeeze(dim=0),
                                         fc_state_step=fc_state_step.squeeze(dim=0) if fc_state_step is not None else None,
                                         ts_id=pred_data.ts_id,
                                         single_ctx=True)
        selected_eps = self._retrieve_epsilon(current_ctx)
        q_conformal_low, q_conformal_high, _ = self._calc_conformal_quantiles(selected_eps, alpha, self._no_beta_bins)
        pred_int = Y_hat + score.resolve(q_conformal_low, **pred_data.score_param),\
            Y_hat + score.resolve(q_conformal_high, **pred_data.score_param)
        prediction_result = PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)
        prediction_result.eps_ctx = current_ctx
        return prediction_result

    @abstractmethod
    def _retrieve_epsilon(self, current_ctx) -> torch.tensor:
        """
        Retrieve the relevant epsilons for m contexts
        :param current_ctx:         [m, ctx_size]
        :return: relevant epsilon   [m, no_of_epsilon]
        """
        pass

    def _calc_context(self, X_past, Y_past, eps_past, X_step, Y_hat_step, ts_id, fc_state_step, single_ctx):
        """
        Calculate context for m or a single observations
        :return: [m: ctx_size] (m=1 for single_ctx == True)
        """
        if single_ctx:
            return self._ctx_gen.calc_single(X_past=X_past, Y_past=Y_past, eps_past=eps_past, X_step=X_step,
                                             Y_hat_step=Y_hat_step, fc_state_step=fc_state_step,
                                             ts_id_enc=torch.tensor([self._ctx_gen.get_ts_id_enc(ts_id)], dtype=torch.long))
        else:
            return self._ctx_gen.calc(X_past=X_past, Y_past=Y_past, eps_past=eps_past, X_step=X_step,
                                      Y_hat_step=Y_hat_step, fc_state_step=fc_state_step,
                                      ts_id_enc=torch.tensor([self._ctx_gen.get_ts_id_enc(ts_id)], dtype=torch.long))

    def _check_pred_data(self, pred_data: PIPredictionStepData):
        assert pred_data.alpha is not None
        assert pred_data.X_step.shape[0] == 1  # Only single step prediction allowed for now

    def required_past_len(self) -> Tuple[int, int]:
        if self._ctx_gen.use_eps_past:
            return self._past_window, sys.maxsize
        else:
            return 0, sys.maxsize


class EpsSelectionPIStat(EpsSelectionPIBase):
    """
    Epsilon Selection Model which uses naive (not learned) techniques to select the right epsilons
    """
    def __init__(self, **kwargs):
        EpsSelectionPIBase.__init__(
            self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT, ),
            ctx_mode=kwargs['ctx_mode'], past_window=kwargs['past_window'], no_of_beta_bins=kwargs['no_of_beta_bins'],
            ts_ids=kwargs['ts_ids'])
        sim_mode = kwargs['sim_mode']
        if sim_mode == "pearson":
            self.ctx_sim = ctx_sim_pearson
        elif sim_mode == 'knn':
            self.ctx_sim = ctx_sim_knn
        else:
            raise ValueError(f"Sim mode {sim_mode} not available!")
        self._keep_calib_eps = kwargs['keep_calib_eps']
        self._eps_memory = FiFoMemory(kwargs['eps_mem_size'], store_step_no=False)
        self._retrieve_mode = kwargs['retrieve_mode']
        self._topk_adaptive = kwargs['topk_adaptive']
        self._topk_used_share = kwargs['topk_used_share']
        self._topk_share_adaptive = None
        self._online_memory = kwargs['online_memory']

        # Throw assert to make sweep runs more efficent
        if self._retrieve_mode == 'sample':
            assert self._topk_used_share is None
            assert self._topk_adaptive is None
        elif self._retrieve_mode == 'topk':
            assert self._topk_used_share is not None

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        c_result = self._forcast_service.predict(
            FCPredictionData(ts_id=calib_data.ts_id, X_past=calib_data.X_pre_calib, Y_past=calib_data.Y_pre_calib,
                             X_step=calib_data.X_calib, step_offset=calib_data.step_offset))
        Y_hat = c_result.point
        fc_state_step = c_result.state
        eps_calib = score.get(Y_hat=Y_hat, Y=calib_data.Y_calib, **calib_data.score_param)
        calib_size = calib_data.Y_calib.shape[0]
        real_calib_offset = self._past_window if self._ctx_gen.use_eps_past else 0
        ctx_calib = None
        for calib_step in range(real_calib_offset, calib_size):
            ctx = self._calc_context(
                X_past=torch.cat((calib_data.X_pre_calib, calib_data.X_calib[:calib_step]))[-self._past_window:],
                Y_past=torch.cat((calib_data.Y_pre_calib, calib_data.Y_calib[:calib_step]))[-self._past_window:],
                eps_past=eps_calib[:calib_step][-self._past_window:] if self._ctx_gen.use_eps_past else None,
                X_step=calib_data.X_calib[calib_step],
                Y_hat_step=Y_hat[calib_step],
                fc_state_step=fc_state_step[calib_step] if fc_state_step is not None else None,
                ts_id=calib_data.ts_id,
                single_ctx=True)
            if ctx_calib is None:
                ctx_size = ctx.shape[1]
                ctx_calib = torch.empty((calib_size - real_calib_offset, ctx_size))
            ctx_calib[calib_step - real_calib_offset] = ctx.squeeze()
        if self._keep_calib_eps:
            self._eps_memory.add_freezed(ctx_calib, torch.tensor(eps_calib[real_calib_offset:]).reshape(-1, 1))
        else:
            self._eps_memory.add_transient(ctx_calib, torch.tensor(eps_calib[real_calib_offset:]).reshape(-1, 1))
        return PICalibArtifacts(fc_Y_hat=Y_hat, eps=eps_calib)

    def pre_predict(self, **kwargs):
        super().pre_predict(**kwargs)
        self._topk_share_adaptive = self._topk_used_share  # Reset

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        if self._online_memory:
            step_eps = score.get(Y_hat=pred_result.fc_Y_hat, Y=Y_step, **pred_data.score_param)
            self._eps_memory.add_transient(pred_result.eps_ctx, step_eps.reshape(1, -1))
        if self._retrieve_mode == 'topk' and self._topk_adaptive is not None:
            pred_int = pred_result.pred_interval
            err_step = -1 if pred_int[0] <= Y_step <= pred_int[1] else 1
            self._topk_share_adaptive = self._topk_share_adaptive + self._topk_adaptive * err_step
            self._topk_share_adaptive = max(0.05, min(1, self._topk_share_adaptive))


    def _retrieve_epsilon(self, current_ctx) -> torch.tensor:
        similarity_weights, high_is_nearest = self.ctx_sim(current_ctx.squeeze(), self._eps_memory.ctx)
        if self._retrieve_mode == 'topk':
            top_k = int(self._eps_memory.ctx.shape[0] * self._topk_share_adaptive)
            _, selected = torch.topk(similarity_weights, top_k, largest=high_is_nearest)
        elif self._retrieve_mode == 'sample':
            if not high_is_nearest:
                similarity_weights = 1 / similarity_weights
            selected = torch.multinomial(torch.abs(similarity_weights), num_samples=1000, replacement=True)
        else:
            raise ValueError("Not supported!")
        return self._eps_memory.eps[selected][None, :]

    def model_ready(self):
        return not self._eps_memory.empty

    @property
    def can_handle_different_alpha(self):
        return True


def ctx_sim_pearson(ctx_current, ctx_mem):
    return (ctx_mem * ctx_current).sum(dim=1) /\
           ((ctx_mem * ctx_mem).sum(dim=1) * (ctx_current * ctx_current).sum()).sqrt(), True


def ctx_sim_knn(ctx_current, ctx_mem):
    std, mean = torch.std_mean(ctx_mem, dim=0)
    ctx_current_scaled = (ctx_current - mean) / std
    ctx_mem_scaled = (ctx_mem - mean.T) / std.T
    #ctx_current_scaled = ctx_current
    #ctx_mem_scaled = ctx_mem
    if len(ctx_current_scaled.shape) == 1:
        ctx_current_scaled = ctx_current_scaled.unsqueeze(0)
    return torch.cdist(torch.unsqueeze(ctx_current_scaled, 0), torch.unsqueeze(ctx_mem_scaled, 0), p=2).squeeze(), False
