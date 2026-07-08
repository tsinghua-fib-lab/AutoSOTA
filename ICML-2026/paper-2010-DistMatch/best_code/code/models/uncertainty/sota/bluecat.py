from typing import Tuple, Optional, List

import torch

from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.components.eps_memory import FiFoMemory
from models.uncertainty.pi_base import PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts
from models.uncertainty.pi_eps_sel_base import EpsSelectionPIBase
from utils.calc_torch import calc_residuals


class Bluecat(EpsSelectionPIBase):

    def __init__(self, **kwargs):
        EpsSelectionPIBase.__init__(
            self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT, ),
            ctx_mode='none_and_yhat', past_window=0, no_of_beta_bins=1, ts_ids=kwargs['ts_ids']
        )
        self._used_calibration_set = kwargs['used_calibration_set']
        assert self._used_calibration_set in ['calib', 'train', 'both']
        self._quantile_mode = kwargs['quantile_mode']
        self._sample_m = kwargs['sample_m']
        self._memory = FiFoMemory(kwargs['eps_mem_size'], store_step_no=False)
        self._online_memory = kwargs['online_memory']

    def _calibrate(self, calib_data: [PICalibData], alphas, **kwargs) -> [PICalibArtifacts]:
        pass

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        self._memory.clear()
        ctxs = []
        eps = []
        if self._used_calibration_set in ['calib', 'both']:
            c_result = self._forcast_service.predict(
                FCPredictionData(ts_id=calib_data.ts_id, X_past=calib_data.X_pre_calib, Y_past=calib_data.Y_pre_calib,
                                 X_step=calib_data.X_calib, step_offset=calib_data.step_offset))
            Y_hat = c_result.point
            fc_state_step = c_result.state
            eps_calib = calc_residuals(Y_hat=Y_hat, Y=calib_data.Y_calib)
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
            ctxs.append(ctx_calib)
            eps.append(torch.tensor(eps_calib[real_calib_offset:]).reshape(-1, 1))
        if self._used_calibration_set in ['train', 'both']:
            # Use Training Data for Calibration as they do it in the original paper
            Y_hat = self._forcast_service.predict(
                FCPredictionData(ts_id=calib_data.ts_id, X_past=torch.tensor([]), Y_past=torch.tensor([]),
                                 X_step=calib_data.X_pre_calib, step_offset=0)).point
            eps_calib = calc_residuals(Y_hat=Y_hat, Y=calib_data.Y_pre_calib)
            # Ctxt is always y_hat
            ctxs.append(Y_hat)
            eps.append(eps_calib)

        if self._used_calibration_set == 'both':
            ctxs = torch.cat(ctxs, dim=0)
            eps = torch.cat(eps, dim=0)
        else:
            ctxs = ctxs[0]
            eps = eps[0]
        self._memory.add_transient(ctxs, eps)
        return PICalibArtifacts(fc_Y_hat=ctxs, eps=eps)

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        if self._quantile_mode == "empirical":
            # EmpQuant
            return super()._predict_step(pred_data, **kwargs)
        elif self._quantile_mode == "k-moments":
            raise NotImplemented("K-Moments quantile not implemented")
            # K-Moments
            # 1a) Check y_hat find the closest in calib set
            # 1b) Take all m smaller and m bigger together in a set (if no m smaller or m bigger then use less)
            # 2 Calc k-moments stuff
            # 2a) Calc mean of set
            # 2b) Fit the PBF distribution on the stochastic simulated data
            # 2c)
            # 2d)
            # 3) Take set - weight with this Fx from k-moments - and sum -> bounds
        else:
            raise ValueError("Invalid Quantile Mode")

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        if self._used_calibration_set in ['calib', 'both'] and self._online_memory:
            step_eps = calc_residuals(Y_hat=pred_result.fc_Y_hat, Y=Y_step)
            self._memory.add_transient(pred_result.eps_ctx, step_eps.reshape(1, -1))

    def _retrieve_epsilon(self, current_ctx) -> torch.tensor:
        current_yhat = current_ctx.squeeze()
        assert len(current_yhat.shape) == 0
        # 1a) Check y_hat (=context) find the closest in calib set
        sorted_yhat, sort_idx = torch.sort(self._memory.ctx.squeeze())
        closest_idx_idx = torch.argmin(torch.abs(sorted_yhat - current_yhat))  # Todo: Problem that first is selected?
        # 1b) Take all m smaller and m bigger together in a set (if no m smaller or m bigger then use less)
        set_min_idx_idx = max(0, closest_idx_idx - self._sample_m)
        set_max_idx_idx = min(len(self._memory.eps), closest_idx_idx + self._sample_m)
        selected_idx = sort_idx[set_min_idx_idx:set_max_idx_idx]
        return torch.index_select(self._memory.eps, 0, selected_idx).T

    @property
    def can_handle_different_alpha(self):
        return True

    def model_ready(self):
        return not self._memory.empty
