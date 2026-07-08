import logging
from typing import Optional, Tuple, List

import torch
from torch import nn
from torchmetrics.functional import pairwise_cosine_similarity

from models.uncertainty.ml_base import CalibTrainerMixin, EpsCtxMemoryMixin, BATCH_MODE_ONE_TS
from models.uncertainty.score_service import score
from models.base_model import BaseModel
from models.forcast.forcast_base import PredictionOutputType, FCPredictionData
from models.uncertainty.components.eps_ctx_encode import FcModel
from models.uncertainty.pi_base import PIModelPrediction, PIPredictionStepData, PICalibData, PICalibArtifacts
from models.uncertainty.pi_eps_sel_base import EpsSelectionPIBase


LOGGER = logging.getLogger(__name__)


class EpsSelectionPIML(BaseModel, EpsSelectionPIBase, CalibTrainerMixin, EpsCtxMemoryMixin):

    def __init__(self, **kwargs) -> None:
        BaseModel.__init__(self)
        EpsSelectionPIBase.__init__(
            self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,),
            ctx_mode=kwargs['ctx_mode'], past_window=kwargs['past_window'], no_of_beta_bins=kwargs['no_of_beta_bins'],
            ts_ids=kwargs['ts_ids'])
        CalibTrainerMixin.__init__(self, batch_mode=BATCH_MODE_ONE_TS,
                                   with_loss_weight=kwargs['with_loss_weight'],
                                   coverage_loss_weight=kwargs['coverage_loss_weight'])
        EpsCtxMemoryMixin.__init__(self, mem_size=kwargs['eps_mem_size'], keep_calib_eps=kwargs['keep_calib_eps'])
        self._enc_model: nn.Module = FcModel(input_dim=self._ctx_gen.context_size(kwargs['no_x_features'], self._past_window,
                                                                                  fc_state_dim=kwargs['fc_state_dim']),
                                             out_dim=50, hidden=(50,))
        self._topk_eps = 40

        # Calib Training Vars
        self._use_pre_calib_eps_for_calib = False
        self._train_with_beta = False

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
            calib_artifacts.append(PICalibArtifacts(fc_Y_hat=Y_hat[-1], fc_state_step=fc_state_step[-1]))

        trainer_config = kwargs['trainer_config']
        experiment_config = kwargs['experiment_config']
        self._train_model(calib_data, Y_hat=Y_hat, fc_state_step=fc_state_step, alphas=alphas,
                          experiment_config=experiment_config, trainer_config=trainer_config)

        return calib_artifacts

    def calibrate_individual(self, calib_data: PICalibData, alpha, calib_artifact: Optional[PICalibArtifacts],
                             mix_calib_data: Optional[List[PICalibData]],
                             mix_calib_artifact: Optional[List[PICalibArtifacts]]) -> PICalibArtifacts:
        LOGGER.info("Execute Model calibration (Fill memory with reference eps).")
        self._fill_memory(calib_data, calib_artifact)
        # Fill the memory with encoded calibration data
        return calib_artifact

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        # Update memory
        encoded_eps_ctx = pred_result.eps_ctx
        eps = score.get(Y=Y_step, Y_hat=pred_result.fc_Y_hat, **pred_data.score_param)
        self._memory.add_transient(encoded_eps_ctx, eps)

    def _retrieve_epsilon(self, current_ctx) -> torch.tensor:
        sel_idx = self._sel_reference_ctx_idx(current_ctx, self._memory.ctx, self._topk_eps, ctx_is_reference=False)
        return self._memory.eps[sel_idx]

    def _calc_context(self, **kwargs):
        ctx = super()._calc_context(**kwargs)    # [m, ctx_size]
        return self._encode_ctx(context=ctx, step_no=None)[0]   # [m, ctx_emb_size]

    def _get_calib_ctx(self, calib_data: PICalibData, Y_hat) -> Tuple[torch.tensor, int]:
        return self._ctx_gen.calib_data_to_ctx(calib_data, Y_hat=Y_hat, past_window=self._past_window,
                                               use_pre_calib_eps_for_calib=self._use_pre_calib_eps_for_calib)

    def _encode_ctx(self, context, step_no) -> Tuple[torch.tensor, torch.tensor]:
        return self._enc_model(context=context, step_no=step_no), None

    @staticmethod
    def _sel_reference_ctx_idx(ctx_encoded, reference_ctx_encoded, top_k: int, ctx_is_reference: bool,
                               mask: torch.tensor=None):
        """
        :param ctx_encoded:                 [batch_size, ctx_emb_size]
        :param reference_ctx_encoded:       [reference_size, ctx_emb_size]  (in calib both will be the same)
        :param top_k:                       amount of similar reference context indices one want to retrieve
        :param ctx_is_reference:            if ctx_encoded and reference_ctx_encoded are the same and one does not want
                                            to retrieve the "own index"
        :param mask:                        mask to omit usage of certain epsilon (e.g. if only past allowed)
        :return: selected_idx:              [batch_size, top_k (values from range 0:reference_size)]
        """
        # [batch_size, memory_size]
        cos_sim = pairwise_cosine_similarity(ctx_encoded, reference_ctx_encoded, zero_diagonal=ctx_is_reference)
        if mask is not None:
            cos_sim = cos_sim * mask    #ToDo Mask set to 0 (but similarity could be negative) Problem?
        #ToDo - How to select the best ones - Top k might use unrelated ones when situation is rare
        _, selected_idx = torch.topk(cos_sim, top_k, dim=1)
        return selected_idx

    def model_ready(self):
        return not self._memory.empty

    @property
    def can_handle_different_alpha(self):
        return True

    #
    # Methods for Calibration Training
    #

    def forward(self, *args, **kwargs):
        """Forward step in the training (during calibration)"""
        torch.autograd.set_detect_anomaly(True)
        ctx, Y, Y_hat, alpha, step_no = kwargs['ctx_data'].detach(), kwargs['Y'].detach(), kwargs['Y_hat'].detach(),\
                                     kwargs['alpha'], kwargs['step_no'].detach()
        eps = score.get(Y=Y, Y_hat=Y_hat).detach()  # [batch_size, *]

        batch_encoded = self._encode_ctx(context=ctx, step_no=step_no)[0]    # [batch_size, ctx_emb_size]
        #relevant_idx = self._sel_reference_ctx_idx(
        #    batch_encoded, batch_encoded, self._topk_eps, ctx_is_reference=True, mask=mask)  # [batch_size, k (0:batch_size)]
        # relevant_eps = eps[relevant_idx]  # [batch_size: k] # TODO:
        # eps_q_low, eps_q_high, beta = self._calc_conformal_quantiles(relevant_eps, alpha, no_beta_bins=0)
        # return dict(q_low=q_low, q_high=q_high, low_alpha=beta, high_alpha=(1 - alpha + beta))
        # wandb.log({'cos_sim': plt.imshow(cos_sim.detach())})
        cos_sim = torch.nn.functional.cosine_similarity(batch_encoded[:, :, None], batch_encoded.t()[None, :, :])
        eps_selected = self._retrieve_with_cos_sim_share_(cos_sim, eps)
        eps_q_low, eps_q_high, beta = self._calc_conformal_quantiles(
            eps_selected, alpha, no_beta_bins=self._no_beta_bins if self._train_with_beta else 0)
        q_low = Y_hat + score.resolve(eps_q_low.T)
        q_high = Y_hat + score.resolve(eps_q_high.T)
        return dict(q_low=q_low, q_high=q_high, low_alpha=beta, high_alpha=(1 - alpha + beta))

    def _retrieve_with_cos_sim_share_(self, cos_sim, eps):
        top_share = 0.3
        shift_sim = torch.quantile(cos_sim, (1 - top_share), dim=1)
        cos_sim = cos_sim - shift_sim - 0.0000001
        topk = int(cos_sim.shape[0] * top_share)
        eps_min = torch.min(eps) - 1
        eps = eps - eps_min
        eps_per_sample = eps.T.repeat(cos_sim.shape[0], 1)
        eps_per_sample = eps_per_sample * (nn.functional.relu(cos_sim, inplace=False) / cos_sim)
        eps_per_sample = eps_per_sample + eps_min
        eps_selected, eps_selected_idx = torch.topk(eps_per_sample, k=topk, dim=1)
        return eps_selected

    def _retrieve_with_cos_sim_sample(self, cos_sim, eps):
        pass #TODO Sample epsilson based on similarity

    def _record_attention(self, association_matrix, query_steps, key_steps):
        pass #ToDo

    def _get_constructor_parameters(self) -> dict:
        pass
