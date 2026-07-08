import sys
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.forcast.forcast_base import FCPredictionData, PredictionOutputType
from models.uncertainty.components.eps_ctx_encode import FcModel
from models.uncertainty.components.eps_ctx_gen import EpsilonContextGen
from models.uncertainty.ml_base import CalibTrainerMixin, EpsCtxMemoryMixin, BATCH_MODE_ONE_TS
from models.uncertainty.pi_base import PIModel, PIPredictionStepData, PIModelPrediction, PICalibData, PICalibArtifacts
from models.uncertainty.score_service import score


class EpsPredictionLSTM(BaseModel, PIModel, CalibTrainerMixin, EpsCtxMemoryMixin):

    def __init__(self, **kwargs):
        BaseModel.__init__(self)
        PIModel.__init__(self, use_dedicated_calibration=True, fc_prediction_out_modes=(PredictionOutputType.POINT,))
        # Train/Predict Mode
        self._train_many_to_one = True
        self._train_with_subsequence = True
        self._train_subsequence_len = kwargs['eps_mem_size']   # Only relevant when train_sequence_windowed
        self._pred_many_to_one = True
        CalibTrainerMixin.__init__(self, batch_mode=BATCH_MODE_ONE_TS,
                                   with_loss_weight=kwargs['with_loss_weight'],
                                   coverage_loss_weight=kwargs['coverage_loss_weight'],
                                   chung_loss_weight=kwargs['chung_loss_weight'],
                                   all_alpha_in_one_batch=True,
                                   batch_size=10 if self._train_with_subsequence else 1,
                                   split_in_subsequence_of_size=self._train_subsequence_len if self._train_with_subsequence else None)
        EpsCtxMemoryMixin.__init__(self, mem_size=kwargs['eps_mem_size'], keep_calib_eps=kwargs['keep_calib_eps'])
        # Context
        self._ctx_past_window = kwargs['ctx_past_window']
        self._use_pre_calib_eps_for_calib = False
        self._ctx_gen = EpsilonContextGen(mode=kwargs['ctx_mode'], ts_ids=kwargs['ts_ids'])
        # Network
        self._pre_encode_context = False
        self._alpha_in_final_layer = False
        self._predict_beta = False
        self._lstm_layers = 1
        self._project_with_lstm = False

        # Init Network
        if self._pre_encode_context:
            ctx_size = self._ctx_gen.context_size(kwargs['no_x_features'], self._ctx_past_window,
                                                  fc_state_dim=kwargs['fc_state_dim'])
            ctx_encoded_size = ctx_size // 2
            ctx_enc_hidden = ctx_size
            self._ctx_encoding = FcModel(input_dim=ctx_size, out_dim=ctx_encoded_size, hidden=(ctx_enc_hidden,))
        else:
            ctx_encoded_size = self._ctx_gen.context_size(kwargs['no_x_features'], self._ctx_past_window,
                                                          fc_state_dim=kwargs['fc_state_dim'])
            self._ctx_encoding = lambda **enc_args: enc_args

        if self._predict_beta:
            # Predict beta share of alpha
            self._beta_network = nn.Sequential(FcModel(input_dim=ctx_encoded_size + 1, out_dim=1, hidden=()),
                                               nn.Sigmoid())

        lstm_hidden = ctx_encoded_size // 2  #TODO
        self._lstm = nn.LSTM(
            batch_first=True,
            input_size=ctx_encoded_size + 2,  # R (Query) size + Alpha/Beta
            hidden_size=lstm_hidden,
            proj_size=2 if self._project_with_lstm else 0,                     # Epsilon size (Values)
            num_layers=self._lstm_layers
        )
        self._lstm_state = None
        if self._alpha_in_final_layer or not self._project_with_lstm:
            self._final_layer = list()
            for _ in range(1):
                dim = lstm_hidden + (1 if self._alpha_in_final_layer else 0)
                self._final_layer.append(FcModel(input_dim=dim, out_dim=2, hidden=(), dropout=0.4, dropout_at_first=True))

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
        # Fill Memory
        self._fill_memory(calib_data, calib_artifact)
        return calib_artifact

    def pre_predict(self, **kwargs):
        # Run through end of calib to get initial hidden state
        _, _, _, (lstm_hn, lstm_cn) = self._forward(
            current_ctx=self._memory.ctx_chronological, alpha=kwargs['alpha'])
        self._lstm_state = (lstm_hn, lstm_cn)

    def _predict_step(self, pred_data: PIPredictionStepData, **kwargs) -> PIModelPrediction:
        # Feed through model and retrieve eps_lower, eps_higher
        alpha, X_step, X_past, Y_past, eps_past = pred_data.alpha, pred_data.X_step, pred_data.X_past,\
                                                  pred_data.Y_past, pred_data.eps_past
        # Calculate y_hat and prediction interval for current step
        fc_result = self._forcast_service.predict(
            FCPredictionData(ts_id=ts_id, X_past=X_past, Y_past=Y_past, X_step=X_step,
                             step_offset=step_abs))
        Y_hat = fc_result.point
        fc_state_step = fc_result.state
        ctx = self._ctx_gen.calc_single(
            X_past=X_past[-self._ctx_past_window:], Y_past=Y_past[-self._ctx_past_window:],
            eps_past=eps_past[-self._ctx_past_window:] if eps_past is not None else None,
            X_step=X_step.squeeze(dim=0), Y_hat_step=Y_hat.squeeze(dim=0),
            fc_state_step=fc_state_step.squeeze(dim=0) if fc_state_step is not None else None,
            ts_id_enc=torch.tensor([self._ctx_gen.get_ts_id_enc(pred_data.ts_id)], dtype=torch.long))
        ctx_encoded = self._encode_ctx(context=ctx, step_no=None)[0]
        if self._pred_many_to_one:
            _, _, _, (lstm_hn, lstm_cn) = self._forward(
                current_ctx=self._memory.ctx_chronological, alpha=alpha)
        else:
            (lstm_hn, lstm_cn) = self._lstm_state
        eps_q_low, eps_q_high, _, (lstm_hn, lstm_cn) = self._forward(
            current_ctx=ctx_encoded, alpha=alpha, lstm_state=(lstm_hn, lstm_cn))
        self._lstm_state = (lstm_hn, lstm_cn)

        pred_int = Y_hat + score.resolve(eps_q_low, **pred_data.score_param), Y_hat + score.resolve(eps_q_high, **pred_data.score_param)
        prediction_result = PIModelPrediction(pred_interval=pred_int, fc_Y_hat=Y_hat)
        prediction_result.eps_ctx = ctx_encoded
        return prediction_result

    def _post_predict_step(self, Y_step, pred_result: PIModelPrediction, pred_data: PIPredictionStepData, **kwargs):
        # Update memory
        encoded_eps_ctx = pred_result.eps_ctx
        eps = score.get(Y=Y_step, Y_hat=pred_result.fc_Y_hat)
        self._memory.add_transient(encoded_eps_ctx, eps)

    def model_ready(self):
        return not self._memory.empty

    @property
    def can_handle_different_alpha(self):
        return True

    def required_past_len(self) -> Tuple[int, int]:
        if self._ctx_gen.use_eps_past:
            return self._ctx_past_window, sys.maxsize
        else:
            return 0, sys.maxsize

    def _encode_ctx(self, context, step_no) -> Tuple[torch.tensor, torch.tensor]:
        return self._ctx_encoding(context=context, step_no=step_no), None

    def _get_calib_ctx(self, calib_data, Y_hat, fc_state_step=None) -> Tuple[torch.tensor, int, int]:
        return self._ctx_gen.calib_data_to_ctx(calib_data, Y_hat=Y_hat, past_window=self._ctx_past_window,
                                               fc_state_step=fc_state_step,
                                               use_pre_calib_eps_for_calib=self._use_pre_calib_eps_for_calib)

    #
    # Network Module methods
    #

    def _forward(self, current_ctx, alpha, lstm_state=None):
        """
        :param current_ctx: [batch, sequence_len, features] or [sequence_len, features]
        :param alpha:       float or same dim as current_ctx
        :param lstm_state:  Tuple[T[layers, batch, lstm_dim], T[layers, batch, lstm_dim]]
        :return: eps_q_low (same as ctx with feature_dim=1), eps_q_high (same as ctx with feature_dim=1), beta, (lstm_hn, lstm_cn),
        """
        squeeze = False
        if len(current_ctx.shape) == 2:
            squeeze = True
            current_ctx = current_ctx.unsqueeze(0)
            if not isinstance(alpha, float):
                alpha = alpha.unsqueeze(0)
        alpha = torch.full((current_ctx.shape[0], current_ctx.shape[1], 1), fill_value=alpha) if isinstance(alpha, float) else alpha
        if self._predict_beta:
            beta = alpha * self._beta_network(torch.cat((current_ctx, alpha), dim=2))
        else:
            beta = torch.div(alpha, 2)

        current_input = torch.cat((current_ctx, beta - alpha + 1, beta), dim=2)
        if lstm_state:
            lstm_out, (lstm_hn, lstm_cn) = self._lstm(current_input, lstm_state)
        else:
            lstm_out, (lstm_hn, lstm_cn) = self._lstm(current_input)

        no_batches = lstm_out.shape[0]
        batch_len = lstm_out.shape[1]
        if self._alpha_in_final_layer:
            feat_len = lstm_out.shape[2] + 1
            input_final = torch.cat((lstm_out, alpha), dim=2).reshape(-1, feat_len)
            final_out = self._final_layer[0](input_final).reshape(no_batches, batch_len, -1)
            eps_q_high, eps_q_low = final_out[:, :, 0].unsqueeze(2), final_out[:, :, 1].unsqueeze(2)
        elif not self._project_with_lstm:
            feat_len = lstm_out.shape[2]
            input_final = lstm_out.reshape(-1, feat_len)
            final_out = self._final_layer[0](input_final).reshape(no_batches, batch_len, -1)
            eps_q_high, eps_q_low = final_out[:, :, 0].unsqueeze(2), final_out[:, :, 1].unsqueeze(2)
        else:
            eps_q_low, eps_q_high = lstm_out[:, :, 0].unsqueeze(2), lstm_out[:, :, 1].unsqueeze(2)

        if squeeze:
            eps_q_low, eps_q_high, beta = eps_q_low.squeeze(0), eps_q_high.squeeze(0), beta.squeeze(0)
        return eps_q_low, eps_q_high, beta, (lstm_hn, lstm_cn)

    def forward(self, *args, **kwargs):
        """
        Forward method used in !training!
        """
        ctx, Y, Y_hat, alpha, step_no = kwargs['ctx_data'].detach(), kwargs['Y'].detach(), kwargs['Y_hat'].detach(),\
                                     kwargs['alpha'].detach(), kwargs['step_no'].detach()
        # Alpha is here a tensor of all alphas!
        batches = ctx.shape[0]
        seq_len = ctx.shape[1]
        ctx_features = ctx.shape[2]
        y_features = Y_hat.shape[2]
        no_alphas = alpha.shape[1]
        alphas = alpha.unsqueeze(2)
        train_batch_encoded = self._encode_ctx(context=ctx.reshape(-1, ctx_features), step_no=step_no)[0]\
            .reshape(batches, seq_len, -1)
        alphas = alphas.permute(1, 0, 2).reshape(-1, 1, 1)   # First have to be the first alpha from all batches
        alphas = alphas.repeat_interleave(seq_len, dim=1)
        train_batch_encoded = train_batch_encoded.repeat(no_alphas, 1, 1)
        eps_q_low, eps_q_high, beta, _ = self._forward(current_ctx=train_batch_encoded, alpha=alphas)
        if self._train_many_to_one:
            res = lambda t: t[:, -1, :]  # Only Last element gets loss
        else:
            res = lambda t: t.reshape(-1, 1)  # All elements get loss

        eps_q_low, eps_q_high, beta, alphas = res(eps_q_low), res(eps_q_high), res(beta), res(alphas)

        Y_hat = res(Y_hat).repeat(no_alphas, 1)     # repeat for each alpha
        q_low = Y_hat + eps_q_low
        q_high = Y_hat + eps_q_high
        return dict(q_low=q_low, q_high=q_high, low_alpha=beta, high_alpha=(beta - alphas + 1))

    def _get_constructor_parameters(self) -> dict:
        pass
