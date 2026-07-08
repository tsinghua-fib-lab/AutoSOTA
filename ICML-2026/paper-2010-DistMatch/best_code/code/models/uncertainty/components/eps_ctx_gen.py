from abc import abstractmethod
from typing import List, Optional

import torch
import torch.nn.functional as F

from models.uncertainty.pi_base import PICalibData
from utils.calc_torch import calc_residuals, unfold_window

_POSTFIX_EPS = "_and_eps"
_POSTFIX_Y_HAT = "_and_yhat"
_POSTFIX_TS_ONE_HOT = "_and_tsOneHot"


class EpsilonContextGen:

    def __init__(self, mode, ts_ids: List[str] = None):
        """
        Base Context Modes:
        - uni
        - uni_past_multi_step
        - multi_step
        - multi
        - fc_state
        - uni_past_multi_step_fc_states
        - multi_step_fc_states
        - none
        each of the 3 with optionally
        + eps (postfix: "_and_eps"
        + Y_hat_step (postfix: "_and_yhat")
        + TsOneHot (postfix: "_and_tsOneHot")
        """
        super().__init__()
        self._full_mode = mode

        self._use_ts_onehot = mode[-len(_POSTFIX_TS_ONE_HOT):] == _POSTFIX_TS_ONE_HOT
        self._ts_ids = list(set(ts_ids)) if ts_ids else None
        if self._use_ts_onehot:
            assert ts_ids is not None
            assert len(ts_ids) > 0
            mode = mode[:-len(_POSTFIX_TS_ONE_HOT)]

        self._use_y_hat = mode[-len(_POSTFIX_Y_HAT):] == _POSTFIX_Y_HAT
        if self._use_y_hat:
            mode = mode[:-len(_POSTFIX_Y_HAT)]

        self._use_eps = mode[-len(_POSTFIX_EPS):] == _POSTFIX_EPS
        if self._use_eps:
            mode = mode[:-len(_POSTFIX_EPS)]

        self._short_mode = mode
        calc = None
        if mode == 'uni':
            calc = self._ctx_univariate
        elif mode == 'uni_past_multi_step':
            calc = self._ctx_uni_past_multi_current
        elif mode == 'multi':
            calc = self._ctx_multivariant
        elif mode == 'multi_step':
            calc = self._ctx_multi_current
        elif mode == 'fc_states':
            calc = self._ctx_fc_state
        elif mode == 'uni_past_multi_step_fc_states':
            calc = self._ctx_uni_past_multi_current_fc_states
        elif mode == 'multi_step_fc_states':
            calc = self._ctx_multi_current_fc_states
        elif mode == 'none':
            assert self._use_eps or self._use_y_hat or self._use_ts_onehot
            calc = self._ctx_none
        else:
            raise ValueError(f"Unsupported mode: {self._full_mode}")

        if self._use_eps:
            calc_0 = lambda **kwargs: self.cat_eps(get_ctx=calc, **kwargs)
        else:
            calc_0 = calc

        if self._use_y_hat:
            calc_1 = lambda **kwargs: self.cat_y_hat(get_ctx=calc_0, **kwargs)
        else:
            calc_1 = calc_0

        if self._use_ts_onehot:
            calc_2 = lambda **kwargs: self.cat_ts_onehot(get_ctx=calc_1, **kwargs)
        else:
            calc_2 = calc_1

        self._calc_func = calc_2

    def calc(self, X_past, Y_past, eps_past, X_step, Y_hat_step, ts_id_enc, fc_state_step):
        """
        Calculate context for m observations
        :param X_past:          [m, len_window, #X_feature]
        :param Y_past:          [m, len_window, 1*]
        :param eps_past:        [m, len_window, 1*]
        :param X_step:          [m, #X_feature]
        :param Y_hat_step:      [m, 1*]
        :param ts_id_enc:       [m, 1 (int)]
        :param fc_state_step:   [m, state_dim, 1*]
        1*: =1 if not an ensemble or multi predict
        :return:                [m, ctx_size]
        """
        return self._calc_func(X_past=X_past, Y_past=Y_past, X_step=X_step, eps_past=eps_past, Y_hat_step=Y_hat_step,
                               ts_id_enc=ts_id_enc, fc_state_step=fc_state_step)

    def calc_single(self, X_past, Y_past, eps_past, X_step, Y_hat_step, ts_id_enc, fc_state_step):
        """
        Calculate context for a single observations
        :param X_past:              [len_window, #X_feature]
        :param Y_past:              [len_window, 1*]
        :param eps_past:            [len_window, 1*]
        :param X_step:              [#X_feature]
        :param Y_hat_step:          [1*]
        :param ts_id_enc:           [1 (int)]
        :param fc_state_step:       [state_dim, 1*]
        1*: =1 if not an ensemble or multi predict
        :return:                    [1, ctx_size]
        """
        return self.calc(
            X_past=X_past[None, :] if X_past is not None else None,
            Y_past=Y_past[None, :] if Y_past is not None else None,
            eps_past=eps_past[None, :] if eps_past is not None else None,
            X_step=X_step[None, :] if X_step is not None else None,
            Y_hat_step=Y_hat_step[None, :] if Y_hat_step is not None else None,
            ts_id_enc=ts_id_enc[None, :] if ts_id_enc is not None else None,
            fc_state_step=fc_state_step[None, :] if fc_state_step is not None else None
        )

    def calib_data_to_ctx(self, calib_data: PICalibData, Y_hat, past_window, use_pre_calib_eps_for_calib, fc_state_step=None,
                          Y_hat_pre_calib=None):
        # 1) Unfold calib data and create past windows for each timestep
        if (self.use_eps_past and not use_pre_calib_eps_for_calib) or calib_data.Y_calib is None:
            eps = calc_residuals(Y=calib_data.Y_calib, Y_hat=Y_hat)
            # We need to cut of the first window_len from calib
            eps_past_windowed = unfold_window(M=eps[:-1], window_len=past_window)
            X_past_windowed = unfold_window(M=calib_data.X_calib[:-1], window_len=past_window)
            X_step = calib_data.X_calib[past_window:]
            Y_past_windowed = unfold_window(M=calib_data.Y_calib[:-1], window_len=past_window)
            Y_hat_step = Y_hat[past_window:]
            window_offset = past_window
            fc_state_step = fc_state_step[past_window:] if fc_state_step is not None else None
        else:
            window_offset = 0
            if self.use_eps_past:
                eps = calc_residuals(Y=calib_data.Y_calib, Y_hat=Y_hat)
                assert Y_hat_pre_calib is not None
                eps_pre_calib = calc_residuals(Y=calib_data.Y_pre_calib, Y_hat=Y_hat_pre_calib)
                eps_past_windowed = unfold_window(M=eps[:-1], M_past=eps_pre_calib, window_len=past_window)
            else:
                eps_past_windowed = None
            X_past_windowed = unfold_window(M=calib_data.X_calib[:-1], M_past=calib_data.X_pre_calib,
                                            window_len=past_window)
            X_step = calib_data.X_calib
            Y_past_windowed = unfold_window(M=calib_data.Y_calib[:-1], M_past=calib_data.Y_pre_calib,
                                            window_len=past_window)
            Y_hat_step = Y_hat
            fc_state_step = fc_state_step

        # 2) Get context representation [calib_len/calib_len-past_window_len, ctx_size]
        context_data = self.calc(X_past=X_past_windowed, Y_past=Y_past_windowed, eps_past=eps_past_windowed,
                                 X_step=X_step, Y_hat_step=Y_hat_step, fc_state_step=fc_state_step,
                                 ts_id_enc=torch.tensor([self.get_ts_id_enc(calib_data.ts_id)])
                                 .unsqueeze(0).repeat(X_step.shape[0], 1))
        return context_data, window_offset, self.get_ts_id_enc(calib_data.ts_id)

    @property
    def use_eps_past(self):
        return self._use_eps

    @property
    def use_past(self):
        return self._short_mode in ['uni', 'uni_past_multi_step', 'multi', 'uni_past_multi_step_fc_states']\
            or self.use_eps_past

    def context_size(self, no_of_x_features, past_window_len, y_features=1, fc_state_dim=None):
        if self._short_mode == 'uni':
            ctx_len = past_window_len * y_features
        elif self._short_mode == 'uni_past_multi_step':
            ctx_len = past_window_len * y_features + no_of_x_features
        elif self._short_mode == 'multi':
            ctx_len = (past_window_len * (y_features + no_of_x_features)) + no_of_x_features
        elif self._short_mode == 'multi_step':
            ctx_len = no_of_x_features
        elif self._short_mode == 'fc_states':
            assert fc_state_dim is not None
            ctx_len = fc_state_dim
        elif self._short_mode == 'uni_past_multi_step_fc_states':
            assert fc_state_dim is not None
            ctx_len = fc_state_dim + past_window_len * y_features + no_of_x_features
        elif self._short_mode == 'multi_step_fc_states':
            assert fc_state_dim is not None
            ctx_len = fc_state_dim + no_of_x_features
        elif self._short_mode == 'none':
            ctx_len = 0
        else:
            raise ValueError(f"Unsupported mode: {self._full_mode}")
        if self.use_eps_past:
            ctx_len += past_window_len * y_features
        if self._use_y_hat:
            ctx_len += y_features
        if self._use_ts_onehot:
            ctx_len += len(self._ts_ids)
        return ctx_len

    @staticmethod
    def _ctx_univariate(Y_past, **kwargs):
        return Y_past.view(Y_past.shape[0], -1)

    @staticmethod
    def _ctx_multivariant(X_past, Y_past, X_step, **kwargs):
        return torch.cat((X_past.reshape(X_past.shape[0], -1), Y_past.view(Y_past.shape[0], -1),
                          X_step.view(X_step.shape[0], -1)), dim=1)

    @staticmethod
    def _ctx_uni_past_multi_current(Y_past, X_step, **kwargs):
        return torch.cat((Y_past.view(Y_past.shape[0], -1), X_step.view(X_step.shape[0], -1)), dim=1)

    @staticmethod
    def _ctx_multi_current(X_step, **kwargs):
        return X_step.view(X_step.shape[0], -1)

    @staticmethod
    def _ctx_fc_state(fc_state_step, **kwargs):
        return fc_state_step.reshape(fc_state_step.shape[0], -1)

    @staticmethod
    def _ctx_uni_past_multi_current_fc_states(Y_past, X_step, fc_state_step, **kwargs):
        return torch.cat((Y_past.view(Y_past.shape[0], -1), X_step.view(X_step.shape[0], -1),
                          fc_state_step.reshape(fc_state_step.shape[0], -1)), dim=1)

    @staticmethod
    def _ctx_multi_current_fc_states(X_step, fc_state_step, **kwargs):
        return torch.cat((X_step.view(X_step.shape[0], -1), fc_state_step.reshape(fc_state_step.shape[0], -1)), dim=1)

    @staticmethod
    def _ctx_none(**kwargs):
        return torch.tensor([])

    @staticmethod
    def cat_eps(get_ctx, **kwargs):
        tmp = get_ctx(**kwargs)
        eps_past = kwargs['eps_past'].to(device=tmp.device)
        return torch.cat((eps_past.view(eps_past.shape[0], -1), tmp), dim=1)

    @staticmethod
    def cat_y_hat(Y_hat_step, get_ctx, **kwargs):
        return torch.cat((Y_hat_step.view(Y_hat_step.shape[0], -1), get_ctx(**kwargs)), dim=1)

    def cat_ts_onehot(self, ts_id_enc: torch.Tensor, get_ctx, **kwargs):
        tmp = get_ctx(**kwargs)
        ts_one_hot = F.one_hot(ts_id_enc.flatten(), num_classes=len(self._ts_ids)).to(tmp.device)
        return torch.cat((ts_one_hot, tmp), dim=1)

    @staticmethod
    def cat_mean_of_past(value: str, get_ctx, **kwargs):
        """
        :param value: *Y* or *eps* possible
        :return:
        """
        tmp = get_ctx(**kwargs)
        past = kwargs[f'{value}_past']
        past_mean = torch.mean(past, dim=1).to(device=tmp.device)
        return torch.cat((past_mean.view(past_mean.shape[0], 1), tmp), dim=1)

    @staticmethod
    def cat_sd_of_past(value: str, get_ctx, **kwargs):
        """
        :param value: *Y* or *eps* possible
        :return:
        """
        tmp = get_ctx(**kwargs)
        past = kwargs[f'{value}_past']
        past_sd = torch.std(past, dim=1).to(device=tmp.device)
        return torch.cat((past_sd.view(past_sd.shape[0], 1), tmp), dim=1)

    def get_ts_id_enc(self, ts_id: str) -> Optional[int]:
        try:
            return self._ts_ids.index(ts_id)
        except ValueError:
            return None
