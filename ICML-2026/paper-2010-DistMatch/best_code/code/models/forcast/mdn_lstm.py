import itertools
from typing import Dict

import torch
from torch import nn

from models.base_model import BaseModel
from models.uncertainty.components.mdn import MDN


class MDNLSTMManyToOne(BaseModel):

    def __init__(self, input_size, lstm_conf, mdn_conf, dropout, **kwargs):
        super().__init__()
        self._const_params = dict(input_size=input_size, lstm_conf=lstm_conf, mdn_conf=mdn_conf, dropout=dropout,
                                  **kwargs)
        self._lstm = nn.LSTM(input_size=input_size, **lstm_conf, batch_first=True)
        self._mdn = MDN(input_dim=self._lstm.hidden_size, **mdn_conf)
        self._n_targets = mdn_conf['n_targets']
        self._debug_log = False
        if dropout is not None and dropout > 0:
            self._dropout = nn.Dropout(p=dropout)
        else:
            self._dropout = None


    def forward(self, x, h=None, **kwargs) -> Dict:
        lstm_out, (h, c) = self._lstm(x, h)
        lstm_out = lstm_out[:, -1, :]
        if self._dropout is not None:
            lstm_out = self._dropout(lstm_out)
        dist_out, mdn_coef = self._mdn(lstm_out, draw_samples=1)
        if self._debug_log:
            return dict(y_hat=dist_out.unsqueeze(1), mdn_coef=mdn_coef, hidden=dict(lstm_out=lstm_out, h=h, c=c),
                        log=self._prep_debug_log(mdn_coef))
        else:
            return dict(y_hat=dist_out.unsqueeze(1), mdn_coef=mdn_coef, hidden=dict(lstm_out=lstm_out, h=h, c=c))

    def sample_from_mdn_coef(self, mdn_coef, n_samples=500):
        """
        :param mdn_coef:    named 3-tuple each: [n_batches, n_targets, n_components]
        :param n_samples:   number of samples
        :return:
        """
        batch_size = mdn_coef.pi.shape[0]
        #TODO handle more than 1 target
        d = self._mdn.get_mixture_distribution_from_coef(*mdn_coef)
        samples = d.sample((n_samples,))
        return samples.swapaxes(0, 1)

    def _init_embedding(self, x_init):
        init_latent = self._init_emb_nn(x_init)\
            .reshape(x_init.shape[0], 2, self._lstm.num_layers, self._lstm.hidden_size)
        init_latent = init_latent.transpose(0, 1)
        h, c = init_latent[0], init_latent[1]
        return h.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous(),

    def _prep_debug_log(self, mdn_coef):
        log = dict()
        for t, c in itertools.product(range(self._n_targets), range(self._mdn.n_components)):
            log[f'T{t}_C{c}_pi'] = mdn_coef.pi[:, t, c].mean()
            log[f'T{t}_C{c}_mu'] = mdn_coef.mu[:, t, c].mean()
            log[f'T{t}_C{c}_sigma'] = mdn_coef.sigma[:, t, c].mean()
        return log

    def get_loss_func(self, **kwargs):
        target_names = ['target_1']
        def loss(y, mdn_coef, **kwargs):
            batch_size = y.shape[0]
            seq_len = y.shape[1]
            assert seq_len == 1
            assert batch_size == mdn_coef[0].shape[0]
            assert y.shape[2] == self._n_targets and mdn_coef[0].shape[1] == self._n_targets
            distribution = self._mdn.get_mixture_distribution_from_coef(*mdn_coef)
            y = y.view(-1, 1)
            log_loss = - distribution.log_prob(y)  # Negative Log Likelihood
            log_loss = log_loss.view(batch_size, seq_len, self._n_targets)
            return log_loss.mean(), {f"Target_{target_names[t]} Loss": log_loss[:, :, t].mean()
                                     for t in range(self._n_targets)}
        return loss

    def _get_constructor_parameters(self) -> dict:
        return self._const_params

    def get_train_fingerprint(self) -> dict:
        raise NotImplemented("asf")

