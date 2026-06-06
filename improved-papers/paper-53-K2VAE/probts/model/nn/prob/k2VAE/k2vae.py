'''
* @author: EmpyreanMoon
*
* @create: 2025-01-02 15:40
*
* @description: The overview of k2VAE
'''

import torch.nn as nn
import torch
import math
from torch.distributions import Normal

from probts.model.nn.prob.k2VAE.koopman import KPLayer, KPLayerApprox, MLP
from probts.model.nn.prob.k2VAE.kalman import KalmanFilter
from probts.model.nn.prob.k2VAE.auxiliarynet import Transformer as AuxiliaryNet
from probts.model.nn.prob.k2VAE.RevIN import RevIN
from einops import rearrange
import torch.nn.functional as F


class k2VAE(nn.Module):
    def __init__(self, config):
        super(k2VAE, self).__init__()
        self.config = config
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.multistep = config.multistep
        self.dynamic_dim = config.dynamic_dim
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars

        self.freq = math.ceil(self.input_len / self.patch_len)  # patch number of input
        self.step = math.ceil(self.pred_len / self.patch_len)  # patch number of output
        self.padding_len = self.patch_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.koopman = KPLayerApprox(config.dynamic_dim, config.init_koopman) if self.multistep else KPLayer(
            config.dynamic_dim, config.init_koopman)

        self.kalman = KalmanFilter(state_dim=self.dynamic_dim, init=config.init_kalman)

        self.auxiliary = AuxiliaryNet(self.config)

        self.encoder = MLP(f_in=self.patch_len * self.enc_in, f_out=self.dynamic_dim, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.decoder_mu = MLP(f_in=self.dynamic_dim, f_out=self.patch_len * self.enc_in, activation='relu',
                              hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.decoder_sigma = MLP(f_in=self.dynamic_dim, f_out=self.patch_len * self.enc_in, activation='relu',
                                 hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.revin = RevIN(self.enc_in)

    def encode(self, x):
        B, L, C = x.shape
        x = self.revin(x, 'norm')

        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)  # F x B P C, P means seg_len
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)  # B F PC

        res = self.encoder(res)  # B F H

        x_rec, Y = self.koopman(res, self.step)  # B F H, B S H
        x_residual = res - x_rec
        residual = self.auxiliary(x_residual)
        pred, dist = self.kalman(res[:, -1, :], residual, Y)
        pred = pred + residual

        return x_rec, pred, dist

    def decode(self, x_rec, x_pred):
        B, L, C = x_rec.shape
        x_rec = self.decoder_mu(x_rec)  # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C

        x_pred_mu = self.decoder_mu(x_pred)  # B S PC
        x_pred_mu = x_pred_mu.reshape(B, self.step, self.patch_len, self.enc_in)
        x_pred_mu = x_pred_mu.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B S C

        x_pred_sigma = self.decoder_sigma(x_pred)  # B S PC
        x_pred_sigma = x_pred_sigma.reshape(B, self.step, self.patch_len, self.enc_in)
        x_pred_sigma = x_pred_sigma.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B S C

        x_rec = self.revin(x_rec, 'denorm')
        x_pred_mu = self.revin(x_pred_mu, 'denorm')
        x_pred_sigma = self.revin(x_pred_sigma, 'denorm')

        dist = Normal(loc=x_pred_mu, scale=F.softplus(x_pred_sigma) + 1e-6)
        return x_rec, dist

    def forward(self, x):
        x_rec, pred, prior_dist = self.encode(x)
        pred = pred + prior_dist.rsample()
        x_rec, post_dist = self.decode(x_rec, pred)

        return x_rec, prior_dist, post_dist

    @torch.no_grad()
    def sample(self, x, num_samples):
        x_rec, pred, _ = self.encode(x)
        dist_list = []
        sample_list = []
        for i in range(self.config.sample_schedule):
            t = pred + torch.randn_like(pred).to(x.device)
            _, post_dist = self.decode(x_rec, t)
            dist_list.append(post_dist)

        cnt = 0
        k = num_samples // self.config.sample_schedule
        for id, dist in enumerate(dist_list):
            sample_num = k if id < self.config.sample_schedule - 1 else num_samples - cnt
            samples = dist.rsample(sample_shape=(sample_num,))
            sample_list.append(samples)
            cnt += sample_num

        if num_samples == self.config.sample_schedule:
            sample_list = [dist.loc.unsqueeze(0) for dist in dist_list]

        return torch.concat(sample_list, dim=0)
