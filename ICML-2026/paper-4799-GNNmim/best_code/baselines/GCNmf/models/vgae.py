"""
version 1.0
date 2021/02/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import GCNConv


def reparameterize(mu, logvar, training):
    if training:
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    return mu


class VGAE(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0.):
        super(VGAE, self).__init__()
        self.encoder = Encoder(data, nhid, latent_dim, dropout)
        self.decoder = Decoder(dropout)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = reparameterize(mu, logvar, self.training)
        adj_recon = self.decoder(z)
        return {'adj_recon': adj_recon, 'z': z, 'mu': mu, 'logvar': logvar}


class Encoder(nn.Module):
    def __init__(self, data, nhid, latent_dim, dropout):
        super(Encoder, self).__init__()
        nfeat = data.num_features
        self.gc1 = GCNConv(nfeat, nhid, dropout)
        self.gc_mu = GCNConv(nhid, latent_dim, dropout)
        self.gc_logvar = GCNConv(nhid, latent_dim, dropout)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc_mu.reset_parameters()
        self.gc_logvar.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.relu(self.gc1(x, adj))
        mu, logvar = self.gc_mu(x, adj), self.gc_logvar(x, adj)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, dropout):
        super(Decoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, p=self.dropout, training=self.training)
        adj_recon = torch.mm(z, z.t())
        return adj_recon
