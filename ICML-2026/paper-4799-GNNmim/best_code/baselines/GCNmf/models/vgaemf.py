"""
version 1.0
date 2021/02/04
"""

import torch.nn as nn

from . import GCNmfConv
from . import GCNConv
from . import Decoder, reparameterize


class VGAEmf(nn.Module):
    def __init__(self, data, nhid=32, latent_dim=16, dropout=0., n_components=5):
        super(VGAEmf, self).__init__()
        self.gc1 = GCNmfConv(data.num_features, nhid, data, n_components=n_components, dropout=dropout)
        self.gc_mu = GCNConv(nhid, latent_dim, dropout)
        self.gc_logvar = GCNConv(nhid, latent_dim, dropout)
        self.decoder = Decoder(dropout=dropout)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc_mu.reset_parameters()
        self.gc_logvar.reset_parameters()

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc_mu(hidden1, adj), self.gc_logvar(hidden1, adj)

    def forward(self, data):
        x, adj = data.features, data.adj
        mu, logvar = self.encode(x, adj)
        z = reparameterize(mu, logvar, self.training)
        adj_recon = self.decoder(z)
        return {'adj_recon': adj_recon, 'z': z, 'mu': mu, 'logvar': logvar}
