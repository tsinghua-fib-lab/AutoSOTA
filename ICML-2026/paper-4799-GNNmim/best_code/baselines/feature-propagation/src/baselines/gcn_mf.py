"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def ex_relu(mu, sigma):
    is_zero = sigma == 0
    sigma[is_zero] = 1e-10
    sqrt_sigma = torch.sqrt(sigma)
    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (
        torch.div(torch.exp(torch.div(-w * w, 2)), np.sqrt(2 * np.pi))
        + torch.div(w, 2) * (1 + torch.erf(torch.div(w, np.sqrt(2))))
    )
    nr_values = torch.where(is_zero, F.relu(mu), nr_values)
    return nr_values


def init_gmm(features, n_components):
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    init_x = imp.fit_transform(features)
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag").fit(init_x)
    return gmm


class GCNmfConv(nn.Module):
    def __init__(self, in_features, out_features, x, edge_index, n_components, dropout, bias=True):
        super(GCNmfConv, self).__init__()
        self.device = edge_index.device
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.dropout = dropout
        self.features = x.detach().cpu().numpy()
        # Create sparse tensor from edge index
        _, weights = gcn_norm(edge_index, num_nodes=x.shape[0])
        self.adj = torch.sparse.FloatTensor(edge_index, values=weights).to(self.device)
        # Compute element-wise square of adjacency matrix
        self.adj2 = torch.mul(self.adj, self.adj)
        self.logp = Parameter(torch.FloatTensor(n_components))
        self.means = Parameter(torch.FloatTensor(n_components, in_features))
        self.logvars = Parameter(torch.FloatTensor(n_components, in_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.gmm = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        self.gmm = init_gmm(self.features, self.n_components)
        self.logp.data = torch.FloatTensor(np.log(self.gmm.weights_)).to(self.device)
        self.means.data = torch.FloatTensor(self.gmm.means_).to(self.device)
        self.logvars.data = torch.FloatTensor(np.log(self.gmm.covariances_)).to(self.device)

    def calc_responsibility(self, mean_mat, variances):
        dim = self.in_features
        log_n = (
            (-1 / 2) * torch.sum(torch.pow(mean_mat - self.means.unsqueeze(1), 2) / variances.unsqueeze(1), 2)
            - (dim / 2) * np.log(2 * np.pi)
            - (1 / 2) * torch.sum(self.logvars)
        )
        log_prob = self.logp.unsqueeze(1) + log_n
        return torch.softmax(log_prob, dim=0)

    def forward(self, x, edge_index):
        x_imp = x.repeat(self.n_components, 1, 1)
        x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.logvars)
        mean_mat = torch.where(x_isnan, self.means.repeat((x.size(0), 1, 1)).permute(1, 0, 2), x_imp)
        var_mat = torch.where(
            x_isnan,
            variances.repeat((x.size(0), 1, 1)).permute(1, 0, 2).to(self.device),
            torch.zeros(size=x_imp.size(), device=self.device, requires_grad=True),
        )

        # dropout
        dropmat = F.dropout(torch.ones_like(mean_mat), self.dropout, training=self.training)
        mean_mat = mean_mat * dropmat
        var_mat = var_mat * dropmat

        transform_x = torch.matmul(mean_mat, self.weight)
        if self.bias is not None:
            transform_x = torch.add(transform_x, self.bias)
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)
        conv_x = []
        conv_covs = []
        for component_x in transform_x:
            conv_x.append(torch.spmm(self.adj, component_x))
        for component_covs in transform_covs:
            conv_covs.append(torch.spmm(self.adj2, component_covs))
        transform_x = torch.stack(conv_x, dim=0)
        transform_covs = torch.stack(conv_covs, dim=0)
        expected_x = ex_relu(transform_x, transform_covs)

        # calculate responsibility
        gamma = self.calc_responsibility(mean_mat, variances)
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)
        return expected_x
