# From https://github.com/criteo-research/continuous-fairness. Not modified.

from math import pi, sqrt
import numpy as np
import torch


# Independence of 2 variables
def _joint_2(X, Y, density, damping=1e-10):
    X = (X - X.mean()) / (X.std()+1e-12)
    Y = (Y - Y.mean()) / (Y.std()+1e-12)
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], -1)
    joint_density = density(data)

    nbins = int(min(50, 5. / joint_density.std))
    #nbins = np.sqrt( Y.size/5 )
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)

    xx, yy = torch.meshgrid([x_centers, y_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
    h2d = joint_density.pdf(grid) + damping
    h2d /= h2d.sum()
    return h2d

## EQUALIZED ODDS
def hgr_infty(X,Z,Y,density):
    return np.max(hgr_cond(X, Z, Y, density))

## DEMOGRAPHIC PARITY
def hgr(X, Y, density, damping = 1e-10):
    """
    An estimator of the Hirschfeld-Gebelein-Renyi maximum correlation coefficient using Witsenhausen’s Characterization:
    HGR(x,y) is the second highest eigenvalue of the joint density on (x,y). We compute here the second eigenvalue on
    an empirical and discretized density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: numerical value between 0 and 1 (0: independent, 1:linked by a deterministic equation)
    """
    h2d = _joint_2(X, Y, density, damping=damping)
    marginal_x = h2d.sum(dim=1).unsqueeze(1)
    marginal_y = h2d.sum(dim=0).unsqueeze(0)
    Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
    return torch.svd(Q)[1][1]


def chi_2(X, Y, density, damping = 0):
    """
    The \chi^2 divergence between the joint distribution on (x,y) and the product of marginals. This is know to be the
    square of an upper-bound on the Hirschfeld-Gebelein-Renyi maximum correlation coefficient. We compute it here on
    an empirical and discretized density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: numerical value between 0 and \infty (0: independent)
    """
    h2d = _joint_2(X, Y, density, damping=damping)
    marginal_x = h2d.sum(dim=1).unsqueeze(1)
    marginal_y = h2d.sum(dim=0).unsqueeze(0)
    Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
    return ((Q ** 2).sum(dim=[0, 1]) - 1.)


# Independence of conditional variables

def _joint_3(X, Y, Z, density, damping=1e-10):
    # Modified: ass eps for num.
    X = (X - X.mean()) / (X.std()+1e-12)
    Y = (Y - Y.mean()) / (Y.std()+1e-12)
    Z = (Z - Z.mean()) / (Z.std()+1e-12)
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)], -1)
    joint_density = density(data)  # + damping

    nbins = int(min(50, 5. / joint_density.std))
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)
    z_centers = torch.linspace(-2.5, 2.5, nbins)
    xx, yy, zz = torch.meshgrid([x_centers, y_centers, z_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], -1)

    h3d = joint_density.pdf(grid) + damping
    h3d /= h3d.sum()
    return h3d


def hgr_cond(X, Y, Z, density):
    """
    An estimator of the function z -> HGR(x|z, y|z) where HGR is the Hirschfeld-Gebelein-Renyi maximum correlation
    coefficient computed using Witsenhausen’s Characterization: HGR(x,y) is the second highest eigenvalue of the joint
    density on (x,y). We compute here the second eigenvalue on
    an empirical and discretized density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param Z: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: A torch 1-D Tensor of same size as Z. (0: independent, 1:linked by a deterministic equation)
    """
    damping = 1e-10
    h3d = _joint_3(X, Y, Z, density, damping=damping)
    marginal_xz = h3d.sum(dim=1).unsqueeze(1)
    marginal_yz = h3d.sum(dim=0).unsqueeze(0)
    Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
    return np.array(([torch.svd(Q[:, :, i])[1][1] for i in range(Q.shape[2])]))


def chi_2_cond(X, Y, Z, density):
    """
    An estimator of the function z -> chi^2(x|z, y|z) where \chi^2 is the \chi^2 divergence between the joint
    distribution on (x,y) and the product of marginals. This is know to be the square of an upper-bound on the
    Hirschfeld-Gebelein-Renyi maximum correlation coefficient. We compute it here on an empirical and discretized
    density estimated from the input data.
    :param X: A torch 1-D Tensor
    :param Y: A torch 1-D Tensor
    :param Z: A torch 1-D Tensor
    :param density: so far only kde is supported
    :return: A torch 1-D Tensor of same size as Z. (0: independent)
    """
    damping = 0
    h3d = _joint_3(X, Y, Z, density, damping=damping)
    marginal_xz = h3d.sum(dim=1).unsqueeze(1)
    marginal_yz = h3d.sum(dim=0).unsqueeze(0)
    Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
    return ((Q ** 2).sum(dim=[0, 1]) - 1.)


# From https://github.com/criteo-research/continuous-fairness/blob/master/facl/independence/density_estimation/pytorch_kde.py. Not modified.



class kde:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.

    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """
    def __init__(self, x_train):
        n, d = x_train.shape

        self.n = n
        self.d = d

        self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        self.std = self.bandwidth

        self.train_x = x_train

    def pdf(self, x):
        s = x.shape
        d = s[-1]
        s = s[:-1]
        assert d == self.d

        data = x.unsqueeze(-2)

        train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        pdf_values = (
                         torch.exp(-((data - train_x).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
                     ).mean(dim=-1) / sqrt(2 * pi) / self.bandwidth

        return pdf_values


def _unsqueeze_multiple_times(input, axis, times):
    """
    Utils function to unsqueeze tensor to avoid cumbersome code
    :param input: A pytorch Tensor of dimensions (D_1,..., D_k)
    :param axis: the axis to unsqueeze repeatedly
    :param times: the number of repetitions of the unsqueeze
    :return: the unsqueezed tensor. ex: dimensions (D_1,... D_i, 0,0,0, D_{i+1}, ... D_k) for unsqueezing 3x axis i.
    """
    output = input
    for i in range(times):
        output = output.unsqueeze(axis)
    return output