from typing import Union

import numpy as np
import scipy as sp
import torch
from poly_mixing import PolyMixing
from scipy.stats import ortho_group
from torch import nn
from typing_extensions import Literal


def generate_latent_cov_mat(dim=128, df=128):
    """
    Generate a random covariance matrix for the given input dimension baed on a Wishart distribution.
    :param dim: Dimensionality of the covariance matrix.
    :param df: Degrees of freedom for the Wishart distribution.
    :param seed: Random seed for reproducibility.
    :return: A positive semi-definite covariance matrix of shape (dim, dim).
    """
    cov_mat = sp.stats.wishart.rvs(df=df, scale=np.eye(dim))

    return torch.from_numpy(cov_mat).float()


def generate_latent_mvn(Sig, num_samples=1000, generator=None):

    L = torch.linalg.cholesky(Sig)
    W = (
        torch.randn(num_samples, Sig.shape[0], generator=generator) @ L.T
    )  # shape: (samples, SNPs)

    return W


class SmoothLeakyReLU(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * x + (1 - self.alpha) * torch.log(1 + torch.exp(x))


def get_act_fct(act_fct):
    if act_fct == "relu":
        return torch.nn.ReLU, {}, 1
    if act_fct == "leaky_relu":
        return torch.nn.LeakyReLU, {"negative_slope": 0.2}, 1
    elif act_fct == "elu":
        return torch.nn.ELU, {"alpha": 1.0}, 1
    elif act_fct == "max_out":
        raise NotImplementedError
    elif act_fct == "smooth_leaky_relu":
        return SmoothLeakyReLU, {"alpha": 0.2}, 1
    elif act_fct == "softplus":
        return torch.nn.Softplus, {"beta": 1}, 1
    else:
        raise Exception(f"activation function {act_fct} not defined.")


def construct_poly_mixing(degree, latent_dim, output_dim=None, weights=None):
    """
    Returns nn.Sequential object representing the polynomial mixing.
    """
    mixing_net = nn.Sequential(
        PolyMixing(degree, latent_dim, output_dim, weights)
    )

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False

    return mixing_net


def construct_invertible_mlp(
    n: int = 20,
    n_layers: int = 2,
    weight_matrix_init: Union[Literal["pcl"], Literal["rvs"]] = "pcl",
    act_fct: Union[
        Literal["relu"],
        Literal["leaky_relu"],
        Literal["elu"],
        Literal["smooth_leaky_relu"],
        Literal["softplus"],
    ] = "leaky_relu",
    seed=None,
):
    """
    Create an (approximately) invertible mixing network based on an MLP.
    Based on the mixing code by Hyvarinen et al.

    Args:
        n: Dimensionality of the input and output data
        n_layers: Number of layers in the MLP.
        n_iter_cond_thresh: How many random matrices to use as a pool to find weights.
        cond_thresh_ratio: Relative threshold how much the invertibility
            (based on the condition number) can be violated in each layer.
        weight_matrix_init: How to initialize the weight matrices.
        act_fct: Activation function for hidden layers.
    """

    layers = []
    act_fct, act_kwargs, act_fac = get_act_fct(act_fct)

    # Subfuction to normalize mixing matrix
    def l2_normalize(Amat, axis=0):
        # axis: 0=column-normalization, 1=row-normalization
        l2norm = np.sqrt(np.sum(Amat * Amat, axis))
        Amat = Amat / l2norm
        return Amat

    rng = np.random.default_rng(seed)
    for i in range(n_layers):
        lin_layer = nn.Linear(n, n, bias=False)

        if weight_matrix_init == "pcl":
            weight_matrix = rng.uniform(-1, 1, (n, n))
            weight_matrix = l2_normalize(weight_matrix, axis=0)
            condA = np.linalg.cond(weight_matrix)
            print("L{0:d}: cond={1:f}".format(i, condA))
            lin_layer.weight.data = torch.tensor(
                weight_matrix, dtype=torch.float32
            )
        elif weight_matrix_init == "rvs":
            weight_matrix = ortho_group.rvs(n, random_state=seed + i)
            lin_layer.weight.data = torch.tensor(
                weight_matrix, dtype=torch.float32
            )
        elif weight_matrix_init == "expand":
            pass
        else:
            raise Exception(
                f"weight matrix {weight_matrix_init} not implemented"
            )

        layers.append(lin_layer)

        if i < n_layers - 1:
            layers.append(act_fct(**act_kwargs))

    mixing_net = nn.Sequential(*layers)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False

    return mixing_net


def construct_discretize_mlp(
    ns,
    act_fct: Union[
        Literal["relu"],
        Literal["leaky_relu"],
        Literal["elu"],
        Literal["smooth_leaky_relu"],
        Literal["softplus"],
    ] = "leaky_relu",
    seed=None,
):
    if seed is not None:
        # Save current state
        # state = torch.get_rng_state()
        torch.manual_seed(seed)

    layers = []
    act_fct, act_kwargs, act_fac = get_act_fct(act_fct)
    for i in range(len(ns) - 1):
        layers.append(nn.Linear(ns[i], ns[i + 1]))
        if i < len(ns) - 2:  # no ReLU after the last layer
            layers.append(act_fct(**act_kwargs))

    mixing_net = nn.Sequential(*layers)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False

    # if seed is not None: torch.set_rng_state(state)

    return mixing_net


def generate_obs_data(
    V,
    W,
    H,
    eta,  # coef H to V should be of length H.shape[1]
    beta,  # coef instrument to treatment
    alpha1,  # coef H to treatment
    alpha2,  # coef H to response
    theta,  # coef treatment to response
    dim_z=None,
    Beta=None,
    intercept=False,
    mixing_fn=None,
    discretize=False,
    generator=None,
):
    """
    Generate observed instrument data.
    :param Beta: Linear mixing parameters mapping VW to Z.
    :param V: Invalid component in Z.
    :param W: Valid component in Z.
    :param H: Unobserved confounder between V, D, and Y.
    """

    if (Beta is not None) and (dim_z is None):
        dim_z = Beta.shape[1]

    # Currently assuming H is 2-dim and eta is repeated to match V's dim
    V += H @ eta
    VW = torch.cat((V, W), dim=1)

    if (mixing_fn is None) and intercept:
        VW_int = torch.cat(
            (VW, torch.ones(VW.shape[0], 1)), dim=1
        )  # add intercept
        Z = VW_int @ Beta  # shape: (samples, dim_z)
        Z_orig = Z.clone()
    elif (Beta is not None) and (mixing_fn is None):
        Z = VW @ Beta
        Z_orig = Z.clone()
        if discretize:
            raise ValueError(
                "Linear mixing with discretized Z is not implemented."
            )
    elif (Beta is None) and (mixing_fn is not None):
        with torch.no_grad():
            Z = mixing_fn.forward(VW)
            Z_orig = Z.clone()
            # print(f"Z.shape {Z.shape}")
        if discretize:
            logits = Z.view(V.shape[0], dim_z, 3)
            Z = torch.argmax(logits, dim=-1).float()  # convert to float for now

    # print(f"VW shape: {VW.shape}, Beta shape: {Beta.shape}, Z shape: {Z.shape}")

    # print(f"beta: {beta.T}")
    # generate treatment and response
    D = VW @ beta + H @ alpha1 + torch.randn(V.shape[0], 1, generator=generator)
    Y = theta * D + H @ alpha2 + torch.randn(V.shape[0], 1, generator=generator)

    return Z, VW, D, Y, Z_orig
