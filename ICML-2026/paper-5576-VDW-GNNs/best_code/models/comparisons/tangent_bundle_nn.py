#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from github for "Tangent Bundle Convolutional Learning" paper (Battiloro et al. 2024).
@author: Claudio Battiloro
https://github.com/clabat9/Tangent-Bundle-Neural-Networks/blob/main/Journal_repo/architecture.py
https://github.com/clabat9/Tangent-Bundle-Neural-Networks/blob/main/Journal_repo/layers.py
https://github.com/clabat9/Tangent-Bundle-Neural-Networks/blob/main/Journal_repo/utils.py

Note at lines 158-161 of `mainWindSampling.py` (their training file that uses the above files),
there is some results engineering we do not use, dropping 2 worst and NaN MSEs.

-------------------------------------------------------------------------------
Sheaf Laplacian construction + learning logic flow (as implemented in this file)
-------------------------------------------------------------------------------

This file uses the name "Sheaf Laplacian", but the returned operator `Delta_n`
is a *diffusion-like* matrix built from a sheaf Laplacian expression and then
matrix-exponentiated via `scipy.linalg.expm`.

### Geometry / operator construction (TNN path)

Given a point cloud `coord` with shape (n, D):

1) `compute_neighbours(data=coord, epsilon, epsilon_pca)`:
   - For each point i, collects neighbors j within radius sqrt(epsilon_pca)
   - Builds neighbor-offset matrix `X_i` (D x n_i) and distances

2) `compute_weighted_X_i(X_i_collection, distances_collection)`:
   - Applies a kernel to neighbor distances to get weights `D_i`
   - Forms weighted local data matrices:
       B_i = X_i @ diag(D_i)

3) `local_pca(B_i_collection, gamma_svd)`:
   - For each i, computes SVD(B_i) = U_i Sigma_i V_i^T
   - Estimates local intrinsic dimension d_hat_i by thresholding the cumulative
     fraction of singular values at `gamma_svd`
   - Sets a single global `d_hat` as the median of the local estimates
   - Returns local orthonormal frames:
       O_i = U_i[:, :d_hat]     (shape D x d_hat)

4) `build_S_W(O_i_collection, complete_distance_collection)`:
   - Builds scalar kernel weights (Gaussian/Epanechnikov) on full pairwise
     distances (scaled by sqrt(epsilon))
   - For each (i, j): align local frames O_i and O_j to minimize the Frobenius norm of the difference.
       w_ij = D_ij^2
       O_ij_tilde = O_i^T @ O_j
       O_ij = argmin_{Q orthogonal} ||Q - O_ij_tilde||_F = U V^T  (via SVD)
       S_ij = w_ij * O_ij     (stored as a d_hat x d_hat block inside S)
   - Also returns a symmetrized weight matrix W

5) `build_SheafLaplacian(S, W, d_hat, epsilon)`:
   - Applies degree normalizations to W and the block-matrix S
   - Forms a (normalized) sheaf Laplacian-like generator and then exponentiates:
       Delta_n = expm( (1/epsilon) * (D_1^{-1} S_1 - I) )
   - `Delta_n` is what the neural network uses as its shift operator `L`

6) `compute_L(coord, ...)`:
   - Calls `get_laplacians(...)` and returns:
       L = [Delta_n, Delta_n, ..., Delta_n]  (one copy per layer)

### Learning / filtering in `GNNLayer`

In `GNNLayer.forward`, the polynomial filter is implemented as repeated
applications of the shift operator `self.L`:

- **Filter order**: `self.K` (set from `kappa`)
- **Learnable parameters**: `self.W[k]` for k=0..K-1, with shape (K, F_in, F_out)

The forward pass computes a learned polynomial in `L`:

    z = sum_{k=1..K} L^k @ (X @ W_{k-1})

Note: the implementation starts at L^1 (not the identity term L^0). Also, `b`
is defined in `GNNLayer` but is not used in the current forward computation.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from scipy.linalg import expm  # exponentiate matrix

# Hardcoded hyperparameters from their repo ('mainWindSampling.py')
MODEL_NAME: str = "tnn"
MANIFOLD_D: int = 2
FEATURES: list[int] = [8, 4, 1]
if MODEL_NAME == "mnn" or MODEL_NAME == "fmnn":
    FEATURES[-1] = FEATURES[-1] * MANIFOLD_D

# Sheaf Laplacian hyperparameters
GAMMA: float = 0.8
EPSILON: float = 0.5
EPSILON_PCA: float = 0.8

# Training hyperparameters
LEARN_RATE: float =  4e-4
WEIGHT_DECAY: float = 1e-3
KAPPA: list[int] = [2] * len(FEATURES)
SIGMA: nn.Module = nn.Tanh()
READOUT_SIGMA: nn.Module = nn.Identity()
LOSS_FUNCTION: nn.Module = nn.MSELoss(reduction='sum')

# Our addition: optional node-level readout MLP defaults.
USE_READOUT_MLP: bool = True
READOUT_MLP_HIDDEN_DIMS: list[int] = [128, 128]

# Runner defaults, to work with our experiment config logic (can be overridden by CLI/config in scripts)
DEFAULT_REPLICATIONS: int = 5
DEFAULT_KNN_K: int = 4
DEFAULT_SEED: int = 1234
DEFAULT_EPOCHS: int = 2000
DEFAULT_PATIENCE: int = 100
DEFAULT_PLATEAU_PATIENCE: int = 100
DEFAULT_PLATEAU_FACTOR: float = 0.5
DEFAULT_PLATEAU_MIN_LR: float = 1e-5
DEFAULT_PLATEAU_MAX_RESTARTS: int = 1
DEFAULT_VECTOR_FEAT_KEY: str = "v"


def set_hparams(
    n: int,
    L: list[torch.Tensor],  # use compute_L for this
    in_features: int,
    features: list[int] = FEATURES,
    lr: float = LEARN_RATE,
    weight_decay: float = WEIGHT_DECAY,
    sigma: nn.Module = SIGMA,
    readout_sigma: nn.Module = READOUT_SIGMA,
    kappa: int = KAPPA,
    loss_function: nn.Module = LOSS_FUNCTION,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    """
    Set hyperparameters for `TNN(**hparams).to(device)` constructor call.
    """
    hparams = {
        'in_features': in_features,
        'features': features,
        'L': L,
        'lr': lr,
        'weight_decay': weight_decay,
        'sigma': sigma,
        'readout_sigma': readout_sigma,
        'kappa': kappa,
        'n': n,
        'loss_function': loss_function,
        'device': device
    }
    return hparams


def compute_L(
    coord: np.ndarray,
    features: list[int] = FEATURES,
    epsilon: float = EPSILON, 
    epsilon_pca: float = EPSILON_PCA, 
    gamma: float = GAMMA, 
    tnn_or_mnn: str = MODEL_NAME
):
    (
        Delta_n_numpy, S, W, O_i_collection, d_hat, B_i_collection
    ) = get_laplacians(coord, epsilon, epsilon_pca, gamma, tnn_or_mnn)
    L = len(features) * [torch.from_numpy(Delta_n_numpy).to(torch.float32)]
    return L

# ------------------------------------------------------------
# From here: copied from github repo
# ------------------------------------------------------------

# Tangent Bundle Neural Network
class TNN(pl.LightningModule):

    def __init__(self, in_features, L, features,
                 lr, weight_decay, sigma, readout_sigma, kappa, n,
                 loss_function, device):
        """
        Parameters
        ----------
        in_features : Input features
        L : List of Shift Operators (one per layer)
        features : List of hidden features
        lr: optimizer's learning rate
        weight_decay: Weight decay multiplier
        sigma : Non-linearity
        readout_sigma: Non-linearity of the last layer
        kappa : List of filters order
        n: Number of manifold points
        loss_function: Loss function
        device : device
        """
        super(TNN, self).__init__()
        self.lr = lr
        self.n = n
        self.weight_decay = weight_decay
        self.L = [l.to(device) for l in L]
        ops = []
        self.sigma = sigma
        self.readout_sigma = readout_sigma
        in_features = [in_features] + [features[l] for l in range(len(features))]
        self.N_layers = len(in_features)
        self.min_mse_train = 1e20
        self.min_mse_val = 1e20
        self.loss_fn = loss_function
        for l in range(self.N_layers-1):
            if l == self.N_layers-2:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "sigma": self.readout_sigma}
            else:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "sigma": self.sigma}
            tnn_layer = GNNLayer(**hparams).to(device)
            ops.extend([tnn_layer])
        self.tnn = nn.Sequential(*ops)

    def forward(self, x):
        return self.tnn(x)

    def training_step(self, batch, batch_idx):
        try:
            x, y, mask = batch
            y_hat = self(x)
            y_trim = y[mask, :]
            y_hat_trim = y_hat[mask, :]
            loss = self.loss_fn(y_hat_trim, y_trim)
        except:
            x, y  = batch
            y_hat = self(x)
            y_trim = y
            y_hat_trim = y_hat
            loss = self.loss_fn(y_hat_trim, x)
        self.mse_train = ((y_trim - y_hat_trim).square()).sum() / self.n
        self.min_mse_train = min(self.mse_train, self.min_mse_train)
        self.log('train_mse', self.mse_train, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        try:
            x, y, mask = batch
            y_hat = self(x)
            y_trim = y[mask, :]
            y_hat_trim = y_hat[mask, :]
        except:
            x, y  = batch
            y_hat = self(x)
            y_trim = y
            y_hat_trim = y_hat
        loss = self.loss_fn(y_hat_trim, y_trim)
        self.mse_val = ((y_trim - y_hat_trim).square()).sum() / self.n
        self.min_mse_val = min(self.mse_val, self.min_mse_val)
        self.log('test_mse', self.mse_val, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('test_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'train_loss'}

    
# Manifold Neural Network (redundant in our experiments, just to keep things separated)
class MNN(pl.LightningModule):

    def __init__(self, in_features, L, features,
                 lr, weight_decay, sigma, readout_sigma, kappa, n,
                 loss_function, device):
        """
        Parameters
        ----------
        in_features : Input features
        L : List of Shift Operators (one per layer)
        features : List of hidden features
        lr: optimizer's learning rate
        weight_decay: Weight decay multiplier
        sigma : Non-linearity
        readout_sigma: Non-linearity of the last layer
        kappa : List of filters order
        n: Number of manifold points
        loss_function: Loss function
        device : device
        """
        super(MNN, self).__init__()
        self.lr = lr
        self.n = n
        self.weight_decay = weight_decay
        self.L = [l.to(device) for l in L]
        ops = []
        self.sigma = sigma
        self.readout_sigma = readout_sigma
        in_features = [in_features] + [features[l]
                                       for l in range(len(features))]
        self.N_layers = len(in_features)
        self.min_mse_train = 1e20
        self.min_mse_val = 1e20
        self.loss_fn = loss_function
        for l in range(self.N_layers-1):
            if l == self.N_layers-2:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "sigma": self.readout_sigma}
            else:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "sigma": self.sigma}
            mnn_layer = GNNLayer(**hparams).to(device)
            ops.extend([mnn_layer])
        self.mnn = nn.Sequential(*ops)

    def forward(self, x):
        return self.mnn(x)

    def training_step(self, batch, batch_idx):
        try:
            x, y, mask = batch
            y_hat = self(x)
            y_trim = y[mask, :]
            y_hat_trim = y_hat[mask, :]
            loss = self.loss_fn(y_hat_trim, y_trim)
        except:
            x, y  = batch
            y_hat = self(x)
            y_trim = y
            y_hat_trim = y_hat
            loss = self.loss_fn(y_hat_trim, x)
        self.mse_train = ((y_trim - y_hat_trim).square()).sum() / self.n
        self.min_mse_train = min(self.mse_train, self.min_mse_train)
        self.log('train_mse', self.mse_train, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        try:
            x, y, mask = batch
            y_hat = self(x)
            y_trim = y[mask, :]
            y_hat_trim = y_hat[mask, :]
        except:
            x, y  = batch
            y_hat = self(x)
            y_trim = y
            y_hat_trim = y_hat
        loss = self.loss_fn(y_hat_trim, y_trim)
        self.mse_val = ((y_trim - y_hat_trim).square()).sum() / self.n
        self.min_mse_val = min(self.mse_val, self.min_mse_val)
        self.log('test_mse', self.mse_val, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('test_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'train_loss'}

# Recurrent Tangent Bundle Neural Network
class RTNN(pl.LightningModule):

    def __init__(self, in_features, time_window, L,
                 lr, weight_decay, sigma, kappa,
                 loss_function, device):
        """
        Parameters
        ----------
        in_features : Input features
        time_window: Prediction time window
        L : List of Shift Operators (one per layer)
        features : List of hidden features
        lr: optimizer's learning rate
        weight_decay: Weight decay multiplier
        sigma : Non-linearity
        readout_sigma: Non-linearity of the last layer
        kappa : List of filters order
        n: Number of manifold points
        loss_function: Loss function
        device : device
        """
        super(RTNN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.L = [l.to(device) for l in L]
        ops = []
        self.sigma = sigma
        self.N_layers = len(in_features)
        self.min_mse_train = 1e20
        self.min_mse_val = 1e20
        self.loss_fn = loss_function
        for l in range(self.N_layers-1):
            if l == self.N_layers-2:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "time_window": time_window,
                           "sigma": self.sigma}
            else:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "time_window": time_window,
                           "sigma": self.sigma}
            rtnn_layer = RGNNLayer(**hparams).to(device)
            ops.extend([rtnn_layer])
        self.rtnn = nn.Sequential(*ops)

    def forward(self, x):
        return self.rtnn(x)

    def training_step(self, batch, batch_idx):
        xt, xT = batch
        xT_hat = self(xt).to(self.device).double()
        loss = self.loss_fn(xT_hat, xT)
        self.mse_train = ((xT - xT_hat).square()).sum() / np.prod(xT.shape)
        self.min_mse_train = min(self.mse_train, self.min_mse_train)
        self.log('train_mse', self.mse_train, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        xt, xT = batch
        xT_hat = self(xt).to(self.device)
        loss = self.loss_fn(xT_hat, xT)
        self.mse_val = ((xT - xT_hat).square()).sum() / np.prod(xT.shape)
        self.min_mse_val = min(self.mse_val, self.min_mse_val)
        self.log('test_mse', self.mse_val, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('test_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'train_loss'}


# Recurrent Manifold Neural Network (redundant in our experiments, just to keep things separated)
class RMNN(pl.LightningModule):

    def __init__(self, in_features, time_window, L,
                 lr, weight_decay, sigma, kappa,
                 loss_function, device):
        """
        Parameters
        ----------
        in_features : Input features
        time_window: Prediction time window
        L : List of Shift Operators (one per layer)
        features : List of hidden features
        lr: optimizer's learning rate
        weight_decay: Weight decay multiplier
        sigma : Non-linearity
        readout_sigma: Non-linearity of the last layer
        kappa : List of filters order
        n: Number of manifold points
        loss_function: Loss function
        device : device
        """
        super(RMNN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.L = [l.to(device) for l in L]
        ops = []
        self.sigma = sigma
        self.N_layers = len(in_features)
        self.min_mse_train = 1e20
        self.min_mse_val = 1e20
        self.loss_fn = loss_function
        for l in range(self.N_layers-1):
            if l == self.N_layers-2:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "time_window": time_window,
                           "sigma": self.sigma}
            else:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "time_window": time_window,
                           "sigma": self.sigma}
            simplicial_attention_layer = RGNNLayer(**hparams).to(device)
            ops.extend([simplicial_attention_layer])
        self.rmnn = nn.Sequential(*ops)

    def forward(self, x):
        return self.rmnn(x)

    def training_step(self, batch, batch_idx):
        xt, xT = batch
        xT_hat = self(xt).to(self.device).double()
        loss = self.loss_fn(xT_hat, xT)
        self.mse_train = ((xT - xT_hat).square()).sum() / np.prod(xT.shape)
        self.min_mse_train = min(self.mse_train, self.min_mse_train)
        self.log('train_mse', self.mse_train, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        xt, xT = batch
        xT_hat = self(xt).to(self.device)
        loss = self.loss_fn(xT_hat, xT)
        self.mse_val = ((xT - xT_hat).square()).sum() / np.prod(xT.shape)
        self.min_mse_val = min(self.mse_val, self.min_mse_val)
        self.log('test_mse', self.mse_val, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('test_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'train_loss'}


# Graph Convolutional Neural Network Layer
class GNNLayer(nn.Module):
    
    def __init__(self, F_in, F_out, L, kappa,device, sigma):
        """
        Parameters
        ----------
        F_in: Numer of input signals
        F_out: Numer of outpu signals
        L: Shift Operator
        kappa: Filters order
        device: Device
        sigma: non-linearity
        """
        super(GNNLayer, self).__init__()
        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.sigma = sigma
        self.L = L
        if self.L.type() == 'torch.cuda.DoubleTensor':
            self.W = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device).double())
            self.b = nn.Parameter(torch.empty(size=(1, 1)).to(device).double())
        else:
            self.W = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device))
            self.b = nn.Parameter(torch.empty(size=(1, 1)).to(device))
        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        nn.init.xavier_uniform_(self.b.data, gain=gain)

    def forward(self, x):
        alpha_zero = torch.clone(self.L)
        data = torch.clone(x)
        alpha_k = torch.clone(alpha_zero)
        try:
            z_i = alpha_k @ torch.clone(data  @ self.W[0])
        except:
            alpha_k = alpha_k.to(data.device)
            z_i = alpha_k @ torch.clone(data  @ self.W[0])
        for k in range(1, self.K):
            alpha_k = alpha_k @ alpha_zero
            z_i += alpha_k  @  data  @ self.W[k]
        out = self.sigma(z_i)
        return out


# Graph Convolutional Neural Network Layer
class RGNNLayer(nn.Module):

    def __init__(self, F_in, F_out, L, kappa,device, sigma, time_window):
        """
        Parameters
        ----------
        F_in: Numer of input signals 
        F_out: Numer of outpu signals 
        L: Shift Operator
        kappa: Filters order 
        device: Device
        sigma: non-linearity 
        time_window: Prediction time window 
        """
        super(RGNNLayer, self).__init__()
        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.sigma = sigma
        self.time_window = time_window
        self.L = L
        if self.L.type() == 'torch.cuda.DoubleTensor':
            self.W = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device).double())
            self.H = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device).double())
            self.b = nn.Parameter(torch.empty(size=(1, 1)).to(device).double())
        else:
            self.W = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device))
            self.H = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device))
            self.b = nn.Parameter(torch.empty(size=(1, 1)).to(device))
        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        nn.init.xavier_uniform_(self.H.data, gain=gain)
        nn.init.xavier_uniform_(self.b.data, gain=gain)

    def forward(self, x):
        # x is batch_sizeXhow_many_time_slotsXnumber_of_nodesXnumber_of_features
        alpha_zero = torch.clone(self.L)
        data = torch.clone(x).to(self.device).double()
        out = torch.zeros(data.shape)
        for data_point in range(data.shape[0]): # Batch Loop: inefficient, can be improved with PyTorch Geometric
            hidden_state = torch.zeros(data.shape[2:])
            for t in range(self.time_window): # Time Loop 
                alpha_k = torch.clone(alpha_zero)
                hidden_state = hidden_state.to(self.device).double()
                try:
                    z_i = alpha_k @ torch.clone(data[data_point,t,:,:]  @ self.W[0]) + alpha_k @ torch.clone(hidden_state  @ self.H[0])
                except:
                    alpha_k = alpha_k.to(data.device)
                    z_i = alpha_k @ torch.clone(data[data_point,t,:,:]  @ self.W[0]) + alpha_k @ torch.clone(hidden_state  @ self.H[0])
                for k in range(1, self.K):
                    alpha_k = alpha_k @ alpha_zero
                    z_i += alpha_k  @  data[data_point,t,:,:]  @ self.W[k] + alpha_k @ torch.clone(hidden_state  @ self.H[k])
                hidden_state = self.sigma(z_i)
                out[data_point,t,:,:] = hidden_state
        return out


# ------------------------------------------------------------------
# Sheaf Laplacian Utils
# https://github.com/clabat9/Tangent-Bundle-Neural-Networks/blob/main/Journal_repo/utils.py
# ------------------------------------------------------------------

def compute_neighbours(
    data,
    epsilon,
    epsilon_pca, 
    option = 'mean_shift'
):
    """
    Compute neighbours for each point in the data, using
    Euclidean distance, keeping only points within 
    epsilon_pca**0.5 of the point.
    """
    n = data.shape[0]
    X_i_collection = []
    neighbours_collection = np.zeros((n,n))
    distances_collection = []
    complete_distance_collection = []
    for point in range(n):
        x_i = data[point,:]
        x_i_dists = np.sum((x_i - data)**2, 1)**.5
        neigh = (x_i_dists > 0.0) * (x_i_dists < epsilon_pca**.5) 
        tmp_neigh = data[neigh,:]
        tmp_dist_trim_scaled_pcs = x_i_dists[neigh]/epsilon_pca**.5
        if option == 'point_shift':
            X_i_collection.append((tmp_neigh - x_i).T)
        if option == 'mean_shift':   
            X_i_collection.append((tmp_neigh - np.mean(tmp_neigh,0)).T)
        distances_collection.append(tmp_dist_trim_scaled_pcs)
        complete_distance_collection.append(x_i_dists/epsilon**.5)
        neighbours_collection[point,:] = neigh
    return X_i_collection,  distances_collection, neigh, complete_distance_collection


def truncated_gaussian_kernel(distances_collection):
    n = len(distances_collection)
    D_i_collection = []
    for point in range(n):
        dist = distances_collection[point]
        kernel_dist = np.sqrt(np.exp(-dist**2)) * (dist < 1.0) * (dist > 0.0) 
        D_i_collection.append(kernel_dist)
    return D_i_collection


def epanechnikov_kernel(distances_collection):
    n = len(distances_collection)
    D_i_collection = []
    for point in range(n):
        dist = distances_collection[point]
        kernel_dist = np.sqrt((1-dist**2)) * (dist < 1.0) * (dist > 0.0) 
        D_i_collection.append(kernel_dist)
    return D_i_collection
        
        
def compute_weighted_X_i(X_i_collection,distances_collection,option = 'epanechnikov'):
    n = len(X_i_collection)
    B_i_collection = []
    if option == 'epanechnikov': 
        D_i_collection = epanechnikov_kernel(distances_collection)
    if option == 'gaussian': 
        D_i_collection = truncated_gaussian_kernel(distances_collection) 
    for point in range(n):
        B_i = X_i_collection[point]@np.diag(D_i_collection[point])
        B_i_collection.append(B_i)
    return B_i_collection


def local_pca(B_i_collection, gamma):
    n = len(B_i_collection)
    U_i_collection = []
    dhat_i_collection = []
    for point in range(n):
        U_i,sigma_i,_ = np.linalg.svd(B_i_collection[point],full_matrices=False)
        U_i_collection.append(U_i)
        tmp_cumsum = np.sort(np.cumsum(sigma_i)/np.sum(sigma_i))
        d_hat_i = np.where(tmp_cumsum>gamma)[0][0]+1
        dhat_i_collection.append(d_hat_i)
    d_hat = int(np.median(dhat_i_collection))
    O_i_collection = []
    for point in range(n):
        O_i = U_i_collection[point][:,:d_hat]
        O_i_collection.append(O_i)
    return O_i_collection, d_hat


def build_S_W(O_i_collection, complete_distance_collection, option = 'gaussian'):
    n = len(O_i_collection)
    d_hat = O_i_collection[0].shape[1]
    S = np.zeros((n*d_hat,n*d_hat))
    if option == 'epanechnikov': 
        D_i_collection = epanechnikov_kernel(complete_distance_collection)
    if option == 'gaussian': 
        D_i_collection = truncated_gaussian_kernel(complete_distance_collection) 
    for point_i in range(n):
        for point_j in range(n):
            w_ij = D_i_collection[point_i][point_j]**2
            O_ij_tilde = O_i_collection[point_i].T@O_i_collection[point_j]
            U_i,_,Vt_i = np.linalg.svd(O_ij_tilde,full_matrices=False)
            O_ij = U_i@Vt_i
            S[point_i*d_hat:(point_i+1)*d_hat,point_j*d_hat:(point_j+1)*d_hat,]=w_ij*O_ij
    W = np.array(D_i_collection)       
    return S, (W+W.T)/2

def build_SheafLaplacian(S,W,d_hat,epsilon):
    D_cal_inv = np.diag(1/np.sum(W,1))
    W_1 = D_cal_inv @ W @ D_cal_inv
    D_cal_inv_block = np.kron(D_cal_inv, np.eye(d_hat))
    S_1 = D_cal_inv_block @ S @ D_cal_inv_block
    D_1_cal_inv = np.diag(1/np.sum(W_1,1))
    D_1_inv = np.kron(D_1_cal_inv, np.eye(d_hat))
    Delta_n = (1/epsilon) * (D_1_inv@S_1 - np.eye(S.shape[0]))
    Delta_n = expm(Delta_n)
    #Delta_n[Delta_n < 1e-10] = 0            
    return Delta_n
    
def get_laplacians(data, epsilon, epsilon_pca, gamma_svd,tnn_or_gnn):
    if tnn_or_gnn == "tnn":
        X_i_collection, distances_collection, _, complete_distance_collection = compute_neighbours(
            data,
            epsilon,
            epsilon_pca
        )
        B_i_collection = compute_weighted_X_i(X_i_collection,distances_collection)
        O_i_collection, d_hat = local_pca(B_i_collection, gamma_svd)
        S,W = build_S_W(O_i_collection, complete_distance_collection)
        Delta_n = (1/epsilon) * build_SheafLaplacian(S,W, d_hat, epsilon)
        return Delta_n, S, W, O_i_collection, d_hat, B_i_collection, X_i_collection, distances_collection, complete_distance_collection
    else:
        Delta_n = build_CloudLaplacian(data, heat_kernel_t=epsilon)
        return Delta_n

def project_data(data, O_i_collection):
    d_hat = O_i_collection[0].shape[1]
    data_proj = np.zeros((data.shape[0]*d_hat,1))
    for point in range(len(O_i_collection)):
        if data.shape[1] == d_hat:
            data_proj[point*d_hat:(point+1)*d_hat,:] = np.expand_dims(data[point,:],1)
        else:
            data_proj[point*d_hat:(point+1)*d_hat,:] = np.expand_dims(O_i_collection[point].T@data[point,:],1)
    return data_proj

def topk(input, k, axis=None, ascending=False):
    if not ascending:
        input *= -1
    ind = np.argsort(input, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind, axis=axis) 
    return val

# %% Cloud Laplacian Utils
#  From https://github.com/tegusi/RGCNN

def get_pairwise_euclidean_distance_matrix(tensor):
    """Compute pairwise distance of a tensor.
    Args:
        tensor: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    tensor = torch.tensor(tensor)
    adj_matrix = torch.cdist(tensor,tensor)
    return adj_matrix

def get_pairwise_distance_matrix(tensor, t):
    """Compute pairwise distance of a tensor.
    Args:
        tensor: tensor (batch_size, num_points, num_dims)
        t: scalar
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    # t = 10.55 # Average distance of CIFAR10
    # t = 10.55**2 # Average distance square of CIFAR10
    if len(tensor.shape)== 2:
        tensor = np.expand_dims(tensor,0)
    tensor = torch.tensor(tensor)
    adj_matrix = torch.squeeze(torch.cdist(tensor,tensor))
    adj_matrix = torch.square(adj_matrix)
    adj_matrix = torch.div( adj_matrix, -4*t)
    adj_matrix = torch.exp(adj_matrix)
    adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements


    return adj_matrix

def build_CloudLaplacian(imgs, normalize_exp = True, heat_kernel_t = 10, clamp_value=None):

    adj_matrix = get_pairwise_distance_matrix(imgs, heat_kernel_t)
    # Remove large values
    if clamp_value!=None:
        zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
        adj_matrix = torch.where(adj_matrix > clamp_value, adj_matrix, zero_tensor)

    if normalize_exp:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0])
        D = torch.diag(1 / torch.sqrt(D))
        L = (torch.matmul(torch.matmul(D, adj_matrix), D) - eye).numpy()
        L = expm(-L)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = (D - adj_matrix).numpy()
    return L

def get_laplacian_from_adj(adj_matrix, normalize = False, heat_kernel_t = 10, clamp_value=None):

    # Remove small values
    adj_matrix = torch.square(adj_matrix)

    adj_matrix = torch.div( adj_matrix, -4*heat_kernel_t)
    adj_matrix = torch.exp(adj_matrix)
    adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements

    if clamp_value!=None:
        # remove large values
        zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
        adj_matrix = torch.where(adj_matrix < clamp_value, adj_matrix, zero_tensor)

    if normalize:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0]).to('cuda') # Juan Modified This
        D = torch.diag(1 / torch.sqrt(D))
        L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = D - adj_matrix
    L= L.fill_diagonal_(0)
    return L

def get_gau_adj_from_adj(X_unlab, adj_matrix, normalize = False, heat_kernel_t = 10, clamp_value=None):

    # Remove small values
    adj_matrix = torch.square(adj_matrix)

    # adj_matrix = torch.div( adj_matrix, 4*heat_kernel_t)
    # adj_matrix = torch.div( adj_matrix, 0.4)
    adj_matrix = torch.div( adj_matrix, 0.2)

    # adj_matrix= torch.div(adj_matrix, torch.max(adj_matrix))

    adj_matrix = torch.exp(-adj_matrix)
    # adj_matrix= torch.div(adj_matrix, torch.max(adj_matrix))

    # e, V = np.linalg.eig(adj_matrix.cpu().detach().numpy())

    adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements
    # adj_matrix= torch.div(adj_matrix, torch.max(adj_matrix))
    zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
    # adj_matrix = torch.where(adj_matrix < 1e-5,  zero_tensor, adj_matrix)

    if clamp_value!=None:
        # remove large values
        zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
        adj_matrix = torch.where(adj_matrix < clamp_value, adj_matrix, zero_tensor)
    
    # Remove path through obstacles
    zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')

    for i in range(X_unlab.shape[0]):
        for j in range(i+1, X_unlab.shape[0]):
            x1 = X_unlab[i, 0]
            y1 = X_unlab[i, 1]
            x2 = X_unlab[j, 0]
            y2 = X_unlab[j, 1]
            kk = (y2 - y1) / (x2 - x1)
            if (5- x1) * kk + y1 <=10 and (5- x1) * kk + y1 >= 3 and x1 < 5 and x2 > 5:
                # print(adj_matrix[i, j])
                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
            if (5- x1) * kk + y1 <=10 and (5- x1) * kk + y1 >= 3 and x2 < 5 and x1 > 5:
                # print(adj_matrix[i, j])

                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
            if (15- x1) * kk + y1 <= 7 and (15- x1) * kk + y1 >= 0 and x2 < 15 and x1 > 15:
                # print(adj_matrix[i, j])

                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
            if (15- x1) * kk + y1 <= 7 and (15- x1) * kk + y1 >= 0 and x1 < 15 and x2 > 15:
                # print(adj_matrix[i, j])

                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
    return adj_matrix

def get_euclidean_laplacian_from_adj(adj_matrix, normalize = False, clamp_value=None):

    # Remove small values
    adj_matrix = torch.square(adj_matrix)

    # adj_matrix = torch.div( adj_matrix, -4*heat_kernel_t)
    # adj_matrix = torch.exp(adj_matrix)
    # adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements

    if clamp_value!=None:
        zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
        adj_matrix = torch.where(adj_matrix < clamp_value, adj_matrix, zero_tensor)

    if normalize:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0]).to('cuda') # Juan Modified This
        D = torch.diag(1 / torch.sqrt(D))
        L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = D - adj_matrix
    return L


def projsplx(tensor):
    hk1 = np.argsort(tensor)
    vals = tensor[hk1]
    n = len(vals)
    Flag = True
    i = n - 1
    while Flag:
        ti = (torch.sum(vals[i + 1:]) - 1) / (n - i)
        if ti >= vals[i]:
            Flag = False
            that = ti
        else:
            i = i - 1
        if i == 0:
            Flag = False
            that = (torch.sum(vals) - 1) / n
    vals = torch.nn.functional.relu(vals - that)
    vals = vals/torch.sum(vals).item()
    return vals[np.argsort(hk1)]