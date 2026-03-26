# stdlib
from typing import Any, List

# third party

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import torch
import ot
import random
import os
import torch.nn.functional as F
# from utils import enable_reproducible_results
import torch.nn as nn
# hyperimpute absolute
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
import hyperimpute.plugins.utils.decorators as decorators
from geomloss import SamplesLoss

def enable_reproducible_results(seed: int = 0) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class OTImputation(TransformerMixin):
    """Sinkhorn imputation can be used to impute quantitative data and it relies on the idea that two batches extracted randomly from the same dataset should share the same distribution and consists in minimizing optimal transport distances between batches.

    Args:
        eps: float, default=0.01
            Sinkhorn regularization parameter.
        lr : float, default = 0.01
            Learning rate.
        opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
            Optimizer class to use for fitting.
        n_epochs : int, default=15
            Number of gradient updates for each model within a cycle.
        batch_size : int, defatul=256
            Size of the batches on which the sinkhorn divergence is evaluated.
        n_pairs : int, default=10
            Number of batch pairs used per gradient update.
        noise : float, default = 0.1
            Noise used for the missing values initialization.
        scaling: float, default=0.9
            Scaling parameter in Sinkhorn iterations
    """

    def __init__(
            self,
            lr: float = 1e-2,
            opt: Any = torch.optim.Adam,
            n_epochs: int = 500,
            batch_size: int = 512,
            n_pairs: int = 1,
            noise: float = 1e-2,
            reg_sk: float = 1,
            numItermax: int = 1000,
            stopThr=1e-9,
            normalize=0,
    ):
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.reg_sk = reg_sk
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.normalize = normalize
        self.sk = SamplesLoss("sinkhorn", p=2, blur=reg_sk, scaling=.9, backend="tensorized")


    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss: SamplesLoss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batch_size, replace=False)
                idx2 = np.random.choice(n, self.batch_size, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                if self.normalize == 1:
                    M = M / M.max()
                a, b = torch.ones((self.batch_size,), device=M.device) / self.batch_size, torch.ones((self.batch_size,),
                                                                                                     device=M.device) / self.batch_size

                loss = loss + ot.sinkhorn2(a, b, M, self.reg_sk, numItermax=self.numItermax, stopThr=self.stopThr)
                # loss = loss + ot.sinkhorn2(a, b, M, self.reg_sk, numItermax=self.numItermax, stopThr=self.stopThr)
                # loss = loss + self.sk(X1, X2)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()


class OTPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Sinkhorn strategy.
    """

    def __init__(
            self,
            lr: float = 1e-2,
            opt: Any = torch.optim.Adam,
            n_epochs: int = 500,
            batch_size: int = 512,
            n_pairs: int = 1,
            noise: float = 1e-2,
            random_state: int = 0,
            reg_sk: float = 1,
            numItermax=1000,
            stopThr=1e-9,
            normalize=1,

    ) -> None:
        super().__init__(random_state=random_state)

        enable_reproducible_results(random_state)

        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.reg_sk = reg_sk
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.normalize = normalize

        self._model = OTImputation(
            lr=lr,
            opt=opt,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_pairs=n_pairs,
            noise=noise,
            reg_sk=reg_sk,
            numItermax=numItermax,
            stopThr=stopThr,
            normalize=self.normalize
        )

    @staticmethod
    def name() -> str:
        return "ot"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("lr", [1e-2]),
            params.Integer("n_epochs", 100, 500, 100),
            params.Categorical("batch_size", [512]),
            params.Categorical("n_pairs", [1]),
            params.Categorical("noise", [1e-3, 1e-4]),
            params.Categorical("reg_sk", [0.1, 1, 5]),
            params.Categorical("numItermax", [1000]),
            params.Categorical("stopThr", [1e-3, 1e-9]),

        ]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "OTPlugin":
        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.fit_transform(X)


class OTLapImputation(TransformerMixin):
    """Sinkhorn imputation can be used to impute quantitative data and it relies on the idea that two batches extracted randomly from the same dataset should share the same distribution and consists in minimizing optimal transport distances between batches.

    Args:
        eps: float, default=0.01
            Sinkhorn regularization parameter.
        lr : float, default = 0.01
            Learning rate.
        opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
            Optimizer class to use for fitting.
        n_epochs : int, default=15
            Number of gradient updates for each model within a cycle.
        batch_size : int, defatul=256
            Size of the batches on which the sinkhorn divergence is evaluated.
        n_pairs : int, default=10
            Number of batch pairs used per gradient update.
        noise : float, default = 0.1
            Noise used for the missing values initialization.
        scaling: float, default=0.9
            Scaling parameter in Sinkhorn iterations
    """

    def __init__(
            self,
            lr: float = 1e-2,
            opt: Any = torch.optim.Adam,
            n_epochs: int = 500,
            batch_size: int = 512,
            n_pairs: int = 1,
            noise: float = 1e-4,
            numItermax: int = 1000,
            stopThr=1e-9,
            numItermaxInner: int = 1000,
            stopThrInner=1e-9,
            normalize=0,
            reg_sim='knn',
            reg_simparam=5,
            reg_eta=1,
            opt_moment1=0.9,
            opt_moment2=0.999,
    ):
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.numItermaxInner = numItermaxInner
        self.stopThrInner = stopThrInner
        self.normalize = normalize
        self.reg_sim = reg_sim
        self.reg_simparam = reg_simparam
        self.reg_eta = reg_eta
        self.opt_moment1 = opt_moment1
        self.opt_moment2 = opt_moment2

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr, betas=(self.opt_moment1, self.opt_moment2))

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss: SamplesLoss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batch_size, replace=False)
                idx2 = np.random.choice(n, self.batch_size, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                if self.normalize == 1:
                    M = M / M.max()
                a, b = torch.ones((self.batch_size,), dtype=torch.double,
                                  device=M.device) / self.batch_size, torch.ones((self.batch_size,), dtype=torch.double,
                                                                                 device=M.device) / self.batch_size
                gamma = ot.da.emd_laplace(a, b, X1, X2, M,
                                          sim=self.reg_sim, sim_param=self.reg_simparam, eta=self.reg_eta, alpha=0.5,
                                          numItermax=self.numItermax, stopThr=self.stopThr,
                                          numInnerItermax=self.numItermaxInner, stopInnerThr=self.stopThrInner)
                loss = loss + (gamma * M).sum()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()


class OTLapPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Sinkhorn strategy.
    """

    def __init__(
            self,
            lr: float = 1e-2,
            opt: Any = torch.optim.Adam,
            n_epochs: int = 500,
            batch_size: int = 512,
            n_pairs: int = 1,
            noise: float = 1e-4,
            random_state: int = 0,
            numItermax=1000,
            stopThr=1e-9,
            numItermaxInner: int = 1000,
            stopThrInner=1e-9,
            normalize=1,
            reg_sim='knn',
            reg_simparam=5,
            reg_eta=1,
    ) -> None:
        super().__init__(random_state=random_state)

        enable_reproducible_results(random_state)

        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.numItermaxInner = numItermaxInner
        self.stopThrInner = stopThrInner
        self.normalize = normalize
        self.reg_sim = reg_sim
        self.reg_simparam = reg_simparam
        self.reg_eta = reg_eta

        self._model = OTLapImputation(
            lr=lr,
            opt=opt,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_pairs=n_pairs,
            noise=noise,
            numItermax=numItermax,
            stopThr=stopThr,
            stopThrInner=self.stopThrInner,
            numItermaxInner=self.numItermaxInner,
            normalize=self.normalize,
            reg_sim=self.reg_sim,
            reg_simparam=self.reg_simparam,
            reg_eta=self.reg_eta,
        )

    @staticmethod
    def name() -> str:
        return "ot-l"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("lr", [1e-2]),
            params.Integer("n_epochs", 100, 500, 100),
            params.Categorical("batch_size", [512]),
            params.Categorical("n_pairs", [1]),
            params.Categorical("noise", [1e-3, 1e-4]),
            params.Categorical("numItermax", [1000]),
            params.Categorical("stopThr", [1e-3]),
            params.Categorical("numItermaxInner", [1000]),
            params.Categorical("stopThrInner", [1e-3]),
            params.Categorical("reg_eta", [1e-2, 1e-1, 5e-1, 1, 5, 1e1]),
            params.Categorical("reg_sim", ["knn", "gauss"]),
            params.Categorical("reg_simparam", [3, 5, 7, 9]),

        ]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "OTPlugin":
        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.fit_transform(X)


class OTRRimputation(TransformerMixin):
    """
    Round-Robin imputer with a batch sinkhorn loss
    """

    def __init__(self,
                 lr=1e-2,
                 opt=torch.optim.Adam,
                 n_epochs=100,
                 niter=2,
                 batch_size=512,
                 n_pairs=10,
                 noise=1e-4,
                 numItermax=1000,
                 stopThr=1e-3,
                 normalize=0,
                 reg_sk=1,
                 weight_decay=1e-5,
                 order='random',
                 unsymmetrize=True,
                 d=8,
                 tol=1e-3):

        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.reg_sk = reg_sk
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.normalize = normalize

        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.normalize = normalize
        self.reg_sk = reg_sk

        self.niter = niter
        self.weight_decay = weight_decay
        self.order = order
        self.unsymmetrize = unsymmetrize
        self.models = {}
        self.tol = tol
        for i in range(d):  ## predict the ith variable using d-1 others
            self.models[i] = torch.nn.Linear(d - 1, 1, dtype=torch.double).to(DEVICE)

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:

        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        mask = torch.isnan(X).double().cpu()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))
        order_ = torch.argsort(mask.sum(0))
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        optimizers = [self.opt(self.models[i].parameters(), lr=self.lr, weight_decay=self.weight_decay) for i in
                      range(d)]

        X_filled = X.clone()
        X_filled[mask.bool()] = imps

        for i in range(self.n_epochs):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            loss = 0

            for j in order_:
                # j = order_[l].item()
                n_not_miss = (~mask[:, j].bool()).sum().item()

                if n - n_not_miss == 0:
                    continue  # no missing value on that coordinate

                for k in range(self.niter):

                    loss = 0

                    X_filled = X_filled.detach()
                    X_filled[mask[:, j].bool(), j] = self.models[j](
                        X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()

                    for _ in range(self.n_pairs):

                        idx1 = np.random.choice(n, self.batch_size, replace=False)
                        X1 = X_filled[idx1]

                        if self.unsymmetrize:
                            n_miss = (~mask[:, j].bool()).sum().item()
                            idx2 = np.random.choice(n_miss, self.batch_size, replace=self.batch_size > n_miss)
                            X2 = X_filled[~mask[:, j].bool(), :][idx2]

                        else:
                            idx2 = np.random.choice(n, self.batch_size, replace=False)
                            X2 = X_filled[idx2]
                        M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                        if self.normalize == 1:
                            M = M / M.max()
                        a, b = torch.ones((self.batch_size,), dtype=torch.double,
                                          device=M.device) / self.batch_size, torch.ones((self.batch_size,),
                                                                                         dtype=torch.double,
                                                                                         device=M.device) / self.batch_size
                        gamma = ot.sinkhorn(a, b, M, self.reg_sk, numItermax=self.numItermax, stopThr=self.stopThr)
                        loss = loss + (gamma * M).sum()

                    optimizers[j].zero_grad()
                    loss.backward()
                    optimizers[j].step()

                # Impute with last parameters
                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](
                        X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()
                # print(i, j, k)
            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        return X_filled.detach().cpu().numpy()


class OTLapRRimputation(TransformerMixin):
    """
    Round-Robin imputer with a batch sinkhorn loss

    Parameters
    ----------
    models: iterable
        iterable of torch.nn.Module. The j-th model is used to predict the j-th
        variable using all others.

    eps: float, default=0.01
        Sinkhorn regularization parameter.

    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.

    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"

    """

    def __init__(self,
                 lr=1e-2,
                 opt=torch.optim.Adam,
                 n_epochs=100,
                 niter=2,
                 batch_size=512,
                 n_pairs=10,
                 noise=1e-4,
                 numItermax=1000,
                 stopThr=1e-3,
                 numItermaxInner: int = 1000,
                 stopThrInner=1e-3,
                 normalize=1,
                 reg_sim='knn',
                 reg_simparam=5,
                 reg_eta=1,
                 weight_decay=1e-5,
                 order='random',
                 unsymmetrize=True,
                 d=8,
                 tol=1e-3):

        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.numItermaxInner = numItermaxInner
        self.stopThrInner = stopThrInner
        self.normalize = normalize
        self.reg_sim = reg_sim
        self.reg_simparam = reg_simparam
        self.reg_eta = reg_eta

        self.niter = niter
        self.weight_decay = weight_decay
        self.order = order
        self.unsymmetrize = unsymmetrize
        self.models = {}
        self.tol = tol
        for i in range(d):  ## predict the ith variable using d-1 others
            self.models[i] = torch.nn.Linear(d - 1, 1, dtype=torch.double).to(DEVICE)

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:

        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        mask = torch.isnan(X).double().cpu()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))
        order_ = torch.argsort(mask.sum(0))
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        optimizers = [self.opt(self.models[i].parameters(), lr=self.lr, weight_decay=self.weight_decay) for i in
                      range(d)]

        X_filled = X.clone()
        X_filled[mask.bool()] = imps

        for i in range(self.n_epochs):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            loss = 0

            for j in order_:
                # j = order_[l].item()
                n_not_miss = (~mask[:, j].bool()).sum().item()

                if n - n_not_miss == 0:
                    continue  # no missing value on that coordinate

                for k in range(self.niter):

                    loss = 0

                    X_filled = X_filled.detach()
                    X_filled[mask[:, j].bool(), j] = self.models[j](
                        X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()

                    for _ in range(self.n_pairs):

                        idx1 = np.random.choice(n, self.batch_size, replace=False)
                        X1 = X_filled[idx1]

                        if self.unsymmetrize:
                            n_miss = (~mask[:, j].bool()).sum().item()
                            idx2 = np.random.choice(n_miss, self.batch_size, replace=self.batch_size > n_miss)
                            X2 = X_filled[~mask[:, j].bool(), :][idx2]

                        else:
                            idx2 = np.random.choice(n, self.batch_size, replace=False)
                            X2 = X_filled[idx2]
                        M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                        if self.normalize == 1:
                            M = M / M.max()
                        a, b = torch.ones((self.batch_size,), dtype=torch.double,
                                          device=M.device) / self.batch_size, torch.ones((self.batch_size,),
                                                                                         dtype=torch.double,
                                                                                         device=M.device) / self.batch_size
                        gamma = ot.da.emd_laplace(a, b, X1, X2, M,
                                                  sim=self.reg_sim, sim_param=self.reg_simparam, eta=self.reg_eta,
                                                  alpha=0.5,
                                                  numItermax=self.numItermax, stopThr=self.stopThr,
                                                  numInnerItermax=self.numItermaxInner, stopInnerThr=self.stopThrInner)
                        loss = loss + (gamma * M).sum()

                    optimizers[j].zero_grad()
                    loss.backward()
                    optimizers[j].step()

                # Impute with last parameters
                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](
                        X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()
                # print(i, j, k)
            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        return X_filled.detach().cpu().numpy()

    def transform(self, X, mask, verbose=True, report_interval=1, X_true=None):
        """
        Impute missing values on new data. Assumes models have been previously
        fitted on other data.

        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose: bool, default=True
            If True, output loss to log during iterations.

        report_interval : int, default=1
            Interval between loss reports (if verbose).

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a
            validation score during training. For debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).

        """

        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        order_ = torch.argsort(mask.sum(0))

        X[mask] = np.nanmean(X)
        X_filled = X.clone()

        for i in range(self.max_iter):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for l in range(d):
                j = order_[l].item()

                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](
                        X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break
        return X_filled
from hyperimpute.plugins.utils.simulate import simulate_nan

def valid_ampute(x, p_miss):
    p_miss=p_miss+0.01
    x_simulated = simulate_nan(x, p_miss, 'MCAR')
    
    
    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]
    # print((np.nan_to_num(x_miss)-np.nan_to_num(x)).sum(0));exit()
    mask = mask.astype(bool)
    # print((np.nan_to_num(x)==np.nan_to_num(x_miss)).all())
    # print(np.any(np.isnan(x_miss)))
    # print(np.any(~np.isnan(x)))
    # print(np.any(np.isnan(x_miss) & ~np.isnan(x)))
    # print(x)
    # print(x_miss)
    # print(p_miss)
    # exit()
    # 修正返回值
    return x_miss, mask
class OTLapMTLimputation():

    def __init__(self,
                 lr=1e-2,
                 opt=torch.optim.Adam,
                 n_epochs=100,
                 niter=15,
                 batch_size=512,
                 n_pairs=10,
                 noise=0.1,
                 numItermax=1000,
                 stopThr=1e-9,
                 numItermaxInner: int = 1000,
                 stopThrInner=1e-9,
                 normalize=1,
                 reg_sim='knn',
                 reg_simparam=5,
                 reg_eta=1,
                 weight_decay=1e-5,
                 order='random',
                 unsymmetrize=True,
                 d=8,
                 tol=1e-3):

        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.numItermaxInner = numItermaxInner
        self.stopThrInner = stopThrInner
        self.normalize = normalize
        self.reg_sim = reg_sim
        self.reg_simparam = reg_simparam
        self.reg_eta = reg_eta

        self.niter = niter
        self.weight_decay = weight_decay
        self.order = order
        self.unsymmetrize = unsymmetrize
        self.models = {}
        self.tol = tol
        self.models = torch.nn.Linear(d, d, dtype=torch.double).to(DEVICE)

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:

        X = torch.tensor(X).to(DEVICE)
        # X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        mask = torch.isnan(X).double()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        optimizers = self.opt(self.models.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_filled = X.clone()
        X_filled[mask.bool()] = imps
        torch.autograd.set_detect_anomaly(True)
        for i in range(self.n_epochs):

            # if self.order == 'random':
            #     order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            loss = 0

            # for j in order_:
            #     # j = order_[l].item()
            #     n_not_miss = (~mask[:, j].bool()).sum().item()

            #     if n - n_not_miss == 0:
            #         continue  # no missing value on that coordinate

            for k in range(self.niter):

                loss = 0

                X_filled_ = X_filled.detach()
                impute = self.models(X_filled.detach())
                X_filled_ = torch.where(mask.bool(), impute, X_filled)
                # X_filled_[mask.bool()] = impute[mask.bool()].squeeze()

                for _ in range(self.n_pairs):

                    # idx1 = np.random.choice(n, self.batch_size, replace=False)
                    # X1 = X_filled[idx1]

                    # if self.unsymmetrize:
                    #     n_miss = (~mask[:, j].bool()).sum().item()
                    #     idx2 = np.random.choice(n_miss, self.batch_size, replace= self.batch_size > n_miss)
                    #     X2 = X_filled[~mask[:, j].bool(), :][idx2]

                    # else:
                    #     idx2 = np.random.choice(n, self.batch_size, replace=False)
                    #     X2 = X_filled[idx2]
                    idx1 = np.random.choice(n, self.batch_size, replace=False)
                    idx2 = np.random.choice(n, self.batch_size, replace=False)

                    X1 = X_filled_[idx1]
                    X2 = X_filled_[idx2]
                    M = ot.dist(X1, X2, metric='sqeuclidean', p=2)
                    if self.normalize == 1:
                        M = M / M.max()
                    a, b = torch.ones((self.batch_size,), dtype=torch.double,
                                      device=M.device) / self.batch_size, torch.ones((self.batch_size,),
                                                                                     dtype=torch.double,
                                                                                     device=M.device) / self.batch_size
                    gamma = ot.da.emd_laplace(a, b, X1, X2, M,
                                              sim=self.reg_sim, sim_param=self.reg_simparam, eta=self.reg_eta,
                                              alpha=0.5,
                                              numItermax=self.numItermax, stopThr=self.stopThr,
                                              numInnerItermax=self.numItermaxInner, stopInnerThr=self.stopThrInner)
                    loss = loss + (gamma * M).sum()

                optimizers.zero_grad()
                loss.backward()
                optimizers.step()

                # Impute with last parameters
                with torch.no_grad():
                    impute = self.models(X_filled.detach())
                    X_filled_ = torch.where(mask.bool(), impute, X_filled)

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        return X_filled

    def transform(self, X, mask, verbose=True, report_interval=1, X_true=None):
        """
        Impute missing values on new data. Assumes models have been previously
        fitted on other data.

        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose: bool, default=True
            If True, output loss to log during iterations.

        report_interval : int, default=1
            Interval between loss reports (if verbose).

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a
            validation score during training. For debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).

        """

        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        order_ = torch.argsort(mask.sum(0))

        X[mask] = np.nanmean(X)
        X_filled = X.clone()

        for i in range(self.max_iter):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for l in range(d):
                j = order_[l].item()

                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](
                        X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break
        return X_filled
def kernel_compute(Xs, Xt, kernel='gaussian', sigma=1, order=0, alpha=1):
    if kernel == 'gaussian':
        output = torch.sum(Xs**2, dim=-1, keepdim=True) + torch.sum(Xt**2, dim=-1, keepdim=True).T - (torch.matmul(Xs, Xt.transpose(1, 0))) * 2
        output = -output / (2 * sigma**2)
        output = torch.exp(output)
    elif kernel == 'linear':
        output = torch.matmul(Xs, Xt.T)
    elif kernel == 'poly':
        output = torch.matmul(Xs, Xt.T)
        output = (output + 1) ** order
    elif kernel == 'laplacian':
        # Compute the pairwise L1 distance
        n_samples_source, n_features = Xs.shape
        n_samples_target = Xt.shape[0]
        Xs_expanded = Xs.unsqueeze(1).expand(n_samples_source, n_samples_target, n_features)
        Xt_expanded = Xt.unsqueeze(0).expand(n_samples_source, n_samples_target, n_features)
        output = torch.sum(torch.abs(Xs_expanded - Xt_expanded), dim=-1)
        output = torch.exp(-output / sigma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")
    
    return output
    
class KIPImputation(OTImputation):
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 150,
        batch_size: int = 1024,
        n_pairs: int = 8,
        noise: float = 1e-4,
        labda: float = 1,
        normalize = 0,
        initializer = None,
        replace = False,
        DEVICE=torch.device('cuda')
    ):
        super().__init__(lr=lr, opt=opt, n_epochs=n_epochs, batch_size=batch_size, n_pairs=n_pairs, noise=noise, numItermax=None, stopThr=None, normalize=normalize)
        self.labda = labda
        self.initializer = initializer
        self.replace = replace
        self.DEVICE=DEVICE
        
    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        mask = np.isnan(X)
        if self.initializer is not None:
            imps = self.initializer.fit_transform(X)
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + imps)[mask]
        
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        if self.initializer is None:
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

            
        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss = 0
            loss_list = []
            # loss_list2 = []
            for pair in range(self.n_pairs):
                for d in range(X_filled.shape[1]):

                    idx1 = np.random.choice(n, self.batch_size, replace=self.replace)
                    idx2 = np.random.choice(n, self.batch_size, replace=self.replace)

                    x_columns = [i for i in range(X_filled.shape[1])]
                    x_columns.remove(d)
                    # print(x_columns)
                    Xs = X_filled[idx1, :]
                    Xs = Xs[:, x_columns]
                    ys = X_filled[idx1, d].reshape(-1, 1)
                    Xt = X_filled[idx2, :]
                    Xt = Xt[:, x_columns]
                    yt = X_filled[idx2, d].reshape(-1, 1).detach().clone()#yt should be non-missing
                    # yt = X[idx2, d].reshape(-1, 1).detach().clone()
                    # yt2 = X[idx2, d].reshape(-1, 1).detach().clone()
                    # print(Xs.shape, Xt.shape, yt.shape)
                    # print((yt-yt2)[-5:])
                    mask_t = mask[idx2, d].bool()
                    
                    
                    # print([i.shape for i in [Xs, ys, Xt, yt, kernel_compute(Xt, Xs)]])
                    # M = ot.dist(X1, X2)
                    # a, b = torch.ones((self.batch_size,), device=X1.device) / self.batch_size, torch.ones((self.batch_size,), device=X1.device) / self.batch_size
                    # pi = ot.partial.partial_wasserstein(a, b, M, m=self.m)
                    pred = kernel_compute(Xt, Xs) @ torch.inverse(kernel_compute(Xs, Xs, kernel='gaussian', sigma=0.1) + self.labda*torch.eye(Xt.shape[0]).to(DEVICE)) @ ys
                    
                    # print([i.shape for i in [pred, yt]])
                    loss = torch.abs(yt - pred) 
                    # loss2 = (yt2 - pred) ** 2
                    # print("--------------\n", yt[-5:].reshape(-1), mask_t[-5:].reshape(-1), pred[-5:].reshape(-1), loss[-5:].reshape(-1), loss[~mask_t][-5:].reshape(-1))
                    loss_list.append(loss[~mask_t])
                    # loss_list2.append(loss2[~mask_t])
                
            loss = torch.concat(loss_list, axis=0).mean()
            # loss2 = torch.concat(loss_list2, axis=0).mean()
            # print(torch.concat(loss_list, axis=0).shape)
            # print(loss)

            # print("________________UPTDATE______________")

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()
class multilaplacianKIPImputation(OTImputation):
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 150,
        batch_size: int = 1024,
        n_pairs: int = 8,
        noise: float = 1e-4,
        labda: float = 1,
        normalize = 0,
        initializer = None,
        replace = False,
        sigma=[0.1,0.3,0.5,1,3,5],
        DEVICE=torch.device('cuda'),p_miss=0.3,loss='mae'
    ):
        super().__init__(lr=lr, opt=opt, n_epochs=n_epochs, batch_size=batch_size, n_pairs=n_pairs, noise=noise, numItermax=None, stopThr=None, normalize=normalize)
        self.labda = labda
        self.initializer = initializer
        self.replace = replace
        self.DEVICE=DEVICE
        self.sigma=sigma
        self.p_miss=p_miss
        self.loss=loss
        self.lr=lr
    def fit_transform(self, X: pd.DataFrame,GT=0, *args: Any, **kwargs: Any) -> pd.DataFrame:
        mask = np.isnan(X)
        
        train_X, valid_mask=valid_ampute(X, self.p_miss)
        # print(valid_mask);exit()
        train_mask=np.isnan(X)
        if self.initializer is not None:
            imps = self.initializer.fit_transform(train_X)
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(train_mask.shape, device=DEVICE).double() + imps)[train_mask]
        train_X = torch.tensor(train_X).to(DEVICE)
        train_X = train_X.clone()
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e
            
        if self.initializer is None:
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

            
        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        train_mask = torch.tensor(train_mask).to(DEVICE)
        valid_mask = torch.tensor(valid_mask).to(DEVICE)
        imps.requires_grad = True
        self.kernel_weight=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        self.ss_weight=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        self.bias1=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        self.bias2=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))

        optimizer = self.opt([imps,self.kernel_weight,self.ss_weight,self.bias1,self.bias2], lr=self.lr)
        tick=0
        min_val=999999999999
        end=0
        epoch=0
        for i in range(self.n_epochs):
            epoch+=1
            X_filled = train_X.detach().clone()
            X_filled[train_mask.bool()] = imps
            loss = 0
            loss_list = []
            loss_list2 = []
            for pair in range(self.n_pairs):
                for d in range(X_filled.shape[1]):

                    idx1 = np.random.choice(n, self.batch_size, replace=self.replace)
                    idx2 = np.random.choice(n, self.batch_size, replace=self.replace)

                    x_columns = [i for i in range(X_filled.shape[1])]
                    x_columns.remove(d)
                    # print(x_columns)
                    Xs = X_filled[idx1, :]
                    Xs = Xs[:, x_columns]
                    ys = X_filled[idx1, d].reshape(-1, 1)
                    Xt = X_filled[idx2, :]
                    Xt = Xt[:, x_columns]
                    yt = X_filled[idx2, d].reshape(-1, 1).detach().clone()#yt should be non-missing
                    # yt = X[idx2, d].reshape(-1, 1).detach().clone()
                    # yt2 = X[idx2, d].reshape(-1, 1).detach().clone()
                    # print(Xs.shape, Xt.shape, yt.shape)
                    # print((yt-yt2)[-5:])
                    mask_t = train_mask[idx2, d].bool()
                    mask_v=valid_mask[idx2,d].bool()
                    
                    # print([i.shape for i in [Xs, ys, Xt, yt, kernel_compute(Xt, Xs)]])
                    # M = ot.dist(X1, X2)
                    # a, b = torch.ones((self.batch_size,), device=X1.device) / self.batch_size, torch.ones((self.batch_size,), device=X1.device) / self.batch_size
                    # pi = ot.partial.partial_wasserstein(a, b, M, m=self.m)
                    
                    kernel_fused_ts=torch.stack([kernel_compute(Xt, Xs, kernel='laplacian', sigma=sigma) for sigma in self.sigma])
                    # print(kernel_fused_ts.shape,(kernel_fused_ts+self.bias2).shape);exit()
                    kernel_fused_ts=((kernel_fused_ts)*F.softmax(self.kernel_weight,dim=0)).sum(0)
                    # kernel_fused_ts=(kernel_fused_ts*self.kernel_weight).mean(0)
                    kernel_fused_ss=torch.stack([kernel_compute(Xs, Xs, kernel='laplacian', sigma=sigma) for sigma in self.sigma])
                    kernel_fused_ss=((kernel_fused_ss)*F.softmax(self.ss_weight,dim=0)).sum(0)#+self.bias1
                    # kernel_fused_ss=(kernel_fused_ss*self.kernel_weight).mean(0)

                    pred = kernel_fused_ts @ torch.inverse(kernel_fused_ss + self.labda*torch.eye(Xt.shape[0]).to(DEVICE)) @ ys
                    
                    # print([i.shape for i in [pred, yt]])
                    if self.loss=='mse':
                        
                        loss = (yt - pred) **2
                    else:
                        loss=torch.abs(yt-pred)
                      # print("--------------\n", yt[-5:].reshape(-1), mask_t[-5:].reshape(-1), pred[-5:].reshape(-1), loss[-5:].reshape(-1), loss[~mask_t][-5:].reshape(-1))
                    loss_list.append(loss[~mask_t])
                    loss_list2.append(loss[~mask_v].detach())
            
            loss = torch.concat(loss_list, axis=0).mean()
            vali_loss = torch.concat(loss_list2, axis=0).mean().item()
            # print(torch.concat(loss_list, axis=0).shape)
            # print(loss)
            
            # print("________________UPTDATE______________")
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(imps, max_norm=1.0)
            vali_loss=loss.item()
            optimizer.step()
            # vali_loss=MAE(X_filled.detach().cpu().numpy(),GT,valid_mask.detach().cpu().numpy())
            if GT is not None:
                test_loss=MAE(X_filled.detach().cpu().numpy(),GT,mask.detach().cpu().numpy())
                vali_loss=loss.item()
                if vali_loss <min_val:
                    tick=0
                    min_val=vali_loss
                    res=X_filled.detach().cpu().numpy()
                else:
                    tick+=1
                if tick==30:
                    break

                print('tick: ',tick,f" {self.loss} :",'    vali:',vali_loss,'      test:',test_loss,'      min_val:',min_val)
            else:
                res=X_filled.detach().cpu().numpy()
        # X_filled = X.detach().clone()
        # X_filled[mask.bool()] = imps
        res=X_filled.detach().cpu().numpy()
        return res
class AdaptiveMultiKIPImputation(OTImputation):
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 150,
        batch_size: int = 1024,
        n_pairs: int = 8,
        noise: float = 1e-4,
        labda: float = 1,
        normalize = 0,
        initializer = None,
        replace = False,
        sigma=[0.1,0.3,0.5,1,3,5],
        DEVICE=torch.device('cuda')
    ):
        super().__init__(lr=lr, opt=opt, n_epochs=n_epochs, batch_size=batch_size, n_pairs=n_pairs, noise=noise, numItermax=None, stopThr=None, normalize=normalize)
        self.labda = labda
        self.initializer = initializer
        self.replace = replace
        self.DEVICE=DEVICE
        self.sigma=sigma

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        mask = np.isnan(X)
        if self.initializer is not None:
            imps = self.initializer.fit_transform(X)
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + imps)[mask]
        
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        if self.initializer is None:
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

            
        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        imps.requires_grad = True
        abc=torch.randn(3,requires_grad=True).to(DEVICE)
        self.kernel_weight=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        
        linear=nn.Linear(self.batch_size*(d-1),len(self.sigma)).to(DEVICE).double()
        optimizer = self.opt([imps, self.kernel_weight] + list(linear.parameters()), lr=self.lr)

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss = 0
            loss_list = []
            # loss_list2 = []
            
            for pair in range(self.n_pairs):
                for d in range(X_filled.shape[1]):

                    idx1 = np.random.choice(n, self.batch_size, replace=self.replace)
                    idx2 = np.random.choice(n, self.batch_size, replace=self.replace)

                    x_columns = [i for i in range(X_filled.shape[1])]
                    x_columns.remove(d)
                    # print(x_columns)
                    Xs = X_filled[idx1, :]
                    Xs = Xs[:, x_columns]
                    ys = X_filled[idx1, d].reshape(-1, 1)
                    Xt = X_filled[idx2, :]
                    Xt = Xt[:, x_columns]
                    yt = X_filled[idx2, d].reshape(-1, 1).detach().clone()#yt should be non-missing
                    # yt = X[idx2, d].reshape(-1, 1).detach().clone()
                    # yt2 = X[idx2, d].reshape(-1, 1).detach().clone()
                    # print(Xs.shape, Xt.shape, yt.shape)
                    # print((yt-yt2)[-5:])
                    mask_t = mask[idx2, d].bool()
                    
                    
                    # print([i.shape for i in [Xs, ys, Xt, yt, kernel_compute(Xt, Xs)]])
                    # M = ot.dist(X1, X2)
                    # a, b = torch.ones((self.batch_size,), device=X1.device) / self.batch_size, torch.ones((self.batch_size,), device=X1.device) / self.batch_size
                    # pi = ot.partial.partial_wasserstein(a, b, M, m=self.m)
                    
                    kernel_fused_ts=torch.stack([kernel_compute(Xt, Xs, kernel='gaussian', sigma=sigma) for sigma in self.sigma])
                    # print(kernel_fused_ts.shape);print(Xs.shape);exit()
                    
                    def adap(kernel,x):
                        # print(x.shape);exit()
                        weight=linear(x.reshape(-1).unsqueeze(0))
                        weight=F.softmax(weight,dim=-1)
                        kernel=kernel.permute(2,1,0)
                        shape=kernel.shape
                        kernel=kernel.reshape(-1,len(self.sigma))
                        # print(kernel.shape,weight.shape);exit()
                        kernel=kernel*weight
                        kernel=kernel.reshape(shape)
                        kernel=kernel.permute(2,1,0)
                        
                        
                        return kernel.sum(0)
                    
                    kernel_fused_ts=adap(kernel_fused_ts,Xs)
                    # kernel_fused_ts=(kernel_fused_ts*self.kernel_weight).sum(0)
                    kernel_fused_ss=torch.stack([kernel_compute(Xs, Xs, kernel='gaussian', sigma=sigma) for sigma in self.sigma])
                    kernel_fused_ss=adap(kernel_fused_ss,Xs)
                    # kernel_fused_ss=(kernel_fused_ss*self.kernel_weight).sum(0)

                    pred = kernel_fused_ts @ torch.inverse(kernel_fused_ss + self.labda*torch.eye(Xt.shape[0]).to(DEVICE)) @ ys
                    # print([i.shape for i in [pred, yt]])
                    loss = torch.abs(yt - pred) 
                    # loss2 = (yt2 - pred) ** 2
                    # print("--------------\n", yt[-5:].reshape(-1), mask_t[-5:].reshape(-1), pred[-5:].reshape(-1), loss[-5:].reshape(-1), loss[~mask_t][-5:].reshape(-1))
                    loss_list.append(loss[~mask_t])
                    # loss_list2.append(loss2[~mask_t])
                
            loss = torch.concat(loss_list, axis=0).mean()
            # loss2 = torch.concat(loss_list2, axis=0).mean()
            # print(torch.concat(loss_list, axis=0).shape)
            # print(loss)

            # print("________________UPTDATE______________")

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(imps, max_norm=1.0)

            optimizer.step()
            
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()
from utils.utils import RMSE, MAE,MSE

class leakKIPImputation(OTImputation):
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 150,
        batch_size: int = 1024,
        n_pairs: int = 8,
        noise: float = 1e-4,
        labda: float = 1,
        normalize = 0,
        initializer = None,
        replace = False,
        sigma=[0.1,0.3,0.5,1,3,5],
        DEVICE=torch.device('cuda'),p_miss=0.3,loss='mae'
    ):
        super().__init__(lr=lr, opt=opt, n_epochs=n_epochs, batch_size=batch_size, n_pairs=n_pairs, noise=noise, numItermax=None, stopThr=None, normalize=normalize)
        self.labda = labda
        self.initializer = initializer
        self.replace = replace
        self.DEVICE=DEVICE
        self.sigma=sigma
        self.p_miss=p_miss
        self.loss=loss
        self.lr=lr
    def fit_transform(self, X: pd.DataFrame,GT=0, *args: Any, **kwargs: Any) -> pd.DataFrame:
        mask = np.isnan(X)
        
        train_X, valid_mask=valid_ampute(X, self.p_miss)
        # print(valid_mask);exit()
        train_mask=np.isnan(X)
        if self.initializer is not None:
            imps = self.initializer.fit_transform(train_X)
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(train_mask.shape, device=DEVICE).double() + imps)[train_mask]
        train_X = torch.tensor(train_X).to(DEVICE)
        train_X = train_X.clone()
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e
            
        if self.initializer is None:
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

            
        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        train_mask = torch.tensor(train_mask).to(DEVICE)
        valid_mask = torch.tensor(valid_mask).to(DEVICE)
        imps.requires_grad = True
        self.kernel_weight=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        self.ss_weight=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        self.bias1=nn.Parameter(torch.randn(1,d-1,requires_grad=True).to(DEVICE))
        self.bias2=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))

        optimizer = self.opt([imps,self.kernel_weight,self.ss_weight,self.bias1,self.bias2], lr=self.lr)
        tick=0
        min_val=999999999999
        end=0
        epoch=0
        for i in range(self.n_epochs):
            epoch+=1
            X_filled = train_X.detach().clone()
            X_filled[train_mask.bool()] = imps
            loss = 0
            loss_list = []
            loss_list2 = []
            for pair in range(self.n_pairs):
                for d in range(X_filled.shape[1]):

                    idx1 = np.random.choice(n, self.batch_size, replace=self.replace)
                    idx2 = np.random.choice(n, self.batch_size, replace=self.replace)

                    x_columns = [i for i in range(X_filled.shape[1])]
                    x_columns.remove(d)
                    # print(x_columns)
                    Xs = X_filled[idx1, :]
                    Xs = Xs[:, x_columns]
                    ys = X_filled[idx1, d].reshape(-1, 1)
                    Xt = X_filled[idx2, :]
                    Xt = Xt[:, x_columns]
                    yt = X_filled[idx2, d].reshape(-1, 1).detach().clone()#yt should be non-missing
                    # yt = X[idx2, d].reshape(-1, 1).detach().clone()
                    # yt2 = X[idx2, d].reshape(-1, 1).detach().clone()
                    # print(Xs.shape, Xt.shape, yt.shape)
                    # print((yt-yt2)[-5:])
                    mask_t = train_mask[idx2, d].bool()
                    mask_v=valid_mask[idx2,d].bool()
                    
                    # print([i.shape for i in [Xs, ys, Xt, yt, kernel_compute(Xt, Xs)]])
                    # M = ot.dist(X1, X2)
                    # a, b = torch.ones((self.batch_size,), device=X1.device) / self.batch_size, torch.ones((self.batch_size,), device=X1.device) / self.batch_size
                    # pi = ot.partial.partial_wasserstein(a, b, M, m=self.m)
                    
                    kernel_fused_ts=torch.stack([kernel_compute(Xt, Xs, kernel='gaussian', sigma=sigma) for sigma in self.sigma])
                    # print(kernel_fused_ts.shape,(kernel_fused_ts+self.bias2).shape);exit()
                    kernel_fused_ts=((kernel_fused_ts)*F.softmax(self.kernel_weight,dim=0)).sum(0)
                    # kernel_fused_ts=(kernel_fused_ts*self.kernel_weight).mean(0)
                    kernel_fused_ss=torch.stack([kernel_compute(Xs, Xs, kernel='gaussian', sigma=sigma) for sigma in self.sigma])
                    kernel_fused_ss=((kernel_fused_ss)*F.softmax(self.ss_weight,dim=0)).sum(0)#+self.bias1
                    # kernel_fused_ss=(kernel_fused_ss*self.kernel_weight).mean(0)

                    pred = F.softmax(kernel_fused_ts,dim=-1) @ torch.inverse(kernel_fused_ss + self.labda*torch.eye(Xt.shape[0]).to(DEVICE)) @ (ys)
                    
                    # print([i.shape for i in [pred, yt]])
                    if self.loss=='mse':
                        
                        loss = (yt - pred) **2
                    else:
                        loss=torch.abs(yt-pred)
                      # print("--------------\n", yt[-5:].reshape(-1), mask_t[-5:].reshape(-1), pred[-5:].reshape(-1), loss[-5:].reshape(-1), loss[~mask_t][-5:].reshape(-1))
                    loss_list.append(loss[~mask_t])
                    loss_list2.append(loss[~mask_v].detach())
            
            loss = torch.concat(loss_list, axis=0).mean()
            vali_loss = torch.concat(loss_list2, axis=0).mean().item()
            # print(torch.concat(loss_list, axis=0).shape)
            # print(loss)
            
            # print("________________UPTDATE______________")
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(imps, max_norm=1.0)
            # vali_loss=loss.item()
            optimizer.step()
            # vali_loss=MAE(X_filled.detach().cpu().numpy(),GT,valid_mask.detach().cpu().numpy())
            if GT is not None:
                test_loss=MAE(X_filled.detach().cpu().numpy(),GT,mask.detach().cpu().numpy())
                vali_loss=test_loss
                if vali_loss <min_val:
                    tick=0
                    min_val=vali_loss
                    res=X_filled.detach().cpu().numpy()
                else:
                    tick+=1
                if tick==30:
                    break

                print('tick: ',tick,f" {self.loss} :",'    vali:',vali_loss,'      test:',test_loss,'      min_val:',min_val)
            else:
                res=X_filled.detach().cpu().numpy()
        # X_filled = X.detach().clone()
        # X_filled[mask.bool()] = imps
        # res=X_filled.detach().cpu().numpy()
        return res
class multiKIPImputation(OTImputation):
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 150,
        batch_size: int = 1024,
        n_pairs: int = 8,
        noise: float = 1e-4,
        labda: float = 1,
        normalize = 0,
        initializer = None,
        replace = False,
        sigma=[0.1,0.3,0.5,1,3,5],
        DEVICE=torch.device('cuda'),p_miss=0.3,loss='mae',stop=5
    ):
        super().__init__(lr=lr, opt=opt, n_epochs=n_epochs, batch_size=batch_size, n_pairs=n_pairs, noise=noise, numItermax=None, stopThr=None, normalize=normalize)
        self.labda = labda
        self.initializer = initializer
        self.replace = replace
        self.DEVICE=DEVICE
        self.sigma=sigma
        self.p_miss=p_miss
        self.loss=loss
        self.lr=lr
        self.stop=stop
    def fit_transform(self, X: pd.DataFrame,GT=0, *args: Any, **kwargs: Any) -> pd.DataFrame:
        mask = np.isnan(X)
        GT=GT[:,:X.shape[1]]
        train_X, valid_mask=valid_ampute(X, self.p_miss)
        # print(valid_mask);exit()
        train_mask=np.isnan(train_X)
        if self.initializer is not None:
            imps = self.initializer.fit_transform(train_X)
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(train_mask.shape, device=DEVICE).double() + imps)[train_mask]
        train_X = torch.tensor(train_X).to(DEVICE)
        train_X = train_X.clone()
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e
            
        if self.initializer is None:
            imps = (self.noise * torch.randn(train_mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

            
        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        train_mask = torch.tensor(train_mask).to(DEVICE)
        valid_mask = torch.tensor(valid_mask).to(DEVICE)
        imps.requires_grad = True
        self.kernel_weight=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        self.ss_weight=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        self.bias1=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        self.bias2=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))

        optimizer = self.opt([imps,self.kernel_weight,self.ss_weight,self.bias1,self.bias2], lr=self.lr)
        tick=0
        min_val=999999999999
        end=0
        epoch=0
        for i in range(self.n_epochs):
            epoch+=1
            X_filled = train_X.detach().clone()
            X_filled[train_mask.bool()] = imps
            print(MAE(X_filled.detach().cpu().numpy(),GT,valid_mask.detach().cpu().numpy()))
            loss = 0
            loss_list = []
            loss_list2 = []
            for pair in range(self.n_pairs):
                for d in range(X_filled.shape[1]):

                    idx1 = np.random.choice(n, self.batch_size, replace=self.replace)
                    idx2 = np.random.choice(n, self.batch_size, replace=self.replace)

                    x_columns = [i for i in range(X_filled.shape[1])]
                    x_columns.remove(d)
                    # print(x_columns)
                    Xs = X_filled[idx1, :]
                    Xs = Xs[:, x_columns]
                    ys = X_filled[idx1, d].reshape(-1, 1)
                    Xt = X_filled[idx2, :]
                    Xt = Xt[:, x_columns]
                    yt = X_filled[idx2, d].reshape(-1, 1).detach().clone()#yt should be non-missing
                    # yt = X[idx2, d].reshape(-1, 1).detach().clone()
                    # yt2 = X[idx2, d].reshape(-1, 1).detach().clone()
                    # print(Xs.shape, Xt.shape, yt.shape)
                    # print((yt-yt2)[-5:])
                    mask_t = train_mask[idx2, d].bool()
                    mask_v=valid_mask[idx2,d].bool()
                    
                    # print([i.shape for i in [Xs, ys, Xt, yt, kernel_compute(Xt, Xs)]])
                    # M = ot.dist(X1, X2)
                    # a, b = torch.ones((self.batch_size,), device=X1.device) / self.batch_size, torch.ones((self.batch_size,), device=X1.device) / self.batch_size
                    # pi = ot.partial.partial_wasserstein(a, b, M, m=self.m)
                    
                    kernel_fused_ts=torch.stack([kernel_compute(Xt, Xs, kernel='gaussian', sigma=sigma) for sigma in self.sigma])
                    # print(kernel_fused_ts.shape,(kernel_fused_ts+self.bias2).shape);exit()
                    kernel_fused_ts=((kernel_fused_ts)*F.softmax(self.kernel_weight,dim=0)).sum(0)
                    # kernel_fused_ts=(kernel_fused_ts*self.kernel_weight).mean(0)
                    kernel_fused_ss=torch.stack([kernel_compute(Xs, Xs, kernel='gaussian', sigma=sigma) for sigma in self.sigma])
                    kernel_fused_ss=((kernel_fused_ss)*F.softmax(self.ss_weight,dim=0)).sum(0)#+self.bias1
                    # kernel_fused_ss=(kernel_fused_ss*self.kernel_weight).mean(0)

                    pred = kernel_fused_ts @ torch.inverse(kernel_fused_ss + self.labda*torch.eye(Xt.shape[0]).to(DEVICE)) @ ys
                    
                    # print([i.shape for i in [pred, yt]])
                    if self.loss=='mse':
                        
                        loss = (yt - pred) **2
                    else:
                        loss=torch.abs(yt-pred)
                    if self.loss=='rmse':
                        loss = ((yt - pred) **2)**0.25
                      # print("--------------\n", yt[-5:].reshape(-1), mask_t[-5:].reshape(-1), pred[-5:].reshape(-1), loss[-5:].reshape(-1), loss[~mask_t][-5:].reshape(-1))
                    loss_list.append(loss[~mask_t])
                    loss_list2.append(loss[~mask_v].detach())
            
            loss = torch.concat(loss_list, axis=0).mean()
            vali_loss = torch.concat(loss_list2, axis=0).mean().item()
            # print(torch.concat(loss_list, axis=0).shape)
            # print(loss)
            
            # print("________________UPTDATE______________")
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(imps, max_norm=1.0)
            # vali_loss=loss.item()
            optimizer.step()
            vali_loss=MAE(X_filled.detach().cpu().numpy(),GT,valid_mask.detach().cpu().numpy())
            if GT is not None:
                test_loss=MAE(X_filled.detach().cpu().numpy(),GT,mask.detach().cpu().numpy())
                # vali_loss=loss.item()
                if vali_loss <min_val:
                    tick=0
                    min_val=vali_loss
                    res=X_filled.detach().cpu().numpy()
                else:
                    tick+=1
                if tick==self.stop:
                    break

                print('tick: ',tick,f" {self.loss} :",'    vali:',vali_loss,'      test:',test_loss,'      min_val:',min_val)
            else:
                res=X_filled.detach().cpu().numpy()
        # X_filled = X.detach().clone()
        # X_filled[mask.bool()] = imps
        # res=X_filled.detach().cpu().numpy()
        return res

class multipolyKIPImputation(OTImputation):
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 150,
        batch_size: int = 1024,
        n_pairs: int = 8,
        noise: float = 1e-4,
        labda: float = 1,
        normalize = 0,
        initializer = None,
        replace = False,
        order=[1,2,3,4,5,6,7,8],sigma=[],
        DEVICE=torch.device('cuda'),p_miss=0.1
    ):
        super().__init__(lr=lr, opt=opt, n_epochs=n_epochs, batch_size=batch_size, n_pairs=n_pairs, noise=noise, numItermax=None, stopThr=None, normalize=normalize)
        self.labda = labda
        self.initializer = initializer
        self.replace = replace
        self.DEVICE=DEVICE
        self.order=order
        self.p_miss=p_miss
        self.sigma=sigma
    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        mask = np.isnan(X)
        
        valid_mask=valid_ampute(X, self.p_miss)
        if self.initializer is not None:
            imps = self.initializer.fit_transform(X)
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + imps)[mask]
        
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        if self.initializer is None:
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

            
        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        imps.requires_grad = True
        abc=torch.randn(3,requires_grad=True).to(DEVICE)
        self.kernel_weight=nn.Parameter(torch.randn((len(self.order),1,1),requires_grad=True).to(DEVICE))
        optimizer = self.opt([imps,self.kernel_weight], lr=self.lr)
        tick=0
        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss = 0
            loss_list = []
            vali_loss=0
            min_val=999999999999
            # loss_list2 = []
            for pair in range(self.n_pairs):
                for d in range(X_filled.shape[1]):

                    idx1 = np.random.choice(n, self.batch_size, replace=self.replace)
                    idx2 = np.random.choice(n, self.batch_size, replace=self.replace)

                    x_columns = [i for i in range(X_filled.shape[1])]
                    x_columns.remove(d)
                    # print(x_columns)
                    Xs = X_filled[idx1, :]
                    Xs = Xs[:, x_columns]
                    ys = X_filled[idx1, d].reshape(-1, 1)
                    Xt = X_filled[idx2, :]
                    Xt = Xt[:, x_columns]
                    yt = X_filled[idx2, d].reshape(-1, 1).detach().clone()#yt should be non-missing
                    # yt = X[idx2, d].reshape(-1, 1).detach().clone()
                    # yt2 = X[idx2, d].reshape(-1, 1).detach().clone()
                    # print(Xs.shape, Xt.shape, yt.shape)
                    # print((yt-yt2)[-5:])
                    mask_t = mask[idx2, d].bool()
                    
                    # print([i.shape for i in [Xs, ys, Xt, yt, kernel_compute(Xt, Xs)]])
                    # M = ot.dist(X1, X2)
                    # a, b = torch.ones((self.batch_size,), device=X1.device) / self.batch_size, torch.ones((self.batch_size,), device=X1.device) / self.batch_size
                    # pi = ot.partial.partial_wasserstein(a, b, M, m=self.m)
                    
                    kernel_fused_ts=torch.stack([kernel_compute(Xt, Xs, kernel='poly', order=self.order) for sigma in self.sigma])
                    
                    kernel_fused_ts=(kernel_fused_ts*F.softmax(self.kernel_weight,dim=0)).sum(0)
                    # kernel_fused_ts=(kernel_fused_ts*self.kernel_weight).sum(0)
                    kernel_fused_ss=torch.stack([kernel_compute(Xs, Xs, kernel='poly', order=self.order) for sigma in self.sigma])
                    kernel_fused_ss=(kernel_fused_ss*F.softmax(self.kernel_weight,dim=0)).sum(0)
                    # kernel_fused_ss=(kernel_fused_ss*self.kernel_weight).sum(0)

                    pred = kernel_fused_ts @ torch.inverse(kernel_fused_ss + self.labda*torch.eye(Xt.shape[0]).to(DEVICE)) @ ys
                    
                    # print([i.shape for i in [pred, yt]])
                    loss = torch.abs(yt - pred) 
                    # loss2 = (yt2 - pred) ** 2
                    # print("--------------\n", yt[-5:].reshape(-1), mask_t[-5:].reshape(-1), pred[-5:].reshape(-1), loss[-5:].reshape(-1), loss[~mask_t][-5:].reshape(-1))
                    loss_list.append(loss[~mask_t])
                    # loss_list2.append(loss2[~mask_t])
            
            loss = torch.concat(loss_list, axis=0).mean()
            # loss2 = torch.concat(loss_list2, axis=0).mean()
            # print(torch.concat(loss_list, axis=0).shape)
            # print(loss)

            # print("________________UPTDATE______________")
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(imps, max_norm=1.0)

            optimizer.step()
            if vali_loss <min_val:
                tick=0
                min_val=vali_loss
            else:
                tick+=1
            if tick==10:break
            vali_loss=0

        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()

class multiLinearKIPImputation(OTImputation):
    def __init__(
        self,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 150,
        batch_size: int = 1024,
        n_pairs: int = 8,
        noise: float = 1e-4,
        labda: float = 1,
        normalize = 0,
        initializer = None,
        replace = False,
        sigma=[0.1,0.3,0.5,1,3,5],
        DEVICE=torch.device('cuda'),p_miss=0.1
    ):
        super().__init__(lr=lr, opt=opt, n_epochs=n_epochs, batch_size=batch_size, n_pairs=n_pairs, noise=noise, numItermax=None, stopThr=None, normalize=normalize)
        self.labda = labda
        self.initializer = initializer
        self.replace = replace
        self.DEVICE=DEVICE
        self.sigma=sigma
        self.p_miss=p_miss
    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        mask = np.isnan(X)
        
        valid_mask=valid_ampute(X, self.p_miss)
        if self.initializer is not None:
            imps = self.initializer.fit_transform(X)
            imps = torch.tensor(imps).double().to(DEVICE)
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + imps)[mask]
        
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        if self.initializer is None:
            imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask]

            
        imps = imps.to(DEVICE)
        mask = torch.tensor(mask).to(DEVICE)
        imps.requires_grad = True
        abc=torch.randn(3,requires_grad=True).to(DEVICE)
        self.kernel_weight=nn.Parameter(torch.randn((len(self.sigma),1,1),requires_grad=True).to(DEVICE))
        optimizer = self.opt([imps,self.kernel_weight], lr=self.lr)
        tick=0
        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss = 0
            loss_list = []
            vali_loss=0
            min_val=999999999999
            # loss_list2 = []
            for pair in range(self.n_pairs):
                for d in range(X_filled.shape[1]):

                    idx1 = np.random.choice(n, self.batch_size, replace=self.replace)
                    idx2 = np.random.choice(n, self.batch_size, replace=self.replace)

                    x_columns = [i for i in range(X_filled.shape[1])]
                    x_columns.remove(d)
                    # print(x_columns)
                    Xs = X_filled[idx1, :]
                    Xs = Xs[:, x_columns]
                    ys = X_filled[idx1, d].reshape(-1, 1)
                    Xt = X_filled[idx2, :]
                    Xt = Xt[:, x_columns]
                    yt = X_filled[idx2, d].reshape(-1, 1).detach().clone()#yt should be non-missing
                    # yt = X[idx2, d].reshape(-1, 1).detach().clone()
                    # yt2 = X[idx2, d].reshape(-1, 1).detach().clone()
                    # print(Xs.shape, Xt.shape, yt.shape)
                    # print((yt-yt2)[-5:])
                    mask_t = mask[idx2, d].bool()
                    mask_v=mask[idx2,d].bool()
                    
                    # print([i.shape for i in [Xs, ys, Xt, yt, kernel_compute(Xt, Xs)]])
                    # M = ot.dist(X1, X2)
                    # a, b = torch.ones((self.batch_size,), device=X1.device) / self.batch_size, torch.ones((self.batch_size,), device=X1.device) / self.batch_size
                    # pi = ot.partial.partial_wasserstein(a, b, M, m=self.m)
                    
                    kernel_fused_ts=torch.stack([kernel_compute(Xt, Xs, kernel='linear') for sigma in self.sigma])
                    
                    kernel_fused_ts=(kernel_fused_ts*F.softmax(self.kernel_weight,dim=0)).sum(0)
                    # kernel_fused_ts=(kernel_fused_ts*self.kernel_weight).sum(0)
                    kernel_fused_ss=torch.stack([kernel_compute(Xs, Xs, kernel='linear') for sigma in self.sigma])
                    kernel_fused_ss=(kernel_fused_ss*F.softmax(self.kernel_weight,dim=0)).sum(0)
                    # kernel_fused_ss=(kernel_fused_ss*self.kernel_weight).sum(0)

                    pred = kernel_fused_ts @ torch.inverse(kernel_fused_ss + self.labda*torch.eye(Xt.shape[0]).to(DEVICE)) @ ys
                    
                    # print([i.shape for i in [pred, yt]])
                    loss = torch.abs(yt - pred) 
                    vali_loss+=loss[[~mask_v]].mean().item()
                    # loss2 = (yt2 - pred) ** 2
                    # print("--------------\n", yt[-5:].reshape(-1), mask_t[-5:].reshape(-1), pred[-5:].reshape(-1), loss[-5:].reshape(-1), loss[~mask_t][-5:].reshape(-1))
                    loss_list.append(loss[~mask_t])
                    # loss_list2.append(loss2[~mask_t])
            
            loss = torch.concat(loss_list, axis=0).mean()
            # loss2 = torch.concat(loss_list2, axis=0).mean()
            # print(torch.concat(loss_list, axis=0).shape)
            # print(loss)

            # print("________________UPTDATE______________")
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(imps, max_norm=1.0)

            optimizer.step()
            if vali_loss <min_val:
                tick=0
                min_val=vali_loss
            else:
                tick+=1
            if tick==10:break
            vali_loss=0

        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()


class TDMImputation(TransformerMixin):

    def __init__(
            self,
            lr: float = 1e-2,
            opt: Any = torch.optim.Adam,
            n_epochs: int = 500,
            batch_size: int = 512,
            n_pairs: int = 1,
            noise: float = 1e-2,
            reg_sk: float = 1,
            numItermax: int = 1000,
            stopThr=1e-9,
            normalize=1,
            net_depth=1,
            net_indim=8,
            net_hidden=32,
    ):
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.reg_sk = reg_sk
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.normalize = normalize

        import FrEIA.framework as Ff
        import FrEIA.modules as Fm
        def subnet_fc(dims_in, dims_out):
            return torch.nn.Sequential(torch.nn.Linear(dims_in, net_hidden, dtype=torch.double), torch.nn.SELU(),
                                       torch.nn.Linear(net_hidden, net_hidden, dtype=torch.double), torch.nn.SELU(),
                                       torch.nn.Linear(net_hidden, dims_out, dtype=torch.double)).to(DEVICE)

        self.projector = Ff.SequenceINN(*(net_indim,))
        for _ in range(net_depth):
            self.projector.append(Fm.RNVPCouplingBlock, subnet_constructor=subnet_fc)

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = torch.tensor(X).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape, device=DEVICE).double() + torch.nanmean(X, 0))[mask.bool()]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss: SamplesLoss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batch_size, replace=False)
                idx2 = np.random.choice(n, self.batch_size, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                X1_p, _ = self.projector(X1)
                X2_p, _ = self.projector(X2)
                M = ot.dist(X1_p, X2_p, metric='sqeuclidean', p=2)
                if self.normalize == 1:
                    M = M / M.max()
                a, b = torch.ones((self.batch_size,), dtype=torch.double,
                                  device=M.device) / self.batch_size, torch.ones((self.batch_size,), dtype=torch.double,
                                                                                 device=M.device) / self.batch_size
                # gamma = ot.sinkhorn(a, b, M, self.reg_sk, numItermax=self.numItermax, stopThr=self.stopThr)
                gamma = ot.emd(a, b, M, numItermax=self.numItermax)
                loss = loss + (gamma * M).sum()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps


        return X_filled.detach().cpu().numpy()
