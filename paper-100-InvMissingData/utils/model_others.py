from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from sklearn.base import TransformerMixin
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from typing import Any, List
import torch
import torch.nn as nn
from hyperimpute.plugins.core.device import DEVICE
import numpy as np


class KNNImputation(TransformerMixin):
    def __init__(self, k=3, weights="distance",metric="nan_euclidean"):
        super().__init__()
        self._model = KNNImputer(n_neighbors=k, weights=weights,metric=metric)

    def fit_transform(self, X: pd.DataFrame, *args, **kwargs):
        model = self._model.fit(X)
        X_filled = model.transform(X)
        return X_filled

class MissForestImputation(TransformerMixin):
    def __init__(self, n_trees=3, max_depth=3, min_samples_split=2, max_iter=100, tol=1e-3, random_state=0):
        super().__init__()
        estimator = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        self._model = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            tol=tol,
            initial_strategy='mean',
            imputation_order='ascending',
            sample_posterior=False,
            random_state=random_state
            )
    def fit_transform(self, X: pd.DataFrame, *args, **kwargs):
        model = self._model.fit(X)
        X_filled = model.transform(X)
        return X_filled
    

class MF(nn.Module):
    def __init__(self, sample_num, feature_num, embedding_size):
        super().__init__()
        self.sample_embedding = torch.nn.Embedding(sample_num, embedding_size)
        self.sample_bias = torch.nn.Embedding(sample_num, 1)
        self.feature_embedding = torch.nn.Embedding(feature_num, embedding_size)
        self.feature_bias = torch.nn.Embedding(feature_num, 1)

    def forward(self, x):
        sample_embed = self.sample_embedding(x[:, 0])
        feature_embed = self.feature_embedding(x[:, 1])
        score = (sample_embed * feature_embed).mean(1) + self.sample_bias(x[:, 0]) + self.feature_bias(x[:, 1])
        return torch.sigmoid(score)
    
class MFImputation(TransformerMixin):
    
    def __init__(
        self,
        sample_num: int = 10000,
        feature_num: int = 1000,
        lr: float = 1e-2,
        opt: Any = torch.optim.SGD,
        n_epochs: int = 500,
        batch_size: int = 512,
        embedding_size: int = 4,
        
    ):
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = MF(sample_num, feature_num, embedding_size).to(DEVICE)
        

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        mask = torch.isnan(X).double().cpu()
        mask = mask.to(DEVICE)
        idx = torch.nonzero(torch.isnan(X)==False)

        optimizer = self.opt(self.model.parameters(), lr=self.lr)

        for _ in range(self.n_epochs):
            loss = 0
            _idx = np.random.choice(len(idx), self.batch_size, replace=False)
            _idx = idx[_idx]
            _X_imp = self.model(_idx)
            _X = torch.tensor([X[row, col] for row, col in _idx], device=DEVICE)
            loss = torch.nn.functional.mse_loss(_X_imp, _X)
            reg = (self.model.sample_embedding(_idx[:, 0])**2).mean() 
            + (self.model.feature_embedding(_idx[:, 1])**2).mean() 
            + (self.model.sample_bias(_idx[:, 0])**2).mean() 
            + (self.model.feature_bias(_idx[:, 1])**2).mean()
            loss += 1e3 * reg

            # if torch.isnan(loss).any() or torch.isinf(loss).any():
            #     break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        X_filled = X.detach().clone()
        sample_embed = self.model.sample_embedding(torch.tensor([i for i in range(n)], device=DEVICE))
        feature_embed = self.model.feature_embedding(torch.tensor([i for i in range(d)], device=DEVICE))
        sample_bias = self.model.sample_bias(torch.tensor([i for i in range(n)], device=DEVICE))
        feature_bias = self.model.feature_bias(torch.tensor([i for i in range(d)], device=DEVICE))
        X_imp = torch.sigmoid(torch.matmul(sample_embed, feature_embed.transpose(0, 1)) + sample_bias.reshape(n, 1) + feature_bias.reshape(1, d))
        X_filled[mask.bool()] = X_imp[mask.bool()]

        return X_filled.detach().cpu().numpy()