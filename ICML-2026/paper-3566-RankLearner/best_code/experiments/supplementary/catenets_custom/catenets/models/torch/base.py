import abc
from typing import Optional

import numpy as np
import torch
from torch import nn

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_R,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.torch.utils.decorators import benchmark, check_input_train
from catenets.models.torch.utils.model_utils import make_val_split
from catenets.models.torch.utils.weight_utils import compute_importance_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
}


class BasicNet(nn.Module):
    """
    Basic hypothesis neural net.

    Parameters
    ----------
    n_unit_in: int
        Number of features
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    binary_y: bool, default False
        Whether the outcome is binary. Impacts the loss function.
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
    lr: float
        learning rate for optimizer. step_size equivalent in the JAX version.
    weight_decay: float
        l2 (ridge) penalty for the weights.
    n_iter: int
        Maximum number of iterations.
    batch_size: int
        Batch size
    n_iter_print: int
        Number of iterations after which to print updates and check the validation loss.
    seed: int
        Seed used
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    clipping_value: int, default 1
        Gradients clipping value
    """

    def __init__(
        self,
        name: str,
        n_unit_in: int,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        binary_y: bool = False,
        nonlin: str = DEFAULT_NONLIN,
        lr: float = DEFAULT_STEP_SIZE,
        weight_decay: float = DEFAULT_PENALTY_L2,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        clipping_value: int = 1,
        batch_norm: bool = True,
        early_stopping: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ) -> None:
        super(BasicNet, self).__init__()

        self.name = name
        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]

        if n_layers_out > 0:
            if batch_norm:
                layers = [
                    nn.Linear(n_unit_in, n_units_out),
                    nn.BatchNorm1d(n_units_out),
                    NL(),
                ]
            else:
                layers = [nn.Linear(n_unit_in, n_units_out), NL()]

            # add required number of layers
            for i in range(n_layers_out - 1):
                if dropout:
                    layers.extend([nn.Dropout(dropout_prob)])
                if batch_norm:
                    layers.extend(
                        [
                            nn.Linear(n_units_out, n_units_out),
                            nn.BatchNorm1d(n_units_out),
                            NL(),
                        ]
                    )
                else:
                    layers.extend(
                        [
                            nn.Linear(n_units_out, n_units_out),
                            NL(),
                        ]
                    )

            # add final layers
            layers.append(nn.Linear(n_units_out, 1))
        else:
            layers = [nn.Linear(n_unit_in, 1)]

        if binary_y:
            layers.append(nn.Sigmoid())

        # return final architecture
        self.model = nn.Sequential(*layers).to(DEVICE)
        self.binary_y = binary_y

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def fit(
        self, X: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> "BasicNet":
        self.train()

        X = self._check_tensor(X)
        y = self._check_tensor(y).squeeze()

        # get validation split (can be none)
        X, y, X_val, y_val, val_string = make_val_split(
            X, y, val_split_prop=self.val_split_prop, seed=self.seed
        )
        y_val = y_val.squeeze()
        n = X.shape[0]  # could be different from before due to split

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        # do training
        val_loss_best = LARGE_VAL
        patience = 0
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                self.optimizer.zero_grad()

                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]

                X_next = X[idx_next]
                y_next = y[idx_next]

                weight_next = None
                if weight is not None:
                    weight_next = weight[idx_next].detach()

                loss = nn.BCELoss(weight=weight_next) if self.binary_y else nn.MSELoss()

                preds = self.forward(X_next).squeeze()

                batch_loss = loss(preds, y_next)

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                loss = nn.BCELoss() if self.binary_y else nn.MSELoss()
                with torch.no_grad():
                    preds = self.forward(X_val).squeeze()
                    val_loss = loss(preds, y_val)

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.patience and i > self.n_iter_min:
                            break

                    if i % self.n_iter_print == 0:
                        log.info(
                            f"[{self.name}] Epoch: {i}, current {val_string} loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
                        )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)


class RepresentationNet(nn.Module):
    """
    Basic representation neural net

    Parameters
    ----------
    n_unit_in: int
        Number of features
    n_layers: int
        Number of shared representation layers before hypothesis layers
    n_units: int
        Number of hidden units in each representation layer
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
    """

    def __init__(
        self,
        n_unit_in: int,
        n_layers: int = DEFAULT_LAYERS_R,
        n_units: int = DEFAULT_UNITS_R,
        nonlin: str = DEFAULT_NONLIN,
        batch_norm: bool = True,
    ) -> None:
        super(RepresentationNet, self).__init__()
        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]

        if batch_norm:
            layers = [nn.Linear(n_unit_in, n_units), nn.BatchNorm1d(n_units), NL()]
        else:
            layers = [nn.Linear(n_unit_in, n_units), NL()]
        # add required number of layers
        for i in range(n_layers - 1):
            if batch_norm:
                layers.extend(
                    [nn.Linear(n_units, n_units), nn.BatchNorm1d(n_units), NL()]
                )
            else:
                layers.extend([nn.Linear(n_units, n_units), NL()])

        self.model = nn.Sequential(*layers).to(DEVICE)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class PropensityNet(nn.Module):
    """
    Basic propensity neural net

    Parameters
    ----------
    name: str
        Display name
    n_unit_in: int
        Number of features
    n_unit_out: int
        Number of output features
    weighting_strategy: str
        Weighting strategy
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Dense
        layer)
    nonlin: string, default 'elu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
    lr: float
        learning rate for optimizer. step_size equivalent in the JAX version.
    weight_decay: float
        l2 (ridge) penalty for the weights.
    n_iter: int
        Maximum number of iterations.
    batch_size: int
        Batch size
    n_iter_print: int
        Number of iterations after which to print updates and check the validation loss.
    seed: int
        Seed used
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    clipping_value: int, default 1
        Gradients clipping value
    """

    def __init__(
        self,
        name: str,
        n_unit_in: int,
        n_unit_out: int,
        weighting_strategy: str,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        nonlin: str = DEFAULT_NONLIN,
        lr: float = DEFAULT_STEP_SIZE,
        weight_decay: float = DEFAULT_PENALTY_L2,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        clipping_value: int = 1,
        batch_norm: bool = True,
        early_stopping: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ) -> None:
        super(PropensityNet, self).__init__()
        if nonlin not in list(NONLIN.keys()):
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]

        if batch_norm:
            layers = [
                nn.Linear(in_features=n_unit_in, out_features=n_units_out_prop),
                nn.BatchNorm1d(n_units_out_prop),
                NL(),
            ]
        else:
            layers = [
                nn.Linear(in_features=n_unit_in, out_features=n_units_out_prop),
                NL(),
            ]

        for i in range(n_layers_out_prop - 1):
            if dropout:
                layers.extend([nn.Dropout(dropout_prob)])
            if batch_norm:
                layers.extend(
                    [
                        nn.Linear(
                            in_features=n_units_out_prop, out_features=n_units_out_prop
                        ),
                        nn.BatchNorm1d(n_units_out_prop),
                        NL(),
                    ]
                )
            else:
                layers.extend(
                    [
                        nn.Linear(
                            in_features=n_units_out_prop, out_features=n_units_out_prop
                        ),
                        NL(),
                    ]
                )
        layers.extend(
            [
                nn.Linear(in_features=n_units_out_prop, out_features=n_unit_out),
                nn.Softmax(dim=-1),
            ]
        )

        self.model = nn.Sequential(*layers).to(DEVICE)
        self.name = name
        self.weighting_strategy = weighting_strategy
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def get_importance_weights(
        self, X: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        p_pred = self.forward(X).squeeze()[:, 1]
        return compute_importance_weights(p_pred, w, self.weighting_strategy, {})

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        return nn.NLLLoss()(torch.log(y_pred + EPS), y_target)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "PropensityNet":
        self.train()

        X = self._check_tensor(X)
        y = self._check_tensor(y).long()

        # get validation split (can be none)
        X, y, X_val, y_val, val_string = make_val_split(
            X, y, val_split_prop=self.val_split_prop, seed=self.seed
        )
        y_val = y_val.squeeze()
        n = X.shape[0]  # could be different from before due to split

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        # do training
        val_loss_best = LARGE_VAL
        patience = 0
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                self.optimizer.zero_grad()

                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]

                X_next = X[idx_next]
                y_next = y[idx_next].squeeze()

                preds = self.forward(X_next).squeeze()

                batch_loss = self.loss(preds, y_next)

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                self.optimizer.step()
                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    preds = self.forward(X_val).squeeze()
                    val_loss = self.loss(preds, y_val)

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1
                        if patience > self.patience and (
                            (i + 1) * n_batches > self.n_iter_min
                        ):
                            break
                    if i % self.n_iter_print == 0:
                        log.info(
                            f"[{self.name}] Epoch: {i}, current {val_string} loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
                        )

        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)


class BaseCATEEstimator(nn.Module):
    """
    Interface for estimators of CATE.

    The interface has train/forward API for PyTorch-based models and fit/predict API for sklearn-based models.
    """

    def __init__(
        self,
    ) -> None:
        super(BaseCATEEstimator, self).__init__()

    def score(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """
        Return the sqrt PEHE error (oracle metric).

        Parameters
        ----------
        X: torch.Tensor
            Covariate matrix
        y: torch.Tensor
            Expected potential outcome vector
        """
        X = self._check_tensor(X)
        y = self._check_tensor(y)
        if len(X) != len(y):
            raise ValueError("X/y length mismatch for score")
        if y.shape[-1] != 2:
            raise ValueError(f"y has invalid shape {y.shape}")

        hat_te = self.predict(X)

        return torch.sqrt(torch.mean(((y[:, 1] - y[:, 0]) - hat_te) ** 2))

    @abc.abstractmethod
    @check_input_train
    @benchmark
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "BaseCATEEstimator":
        """
        Train method for a CATEModel

        Parameters
        ----------
        X: torch.Tensor
            Covariate matrix
        y: torch.Tensor
            Outcome vector
        w: torch.Tensor
            Treatment indicator
        """
        ...

    @benchmark
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict treatment effect estimates using a CATE estimator.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Covariate matrix
        Returns
        -------
        potential outcomes probabilities
        """
        return self.predict(X, return_po=False, training=True)

    @abc.abstractmethod
    @benchmark
    def predict(
        self, X: torch.Tensor, return_po: bool = False, training: bool = False
    ) -> torch.Tensor:
        """
        Predict treatment effect estimates using a CATE estimator.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Covariate matrix
        return_po: bool, optional
            Return the potential outcomes too
        Returns
        -------
        potential outcomes probabilities
        """
        ...

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)
