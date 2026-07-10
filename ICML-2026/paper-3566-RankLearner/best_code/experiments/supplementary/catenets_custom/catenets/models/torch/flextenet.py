from typing import Any, Callable, List

import numpy as np
import torch
from torch import nn

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIM_P_OUT,
    DEFAULT_DIM_P_R,
    DEFAULT_DIM_S_OUT,
    DEFAULT_DIM_S_R,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_PENALTY_ORTHOGONAL,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.torch.base import DEVICE, BaseCATEEstimator
from catenets.models.torch.utils.model_utils import make_val_split


class FlexTELinearLayer(nn.Module):
    """Layer constructor function for a fully-connected layer. Adapted to allow passing
    treatment indicator through layer without using it"""

    def __init__(
        self,
        name: str,
        dropout: bool = False,
        dropout_prob: float = 0.5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(FlexTELinearLayer, self).__init__()
        self.name = name
        if dropout:
            self.model = nn.Sequential(
                nn.Dropout(dropout_prob), nn.Linear(*args, **kwargs)
            ).to(DEVICE)
        else:
            self.model = nn.Sequential(nn.Linear(*args, **kwargs)).to(DEVICE)

    def forward(self, tensors: List[torch.Tensor]) -> List:
        if len(tensors) != 2:
            raise ValueError(
                "Invalid number of tensor for the FlexLinearLayer layer. It requires the features vector and the treatments vector"
            )

        features_vector = tensors[0]
        treatments_vector = tensors[1]

        return [self.model(features_vector), treatments_vector]


class FlexTESplitLayer(nn.Module):
    """
    Create multitask layer has shape [shared, private_0, private_1]
    """

    def __init__(
        self,
        name: str,
        n_units_in: int,
        n_units_in_p: int,
        n_units_s: int,
        n_units_p: int,
        first_layer: bool,
        dropout: bool = False,
        dropout_prob: float = 0.5,
    ) -> None:
        super(FlexTESplitLayer, self).__init__()
        self.name = name
        self.first_layer = first_layer
        self.n_units_in = n_units_in
        self.n_units_in_p = n_units_in_p
        self.n_units_s = n_units_s
        self.n_units_p = n_units_p

        if dropout:
            self.net_shared = nn.Sequential(
                nn.Dropout(dropout_prob), nn.Linear(n_units_in, n_units_s)
            ).to(DEVICE)
            self.net_p0 = nn.Sequential(
                nn.Dropout(dropout_prob), nn.Linear(n_units_in_p, n_units_p)
            ).to(DEVICE)
            self.net_p1 = nn.Sequential(
                nn.Dropout(dropout_prob), nn.Linear(n_units_in_p, n_units_p)
            ).to(DEVICE)
        else:
            self.net_shared = nn.Sequential(nn.Linear(n_units_in, n_units_s)).to(DEVICE)
            self.net_p0 = nn.Sequential(nn.Linear(n_units_in_p, n_units_p)).to(DEVICE)
            self.net_p1 = nn.Sequential(nn.Linear(n_units_in_p, n_units_p)).to(DEVICE)

    def forward(self, tensors: List[torch.Tensor]) -> List:
        if self.first_layer and len(tensors) != 2:
            raise ValueError(
                "Invalid number of tensor for the FlexSplitLayer layer. It requires the features vector and the treatments vector"
            )
        if not self.first_layer and len(tensors) != 4:
            raise ValueError(
                "Invalid number of tensor for the FlexSplitLayer layer. It requires X_s, X_p0, X_p1 and W as input"
            )

        if self.first_layer:
            X = tensors[0]
            W = tensors[1]

            rep_s = self.net_shared(X)
            rep_p0 = self.net_p0(X)
            rep_p1 = self.net_p1(X)

        else:
            X_s = tensors[0]
            X_p0 = tensors[1]
            X_p1 = tensors[2]
            W = tensors[3]

            rep_s = self.net_shared(X_s)
            rep_p0 = self.net_p0(torch.cat([X_s, X_p0], dim=1))
            rep_p1 = self.net_p1(torch.cat([X_s, X_p1], dim=1))

        return [rep_s, rep_p0, rep_p1, W]


class FlexTEOutputLayer(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_in_p: int,
        private: bool,
        dropout: bool = False,
        dropout_prob: float = 0.5,
    ) -> None:
        super(FlexTEOutputLayer, self).__init__()
        self.private = private
        if dropout:
            self.net_shared = nn.Sequential(
                nn.Dropout(dropout_prob), nn.Linear(n_units_in, 1)
            ).to(DEVICE)
            self.net_p0 = nn.Sequential(
                nn.Dropout(dropout_prob), nn.Linear(n_units_in_p, 1)
            ).to(DEVICE)
            self.net_p1 = nn.Sequential(
                nn.Dropout(dropout_prob), nn.Linear(n_units_in_p, 1)
            ).to(DEVICE)
        else:
            self.net_shared = nn.Sequential(nn.Linear(n_units_in, 1)).to(DEVICE)
            self.net_p0 = nn.Sequential(nn.Linear(n_units_in_p, 1)).to(DEVICE)
            self.net_p1 = nn.Sequential(nn.Linear(n_units_in_p, 1)).to(DEVICE)

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(tensors) != 4:
            raise ValueError(
                "Invalid number of tensor for the FlexSplitLayer layer. It requires X_s, X_p0, X_p1 and W as input"
            )
        X_s = tensors[0]
        X_p0 = tensors[1]
        X_p1 = tensors[2]
        W = tensors[3]

        if self.private:
            rep_p0 = self.net_p0(torch.cat([X_s, X_p0], dim=1)).squeeze()
            rep_p1 = self.net_p1(torch.cat([X_s, X_p1], dim=1)).squeeze()

            return (1 - W) * rep_p0 + W * rep_p1
        else:
            rep_s = self.net_shared(X_s).squeeze()
            rep_p0 = self.net_p0(torch.cat([X_s, X_p0], dim=1)).squeeze()
            rep_p1 = self.net_p1(torch.cat([X_s, X_p1], dim=1)).squeeze()

            return (1 - W) * rep_p0 + W * rep_p1 + rep_s


class ElementWiseParallelActivation(nn.Module):
    """Layer that applies a scalar function elementwise on its inputs.

    Input looks like: X_s, X_p0, X_p1, t = inputs
    """

    def __init__(self, act: Callable, **act_kwargs: Any) -> None:
        super(ElementWiseParallelActivation, self).__init__()
        self.act = act
        self.act_kwargs = act_kwargs

    def forward(self, tensors: List[torch.Tensor]) -> List:
        if len(tensors) != 4:
            raise ValueError(
                "Invalid number of tensor for the ElementWiseParallelActivation layer. It requires X_s, X_p0, X_p1, t as input"
            )

        return [
            self.act(tensors[0], **self.act_kwargs),
            self.act(tensors[1], **self.act_kwargs),
            self.act(tensors[2], **self.act_kwargs),
            tensors[3],
        ]


class ElementWiseSplitActivation(nn.Module):
    """Layer that applies a scalar function elementwise on its inputs.

    Input looks like: X, t = inputs
    """

    def __init__(self, act: Callable, **act_kwargs: Any) -> None:
        super(ElementWiseSplitActivation, self).__init__()
        self.act = act
        self.act_kwargs = act_kwargs

    def forward(self, tensors: List[torch.Tensor]) -> List:
        if len(tensors) != 2:
            raise ValueError(
                "Invalid number of tensor for the ElementWiseSplitActivation layer. It requires X, t as input"
            )

        return [
            self.act(tensors[0], **self.act_kwargs),
            tensors[1],
        ]


class FlexTENet(BaseCATEEstimator):
    """
    CLass implements FlexTENet, an architecture for treatment effect estimation that allows for
    both shared and private information in each layer of the network.

    Parameters
    ----------
    n_unit_in: int
        Number of features
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_units_s_out: int
        Number of hidden units in each shared hypothesis layer
    n_units_p_out: int
        Number of hidden units in each private hypothesis layer
    n_layers_r: int
        Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_s_r: int
        Number of hidden units in each shared representation layer
    n_units_s_r: int
        Number of hidden units in each private representation layer
    private_out: bool, False
        Whether the final prediction layer should be fully private, or retain a shared component.
    weight_decay: float
        l2 (ridge) penalty
    penalty_orthogonal: float
        orthogonalisation penalty
    lr: float
        learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    early_stopping: bool, default True
        Whether to use early stopping
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    opt: str, default 'adam'
        Optimizer to use, accepts 'adam' and 'sgd'
    shared_repr: bool, False
        Whether to use a shared representation block as TARNet
    lr_scale: float
        Whether to scale down the learning rate after unfreezing the private components of the
        network (only used if pretrain_shared=True)
    normalize_ortho: bool, False
        Whether to normalize the orthogonality penalty (by depth of network)
    clipping_value: int, default 1
        Gradients clipping value
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_s_out: int = DEFAULT_DIM_S_OUT,
        n_units_p_out: int = DEFAULT_DIM_P_OUT,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_s_r: int = DEFAULT_DIM_S_R,
        n_units_p_r: int = DEFAULT_DIM_P_R,
        private_out: bool = False,
        weight_decay: float = DEFAULT_PENALTY_L2,
        penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        shared_repr: bool = False,
        normalize_ortho: bool = False,
        mode: int = 1,
        clipping_value: int = 1,
        dropout: bool = False,
        dropout_prob: float = 0.5,
        nonlin: str = "selu",
    ) -> None:
        super(FlexTENet, self).__init__()

        self.binary_y = binary_y
        self.n_layers_r = n_layers_r if n_layers_r else 1
        self.n_layers_out = n_layers_out
        self.n_units_s_out = n_units_s_out
        self.n_units_p_out = n_units_p_out
        self.n_units_s_r = n_units_s_r
        self.n_units_p_r = n_units_p_r
        self.private_out = private_out
        self.mode = mode

        self.penalty_orthogonal = penalty_orthogonal
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.shared_repr = shared_repr
        self.normalize_ortho = normalize_ortho
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.nonlin = nonlin.lower()
        
        self.seed = seed
        self.n_iter_print = n_iter_print

        layers = []

        if shared_repr:  # fully shared representation as in TARNet
            layers.extend(
                [
                    FlexTELinearLayer(
                        "shared_repr_layer_0",
                        dropout,
                        dropout_prob,
                        n_unit_in,
                        n_units_s_r,
                    ),
                    ElementWiseSplitActivation(self._get_activation()),
                ]
            )

            # add required number of layers
            for i in range(self.n_layers_r - 1):
                layers.extend(
                    [
                        FlexTELinearLayer(
                            f"shared_repr_layer_{i + 1}",
                            dropout,
                            dropout_prob,
                            n_units_s_r,
                            n_units_s_r,
                        ),
                        ElementWiseSplitActivation(self._get_activation()),
                    ]
                )

        else:  # shared AND private representations
            layers.extend(
                [
                    FlexTESplitLayer(
                        "shared_private_layer_0",
                        n_unit_in,
                        n_unit_in,
                        n_units_s_r,
                        n_units_p_r,
                        first_layer=True,
                        dropout=dropout,
                        dropout_prob=dropout_prob,
                    ),
                    ElementWiseParallelActivation(self._get_activation()),
                ]
            )

            # add required number of layers
            for i in range(n_layers_r - 1):
                layers.extend(
                    [
                        FlexTESplitLayer(
                            f"shared_private_layer_{i + 1}",
                            n_units_s_r,
                            n_units_s_r + n_units_p_r,
                            n_units_s_r,
                            n_units_p_r,
                            first_layer=False,
                            dropout=dropout,
                            dropout_prob=dropout_prob,
                        ),
                        ElementWiseParallelActivation(self._get_activation()),
                    ]
                )

        # add output layers
        layers.extend(
            [
                FlexTESplitLayer(
                    "output_layer_0",
                    n_units_s_r,
                    n_units_s_r if shared_repr else n_units_s_r + n_units_p_r,
                    n_units_s_out,
                    n_units_p_out,
                    first_layer=(shared_repr),
                    dropout=dropout,
                    dropout_prob=dropout_prob,
                ),
                ElementWiseParallelActivation(self._get_activation()),
            ]
        )

        # add required number of layers
        for i in range(n_layers_out - 1):
            layers.extend(
                [
                    FlexTESplitLayer(
                        f"output_layer_{i + 1}",
                        n_units_s_out,
                        n_units_s_out + n_units_p_out,
                        n_units_s_out,
                        n_units_p_out,
                        first_layer=False,
                        dropout=dropout,
                        dropout_prob=dropout_prob,
                    ),
                    ElementWiseParallelActivation(self._get_activation()),
                ]
            )

        # append final layer
        layers.append(
            FlexTEOutputLayer(
                n_units_s_out,
                n_units_s_out + n_units_p_out,
                private=self.private_out,
                dropout=dropout,
                dropout_prob=dropout_prob,
            )
        )
        if binary_y:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers).to(DEVICE)

    def _get_activation(self) -> nn.Module:
        if self.nonlin == "selu":
            return nn.SELU(inplace=True)
        elif self.nonlin == "relu":
            return nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported nonlinearity: {self.nonlin}")

    def _ortho_penalty_asymmetric(self) -> torch.Tensor:
        def _get_cos_reg(
            params_0: torch.Tensor, params_1: torch.Tensor, normalize: bool
        ) -> torch.Tensor:
            if normalize:
                params_0 = params_0 / torch.linalg.norm(params_0, dim=0)
                params_1 = params_1 / torch.linalg.norm(params_1, dim=0)

            x_min = min(params_0.shape[0], params_1.shape[0])
            y_min = min(params_0.shape[1], params_1.shape[1])

            return (
                torch.linalg.norm(
                    params_0[:x_min, :y_min] * params_1[:x_min, :y_min], "fro"
                )
                ** 2
            )

        def _apply_reg_split_layer(
            layer: FlexTESplitLayer, full: bool = True
        ) -> torch.Tensor:
            _ortho_body = 0
            if full:
                _ortho_body = _get_cos_reg(
                    layer.net_p0[-1].weight,
                    layer.net_p1[-1].weight,
                    self.normalize_ortho,
                )
            _ortho_body += torch.sum(
                _get_cos_reg(
                    layer.net_shared[-1].weight,
                    layer.net_p0[-1].weight,
                    self.normalize_ortho,
                )
                + _get_cos_reg(
                    layer.net_shared[-1].weight,
                    layer.net_p1[-1].weight,
                    self.normalize_ortho,
                )
            )
            return _ortho_body

        ortho_body = 0
        for layer in self.model:
            if not isinstance(layer, (FlexTESplitLayer, FlexTEOutputLayer)):
                continue

            if isinstance(layer, FlexTESplitLayer):
                ortho_body += _apply_reg_split_layer(layer, full=True)

            if self.private_out:
                continue

            ortho_body += _apply_reg_split_layer(layer, full=False)

        return self.penalty_orthogonal * ortho_body

    def loss(
        self,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
    ) -> torch.Tensor:
        def head_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            if self.binary_y:
                return nn.BCELoss()(y_pred, y_true)
            else:
                return (y_pred - y_true) ** 2

        def po_loss() -> torch.Tensor:
            loss0 = torch.mean((1.0 - t_true) * head_loss(y0_pred, y_true))
            loss1 = torch.mean(t_true * head_loss(y1_pred, y_true))

            return loss0 + loss1

        return po_loss() + self._ortho_penalty_asymmetric()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "FlexTENet":
        """
        Fit treatment models.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            The features to fit to
        y : torch.Tensor of shape (n_samples,) or (n_samples, )
            The outcome variable
        w: torch.Tensor of shape (n_samples,)
            The treatment indicator
        """
        self.model.train()

        X = torch.Tensor(X).to(DEVICE)
        y = torch.Tensor(y).squeeze().to(DEVICE)
        w = torch.Tensor(w).squeeze().long().to(DEVICE)

        X, y, w, X_val, y_val, w_val, val_string = make_val_split(
            X, y, w=w, val_split_prop=self.val_split_prop, seed=self.seed
        )

        n = X.shape[0]  # could be different from before due to split

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # training
        val_loss_best = LARGE_VAL
        self.best_val_loss_ = None
        self.last_val_loss_ = None
        patience = 0
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                optimizer.zero_grad()

                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]

                X_next = X[idx_next]
                y_next = y[idx_next].squeeze()
                w_next = w[idx_next].squeeze()

                _, mu0, mu1 = self.predict(X_next, return_po=True, training=True)
                batch_loss = self.loss(mu0, mu1, y_next, w_next)

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    _, mu0, mu1 = self.predict(X_val, return_po=True, training=True)
                    val_loss = self.loss(mu0, mu1, y_val, w_val).detach().cpu()
                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            self.best_val_loss_ = float(val_loss)
                            patience = 0
                        else:
                            patience += 1
                        if patience > self.patience and (
                            (i + 1) * n_batches > self.n_iter_min
                        ):
                            break
                    if i % self.n_iter_print == 0:
                        log.info(
                            f"[FlexTENet] Epoch: {i}, current {val_string} loss: {val_loss} train_loss: {torch.mean(train_loss)}"
                        )

        return self

    def predict(
        self, X: torch.Tensor, return_po: bool = False, training: bool = False
    ) -> torch.Tensor:
        """
        Predict treatment effects and potential outcomes

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        Returns
        -------
        y: array-like of shape (n_samples,)
        """
        if not training:
            self.model.eval()

        X = self._check_tensor(X).float()
        W0 = torch.zeros(X.shape[0]).to(DEVICE)
        W1 = torch.ones(X.shape[0]).to(DEVICE)

        mu0 = self.model([X, W0])
        mu1 = self.model([X, W1])

        te = mu1 - mu0

        if return_po:
            return te, mu0, mu1

        return te
