from typing import Tuple

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
    DEFAULT_PENALTY_DISC,
    DEFAULT_PENALTY_L2,
    DEFAULT_PENALTY_ORTHOGONAL,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_R_BIG_S,
    DEFAULT_UNITS_R_SMALL_S,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.torch.base import (
    DEVICE,
    BaseCATEEstimator,
    BasicNet,
    PropensityNet,
    RepresentationNet,
)
from catenets.models.torch.utils.model_utils import make_val_split

EPS = 1e-8


class SNet(BaseCATEEstimator):
    """
    Class implements SNet as discussed in Curth & van der Schaar (2021). Additionally to the
    version implemented in the AISTATS paper, we also include an implementation that does not
    have propensity heads (set with_prop=False)
    Parameters
    ----------
    n_unit_in: int
        Number of features
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_r: int
        Number of shared & private representation layers before the hypothesis layers.
    n_units_r: int
        Number of hidden units in representation shared before the hypothesis layer.
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Linear
        layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
    n_units_r_small: int
        Number of hidden units in each PO functions private representation
    weight_decay: float
        l2 (ridge) penalty
    lr: float
        learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in the neural net. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
    penalty_disc: float, default zero
        Discrepancy penalty. Defaults to zero as this feature is not tested.
    clipping_value: int, default 1
        Gradients clipping value
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R_BIG_S,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
        penalty_disc: float = DEFAULT_PENALTY_DISC,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        ortho_reg_type: str = "abs",
        patience: int = DEFAULT_PATIENCE,
        clipping_value: int = 1,
        batch_norm: bool = True,
        with_prop: bool = True,
        early_stopping: bool = True,
        prop_loss_multiplier: float = 1,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ) -> None:
        super(SNet, self).__init__()

        self.n_unit_in = n_unit_in
        self.binary_y = binary_y
        self.penalty_orthogonal = penalty_orthogonal
        self.penalty_disc = penalty_disc
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.ortho_reg_type = ortho_reg_type
        self.clipping_value = clipping_value
        self.patience = patience
        self.with_prop = with_prop
        self.early_stopping = early_stopping
        self.n_iter_min = n_iter_min
        self.prop_loss_multiplier = prop_loss_multiplier
        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self._reps_mu0 = RepresentationNet(
            n_unit_in,
            n_units=n_units_r_small,
            n_layers=n_layers_r,
            nonlin=nonlin,
            batch_norm=batch_norm,
        )
        self._reps_mu1 = RepresentationNet(
            n_unit_in,
            n_units=n_units_r_small,
            n_layers=n_layers_r,
            nonlin=nonlin,
            batch_norm=batch_norm,
        )

        self._po_estimators = []

        if self.with_prop:
            self._reps_c = RepresentationNet(
                n_unit_in,
                n_units=n_units_r,
                n_layers=n_layers_r,
                nonlin=nonlin,
                batch_norm=batch_norm,
            )

            self._reps_o = RepresentationNet(
                n_unit_in,
                n_units=n_units_r_small,
                n_layers=n_layers_r,
                nonlin=nonlin,
                batch_norm=batch_norm,
            )

            self._reps_prop = RepresentationNet(
                n_unit_in,
                n_units=n_units_r,
                n_layers=n_layers_r,
                nonlin=nonlin,
                batch_norm=batch_norm,
            )

            for idx in range(2):
                self._po_estimators.append(
                    BasicNet(
                        f"snet_po_estimator_{idx}",
                        n_units_r
                        + n_units_r_small
                        + n_units_r_small,  # (reps_c, reps_o, reps_mu{idx})
                        binary_y=binary_y,
                        n_layers_out=n_layers_out,
                        n_units_out=n_units_out,
                        nonlin=nonlin,
                        batch_norm=batch_norm,
                        dropout_prob=dropout_prob,
                        dropout=dropout,
                    )
                )
            self._propensity_estimator = PropensityNet(
                "snet_propensity_estimator",
                n_units_r + n_units_r,  # reps_c, reps_w
                2,
                "prop",
                n_layers_out_prop=n_layers_out_prop,
                n_units_out_prop=n_units_out_prop,
                nonlin=nonlin,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_prob=dropout_prob,
            ).to(DEVICE)

            params = (
                list(self._reps_c.parameters())
                + list(self._reps_o.parameters())
                + list(self._reps_mu0.parameters())
                + list(self._reps_mu1.parameters())
                + list(self._reps_prop.parameters())
                + list(self._po_estimators[0].parameters())
                + list(self._po_estimators[1].parameters())
                + list(self._propensity_estimator.parameters())
            )
        else:
            self._reps_o = RepresentationNet(
                n_unit_in,
                n_units=n_units_r,
                n_layers=n_layers_r,
                nonlin=nonlin,
                batch_norm=batch_norm,
            )

            for idx in range(2):
                self._po_estimators.append(
                    BasicNet(
                        f"snet_po_estimator_{idx}",
                        n_units_r + n_units_r_small,  # (reps_o, reps_mu{idx})
                        binary_y=binary_y,
                        n_layers_out=n_layers_out,
                        n_units_out=n_units_out,
                        nonlin=nonlin,
                        batch_norm=batch_norm,
                    )
                )

            params = (
                list(self._reps_o.parameters())
                + list(self._reps_mu0.parameters())
                + list(self._reps_mu1.parameters())
                + list(self._po_estimators[0].parameters())
                + list(self._po_estimators[1].parameters())
            )

        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def loss(
        self,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        t_pred: torch.Tensor,
        discrepancy: torch.Tensor,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
    ) -> torch.Tensor:
        def head_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            if self.binary_y:
                return nn.BCELoss()(y_pred, y_true)
            else:
                return (y_pred - y_true) ** 2

        def po_loss(
            y0_pred: torch.Tensor,
            y1_pred: torch.Tensor,
            y_true: torch.Tensor,
            t_true: torch.Tensor,
        ) -> torch.Tensor:
            loss0 = torch.mean((1.0 - t_true) * head_loss(y0_pred, y_true))
            loss1 = torch.mean(t_true * head_loss(y1_pred, y_true))

            return loss0 + loss1

        def prop_loss(
            t_pred: torch.Tensor,
            t_true: torch.Tensor,
        ) -> torch.Tensor:
            if self.with_prop:
                t_pred = t_pred + EPS
                return nn.CrossEntropyLoss()(t_pred, t_true)
            else:
                return 0

        return (
            po_loss(y0_pred, y1_pred, y_true, t_true)
            + self.prop_loss_multiplier * prop_loss(t_pred, t_true)
            + discrepancy
            + self._ortho_reg()
        )

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "SNet":
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
        self.train()

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

        # training
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
                w_next = w[idx_next].squeeze()

                y0_preds, y1_preds, prop_preds, discrepancy = self._step(X_next, w_next)
                batch_loss = self.loss(
                    y0_preds, y1_preds, prop_preds, discrepancy, y_next, w_next
                )

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    y0_preds, y1_preds, prop_preds, discrepancy = self._step(
                        X_val, w_val
                    )
                    val_loss = (
                        self.loss(
                            y0_preds, y1_preds, prop_preds, discrepancy, y_val, w_val
                        )
                        .detach()
                        .cpu()
                    )
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
                            f"[SNet] Epoch: {i}, current {val_string} loss: {val_loss} train_loss: {torch.mean(train_loss)}"
                        )

        return self

    def _ortho_reg(self) -> float:
        def _get_absolute_rowsums(mat: torch) -> torch.Tensor:
            return torch.sum(torch.abs(mat), dim=0)

        def _get_cos_reg(
            params_0: torch.Tensor, params_1: torch.Tensor, normalize: bool = False
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

        reps_o_params = self._reps_o.model[0].weight
        reps_mu0_params = self._reps_mu0.model[0].weight
        reps_mu1_params = self._reps_mu1.model[0].weight

        if self.with_prop:
            reps_c_params = self._reps_c.model[0].weight
            reps_prop_params = self._reps_prop.model[0].weight

        # define ortho-reg function
        if self.ortho_reg_type == "abs":
            col_o = _get_absolute_rowsums(reps_o_params)
            col_mu0 = _get_absolute_rowsums(reps_mu0_params)
            col_mu1 = _get_absolute_rowsums(reps_mu1_params)
            if self.with_prop:
                col_c = _get_absolute_rowsums(reps_c_params)
                col_w = _get_absolute_rowsums(reps_prop_params)

                return self.penalty_orthogonal * torch.sum(
                    col_c * col_o
                    + col_c * col_w
                    + col_c * col_mu1
                    + col_c * col_mu0
                    + col_w * col_o
                    + col_mu0 * col_o
                    + col_o * col_mu1
                    + col_mu0 * col_mu1
                    + col_mu0 * col_w
                    + col_w * col_mu1
                )
            else:
                return self.penalty_orthogonal * torch.sum(
                    +col_mu0 * col_o + col_o * col_mu1 + col_mu0 * col_mu1
                )

        elif self.ortho_reg_type == "fro":
            if self.with_prop:
                return self.penalty_orthogonal * (
                    _get_cos_reg(reps_c_params, reps_o_params)
                    + _get_cos_reg(reps_c_params, reps_mu0_params)
                    + _get_cos_reg(reps_c_params, reps_mu1_params)
                    + _get_cos_reg(reps_c_params, reps_prop_params)
                    + _get_cos_reg(reps_o_params, reps_mu0_params)
                    + _get_cos_reg(reps_o_params, reps_mu1_params)
                    + _get_cos_reg(reps_o_params, reps_prop_params)
                    + _get_cos_reg(reps_mu0_params, reps_mu1_params)
                    + _get_cos_reg(reps_mu0_params, reps_prop_params)
                    + _get_cos_reg(reps_mu1_params, reps_prop_params)
                )
            else:
                return self.penalty_orthogonal * (
                    +_get_cos_reg(reps_o_params, reps_mu0_params)
                    + _get_cos_reg(reps_o_params, reps_mu1_params)
                    + _get_cos_reg(reps_mu0_params, reps_mu1_params)
                )

        else:
            raise ValueError(f"Invalid orth_reg_typ {self.ortho_reg_type}")

    def _maximum_mean_discrepancy(
        self, X: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        n = w.shape[0]
        n_t = torch.sum(w)

        X = X / torch.sqrt(torch.var(X, dim=0) + EPS)
        w = w.unsqueeze(dim=0)

        mean_control = (n / (n - n_t)) * torch.mean((1 - w).T * X, dim=0)
        mean_treated = (n / n_t) * torch.mean(w.T * X, dim=0)

        return torch.sum((mean_treated - mean_control) ** 2)

    def _step(
        self, X: torch.Tensor, w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y0_preds, y1_preds, prop_preds, reps_o = self._forward(X)

        discrepancy = self.penalty_disc * self._maximum_mean_discrepancy(reps_o, w)

        return y0_preds, y1_preds, prop_preds, discrepancy

    def _forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reps_o = self._reps_o(X)
        reps_mu0 = self._reps_mu0(X)
        reps_mu1 = self._reps_mu1(X)

        if self.with_prop:
            reps_c = self._reps_c(X)
            reps_w = self._reps_prop(X)

            reps_po_0 = torch.cat((reps_c, reps_o, reps_mu0), dim=1)
            reps_po_1 = torch.cat((reps_c, reps_o, reps_mu1), dim=1)
            reps_w = torch.cat((reps_c, reps_w), dim=1)
            prop_preds = self._propensity_estimator(reps_w)
        else:
            reps_po_0 = torch.cat((reps_o, reps_mu0), dim=1)
            reps_po_1 = torch.cat((reps_o, reps_mu1), dim=1)
            prop_preds = 0.5 * torch.ones(len(X))  # no probability predictions

        y0_preds = self._po_estimators[0](reps_po_0).squeeze()
        y1_preds = self._po_estimators[1](reps_po_1).squeeze()

        return y0_preds, y1_preds, prop_preds, reps_o

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
            self.eval()

        X = self._check_tensor(X).float()
        y0_preds, y1_preds, _, _ = self._forward(X)

        outcome = y1_preds - y0_preds

        if return_po:
            return outcome, y0_preds, y1_preds

        return outcome
