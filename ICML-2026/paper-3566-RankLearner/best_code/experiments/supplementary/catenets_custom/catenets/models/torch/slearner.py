from typing import Any, Optional

import torch

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_OUT,
    DEFAULT_VAL_SPLIT,
)
from catenets.models.torch.base import (
    DEVICE,
    BaseCATEEstimator,
    BasicNet,
    PropensityNet,
)
from catenets.models.torch.utils.model_utils import predict_wrapper


class SLearner(BaseCATEEstimator):
    """
    S-learner for treatment effect estimation (single learner, treatment indicator just another feature).

    Parameters
    ----------
    n_unit_in: int
        Number of features
    binary_y: bool
        Whether the outcome is binary
    po_estimator: sklearn/PyTorch model, default: None
        Custom potential outcome model. If this parameter is set, the rest of the parameters are ignored.
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Linear
        layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
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
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in the neural net. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
    weighting_strategy: optional str, None
        Whether to include propensity head and which weightening strategy to use
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool,
        po_estimator: Any = None,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = None,
        batch_norm: bool = True,
        early_stopping: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ) -> None:
        super(SLearner, self).__init__()

        self._weighting_strategy = weighting_strategy
        if po_estimator is not None:
            self._po_estimator = po_estimator
        else:
            self._po_estimator = BasicNet(
                "slearner_po_estimator",
                n_unit_in + 1,
                binary_y=binary_y,
                n_layers_out=n_layers_out,
                n_units_out=n_units_out,
                weight_decay=weight_decay,
                lr=lr,
                n_iter=n_iter,
                batch_size=batch_size,
                val_split_prop=val_split_prop,
                n_iter_print=n_iter_print,
                seed=seed,
                nonlin=nonlin,
                batch_norm=batch_norm,
                early_stopping=early_stopping,
                dropout_prob=dropout_prob,
                dropout=dropout,
            ).to(DEVICE)
        if weighting_strategy is not None:
            self._propensity_estimator = PropensityNet(
                "slearner_prop_estimator",
                n_unit_in,
                2,  # number of treatments
                weighting_strategy,
                n_units_out_prop=n_units_out_prop,
                n_layers_out_prop=n_layers_out_prop,
                weight_decay=weight_decay,
                lr=lr,
                n_iter=n_iter,
                batch_size=batch_size,
                n_iter_print=n_iter_print,
                seed=seed,
                nonlin=nonlin,
                val_split_prop=val_split_prop,
                batch_norm=batch_norm,
                early_stopping=early_stopping,
                dropout=dropout,
                dropout_prob=dropout_prob,
            ).to(DEVICE)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "SLearner":
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
        y = torch.Tensor(y).to(DEVICE)
        w = torch.Tensor(w).to(DEVICE)

        # add indicator as additional variable
        X_ext = torch.cat((X, w.reshape((-1, 1))), dim=1).to(DEVICE)

        if not (
            hasattr(self._po_estimator, "train") or hasattr(self._po_estimator, "fit")
        ):
            raise NotImplementedError("invalid po_estimator for the slearner")

        if hasattr(self._po_estimator, "fit"):
            log.info("Fit the sklearn po_estimator")
            self._po_estimator.fit(
                X_ext.detach().cpu().numpy(), y.detach().cpu().numpy()
            )
            return self

        if self._weighting_strategy is None:
            # fit standard S-learner
            log.info("Fit the PyTorch po_estimator")
            self._po_estimator.fit(X_ext, y)
            return self

        # use reweighting within the outcome model
        log.info("Fit the PyTorch po_estimator with the propensity estimator")
        self._propensity_estimator.fit(X, w)
        weights = self._propensity_estimator.get_importance_weights(X, w)
        self._po_estimator.fit(X_ext, y, weight=weights)

        return self

    def _create_extended_matrices(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        X = self._check_tensor(X)

        # create extended matrices
        w_1 = torch.ones((n, 1)).to(DEVICE)
        w_0 = torch.zeros((n, 1)).to(DEVICE)
        X_ext_0 = torch.cat((X, w_0), dim=1).to(DEVICE)
        X_ext_1 = torch.cat((X, w_1), dim=1).to(DEVICE)

        return [X_ext_0, X_ext_1]

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
        X_ext = self._create_extended_matrices(X)

        y = []
        for ext_mat in X_ext:
            y.append(predict_wrapper(self._po_estimator, ext_mat).to(DEVICE))

        outcome = y[1] - y[0]

        if return_po:
            return outcome, y[0], y[1]

        return outcome
