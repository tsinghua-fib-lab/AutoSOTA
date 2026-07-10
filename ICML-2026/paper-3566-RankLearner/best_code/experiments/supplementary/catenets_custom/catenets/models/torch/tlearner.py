import copy
from typing import Any

import torch

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
from catenets.models.torch.base import DEVICE, BaseCATEEstimator, BasicNet
from catenets.models.torch.utils.model_utils import predict_wrapper, train_wrapper


class TLearner(BaseCATEEstimator):
    """
    TLearner class -- two separate functions learned for each Potential Outcome function

    Parameters
    ----------
    n_unit_in: int
        Number of features
    binary_y: bool, default False
        Whether the outcome is binary
    po_estimator: sklearn/PyTorch model, default: None
        Custom plugin model. If this parameter is set, the rest of the parameters are ignored.
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Linear layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
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
        Nonlinearity to use in the neural net. Cat be 'elu', 'relu', 'selu' or 'leaky_relu'.
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool,
        po_estimator: Any = None,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        batch_norm: bool = True,
        early_stopping: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ) -> None:
        super(TLearner, self).__init__()

        self._plug_in: Any = []
        plugins = [f"tlearner_po_estimator_{i}" for i in range(2)]
        if po_estimator is not None:
            for plugin in plugins:
                self._plug_in.append(copy.deepcopy(po_estimator))
        else:
            for plugin in plugins:
                self._plug_in.append(
                    BasicNet(
                        plugin,
                        n_unit_in,
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
                    ).to(DEVICE),
                )

    def predict(
        self, X: torch.Tensor, return_po: bool = False, training: bool = False
    ) -> torch.Tensor:
        """
        Predict treatment effects and potential outcomes
        Parameters
        ----------
        X: torch.Tensor of shape (n_samples, n_features)
            Test-sample features
        return_po: bool
            Return potential outcomes too

        Returns
        -------
        y: torch.Tensor of shape (n_samples,)
        """
        if not training:
            self.eval()

        X = self._check_tensor(X).float()

        y_hat = []
        for widx, plugin in enumerate(self._plug_in):
            y_hat.append(predict_wrapper(plugin, X))

        outcome = y_hat[1] - y_hat[0]

        if return_po:
            return outcome, y_hat[0], y_hat[1]

        return outcome

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "TLearner":
        """
        Train plug-in models.

        Parameters
        ----------
        X : torch.Tensor (n_samples, n_features)
            The features to fit to
        y : torch.Tensor (n_samples,) or (n_samples, )
            The outcome variable
        w: torch.Tensor (n_samples,)
            The treatment indicator
        """
        self.train()

        X = torch.Tensor(X).to(DEVICE)
        y = torch.Tensor(y).to(DEVICE)
        w = torch.Tensor(w).to(DEVICE)

        for widx, plugin in enumerate(self._plug_in):
            train_wrapper(plugin, X[w == widx], y[w == widx])

        return self
