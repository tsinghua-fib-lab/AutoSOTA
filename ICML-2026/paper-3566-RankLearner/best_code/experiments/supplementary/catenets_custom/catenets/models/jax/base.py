"""
Base modules shared across different nets
"""
# Author: Alicia Curth
import abc
from typing import Any, Callable, List, Optional, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import grad, jit, random
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, Elu, Relu, Sigmoid
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import ParameterGrid

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
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
from catenets.models.jax.model_utils import (
    check_shape_1d_data,
    check_X_is_np,
    make_val_split,
)


def ReprBlock(
    n_layers: int = 3, n_units: int = 100, nonlin: str = DEFAULT_NONLIN
) -> Any:
    # Creates a representation block using jax.stax
    # create first layer
    if nonlin == "elu":
        NL = Elu
    elif nonlin == "relu":
        NL = Relu
    elif nonlin == "sigmoid":
        NL = Sigmoid
    else:
        raise ValueError("Unknown nonlinearity")

    layers: Tuple
    layers = (Dense(n_units), NL)

    # add required number of layers
    for i in range(n_layers - 1):
        layers = (*layers, Dense(n_units), NL)

    return stax.serial(*layers)


def OutputHead(
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    binary_y: bool = False,
    n_layers_r: int = 0,
    n_units_r: int = DEFAULT_UNITS_R,
    nonlin: str = DEFAULT_NONLIN,
) -> Any:
    # Creates an output head using jax.stax
    if nonlin == "elu":
        NL = Elu
    elif nonlin == "relu":
        NL = Relu
    elif nonlin == "sigmoid":
        NL = Sigmoid
    else:
        raise ValueError("Unknown nonlinearity")

    layers: Tuple = ()

    # add required number of layers
    for i in range(n_layers_r):
        layers = (*layers, Dense(n_units_r), NL)

    # add required number of layers
    for i in range(n_layers_out):
        layers = (*layers, Dense(n_units_out), NL)

    # return final architecture
    if not binary_y:
        return stax.serial(*layers, Dense(1))
    else:
        return stax.serial(*layers, Dense(1), Sigmoid)


class BaseCATENet(BaseEstimator, RegressorMixin, abc.ABC):
    """
    Base CATENet class to serve as template for all other nets
    """

    def score(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        sample_weight: Optional[jnp.ndarray] = None,
    ) -> float:
        """
        Return the sqrt PEHE error (Oracle metric).

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Covariate matrix
        y: np.array
            Expected potential outcome vector
        """
        X = check_X_is_np(X)
        y = check_X_is_np(y)
        if len(X) != len(y):
            raise ValueError("X/y length mismatch for score")
        if y.shape[-1] != 2:
            raise ValueError(f"y has invalid shape {y.shape}")

        hat_te = self.predict(X)

        return jnp.sqrt(jnp.mean(((y[:, 1] - y[:, 0]) - hat_te) ** 2))

    @abc.abstractmethod
    def _get_train_function(self) -> Callable:
        ...

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        w: jnp.ndarray,
        p: Optional[jnp.ndarray] = None,
    ) -> "BaseCATENet":
        """
        Fit method for a CATENet. Takes covariates, outcome variable and treatment indicator as
        input

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Covariate matrix
        y: np.array
            Outcome vector
        w: np.array
            Treatment indicator
        p: np.array
            Vector of (known) treatment propensities. Currently only supported for TwoStepNets.
        """
        # some quick input checks
        if p is not None:
            raise NotImplementedError("Only two-step-nets take p as input. ")
        X = check_X_is_np(X)
        self._check_inputs(w, p)

        train_func = self._get_train_function()
        train_params = self.get_params()

        self._params, self._predict_funs = train_func(X, y, w, **train_params)

        return self

    @abc.abstractmethod
    def _get_predict_function(self) -> Callable:
        ...

    def predict(
        self, X: jnp.ndarray, return_po: bool = False, return_prop: bool = False
    ) -> jnp.ndarray:
        """
        Predict treatment effect estimates using a CATENet. Depending on method, can also return
        potential outcome estimate and propensity score estimate.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Covariate matrix
        return_po: bool, default False
            Whether to return potential outcome estimate
        return_prop: bool, default False
            Whether to return propensity estimate

        Returns
        -------
        array of CATE estimates, optionally also potential outcomes and propensity
        """
        X = check_X_is_np(X)
        predict_func = self._get_predict_function()
        return predict_func(
            X,
            trained_params=self._params,
            predict_funs=self._predict_funs,
            return_po=return_po,
            return_prop=return_prop,
        )

    @staticmethod
    def _check_inputs(w: jnp.ndarray, p: jnp.ndarray) -> None:
        if p is not None:
            if onp.sum(p > 1) > 0 or onp.sum(p < 0) > 0:
                raise ValueError("p should be in [0,1]")

        if not ((w == 0) | (w == 1)).all():
            raise ValueError("W should be binary")

    def fit_and_select_params(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        w: jnp.ndarray,
        p: Optional[jnp.ndarray] = None,
        param_grid: dict = {},
    ) -> "BaseCATENet":
        # some quick input checks
        if param_grid is None:
            raise ValueError("No param_grid to evaluate. ")
        X = check_X_is_np(X)
        self._check_inputs(w, p)

        param_grid = ParameterGrid(param_grid)
        self_param_dict = self.get_params()
        train_function = self._get_train_function()

        models = []
        losses = []
        param_settings: list = []

        for param_setting in param_grid:
            log.debug(
                "Testing parameter setting: "
                + " ".join(
                    [key + ": " + str(value) for key, value in param_setting.items()]
                )
            )
            # replace params
            train_param_dict = {
                key: (val if key not in param_setting.keys() else param_setting[key])
                for key, val in self_param_dict.items()
            }
            if p is not None:
                params, funs, val_loss = train_function(
                    X, y, w, p=p, return_val_loss=True, **train_param_dict
                )
            else:
                params, funs, val_loss = train_function(
                    X, y, w, return_val_loss=True, **train_param_dict
                )

            models.append((params, funs))
            losses.append(val_loss)

        # save results
        param_settings.extend(param_grid)
        self._selection_results = {
            "param_settings": param_settings,
            "val_losses": losses,
        }

        # find lowest loss and set params
        best_idx = jnp.array(losses).argmin()
        self._params, self._predict_funs = models[best_idx]
        self.set_params(**param_settings[best_idx])

        return self


def train_output_net_only(
    X: jnp.ndarray,
    y: jnp.ndarray,
    binary_y: bool = False,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    n_layers_r: int = 0,
    n_units_r: int = DEFAULT_UNITS_R,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    step_size: float = DEFAULT_STEP_SIZE,
    n_iter: int = DEFAULT_N_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    early_stopping: bool = True,
    patience: int = DEFAULT_PATIENCE,
    n_iter_min: int = DEFAULT_N_ITER_MIN,
    n_iter_print: int = DEFAULT_N_ITER_PRINT,
    seed: int = DEFAULT_SEED,
    return_val_loss: bool = False,
    nonlin: str = DEFAULT_NONLIN,
    avg_objective: bool = False,
) -> Any:
    # function to train a single output head
    # input check
    y = check_shape_1d_data(y)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    # get validation split (can be none)
    X, y, X_val, y_val, val_string = make_val_split(
        X, y, val_split_prop=val_split_prop, seed=seed
    )
    n = X.shape[0]  # could be different from before due to split

    # get output head
    init_fun, predict_fun = OutputHead(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=binary_y,
        n_layers_r=n_layers_r,
        n_units_r=n_units_r,
        nonlin=nonlin,
    )

    # get functions
    if not binary_y:
        # define loss and grad
        @jit
        def loss(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray], penalty: float
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets = batch
            preds = predict_fun(params, inputs)
            weightsq = sum(
                [
                    jnp.sum(params[i][0] ** 2)
                    for i in range(0, 2 * (n_layers_out + n_layers_r) + 1, 2)
                ]
            )
            if not avg_objective:
                return jnp.sum((preds - targets) ** 2) + 0.5 * penalty * weightsq
            else:
                return jnp.average((preds - targets) ** 2) + 0.5 * penalty * weightsq

    else:
        # get loss and grad
        @jit
        def loss(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray], penalty: float
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets = batch
            preds = predict_fun(params, inputs)
            weightsq = sum(
                [
                    jnp.sum(params[i][0] ** 2)
                    for i in range(0, 2 * (n_layers_out + n_layers_r) + 1, 2)
                ]
            )
            if not avg_objective:
                return (
                    -jnp.sum(
                        targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds)
                    )
                    + 0.5 * penalty * weightsq
                )
            else:
                return (
                    -jnp.average(
                        targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds)
                    )
                    + 0.5 * penalty * weightsq
                )

    # set optimization routine
    # set optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    # set update function
    @jit
    def update(i: int, state: dict, batch: jnp.ndarray, penalty: float) -> jnp.ndarray:
        params = get_params(state)
        g_params = grad(loss)(params, batch, penalty)
        return opt_update(i, g_params, state)

    # initialise states
    _, init_params = init_fun(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    l_best = LARGE_VAL
    p_curr = 0

    # do training
    for i in range(n_iter):
        # shuffle data for minibatches
        onp.random.shuffle(train_indices)
        for b in range(n_batches):
            idx_next = train_indices[
                (b * batch_size) : min((b + 1) * batch_size, n - 1)
            ]
            next_batch = X[idx_next, :], y[idx_next, :]
            opt_state = update(i * n_batches + b, opt_state, next_batch, penalty_l2)

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            l_curr = loss(params_curr, (X_val, y_val), penalty_l2)

        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss: {l_curr}")

        if early_stopping and ((i + 1) * n_batches > n_iter_min):
            # check if loss updated
            if l_curr < l_best:
                l_best = l_curr
                p_curr = 0
            else:
                p_curr = p_curr + 1

            if p_curr > patience:
                trained_params = get_params(opt_state)

                if return_val_loss:
                    # return loss without penalty
                    l_final = loss(trained_params, (X_val, y_val), 0)
                    return trained_params, predict_fun, l_final

                return trained_params, predict_fun

    # get final parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss(trained_params, (X_val, y_val), 0)
        return trained_params, predict_fun, l_final

    return trained_params, predict_fun
