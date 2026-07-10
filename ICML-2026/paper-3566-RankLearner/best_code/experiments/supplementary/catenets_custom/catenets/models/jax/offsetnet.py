"""
Module implements OffsetNet, also referred to as the 'reparametrization approach' and 'hard
approach' in "On inductive biases for heterogeneous treatment effect estimation", Curth & vd
Schaar (2021); modeling the POs using a shared prognostic function and
an offset (treatment effect)
"""
# Author: Alicia Curth
from typing import Any, Callable, List, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import grad, jit, random
from jax.example_libraries import optimizers
from jax.example_libraries.stax import sigmoid

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
from catenets.models.jax.base import BaseCATENet, OutputHead
from catenets.models.jax.model_utils import (
    check_shape_1d_data,
    heads_l2_penalty,
    make_val_split,
)


class OffsetNet(BaseCATENet):
    """
    Module implements OffsetNet, also referred to as the 'reparametrization approach' and 'hard
    approach' in Curth & vd Schaar (2021); modeling the POs using a shared prognostic function and
    an offset (treatment effect).

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_layers_r: int
        Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_r: int
        Number of hidden units in each representation layer
    penalty_l2: float
        l2 (ridge) penalty
    step_size: float
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
    penalty_l2_p: float
        l2-penalty for regularizing the offset
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    """

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        penalty_l2_p: float = DEFAULT_PENALTY_L2,
        step_size: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
    ):
        # structure of net
        self.binary_y = binary_y
        self.n_layers_r = n_layers_r
        self.n_layers_out = n_layers_out
        self.n_units_r = n_units_r
        self.n_units_out = n_units_out
        self.nonlin = nonlin

        # penalties
        self.penalty_l2 = penalty_l2
        self.penalty_l2_p = penalty_l2_p

        # training params
        self.step_size = step_size
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min

    def _get_train_function(self) -> Callable:
        return train_offsetnet

    def _get_predict_function(self) -> Callable:
        return predict_offsetnet


def predict_offsetnet(
    X: jnp.ndarray,
    trained_params: jnp.ndarray,
    predict_funs: List[Any],
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    if return_prop:
        raise NotImplementedError("OffsetNet does not implement a propensity model.")

    # unpack inputs
    predict_fun_head = predict_funs[0]
    binary_y = predict_funs[1]
    param_0, param_1 = trained_params[0], trained_params[1]

    # get potential outcomes
    mu_0 = predict_fun_head(param_0, X)
    offset = predict_fun_head(param_1, X)

    if not binary_y:
        if return_po:
            return offset, mu_0, mu_0 + offset
        else:
            return offset
    else:
        # still need to sigmoid
        po_0 = sigmoid(mu_0)
        po_1 = sigmoid(mu_0 + offset)
        if return_po:
            return po_1 - po_0, po_0, po_1
        else:
            return po_1 - po_0


def train_offsetnet(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    binary_y: bool = False,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_r: int = DEFAULT_UNITS_R,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    penalty_l2_p: float = DEFAULT_PENALTY_L2,
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
    avg_objective: bool = True,
) -> Tuple:
    # input check
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    # get validation split (can be none)
    X, y, w, X_val, y_val, w_val, val_string = make_val_split(
        X, y, w, val_split_prop=val_split_prop, seed=seed
    )
    n = X.shape[0]  # could be different from before due to split

    # get output head functions (both heads share same structure)
    init_fun_head, predict_fun_head = OutputHead(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=False,
        n_layers_r=n_layers_r,
        n_units_r=n_units_r,
        nonlin=nonlin,
    )

    def init_fun_offset(rng: float, input_shape: Tuple) -> Tuple:
        # chain together the layers
        # param should look like [param_base, param_offset]
        rng, layer_rng = random.split(rng)
        _, param_base = init_fun_head(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        input_shape, param_offset = init_fun_head(layer_rng, input_shape)
        return input_shape, [param_base, param_offset]

    # Define loss functions
    if not binary_y:

        @jit
        def loss_offsetnet(
            params: jnp.ndarray, batch: jnp.ndarray, penalty: float, penalty_l2_p: float
        ) -> jnp.ndarray:
            # params: list[representation, head_0, head_1]
            # batch: (X, y, w)
            inputs, targets, w = batch
            preds_0 = predict_fun_head(params[0], inputs)
            offset = predict_fun_head(params[1], inputs)
            preds = preds_0 + w * offset
            weightsq_head = heads_l2_penalty(
                params[0],
                params[1],
                n_layers_out + n_layers_r,
                False,
                penalty,
                penalty_l2_p,
            )
            if not avg_objective:
                return jnp.sum((preds - targets) ** 2) + 0.5 * weightsq_head
            else:
                return jnp.average((preds - targets) ** 2) + 0.5 * weightsq_head

    else:

        def loss_offsetnet(
            params: jnp.ndarray,
            batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            penalty: float,
            penalty_l2_p: float,
        ) -> jnp.ndarray:
            # params: list[representation, head_0, head_1]
            # batch: (X, y, w)
            inputs, targets, w = batch
            preds_0 = predict_fun_head(params[0], inputs)
            offset = predict_fun_head(params[1], inputs)
            preds = sigmoid(preds_0 + w * offset)
            weightsq_head = heads_l2_penalty(
                params[0],
                params[1],
                n_layers_out + n_layers_r,
                False,
                penalty,
                penalty_l2_p,
            )
            if not avg_objective:
                return (
                    -jnp.sum(
                        (targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
                    )
                    + 0.5 * weightsq_head
                )
            else:
                n_batch = y.shape[0]
                return (
                    -jnp.sum(
                        (targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
                    )
                    / n_batch
                    + 0.5 * weightsq_head
                )

    # Define optimisation routine
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    @jit
    def update(
        i: int, state: dict, batch: jnp.ndarray, penalty_l2: float, penalty_l2_p: float
    ) -> jnp.ndarray:
        # updating function
        params = get_params(state)
        return opt_update(
            i, grad(loss_offsetnet)(params, batch, penalty_l2, penalty_l2_p), state
        )

    # initialise states
    _, init_params = init_fun_offset(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    l_best = LARGE_VAL
    p_curr = 0

    pred_funs = predict_fun_head, binary_y

    # do training
    for i in range(n_iter):
        # shuffle data for minibatches
        onp.random.shuffle(train_indices)
        for b in range(n_batches):
            idx_next = train_indices[
                (b * batch_size) : min((b + 1) * batch_size, n - 1)
            ]
            next_batch = X[idx_next, :], y[idx_next, :], w[idx_next]
            opt_state = update(
                i * n_batches + b, opt_state, next_batch, penalty_l2, penalty_l2_p
            )

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            l_curr = loss_offsetnet(
                params_curr, (X_val, y_val, w_val), penalty_l2, penalty_l2_p
            )

        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss {l_curr}")

        if early_stopping and ((i + 1) * n_batches > n_iter_min):
            if l_curr < l_best:
                l_best = l_curr
                p_curr = 0
            else:
                p_curr = p_curr + 1

            if p_curr > patience:
                if return_val_loss:
                    # return loss without penalty
                    l_final = loss_offsetnet(params_curr, (X_val, y_val, w_val), 0, 0)
                    return params_curr, pred_funs, l_final

                return params_curr, pred_funs

    # return the parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss_offsetnet(get_params(opt_state), (X_val, y_val, w_val), 0, 0)
        return trained_params, pred_funs, l_final

    return trained_params, pred_funs
