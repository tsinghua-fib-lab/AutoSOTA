"""
Class implements SNet-3, a variation on DR-CFR discussed in
Hassanpour and Greiner (2020) and Wu et al (2020).
"""
# Author: Alicia Curth
from typing import Any, Callable, List, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import grad, jit, random
from jax.example_libraries import optimizers

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_AVG_OBJECTIVE,
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
    DEFAULT_UNITS_R_BIG_S3,
    DEFAULT_UNITS_R_SMALL_S3,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.jax.base import BaseCATENet, OutputHead, ReprBlock
from catenets.models.jax.model_utils import (
    check_shape_1d_data,
    heads_l2_penalty,
    make_val_split,
)
from catenets.models.jax.representation_nets import mmd2_lin


# helper functions to avoid abstract tracer values in jit
def _get_absolute_rowsums(mat: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.abs(mat), axis=1)


def _concatenate_representations(reps: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(reps, axis=1)


class SNet3(BaseCATENet):
    """
    Class implements SNet-3, which is based on Hassanpour & Greiner (2020)'s DR-CFR (Without
    propensity weighting), using an orthogonal regularizer to enforce decomposition similar to
    Wu et al (2020).

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Dense
        layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
    n_layers_r: int
        Number of shared & private representation layers before hypothesis layers
    n_units_r: int
        Number of hidden units in representation layer shared by propensity score and outcome
        function (the 'confounding factor')
    n_units_r_small: int
        Number of hidden units in representation layer NOT shared by propensity score and outcome
        functions (the 'outcome factor' and the 'instrumental factor')
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
    reg_diff: bool, default False
        Whether to regularize the difference between the two potential outcome heads
    penalty_diff: float
        l2-penalty for regularizing the difference between output heads. used only if
        train_separate=False
    same_init: bool, False
        Whether to initialise the two output heads with same values
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    penalty_disc: float, default zero
        Discrepancy penalty. Defaults to zero as this feature is not tested.
    """

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R_BIG_S3,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S3,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
        penalty_disc: float = DEFAULT_PENALTY_DISC,
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
        reg_diff: bool = False,
        penalty_diff: float = DEFAULT_PENALTY_L2,
        same_init: bool = False,
    ) -> None:
        self.binary_y = binary_y

        self.n_layers_r = n_layers_r
        self.n_layers_out = n_layers_out
        self.n_layers_out_prop = n_layers_out_prop
        self.n_units_r = n_units_r
        self.n_units_r_small = n_units_r_small
        self.n_units_out = n_units_out
        self.n_units_out_prop = n_units_out_prop
        self.nonlin = nonlin

        self.penalty_l2 = penalty_l2
        self.penalty_orthogonal = penalty_orthogonal
        self.penalty_disc = penalty_disc
        self.reg_diff = reg_diff
        self.penalty_diff = penalty_diff
        self.same_init = same_init

        self.step_size = step_size
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min

        self.seed = seed
        self.n_iter_print = n_iter_print

    def _get_predict_function(self) -> Callable:
        return predict_snet3

    def _get_train_function(self) -> Callable:
        return train_snet3


# SNET-3 -------------------------------------------------------------
def train_snet3(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    binary_y: bool = False,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_r: int = DEFAULT_UNITS_R_BIG_S3,
    n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S3,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    n_units_out_prop: int = DEFAULT_UNITS_OUT,
    n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    penalty_disc: float = DEFAULT_PENALTY_DISC,
    penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
    step_size: float = DEFAULT_STEP_SIZE,
    n_iter: int = DEFAULT_N_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    early_stopping: bool = True,
    n_iter_min: int = DEFAULT_N_ITER_MIN,
    patience: int = DEFAULT_PATIENCE,
    n_iter_print: int = DEFAULT_N_ITER_PRINT,
    seed: int = DEFAULT_SEED,
    return_val_loss: bool = False,
    reg_diff: bool = False,
    penalty_diff: float = DEFAULT_PENALTY_L2,
    nonlin: str = DEFAULT_NONLIN,
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
    same_init: bool = False,
) -> Any:
    """
    SNet-3, based on the decompostion used in Hassanpour and Greiner (2020)
    """
    # function to train a net with 3 representations
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    if not reg_diff:
        penalty_diff = penalty_l2

    # get validation split (can be none)
    X, y, w, X_val, y_val, w_val, val_string = make_val_split(
        X, y, w, val_split_prop=val_split_prop, seed=seed
    )
    n = X.shape[0]  # could be different from before due to split

    # get representation layers
    init_fun_repr, predict_fun_repr = ReprBlock(
        n_layers=n_layers_r, n_units=n_units_r, nonlin=nonlin
    )
    init_fun_repr_small, predict_fun_repr_small = ReprBlock(
        n_layers=n_layers_r, n_units=n_units_r_small, nonlin=nonlin
    )

    # get output head functions (output heads share same structure)
    init_fun_head_po, predict_fun_head_po = OutputHead(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=binary_y,
        nonlin=nonlin,
    )
    # add propensity head
    init_fun_head_prop, predict_fun_head_prop = OutputHead(
        n_layers_out=n_layers_out_prop,
        n_units_out=n_units_out_prop,
        binary_y=True,
        nonlin=nonlin,
    )

    def init_fun_snet3(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        # param should look like [repr_c, repr_o, repr_t, po_0, po_1, prop]
        # initialise representation layers
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr_c = init_fun_repr(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        input_shape_repr_small, param_repr_o = init_fun_repr_small(
            layer_rng, input_shape
        )
        rng, layer_rng = random.split(rng)
        _, param_repr_w = init_fun_repr_small(layer_rng, input_shape)

        # each head gets two representations
        input_shape_repr = input_shape_repr[:-1] + (
            input_shape_repr[-1] + input_shape_repr_small[-1],
        )

        # initialise output heads
        rng, layer_rng = random.split(rng)
        if same_init:
            # initialise both on same values
            input_shape, param_0 = init_fun_head_po(layer_rng, input_shape_repr)
            input_shape, param_1 = init_fun_head_po(layer_rng, input_shape_repr)
        else:
            input_shape, param_0 = init_fun_head_po(layer_rng, input_shape_repr)
            rng, layer_rng = random.split(rng)
            input_shape, param_1 = init_fun_head_po(layer_rng, input_shape_repr)
        rng, layer_rng = random.split(rng)
        input_shape, param_prop = init_fun_head_prop(layer_rng, input_shape_repr)
        return input_shape, [
            param_repr_c,
            param_repr_o,
            param_repr_w,
            param_0,
            param_1,
            param_prop,
        ]

    # Define loss functions
    # loss functions for the head
    if not binary_y:

        def loss_head(
            params: List,
            batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            penalty: float,
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            preds = predict_fun_head_po(params, inputs)
            return jnp.sum(weights * ((preds - targets) ** 2))

    else:

        def loss_head(
            params: List,
            batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            penalty: float,
        ) -> jnp.ndarray:
            # log loss function
            inputs, targets, weights = batch
            preds = predict_fun_head_po(params, inputs)
            return -jnp.sum(
                weights
                * (targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
            )

    def loss_head_prop(
        params: List,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        penalty: float,
    ) -> jnp.ndarray:
        # log loss function for propensities
        inputs, targets = batch
        preds = predict_fun_head_prop(params, inputs)
        return -jnp.sum(targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))

    # complete loss function for all parts
    @jit
    def loss_snet3(
        params: List,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        penalty_l2: float,
        penalty_orthogonal: float,
        penalty_disc: float,
    ) -> jnp.ndarray:
        # params: list[repr_c, repr_o, repr_t, po_0, po_1, prop]
        # batch: (X, y, w)
        X, y, w = batch

        # get representation
        reps_c = predict_fun_repr(params[0], X)
        reps_o = predict_fun_repr_small(params[1], X)
        reps_w = predict_fun_repr_small(params[2], X)

        # concatenate
        reps_po = _concatenate_representations((reps_c, reps_o))
        reps_prop = _concatenate_representations((reps_c, reps_w))

        # pass down to heads
        loss_0 = loss_head(params[3], (reps_po, y, 1 - w), penalty_l2)
        loss_1 = loss_head(params[4], (reps_po, y, w), penalty_l2)

        # pass down to propensity head
        loss_prop = loss_head_prop(params[5], (reps_prop, w), penalty_l2)
        weightsq_prop = sum(
            [
                jnp.sum(params[5][i][0] ** 2)
                for i in range(0, 2 * n_layers_out_prop + 1, 2)
            ]
        )

        # which variable has impact on which representation
        col_c = _get_absolute_rowsums(params[0][0][0])
        col_o = _get_absolute_rowsums(params[1][0][0])
        col_w = _get_absolute_rowsums(params[2][0][0])
        loss_o = penalty_orthogonal * (
            jnp.sum(col_c * col_o + col_c * col_w + col_w * col_o)
        )

        # is rep_o balanced between groups?
        loss_disc = penalty_disc * mmd2_lin(reps_o, w)

        # weight decay on representations
        weightsq_body = sum(
            [
                sum(
                    [jnp.sum(params[j][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
                )
                for j in range(3)
            ]
        )
        weightsq_head = heads_l2_penalty(
            params[3], params[4], n_layers_out, reg_diff, penalty_l2, penalty_diff
        )

        if not avg_objective:
            return (
                loss_0
                + loss_1
                + loss_prop
                + loss_o
                + loss_disc
                + 0.5 * (penalty_l2 * (weightsq_body + weightsq_prop) + weightsq_head)
            )
        else:
            n_batch = y.shape[0]
            return (
                (loss_0 + loss_1) / n_batch
                + loss_prop / n_batch
                + loss_o
                + loss_disc
                + 0.5 * (penalty_l2 * (weightsq_body + weightsq_prop) + weightsq_head)
            )

    # Define optimisation routine
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    @jit
    def update(
        i: int,
        state: dict,
        batch: jnp.ndarray,
        penalty_l2: float,
        penalty_orthogonal: float,
        penalty_disc: float,
    ) -> jnp.ndarray:
        # updating function
        params = get_params(state)
        return opt_update(
            i,
            grad(loss_snet3)(
                params, batch, penalty_l2, penalty_orthogonal, penalty_disc
            ),
            state,
        )

    # initialise states
    _, init_params = init_fun_snet3(rng_key, input_shape)
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
            next_batch = X[idx_next, :], y[idx_next, :], w[idx_next]
            opt_state = update(
                i * n_batches + b,
                opt_state,
                next_batch,
                penalty_l2,
                penalty_orthogonal,
                penalty_disc,
            )

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            l_curr = loss_snet3(
                params_curr,
                (X_val, y_val, w_val),
                penalty_l2,
                penalty_orthogonal,
                penalty_disc,
            )

        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss {l_curr}")

        if early_stopping and ((i + 1) * n_batches > n_iter_min):
            # check if loss updated
            if l_curr < l_best:
                l_best = l_curr
                p_curr = 0
                params_best = params_curr
            else:
                if onp.isnan(l_curr):
                    # if diverged, return best
                    return params_best, (
                        predict_fun_repr,
                        predict_fun_head_po,
                        predict_fun_head_prop,
                    )
                p_curr = p_curr + 1

            if p_curr > patience:
                if return_val_loss:
                    # return loss without penalty
                    l_final = loss_snet3(params_curr, (X_val, y_val, w_val), 0, 0, 0)
                    return (
                        params_curr,
                        (predict_fun_repr, predict_fun_head_po, predict_fun_head_prop),
                        l_final,
                    )

                return params_curr, (
                    predict_fun_repr,
                    predict_fun_head_po,
                    predict_fun_head_prop,
                )

    # return the parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss_snet3(get_params(opt_state), (X_val, y_val, w_val), 0, 0)
        return (
            trained_params,
            (predict_fun_repr, predict_fun_head_po, predict_fun_head_prop),
            l_final,
        )

    return trained_params, (
        predict_fun_repr,
        predict_fun_head_po,
        predict_fun_head_prop,
    )


def predict_snet3(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head, predict_fun_prop = predict_funs
    param_repr_c, param_repr_o, param_repr_t = (
        trained_params[0],
        trained_params[1],
        trained_params[2],
    )
    param_0, param_1, param_prop = (
        trained_params[3],
        trained_params[4],
        trained_params[5],
    )

    # get representations
    rep_c = predict_fun_repr(param_repr_c, X)
    rep_o = predict_fun_repr(param_repr_o, X)
    rep_w = predict_fun_repr(param_repr_t, X)

    # concatenate
    reps_po = jnp.concatenate((rep_c, rep_o), axis=1)
    reps_prop = jnp.concatenate((rep_c, rep_w), axis=1)

    # get potential outcomes
    mu_0 = predict_fun_head(param_0, reps_po)
    mu_1 = predict_fun_head(param_1, reps_po)

    te = mu_1 - mu_0
    if return_prop:
        # get propensity
        prop = predict_fun_prop(param_prop, reps_prop)

    # stack other outputs
    if return_po:
        if return_prop:
            return te, mu_0, mu_1, prop
        else:
            return te, mu_0, mu_1
    else:
        if return_prop:
            return te, prop
        else:
            return te
