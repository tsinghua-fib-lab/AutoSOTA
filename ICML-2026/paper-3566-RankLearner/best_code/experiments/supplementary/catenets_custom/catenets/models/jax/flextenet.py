"""
Module implements FlexTENet, also referred to as the 'flexible approach' in "On inductive biases
for heterogeneous treatment effect estimation", Curth & vd Schaar (2021).
"""
# Author: Alicia Curth
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import grad, jit, random
from jax.example_libraries import optimizers
from jax.example_libraries.stax import (
    Dense,
    Sigmoid,
    elu,
    glorot_normal,
    normal,
    serial,
)

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
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_PENALTY_ORTHOGONAL,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
    N_SUBSPACES,
)
from catenets.models.jax.base import BaseCATENet
from catenets.models.jax.model_utils import check_shape_1d_data, make_val_split


class FlexTENet(BaseCATENet):
    """
    Module implements FlexTENet, an architecture for treatment effect estimation that allows for
    both shared and private information in each layer of the network.

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
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
    penalty_l2: float
        l2 (ridge) penalty
    penalty_l2_p: float
        l2 (ridge) penalty for private layers
    penalty_orthogonal: float
        orthogonalisation penalty
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
    opt: str, default 'adam'
        Optimizer to use, accepts 'adam' and 'sgd'
    shared_repr: bool, False
        Whether to use a shared representation block as TARNet
    pretrain_shared: bool, False
        Whether to pretrain the shared component of the network while freezing the private
        parameters
    same_init: bool, True
        Whether to use the same initialisation for all private spaces
    lr_scale: float
        Whether to scale down the learning rate after unfreezing the private components of the
        network (only used if pretrain_shared=True)
    normalize_ortho: bool, False
        Whether to normalize the orthogonality penalty (by depth of network)
    """

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_s_out: int = DEFAULT_DIM_S_OUT,
        n_units_p_out: int = DEFAULT_DIM_P_OUT,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_s_r: int = DEFAULT_DIM_S_R,
        n_units_p_r: int = DEFAULT_DIM_P_R,
        private_out: bool = False,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        penalty_l2_p: float = DEFAULT_PENALTY_L2,
        penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
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
        opt: str = "adam",
        shared_repr: bool = False,
        pretrain_shared: bool = False,
        same_init: bool = True,
        lr_scale: float = 10,
        normalize_ortho: bool = False,
    ) -> None:
        self.binary_y = binary_y

        self.n_layers_r = n_layers_r
        self.n_layers_out = n_layers_out
        self.n_units_s_out = n_units_s_out
        self.n_units_p_out = n_units_p_out
        self.n_units_s_r = n_units_s_r
        self.n_units_p_r = n_units_p_r
        self.private_out = private_out

        self.penalty_orthogonal = penalty_orthogonal
        self.penalty_l2 = penalty_l2
        self.penalty_l2_p = penalty_l2_p
        self.step_size = step_size
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.opt = opt
        self.same_init = same_init
        self.shared_repr = shared_repr
        self.normalize_ortho = normalize_ortho
        self.pretrain_shared = pretrain_shared
        self.lr_scale = lr_scale

        self.seed = seed
        self.n_iter_print = n_iter_print
        self.return_val_loss = return_val_loss

    def _get_train_function(self) -> Callable:
        return train_flextenet

    def _get_predict_function(self) -> Callable:
        return predict_flextenet


def train_flextenet(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    binary_y: bool = False,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_s_out: int = DEFAULT_DIM_S_OUT,
    n_units_p_out: int = DEFAULT_DIM_P_OUT,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_s_r: int = DEFAULT_DIM_S_R,
    n_units_p_r: int = DEFAULT_DIM_P_R,
    private_out: bool = False,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    penalty_l2_p: float = DEFAULT_PENALTY_L2,
    penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
    step_size: float = DEFAULT_STEP_SIZE,
    n_iter: int = DEFAULT_N_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    early_stopping: bool = True,
    patience: int = DEFAULT_PATIENCE,
    n_iter_min: int = DEFAULT_N_ITER_MIN,
    avg_objective: bool = True,
    n_iter_print: int = DEFAULT_N_ITER_PRINT,
    seed: int = DEFAULT_SEED,
    return_val_loss: bool = False,
    opt: str = "adam",
    shared_repr: bool = False,
    pretrain_shared: bool = False,
    same_init: bool = True,
    lr_scale: float = 10,
    normalize_ortho: bool = False,
    nonlin: str = DEFAULT_NONLIN,
    n_units_r: Optional[int] = None,
    n_units_out: Optional[int] = None,
) -> Tuple:  # TODO incorporate different nonlins here
    # function to train a single output head
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

    # get output head
    init_fun, predict_fun = FlexTENetArchitecture(
        n_layers_out=n_layers_out,
        n_layers_r=n_layers_r,
        n_units_p_r=n_units_p_r,
        n_units_p_out=n_units_p_out,
        n_units_s_r=n_units_s_r,
        n_units_s_out=n_units_s_out,
        private_out=private_out,
        shared_repr=shared_repr,
        same_init=same_init,
        binary_y=binary_y,
    )

    # get functions
    if not binary_y:
        # define loss and grad
        @jit
        def loss(
            params: jnp.ndarray,
            batch: jnp.ndarray,
            penalty_l2: float,
            penalty_l2_p: float,
            penalty_orthogonal: float,
            mode: int,
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets = batch
            preds = predict_fun(params, inputs, mode=mode)
            penalty = _compute_penalty(
                params,
                n_layers_out,
                n_layers_r,
                private_out,
                penalty_l2,
                penalty_l2_p,
                penalty_orthogonal,
                shared_repr,
                normalize_ortho,
                mode,
            )
            if not avg_objective:
                return jnp.sum((preds - targets) ** 2) + penalty
            else:
                return jnp.average((preds - targets) ** 2) + penalty

    else:
        # get loss and grad
        @jit
        def loss(
            params: jnp.ndarray,
            batch: jnp.ndarray,
            penalty_l2: float,
            penalty_l2_p: float,
            penalty_orthogonal: float,
            mode: int,
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets = batch
            preds = predict_fun(params, inputs, mode=mode)
            penalty = _compute_penalty(
                params,
                n_layers_out,
                n_layers_r,
                private_out,
                penalty_l2,
                penalty_l2_p,
                penalty_orthogonal,
                shared_repr,
                normalize_ortho,
                mode,
            )
            if not avg_objective:
                return (
                    -jnp.sum(
                        targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds)
                    )
                    + penalty
                )
            else:
                return (
                    -jnp.average(
                        targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds)
                    )
                    + penalty
                )

    # set optimization routine
    # set optimizer
    if opt == "adam":
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
    elif opt == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd(step_size=step_size)
    else:
        raise ValueError("opt should be adam or sgd")

    # set update function
    @jit
    def update(
        i: int,
        state: dict,
        batch: jnp.ndarray,
        penalty_l2: float,
        penalty_l2_p: float,
        penalty_orthogonal: float,
        mode: int,
    ) -> jnp.ndarray:
        params = get_params(state)
        g_params = grad(loss)(
            params, batch, penalty_l2, penalty_l2_p, penalty_orthogonal, mode
        )
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
    if not pretrain_shared:  # train entire model together
        for i in range(n_iter):
            # shuffle data for minibatches
            onp.random.shuffle(train_indices)
            for b in range(n_batches):
                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]
                next_batch = (X[idx_next, :], w[idx_next]), y[idx_next, :]
                opt_state = update(
                    i * n_batches + b,
                    opt_state,
                    next_batch,
                    penalty_l2,
                    penalty_l2_p,
                    penalty_orthogonal,
                    mode=1,
                )

            if (i % n_iter_print == 0) or early_stopping:
                params_curr = get_params(opt_state)
                l_curr = loss(
                    params_curr,
                    ((X_val, w_val), y_val),
                    penalty_l2,
                    penalty_l2_p,
                    penalty_orthogonal,
                    mode=1,
                )

            if i % n_iter_print == 0:
                log.debug(f"Epoch: {i}, current {val_string} loss: {l_curr}")

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
                        l_final = loss(
                            trained_params, ((X_val, w_val), y_val), 0, 0, 0, mode=1
                        )
                        return trained_params, predict_fun, l_final

                    return trained_params, predict_fun

        # get final parameters
        trained_params = get_params(opt_state)

        if return_val_loss:
            # return loss without penalty
            l_final = loss(trained_params, ((X_val, w_val), y_val), 0, 0, 0, mode=1)
            return trained_params, predict_fun, l_final

        return trained_params, predict_fun
    else:
        # Step 1: pretrain only shared bit of network (mode=0)
        for i in range(n_iter):
            # shuffle data for minibatches
            onp.random.shuffle(train_indices)
            for b in range(n_batches):
                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]
                next_batch = (X[idx_next, :], w[idx_next]), y[idx_next, :]
                opt_state = update(
                    i * n_batches + b,
                    opt_state,
                    next_batch,
                    penalty_l2,
                    penalty_l2_p,
                    penalty_orthogonal,
                    mode=0,
                )

            if (i % n_iter_print == 0) or early_stopping:
                params_curr = get_params(opt_state)
                l_curr = loss(
                    params_curr,
                    ((X_val, w_val), y_val),
                    penalty_l2,
                    penalty_l2_p,
                    penalty_orthogonal,
                    mode=0,
                )

            if i % n_iter_print == 0:
                log.debug(
                    f"Pre-training epoch: {i}, current {val_string} loss: {l_curr}"
                )

            if early_stopping and ((i + 1) * n_batches > n_iter_min):
                # check if loss updated
                if l_curr < l_best:
                    l_best = l_curr
                    p_curr = 0
                else:
                    p_curr = p_curr + 1

                if p_curr > patience:
                    break

        # get final parameters
        pre_trained_params = get_params(opt_state)

        # Step 2: train also private parts of network (mode=1)
        # set new optimizer
        if opt == "adam":
            opt_init2, opt_update2, get_params2 = optimizers.adam(
                step_size=step_size / lr_scale
            )
        elif opt == "sgd":
            opt_init2, opt_update2, get_params2 = optimizers.sgd(
                step_size=step_size / lr_scale
            )
        else:
            raise ValueError("opt should be adam or sgd")

        # set update function
        @jit
        def update2(
            i: int,
            state: dict,
            batch: jnp.ndarray,
            penalty_l2: float,
            penalty_l2_p: float,
            penalty_orthogonal: float,
            mode: int,
        ) -> Any:
            params = get_params(state)
            g_params = grad(loss)(
                params, batch, penalty_l2, penalty_l2_p, penalty_orthogonal, mode
            )
            return opt_update2(i, g_params, state)

        opt_state = opt_init2(pre_trained_params)
        l_best = LARGE_VAL
        p_curr = 0

        # train full
        for i in range(n_iter):
            # shuffle data for minibatches
            onp.random.shuffle(train_indices)
            for b in range(n_batches):
                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]
                next_batch = (X[idx_next, :], w[idx_next]), y[idx_next, :]
                opt_state = update2(
                    i * n_batches + b,
                    opt_state,
                    next_batch,
                    penalty_l2,
                    penalty_l2_p,
                    penalty_orthogonal,
                    mode=1,
                )

            if (i % n_iter_print == 0) or early_stopping:
                params_curr = get_params2(opt_state)
                l_curr = loss(
                    params_curr,
                    ((X_val, w_val), y_val),
                    penalty_l2,
                    penalty_l2_p,
                    penalty_orthogonal,
                    mode=1,
                )

            if i % n_iter_print == 0:
                log.debug(f"Epoch: {i}, current {val_string} loss: {l_curr}")

            if early_stopping and ((i + 1) * n_batches > n_iter_min):
                # check if loss updated
                if l_curr < l_best:
                    l_best = l_curr
                    p_curr = 0
                else:
                    p_curr = p_curr + 1

                if p_curr > patience:
                    trained_params = get_params2(opt_state)

                    if return_val_loss:
                        # return loss without penalty
                        l_final = loss(
                            trained_params, ((X_val, w_val), y_val), 0, 0, 0, mode=1
                        )
                        return trained_params, predict_fun, l_final

                    return trained_params, predict_fun

        # get final parameters
        trained_params = get_params2(opt_state)

        if return_val_loss:
            # return loss without penalty
            l_final = loss(trained_params, ((X_val, w_val), y_val), 0, 0, 0, mode=1)
            return trained_params, predict_fun, l_final

        return trained_params, predict_fun


def predict_flextenet(
    X: jnp.ndarray,
    trained_params: jnp.ndarray,
    predict_funs: Callable,
    return_po: bool = False,
    return_prop: bool = False,
) -> Any:
    # unpack inputs
    n, _ = X.shape

    W1 = check_shape_1d_data(jnp.ones(n))
    W0 = check_shape_1d_data(jnp.zeros(n))

    # get potential outcomes
    mu_0 = predict_funs(trained_params, (X, W0))
    mu_1 = predict_funs(trained_params, (X, W1))

    te = mu_1 - mu_0
    if return_prop:
        raise ValueError("does not have propensity score estimator")

    # stack other outputs
    if return_po:
        return te, mu_0, mu_1
    else:
        return te


# helper functions for training
def _get_cos_reg(
    params_0: jnp.ndarray, params_1: jnp.ndarray, normalize: bool
) -> jnp.ndarray:
    if normalize:
        params_0 = params_0 / jnp.linalg.norm(params_0, axis=0)
        params_1 = params_1 / jnp.linalg.norm(params_1, axis=0)

    return jnp.linalg.norm(jnp.dot(jnp.transpose(params_0), params_1), "fro") ** 2


def _compute_ortho_penalty_asymmetric(
    params: jnp.ndarray,
    n_layers_out: int,
    n_layers_r: int,
    private_out: int,
    penalty_orthogonal: float,
    shared_repr: bool,
    normalize_ortho: bool,
    mode: int = 1,
) -> float:
    # where to start counting: is there a fully shared representation?
    if shared_repr:
        lb = 2 * n_layers_r
    else:
        lb = 0

    n_in = [
        params[i][0][0].shape[0] for i in range(lb, 2 * (n_layers_out + n_layers_r), 2)
    ]

    ortho_body = _get_cos_reg(params[lb][1][0], params[lb][2][0], normalize_ortho)
    ortho_body = ortho_body + sum(
        [
            _get_cos_reg(
                params[i][0][0],
                params[i][1][0][: n_in[int(i / 2 - lb / 2)], :],
                normalize_ortho,
            )
            + _get_cos_reg(
                params[i][0][0],
                params[i][2][0][: n_in[int(i / 2 - lb / 2)], :],
                normalize_ortho,
            )
            for i in range(lb, 2 * (n_layers_out + n_layers_r), 2)
        ]
    )

    if not private_out:
        # add also orthogonal regularization on final layer
        idx_out = 2 * (n_layers_r + n_layers_out)
        n_idx = params[idx_out][0][0].shape[0]

        ortho_body = (
            ortho_body
            + _get_cos_reg(
                params[idx_out][0][0],
                params[idx_out][1][0][:n_idx, :],
                normalize_ortho,
            )
            + _get_cos_reg(
                params[idx_out][0][0], params[idx_out][2][0][:n_idx, :], normalize_ortho
            )
        )

    return mode * penalty_orthogonal * ortho_body


def _compute_penalty_l2(
    params: jnp.ndarray,
    n_layers_out: int,
    n_layers_r: int,
    private_out: int,
    penalty_l2: float,
    penalty_l2_p: float,
    shared_repr: bool,
    mode: int = 1,
) -> jnp.ndarray:
    n_bodys = N_SUBSPACES

    # compute l2 penalty
    if shared_repr:
        # get representation and then heads
        weightsq_body = penalty_l2 * sum(
            [jnp.sum(params[i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        weightsq_body = weightsq_body + penalty_l2 * sum(
            [
                jnp.sum(params[i][0][0] ** 2)
                for i in range(2 * n_layers_r, 2 * (n_layers_out + n_layers_r), 2)
            ]
        )
        weightsq_body = weightsq_body + penalty_l2_p * mode * sum(
            [
                sum(
                    [
                        jnp.sum(params[i][j][0] ** 2)
                        for i in range(
                            2 * n_layers_r, 2 * (n_layers_out + n_layers_r), 2
                        )
                    ]
                )
                for j in range(1, n_bodys)
            ]
        )
    else:
        weightsq_body = penalty_l2 * sum(
            [
                jnp.sum(params[i][0][0] ** 2)
                for i in range(0, 2 * (n_layers_out + n_layers_r), 2)
            ]
        )
        weightsq_body = weightsq_body + penalty_l2_p * mode * sum(
            [
                sum(
                    [
                        jnp.sum(params[i][j][0] ** 2)
                        for i in range(0, 2 * (n_layers_out + n_layers_r), 2)
                    ]
                )
                for j in range(1, n_bodys)
            ]
        )

    idx_out = 2 * (n_layers_r + n_layers_out)
    if private_out:
        weightsq = (
            weightsq_body
            + penalty_l2 * jnp.sum(params[idx_out][0][0] ** 2)
            + jnp.sum(params[idx_out][1][0] ** 2)
        )
    else:
        weightsq = (
            weightsq_body
            + penalty_l2 * jnp.sum(params[idx_out][0][0] ** 2)
            + penalty_l2_p * mode * jnp.sum(params[idx_out][1][0] ** 2)
            + penalty_l2_p * mode * jnp.sum(params[idx_out][2][0] ** 2)
        )

    return 0.5 * weightsq


def _compute_penalty(
    params: jnp.ndarray,
    n_layers_out: int,
    n_layers_r: int,
    private_out: int,
    penalty_l2: float,
    penalty_l2_p: float,
    penalty_orthogonal: float,
    shared_repr: bool,
    normalize_ortho: bool,
    mode: int = 1,
) -> jnp.ndarray:
    l2_penalty = _compute_penalty_l2(
        params,
        n_layers_out,
        n_layers_r,
        private_out,
        penalty_l2,
        penalty_l2_p,
        shared_repr,
        mode,
    )

    ortho_penalty = _compute_ortho_penalty_asymmetric(
        params,
        n_layers_out,
        n_layers_r,
        private_out,
        penalty_orthogonal,
        shared_repr,
        normalize_ortho,
        mode,
    )

    return l2_penalty + ortho_penalty


# ------------------------------------------------------------
# construction of FlexTENetlayers/architecture
def SplitLayerAsymmetric(
    n_units_s: int, n_units_p: int, first_layer: bool = False, same_init: bool = True
) -> Tuple:
    # create multitask layer has shape [shared, private_0, private_1]
    init_s, apply_s = Dense(n_units_s)
    init_p, apply_p = Dense(n_units_p)

    def init_fun(rng: float, input_shape: Tuple) -> Tuple:
        if first_layer:  # put input shape in expected format
            input_shape = (input_shape, input_shape, input_shape)
        out_shape = (
            input_shape[0][:-1] + (n_units_s,),
            input_shape[1][:-1] + (n_units_p + n_units_s,),
            input_shape[2][:-1] + (n_units_p + n_units_s,),
        )

        rng_1, rng_2, rng_3 = random.split(rng, N_SUBSPACES)
        if same_init:  # use same init for the two private layers
            return out_shape, (
                init_s(rng_1, input_shape[0])[1],
                init_p(rng_2, input_shape[1])[1],
                init_p(rng_2, input_shape[2])[1],
            )
        else:  # initialise all separately
            return out_shape, (
                init_s(rng_1, input_shape[0])[1],
                init_p(rng_2, input_shape[1])[1],
                init_p(rng_3, input_shape[2])[1],
            )

    def apply_fun(params: jnp.ndarray, inputs: jnp.ndarray, **kwargs: Any) -> Tuple:
        mode = kwargs["mode"] if "mode" in kwargs.keys() else 1
        if first_layer:
            # X is the only input
            X, W = inputs
            rep_s = apply_s(params[0], X)
            rep_p0 = mode * apply_p(params[1], X)
            rep_p1 = mode * apply_p(params[2], X)
        else:
            X_s, X_p0, X_p1, W = inputs
            rep_s = apply_s(params[0], X_s)
            rep_p0 = mode * apply_p(params[1], jnp.concatenate([X_s, X_p0], axis=1))
            rep_p1 = mode * apply_p(params[2], jnp.concatenate([X_s, X_p1], axis=1))
        return (rep_s, rep_p0, rep_p1, W)

    return init_fun, apply_fun


def TEOutputLayerAsymmetric(private: bool = True, same_init: bool = True) -> Tuple:
    init_f, apply_f = Dense(1)
    if private:
        # the two output layers are private
        def init_fun(rng: float, input_shape: Tuple) -> Tuple:
            out_shape = input_shape[1][:-1] + (1,)
            rng_1, rng_2 = random.split(rng, N_SUBSPACES - 1)
            return out_shape, (
                init_f(rng_1, input_shape[1])[1],
                init_f(rng_2, input_shape[2])[1],
            )

        def apply_fun(params: jnp.ndarray, inputs: Tuple, **kwargs: Any) -> jnp.ndarray:
            X_s, X_p0, X_p1, W = inputs
            rep_p0 = apply_f(params[0], jnp.concatenate([X_s, X_p0], axis=1))
            rep_p1 = apply_f(params[1], jnp.concatenate([X_s, X_p1], axis=1))
            return (1 - W) * rep_p0 + W * rep_p1

    else:
        # also have a shared piece of output layer
        def init_fun(rng: float, input_shape: Tuple) -> Tuple:
            out_shape = input_shape[1][:-1] + (1,)
            rng_1, rng_2, rng_3 = random.split(rng, N_SUBSPACES)
            if same_init:
                return out_shape, (
                    init_f(rng_1, input_shape[0])[1],
                    init_f(rng_2, input_shape[1])[1],
                    init_f(rng_2, input_shape[2])[1],
                )
            else:
                return out_shape, (
                    init_f(rng_1, input_shape[0])[1],
                    init_f(rng_2, input_shape[1])[1],
                    init_f(rng_3, input_shape[2])[1],
                )

        def apply_fun(params: jnp.ndarray, inputs: Tuple, **kwargs: Any) -> jnp.ndarray:
            mode = kwargs["mode"] if "mode" in kwargs.keys() else 1
            X_s, X_p0, X_p1, W = inputs
            rep_s = apply_f(params[0], X_s)
            rep_p0 = mode * apply_f(params[1], jnp.concatenate([X_s, X_p0], axis=1))
            rep_p1 = mode * apply_f(params[2], jnp.concatenate([X_s, X_p1], axis=1))
            return (1 - W) * rep_p0 + W * rep_p1 + rep_s

    return init_fun, apply_fun


def FlexTENetArchitecture(
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_s_out: int = DEFAULT_DIM_S_OUT,
    n_units_p_out: int = DEFAULT_DIM_P_OUT,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_s_r: int = DEFAULT_DIM_S_R,
    n_units_p_r: int = DEFAULT_DIM_P_R,
    private_out: bool = False,
    binary_y: bool = False,
    shared_repr: bool = False,
    same_init: bool = True,
) -> Any:
    if n_layers_out < 1:
        raise ValueError(
            "FlexTENet needs at least one hidden output layer (else there are no "
            "parameters to be shared)"
        )

    Nonlin_Elu = Elu_parallel
    Layer = SplitLayerAsymmetric
    Head = TEOutputLayerAsymmetric

    # give broader body (as in e.g. TARNet)
    has_body = n_layers_r > 0

    layers: Tuple = ()
    if has_body:
        # representation block first
        if shared_repr:  # fully shared representation as in TARNet
            layers = (DenseW(n_units_s_r), Elu_split)

            # add required number of layers
            for i in range(n_layers_r - 1):
                layers = (*layers, DenseW(n_units_s_r), Elu_split)

        else:  # shared AND private representations
            layers = (
                Layer(n_units_s_r, n_units_p_r, first_layer=True, same_init=same_init),
                Nonlin_Elu,
            )

            # add required number of layers
            for i in range(n_layers_r - 1):
                layers = (
                    *layers,
                    Layer(n_units_s_r, n_units_p_r, same_init=same_init),
                    Nonlin_Elu,
                )
    else:
        layers = ()

    # add output layers
    first_layer = (has_body is False) | (shared_repr is True)
    layers = (
        *layers,
        Layer(
            n_units_s_out, n_units_p_out, first_layer=first_layer, same_init=same_init
        ),
        Nonlin_Elu,
    )

    if n_layers_out > 1:
        # add required number of layers
        for i in range(n_layers_out - 1):
            layers = (
                *layers,
                Layer(n_units_s_out, n_units_p_out, same_init=same_init),
                Nonlin_Elu,
            )

    # return final architecture
    if not binary_y:
        return serial(*layers, Head(private=private_out, same_init=same_init))
    else:
        return serial(*layers, Head(private=private_out, same_init=same_init), Sigmoid)


# ------------------------------------------------
# rewrite some jax.stax code to allow different input types to be passed
def elementwise_split(fun: Callable, **fun_kwargs: Any) -> Tuple:
    """Layer that applies a scalar function elementwise on its inputs. Adapted from original
    jax.stax to skip treatment indicator.

    Input looks like: X, t = inputs"""

    def init_fun(rng: float, input_shape: Tuple) -> Tuple:
        return (input_shape, ())

    def apply_fun(params: jnp.ndarray, inputs: jnp.ndarray, **kwargs: Any) -> Tuple:
        return fun(inputs[0], **fun_kwargs), inputs[1]

    return init_fun, apply_fun


Elu_split = elementwise_split(elu)


def elementwise_parallel(fun: Callable, **fun_kwargs: Any) -> Tuple:
    """Layer that applies a scalar function elementwise on its inputs. Adapted from original
    jax.stax to allow three inputs and to skip treatment indicator.

    Input looks like: X_s, X_p0, X_p1, t = inputs
    """

    def init_fun(rng: float, input_shape: Tuple) -> Tuple:
        return input_shape, ()

    def apply_fun(params: jnp.ndarray, inputs: jnp.ndarray, **kwargs: Any) -> Tuple:
        return (
            fun(inputs[0], **fun_kwargs),
            fun(inputs[1], **fun_kwargs),
            fun(inputs[2], **fun_kwargs),
            inputs[3],
        )

    return init_fun, apply_fun


Elu_parallel = elementwise_parallel(elu)


def DenseW(
    out_dim: int, W_init: Callable = glorot_normal(), b_init: Callable = normal()
) -> Tuple:
    """Layer constructor function for a dense (fully-connected) layer. Adapted to allow passing
    treatment indicator through layer without using it"""

    def init_fun(rng: float, input_shape: Tuple) -> Tuple:
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(
        params: jnp.ndarray, inputs: jnp.ndarray, **kwargs: Any
    ) -> jnp.ndarray:
        W, b = params
        x, t = inputs
        return (jnp.dot(x, W) + b, t)

    return init_fun, apply_fun
