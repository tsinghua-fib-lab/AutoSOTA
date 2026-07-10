"""
Module implements SNet class as discussed in Curth & van der Schaar (2021)
"""
# Author: Alicia Curth
from typing import Callable, List, Tuple

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
from catenets.models.jax.disentangled_nets import (
    _concatenate_representations,
    _get_absolute_rowsums,
)
from catenets.models.jax.flextenet import _get_cos_reg
from catenets.models.jax.model_utils import (
    check_shape_1d_data,
    heads_l2_penalty,
    make_val_split,
)
from catenets.models.jax.representation_nets import mmd2_lin

DEFAULT_UNITS_R_BIG_S = 100
DEFAULT_UNITS_R_SMALL_S = 50


class SNet(BaseCATENet):
    """
    Class implements SNet as discussed in Curth & van der Schaar (2021). Additionally to the
    version implemented in the AISTATS paper, we also include an implementation that does not
    have propensity heads (set with_prop=False)

    Parameters
    ----------
    with_prop: bool, True
        Whether to include propensity head
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
        If withprop=True: Number of hidden units in representation layer shared by propensity score
        and outcome  function (the 'confounding factor') and in the ('instrumental factor')
        If withprop=False: Number of hidden units in representation shared across PO function
    n_units_r_small: int
        If withprop=True: Number of hidden units in representation layer of the 'outcome factor'
        and each PO functions private representation
        if withprop=False: Number of hidden units in each PO functions private representation
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
    ortho_reg_type: str, 'abs'
        Which type of orthogonalization to use. 'abs' uses the (hard) disentanglement described
        in AISTATS paper, 'fro' uses frobenius norm as in FlexTENet
    """

    def __init__(
        self,
        with_prop: bool = True,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R_BIG_S,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
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
        reg_diff: bool = False,
        penalty_diff: float = DEFAULT_PENALTY_L2,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        same_init: bool = False,
        ortho_reg_type: str = "abs",
    ):
        self.with_prop = with_prop
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
        self.ortho_reg_type = ortho_reg_type

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
        if self.with_prop:
            return predict_snet
        else:
            return predict_snet_noprop

    def _get_train_function(self) -> Callable:
        if self.with_prop:
            return train_snet
        else:
            return train_snet_noprop


def train_snet(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    binary_y: bool = False,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_r: int = DEFAULT_UNITS_R_BIG_S,
    n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
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
    patience: int = DEFAULT_PATIENCE,
    n_iter_min: int = DEFAULT_N_ITER_MIN,
    n_iter_print: int = DEFAULT_N_ITER_PRINT,
    seed: int = DEFAULT_SEED,
    return_val_loss: bool = False,
    reg_diff: bool = False,
    penalty_diff: float = DEFAULT_PENALTY_L2,
    nonlin: str = DEFAULT_NONLIN,
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
    with_prop: bool = True,
    same_init: bool = False,
    ortho_reg_type: str = "abs",
) -> Tuple:
    # function to train a net with 5 representations
    if not with_prop:
        raise ValueError("train_snet works only withprop=True")
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

    def init_fun_snet(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        # param should look like [param_repr_c, param_repr_o, param_repr_mu0, param_repr_mu1,
        #                              param_repr_w, param_0, param_1, param_prop]
        # initialise representation layers
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr_c = init_fun_repr(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        input_shape_repr_small, param_repr_o = init_fun_repr_small(
            layer_rng, input_shape
        )
        rng, layer_rng = random.split(rng)
        _, param_repr_mu0 = init_fun_repr_small(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        _, param_repr_mu1 = init_fun_repr_small(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        _, param_repr_w = init_fun_repr(layer_rng, input_shape)

        # prop and mu_0 each get two representations, mu_1 gets 3
        input_shape_repr_prop = input_shape_repr[:-1] + (2 * input_shape_repr[-1],)
        input_shape_repr_mu = input_shape_repr[:-1] + (
            input_shape_repr[-1] + (2 * input_shape_repr_small[-1]),
        )

        # initialise output heads
        rng, layer_rng = random.split(rng)
        if same_init:
            # initialise both on same values
            input_shape, param_0 = init_fun_head_po(layer_rng, input_shape_repr_mu)
            input_shape, param_1 = init_fun_head_po(layer_rng, input_shape_repr_mu)
        else:
            input_shape, param_0 = init_fun_head_po(layer_rng, input_shape_repr_mu)
            rng, layer_rng = random.split(rng)
            input_shape, param_1 = init_fun_head_po(layer_rng, input_shape_repr_mu)

        rng, layer_rng = random.split(rng)
        input_shape, param_prop = init_fun_head_prop(layer_rng, input_shape_repr_prop)
        return input_shape, [
            param_repr_c,
            param_repr_o,
            param_repr_mu0,
            param_repr_mu1,
            param_repr_w,
            param_0,
            param_1,
            param_prop,
        ]

    # Define loss functions
    # loss functions for the head
    if not binary_y:

        def loss_head(
            params: jnp.ndarray,
            batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            penalty: float,
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            preds = predict_fun_head_po(params, inputs)
            return jnp.sum(weights * ((preds - targets) ** 2))

    else:

        def loss_head(
            params: jnp.ndarray,
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
        params: jnp.ndarray, batch: Tuple[jnp.ndarray, jnp.ndarray], penalty: float
    ) -> jnp.ndarray:
        # log loss function for propensities
        inputs, targets = batch
        preds = predict_fun_head_prop(params, inputs)
        return -jnp.sum(targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))

    # define ortho-reg function
    if ortho_reg_type == "abs":

        def ortho_reg(params: jnp.ndarray) -> jnp.ndarray:
            col_c = _get_absolute_rowsums(params[0][0][0])
            col_o = _get_absolute_rowsums(params[1][0][0])
            col_mu0 = _get_absolute_rowsums(params[2][0][0])
            col_mu1 = _get_absolute_rowsums(params[3][0][0])
            col_w = _get_absolute_rowsums(params[4][0][0])
            return jnp.sum(
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

    elif ortho_reg_type == "fro":

        def ortho_reg(params: jnp.ndarray) -> jnp.ndarray:
            return (
                _get_cos_reg(params[0][0][0], params[1][0][0], False)
                + _get_cos_reg(params[0][0][0], params[2][0][0], False)
                + _get_cos_reg(params[0][0][0], params[3][0][0], False)
                + _get_cos_reg(params[0][0][0], params[4][0][0], False)
                + _get_cos_reg(params[1][0][0], params[2][0][0], False)
                + _get_cos_reg(params[1][0][0], params[3][0][0], False)
                + _get_cos_reg(params[1][0][0], params[4][0][0], False)
                + _get_cos_reg(params[2][0][0], params[3][0][0], False)
                + _get_cos_reg(params[2][0][0], params[4][0][0], False)
                + _get_cos_reg(params[3][0][0], params[4][0][0], False)
            )

    else:
        raise NotImplementedError(
            "train_snet_noprop supports only orthogonal regularization "
            "using absolute values or frobenious norms."
        )

    # complete loss function for all parts
    @jit
    def loss_snet(
        params: jnp.ndarray,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        penalty_l2: float,
        penalty_orthogonal: float,
        penalty_disc: float,
    ) -> jnp.ndarray:
        # params: # param should look like [param_repr_c, param_repr_o, param_repr_mu0,
        #              param_repr_mu1, param_repr_w, param_0, param_1, param_prop]
        # batch: (X, y, w)
        X, y, w = batch

        # get representation
        reps_c = predict_fun_repr(params[0], X)
        reps_o = predict_fun_repr_small(params[1], X)
        reps_mu0 = predict_fun_repr_small(params[2], X)
        reps_mu1 = predict_fun_repr_small(params[3], X)
        reps_w = predict_fun_repr(params[4], X)

        # concatenate
        reps_po_0 = _concatenate_representations((reps_c, reps_o, reps_mu0))
        reps_po_1 = _concatenate_representations((reps_c, reps_o, reps_mu1))
        reps_prop = _concatenate_representations((reps_c, reps_w))

        # pass down to heads
        loss_0 = loss_head(params[5], (reps_po_0, y, 1 - w), penalty_l2)
        loss_1 = loss_head(params[6], (reps_po_1, y, w), penalty_l2)

        # pass down to propensity head
        loss_prop = loss_head_prop(params[7], (reps_prop, w), penalty_l2)

        # is rep_o balanced between groups?
        loss_disc = penalty_disc * mmd2_lin(reps_o, w)

        # which variable has impact on which representation -- orthogonal loss
        loss_o = penalty_orthogonal * ortho_reg(params)

        # weight decay on representations
        weightsq_body = sum(
            [
                sum(
                    [jnp.sum(params[j][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
                )
                for j in range(5)
            ]
        )
        weightsq_head = heads_l2_penalty(
            params[5], params[6], n_layers_out, reg_diff, penalty_l2, penalty_diff
        )
        weightsq_prop = sum(
            [
                jnp.sum(params[7][i][0] ** 2)
                for i in range(0, 2 * n_layers_out_prop + 1, 2)
            ]
        )

        if not avg_objective:
            return (
                loss_0
                + loss_1
                + loss_prop
                + loss_disc
                + loss_o
                + 0.5 * (penalty_l2 * (weightsq_body + weightsq_prop) + weightsq_head)
            )
        else:
            n_batch = y.shape[0]
            return (
                (loss_0 + loss_1) / n_batch
                + loss_prop / n_batch
                + loss_disc
                + loss_o
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
            grad(loss_snet)(
                params, batch, penalty_l2, penalty_orthogonal, penalty_disc
            ),
            state,
        )

    # initialise states
    _, init_params = init_fun_snet(rng_key, input_shape)
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
            l_curr = loss_snet(
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
                    l_final = loss_snet(params_curr, (X_val, y_val, w_val), 0, 0, 0)
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
        l_final = loss_snet(get_params(opt_state), (X_val, y_val, w_val), 0, 0)
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


def predict_snet(
    X: jnp.ndarray,
    trained_params: jnp.ndarray,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head, predict_fun_prop = predict_funs
    param_0, param_1, param_prop = (
        trained_params[5],
        trained_params[6],
        trained_params[7],
    )

    reps_c = predict_fun_repr(trained_params[0], X)
    reps_o = predict_fun_repr(trained_params[1], X)
    reps_mu0 = predict_fun_repr(trained_params[2], X)
    reps_mu1 = predict_fun_repr(trained_params[3], X)
    reps_w = predict_fun_repr(trained_params[4], X)

    # concatenate
    reps_po_0 = _concatenate_representations((reps_c, reps_o, reps_mu0))
    reps_po_1 = _concatenate_representations((reps_c, reps_o, reps_mu1))
    reps_prop = _concatenate_representations((reps_c, reps_w))

    # get potential outcomes
    mu_0 = predict_fun_head(param_0, reps_po_0)
    mu_1 = predict_fun_head(param_1, reps_po_1)

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


# SNet without propensity head  ----------------------------------------
def train_snet_noprop(
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
    with_prop: bool = False,
    same_init: bool = False,
    ortho_reg_type: str = "abs",
) -> Tuple:
    """
    SNet but without the propensity head
    """
    if with_prop:
        raise ValueError("train_snet_noprop works only with_prop=False")
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

    def init_fun_snet_noprop(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        # param should look like [repr_o, repr_p0, repr_p1, po_0, po_1]
        # initialise representation layers
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr_o = init_fun_repr(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        input_shape_repr_small, param_repr_p0 = init_fun_repr_small(
            layer_rng, input_shape
        )
        rng, layer_rng = random.split(rng)
        _, param_repr_p1 = init_fun_repr_small(layer_rng, input_shape)

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

        return input_shape, [
            param_repr_o,
            param_repr_p0,
            param_repr_p1,
            param_0,
            param_1,
        ]

    # Define loss functions
    # loss functions for the head
    if not binary_y:

        def loss_head(
            params: jnp.ndarray,
            batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            penalty: float,
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            preds = predict_fun_head_po(params, inputs)
            return jnp.sum(weights * ((preds - targets) ** 2))

    else:

        def loss_head(
            params: jnp.ndarray,
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

    # define ortho-reg function
    if ortho_reg_type == "abs":

        def ortho_reg(params: jnp.ndarray) -> jnp.ndarray:
            col_o = _get_absolute_rowsums(params[0][0][0])
            col_p0 = _get_absolute_rowsums(params[1][0][0])
            col_p1 = _get_absolute_rowsums(params[2][0][0])
            return jnp.sum(col_o * col_p0 + col_o * col_p1 + col_p1 * col_p0)

    elif ortho_reg_type == "fro":

        def ortho_reg(params: jnp.ndarray) -> jnp.ndarray:
            return (
                _get_cos_reg(params[0][0][0], params[1][0][0], False)
                + _get_cos_reg(params[0][0][0], params[2][0][0], False)
                + _get_cos_reg(params[1][0][0], params[2][0][0], False)
            )

    else:
        raise NotImplementedError(
            "train_snet_noprop supports only orthogonal regularization "
            "using absolute values or frobenious norms."
        )

    # complete loss function for all parts
    @jit
    def loss_snet_noprop(
        params: jnp.ndarray,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        penalty_l2: float,
        penalty_orthogonal: float,
    ) -> jnp.ndarray:
        # params: list[repr_o, repr_p0, repr_p1, po_0, po_1]
        # batch: (X, y, w)
        X, y, w = batch

        # get representation
        reps_o = predict_fun_repr(params[0], X)
        reps_p0 = predict_fun_repr_small(params[1], X)
        reps_p1 = predict_fun_repr_small(params[2], X)

        # concatenate
        reps_po0 = _concatenate_representations((reps_o, reps_p0))
        reps_po1 = _concatenate_representations((reps_o, reps_p1))

        # pass down to heads
        loss_0 = loss_head(params[3], (reps_po0, y, 1 - w), penalty_l2)
        loss_1 = loss_head(params[4], (reps_po1, y, w), penalty_l2)

        # which variable has impact on which representation
        loss_o = penalty_orthogonal * ortho_reg(params)

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
                + loss_o
                + 0.5 * (penalty_l2 * weightsq_body + weightsq_head)
            )
        else:
            n_batch = y.shape[0]
            return (
                (loss_0 + loss_1) / n_batch
                + loss_o
                + 0.5 * (penalty_l2 * weightsq_body + weightsq_head)
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
    ) -> jnp.ndarray:
        # updating function
        params = get_params(state)
        return opt_update(
            i,
            grad(loss_snet_noprop)(params, batch, penalty_l2, penalty_orthogonal),
            state,
        )

    # initialise states
    _, init_params = init_fun_snet_noprop(rng_key, input_shape)
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
                i * n_batches + b, opt_state, next_batch, penalty_l2, penalty_orthogonal
            )

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            l_curr = loss_snet_noprop(
                params_curr, (X_val, y_val, w_val), penalty_l2, penalty_orthogonal
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
                    return params_best, (predict_fun_repr, predict_fun_head_po)
                p_curr = p_curr + 1

            if p_curr > patience:
                if return_val_loss:
                    # return loss without penalty
                    l_final = loss_snet_noprop(params_curr, (X_val, y_val, w_val), 0, 0)
                    return params_curr, (predict_fun_repr, predict_fun_head_po), l_final

                return params_curr, (predict_fun_repr, predict_fun_head_po)

    # return the parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss_snet_noprop(get_params(opt_state), (X_val, y_val, w_val), 0, 0)
        return trained_params, (predict_fun_repr, predict_fun_head_po), l_final

    return trained_params, (predict_fun_repr, predict_fun_head_po)


def predict_snet_noprop(
    X: jnp.ndarray,
    trained_params: jnp.ndarray,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:

    if return_prop:
        raise NotImplementedError("SNet5 does not have propensity estimator")

    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr_o, param_repr_po0, param_repr_po1 = (
        trained_params[0],
        trained_params[1],
        trained_params[2],
    )
    param_0, param_1 = trained_params[3], trained_params[4]

    # get representations
    rep_o = predict_fun_repr(param_repr_o, X)
    rep_po0 = predict_fun_repr(param_repr_po0, X)
    rep_po1 = predict_fun_repr(param_repr_po1, X)

    # concatenate
    reps_po0 = jnp.concatenate((rep_o, rep_po0), axis=1)
    reps_po1 = jnp.concatenate((rep_o, rep_po1), axis=1)

    # get potential outcomes
    mu_0 = predict_fun_head(param_0, reps_po0)
    mu_1 = predict_fun_head(param_1, reps_po1)

    te = mu_1 - mu_0

    # stack other outputs
    if return_po:
        return te, mu_0, mu_1
    else:
        return te
