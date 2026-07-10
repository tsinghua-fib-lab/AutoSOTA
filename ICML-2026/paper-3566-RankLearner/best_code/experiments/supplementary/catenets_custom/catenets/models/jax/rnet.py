"""
Implements NN based on R-learner and U-learner (as discussed in Nie & Wager (2017))
"""
# Author: Alicia Curth
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as onp
import pandas as pd
from jax import grad, jit, random
from jax.example_libraries import optimizers
from sklearn.model_selection import StratifiedKFold

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_AVG_OBJECTIVE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CF_FOLDS,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_OUT_T,
    DEFAULT_LAYERS_R,
    DEFAULT_LAYERS_R_T,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_STEP_SIZE_T,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_OUT_T,
    DEFAULT_UNITS_R,
    DEFAULT_UNITS_R_T,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.jax.base import (
    BaseCATENet,
    OutputHead,
    make_val_split,
    train_output_net_only,
)
from catenets.models.jax.model_utils import check_shape_1d_data, check_X_is_np

R_STRATEGY_NAME = "R"
U_STRATEGY_NAME = "U"


class RNet(BaseCATENet):
    """
    Class implements R-learner and U-learner using NNs

    Parameters
    ----------
    second_stage_strategy: str, default 'R'
        Which strategy to use in the second stage ('R' for R-learner, 'U' for U-learner)
    data_split: bool, default False
        Whether to split the data in two folds for estimation
    cross_fit: bool, default False
        Whether to perform cross fitting
    n_cf_folds: int
        Number of crossfitting folds to use
    n_layers_out: int
        First stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        First stage Number of hidden units in each hypothesis layer
    n_layers_r: int
        First stage Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_r: int
        First stage Number of hidden units in each representation layer
    n_layers_out_t: int
        Second stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out_t: int
        Second stage Number of hidden units in each hypothesis layer
    n_layers_r_t: int
        Second stage Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_r_t: int
        Second stage Number of hidden units in each representation layer
    penalty_l2: float
        First stage l2 (ridge) penalty
    penalty_l2_t: float
        Second stage l2 (ridge) penalty
    step_size: float
        First stage learning rate for optimizer
    step_size_t: float
        Second stage learning rate for optimizer
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
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    """

    def __init__(
        self,
        second_stage_strategy: str = R_STRATEGY_NAME,
        data_split: bool = False,
        cross_fit: bool = False,
        n_cf_folds: int = DEFAULT_CF_FOLDS,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
        n_layers_r_t: int = DEFAULT_LAYERS_R_T,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_r: int = DEFAULT_UNITS_R,
        n_units_out_t: int = DEFAULT_UNITS_OUT_T,
        n_units_r_t: int = DEFAULT_UNITS_R_T,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        penalty_l2_t: float = DEFAULT_PENALTY_L2,
        step_size: float = DEFAULT_STEP_SIZE,
        step_size_t: float = DEFAULT_STEP_SIZE_T,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        binary_y: bool = False,
    ) -> None:
        # settings
        self.binary_y = binary_y
        self.second_stage_strategy = second_stage_strategy
        self.data_split = data_split
        self.cross_fit = cross_fit
        self.n_cf_folds = n_cf_folds

        # model architecture hyperparams
        self.n_layers_out = n_layers_out
        self.n_layers_out_t = n_layers_out_t
        self.n_layers_r = n_layers_r
        self.n_layers_r_t = n_layers_r_t
        self.n_units_out = n_units_out
        self.n_units_out_t = n_units_out_t
        self.n_units_r = n_units_r
        self.n_units_r_t = n_units_r_t
        self.nonlin = nonlin

        # other hyperparameters
        self.penalty_l2 = penalty_l2
        self.penalty_l2_t = penalty_l2_t
        self.step_size = step_size
        self.step_size_t = step_size_t
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min

    def _get_train_function(self) -> Callable:
        return train_r_net

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        w: jnp.ndarray,
        p: Optional[jnp.ndarray] = None,
    ) -> "RNet":
        # overwrite super so we can pass p as extra param
        # some quick input checks
        X = check_X_is_np(X)
        self._check_inputs(w, p)

        train_func = self._get_train_function()
        train_params = self.get_params()

        self._params, self._predict_funs = train_func(X, y, w, p, **train_params)

        return self

    def _get_predict_function(self) -> Callable:
        # Two step nets do not need this
        pass

    def predict(
        self, X: jnp.ndarray, return_po: bool = False, return_prop: bool = False
    ) -> jnp.ndarray:
        # check input
        if return_po:
            raise NotImplementedError(
                "TwoStepNets have no Potential outcome predictors."
            )

        if return_prop:
            raise NotImplementedError("TwoStepNets have no Propensity predictors.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._predict_funs(self._params, X)


def train_r_net(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    p: Optional[jnp.ndarray] = None,
    second_stage_strategy: str = R_STRATEGY_NAME,
    data_split: bool = False,
    cross_fit: bool = False,
    n_cf_folds: int = DEFAULT_CF_FOLDS,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_layers_r_t: int = DEFAULT_LAYERS_R_T,
    n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
    n_units_out: int = DEFAULT_UNITS_OUT,
    n_units_r: int = DEFAULT_UNITS_R,
    n_units_out_t: int = DEFAULT_UNITS_OUT_T,
    n_units_r_t: int = DEFAULT_UNITS_R_T,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    penalty_l2_t: float = DEFAULT_PENALTY_L2,
    step_size: float = DEFAULT_STEP_SIZE,
    step_size_t: float = DEFAULT_STEP_SIZE_T,
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
    binary_y: bool = False,
) -> Any:
    # get shape of data
    n, d = X.shape

    if p is not None:
        p = check_shape_1d_data(p)

    # split data as wanted
    if not cross_fit:
        if not data_split:
            log.debug("Training first stage with all data (no data splitting)")
            # use all data for both
            fit_mask = onp.ones(n, dtype=bool)
            pred_mask = onp.ones(n, dtype=bool)
        else:
            log.debug("Training first stage with half of the data (data splitting)")
            # split data in half
            fit_idx = onp.random.choice(n, int(onp.round(n / 2)))
            fit_mask = onp.zeros(n, dtype=bool)

            fit_mask[fit_idx] = 1
            pred_mask = ~fit_mask

        mu_hat, pi_hat = _train_and_predict_r_stage1(
            X,
            y,
            w,
            fit_mask,
            pred_mask,
            n_layers_out=n_layers_out,
            n_layers_r=n_layers_r,
            n_units_out=n_units_out,
            n_units_r=n_units_r,
            penalty_l2=penalty_l2,
            step_size=step_size,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            seed=seed,
            nonlin=nonlin,
            binary_y=binary_y,
        )
        if data_split:
            # keep only prediction data
            X, y, w = X[pred_mask, :], y[pred_mask, :], w[pred_mask, :]

            if p is not None:
                p = p[pred_mask, :]

    else:
        log.debug(f"Training first stage in {n_cf_folds} folds (cross-fitting)")
        # do cross fitting
        mu_hat, pi_hat = onp.zeros((n, 1)), onp.zeros((n, 1))
        splitter = StratifiedKFold(n_splits=n_cf_folds, shuffle=True, random_state=seed)

        fold_count = 1
        for train_idx, test_idx in splitter.split(X, w):
            log.debug(f"Training fold {fold_count}.")
            fold_count = fold_count + 1

            pred_mask = onp.zeros(n, dtype=bool)
            pred_mask[test_idx] = 1
            fit_mask = ~pred_mask

            mu_hat[pred_mask], pi_hat[pred_mask] = _train_and_predict_r_stage1(
                X,
                y,
                w,
                fit_mask,
                pred_mask,
                n_layers_out=n_layers_out,
                n_layers_r=n_layers_r,
                n_units_out=n_units_out,
                n_units_r=n_units_r,
                penalty_l2=penalty_l2,
                step_size=step_size,
                n_iter=n_iter,
                batch_size=batch_size,
                val_split_prop=val_split_prop,
                early_stopping=early_stopping,
                patience=patience,
                n_iter_min=n_iter_min,
                n_iter_print=n_iter_print,
                seed=seed,
                nonlin=nonlin,
                binary_y=binary_y,
            )

    log.debug("Training second stage.")

    if p is not None:
        # use known propensity score
        p = check_shape_1d_data(p)
        pi_hat = p

    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    w_ortho = w - pi_hat
    y_ortho = y - mu_hat

    if second_stage_strategy == R_STRATEGY_NAME:
        return train_r_stage2(
            X,
            y_ortho,
            w_ortho,
            n_layers_out=n_layers_out_t,
            n_units_out=n_units_out_t,
            n_layers_r=n_layers_r_t,
            n_units_r=n_units_r_t,
            penalty_l2=penalty_l2_t,
            step_size=step_size_t,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            seed=seed,
            return_val_loss=return_val_loss,
            nonlin=nonlin,
        )
    elif second_stage_strategy == U_STRATEGY_NAME:
        return train_output_net_only(
            X,
            y_ortho / w_ortho,
            n_layers_out=n_layers_out_t,
            n_units_out=n_units_out_t,
            n_layers_r=n_layers_r_t,
            n_units_r=n_units_r_t,
            penalty_l2=penalty_l2_t,
            step_size=step_size_t,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            seed=seed,
            return_val_loss=return_val_loss,
            nonlin=nonlin,
        )
    else:
        raise ValueError("R-learner only supports strategies R and U.")


def _train_and_predict_r_stage1(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    fit_mask: jnp.ndarray,
    pred_mask: jnp.ndarray,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    n_layers_r: int = DEFAULT_LAYERS_R,
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
    nonlin: str = DEFAULT_NONLIN,
    binary_y: bool = False,
) -> Any:
    if len(w.shape) > 1:
        w = w.reshape((len(w),))

    # split the data
    X_fit, y_fit, w_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]
    X_pred = X[pred_mask, :]

    log.debug("Training output Net")
    params_out, predict_fun_out = train_output_net_only(
        X_fit,
        y_fit,
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        n_layers_r=n_layers_r,
        n_units_r=n_units_r,
        penalty_l2=penalty_l2,
        step_size=step_size,
        n_iter=n_iter,
        batch_size=batch_size,
        val_split_prop=val_split_prop,
        early_stopping=early_stopping,
        patience=patience,
        n_iter_min=n_iter_min,
        n_iter_print=n_iter_print,
        seed=seed,
        nonlin=nonlin,
        binary_y=binary_y,
    )
    mu_hat = predict_fun_out(params_out, X_pred)

    log.debug("Training propensity net")
    params_prop, predict_fun_prop = train_output_net_only(
        X_fit,
        w_fit,
        binary_y=True,
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        n_layers_r=n_layers_r,
        n_units_r=n_units_r,
        penalty_l2=penalty_l2,
        step_size=step_size,
        n_iter=n_iter,
        batch_size=batch_size,
        val_split_prop=val_split_prop,
        early_stopping=early_stopping,
        patience=patience,
        n_iter_min=n_iter_min,
        n_iter_print=n_iter_print,
        seed=seed,
        nonlin=nonlin,
    )
    pi_hat = predict_fun_prop(params_prop, X_pred)

    return mu_hat, pi_hat


def train_r_stage2(
    X: jnp.ndarray,
    y_ortho: jnp.ndarray,
    w_ortho: jnp.ndarray,
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
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
) -> Any:
    # function to train a single output head
    # input check
    y_ortho, w_ortho = check_shape_1d_data(y_ortho), check_shape_1d_data(w_ortho)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    # get validation split (can be none)
    X, y_ortho, w_ortho, X_val, y_val, w_val, val_string = make_val_split(
        X, y_ortho, w_ortho, val_split_prop=val_split_prop, seed=seed, stratify_w=False
    )
    n = X.shape[0]  # could be different from before due to split

    # get output head
    init_fun, predict_fun = OutputHead(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        n_layers_r=n_layers_r,
        n_units_r=n_units_r,
        nonlin=nonlin,
    )

    # define loss and grad
    @jit
    def loss(params: dict, batch: jnp.ndarray, penalty: float) -> jnp.ndarray:
        # mse loss function
        inputs, ortho_targets, ortho_treats = batch
        preds = predict_fun(params, inputs)
        weightsq = sum(
            [
                jnp.sum(params[i][0] ** 2)
                for i in range(0, 2 * (n_layers_out + n_layers_r) + 1, 2)
            ]
        )
        if not avg_objective:
            return (
                jnp.sum((ortho_targets - ortho_treats * preds) ** 2)
                + 0.5 * penalty * weightsq
            )
        else:
            return (
                jnp.average((ortho_targets - ortho_treats * preds) ** 2)
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
        # g_params = optimizers.clip_grads(g_params, 1.0)
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
            next_batch = X[idx_next, :], y_ortho[idx_next, :], w_ortho[idx_next, :]
            opt_state = update(i * n_batches + b, opt_state, next_batch, penalty_l2)

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            l_curr = loss(params_curr, (X_val, y_val, w_val), penalty_l2)

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
                    l_final = loss(trained_params, (X_val, y_val, w_val), 0)
                    return trained_params, predict_fun, l_final

                return trained_params, predict_fun

    # get final parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss(trained_params, (X_val, y_val, w_val), 0)
        return trained_params, predict_fun, l_final

    return trained_params, predict_fun
