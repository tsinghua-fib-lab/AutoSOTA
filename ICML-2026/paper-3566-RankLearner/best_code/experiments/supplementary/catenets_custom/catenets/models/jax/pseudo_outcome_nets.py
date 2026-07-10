"""
Implements Pseudo-outcome based Two-step Nets, namely the DR-learner, the PW-learner and the
RA-learner.
"""
# Author: Alicia Curth
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import numpy as onp
import pandas as pd
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
)
from catenets.models.jax.base import BaseCATENet, train_output_net_only
from catenets.models.jax.disentangled_nets import predict_snet3, train_snet3
from catenets.models.jax.flextenet import predict_flextenet, train_flextenet
from catenets.models.jax.model_utils import check_shape_1d_data, check_X_is_np
from catenets.models.jax.offsetnet import predict_offsetnet, train_offsetnet
from catenets.models.jax.representation_nets import (
    predict_snet1,
    predict_snet2,
    train_snet1,
    train_snet2,
)
from catenets.models.jax.snet import predict_snet, train_snet
from catenets.models.jax.tnet import predict_t_net, train_tnet
from catenets.models.jax.transformation_utils import (
    DR_TRANSFORMATION,
    PW_TRANSFORMATION,
    RA_TRANSFORMATION,
    _get_transformation_function,
)

T_STRATEGY = "T"
S1_STRATEGY = "Tar"
S2_STRATEGY = "S2"
S3_STRATEGY = "S3"
S_STRATEGY = "S"
OFFSET_STRATEGY = "Offset"
FLEX_STRATEGY = "Flex"

ALL_STRATEGIES = [
    T_STRATEGY,
    S1_STRATEGY,
    S2_STRATEGY,
    S3_STRATEGY,
    S_STRATEGY,
    FLEX_STRATEGY,
    OFFSET_STRATEGY,
]


class PseudoOutcomeNet(BaseCATENet):
    """
    Class implements TwoStepLearners based on pseudo-outcome regression as discussed in
    Curth &vd Schaar (2021): RA-learner, PW-learner and DR-learner

    Parameters
    ----------
    first_stage_strategy: str, default 't'
        which nuisance estimator to use in first stage
    first_stage_args: dict
        Any additional arguments to pass to first stage training function
    data_split: bool, default False
        Whether to split the data in two folds for estimation
    cross_fit: bool, default False
        Whether to perform cross fitting
    n_cf_folds: int
        Number of crossfitting folds to use
    transformation: str, default 'AIPW'
        pseudo-outcome to use ('AIPW' for DR-learner, 'HT' for PW learner, 'RA' for RA-learner)
    binary_y: bool, default False
        Whether the outcome is binary
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
        first_stage_strategy: str = T_STRATEGY,
        first_stage_args: Optional[dict] = None,
        data_split: bool = False,
        cross_fit: bool = False,
        n_cf_folds: int = DEFAULT_CF_FOLDS,
        transformation: str = DR_TRANSFORMATION,
        binary_y: bool = False,
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
        rescale_transformation: bool = False,
        nonlin: str = DEFAULT_NONLIN,
    ) -> None:
        # settings
        self.first_stage_strategy = first_stage_strategy
        self.first_stage_args = first_stage_args
        self.binary_y = binary_y
        self.transformation = transformation
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
        self.rescale_transformation = rescale_transformation

    def _get_train_function(self) -> Callable:
        return train_pseudooutcome_net

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        w: jnp.ndarray,
        p: Optional[jnp.ndarray] = None,
    ) -> "PseudoOutcomeNet":
        # overwrite super so we can pass p as extra param
        # some quick input checks
        X = check_X_is_np(X)
        self._check_inputs(w, p)

        train_func = self._get_train_function()
        train_params = self.get_params()

        if "transformation" not in train_params.keys():
            train_params.update({"transformation": self.transformation})

        if self.rescale_transformation:
            self._params, self._predict_funs, self._scale_factor = train_func(
                X, y, w, p, **train_params
            )
        else:
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

        if self.rescale_transformation:
            return 1 / self._scale_factor * self._predict_funs(self._params, X)
        else:
            return self._predict_funs(self._params, X)


class DRNet(PseudoOutcomeNet):
    """Wrapper for DR-learner using PseudoOutcomeNet"""

    def __init__(
        self,
        first_stage_strategy: str = T_STRATEGY,
        data_split: bool = False,
        cross_fit: bool = False,
        n_cf_folds: int = DEFAULT_CF_FOLDS,
        binary_y: bool = False,
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
        rescale_transformation: bool = False,
        nonlin: str = DEFAULT_NONLIN,
        first_stage_args: Optional[dict] = None,
    ) -> None:
        super().__init__(
            first_stage_strategy=first_stage_strategy,
            data_split=data_split,
            cross_fit=cross_fit,
            n_cf_folds=n_cf_folds,
            transformation=DR_TRANSFORMATION,
            binary_y=binary_y,
            n_layers_out=n_layers_out,
            n_layers_r=n_layers_r,
            n_layers_out_t=n_layers_out_t,
            n_layers_r_t=n_layers_r_t,
            n_units_out=n_units_out,
            n_units_r=n_units_r,
            n_units_out_t=n_units_out_t,
            n_units_r_t=n_units_r_t,
            penalty_l2=penalty_l2,
            penalty_l2_t=penalty_l2_t,
            step_size=step_size,
            step_size_t=step_size_t,
            n_iter=n_iter,
            batch_size=batch_size,
            n_iter_min=n_iter_min,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_print=n_iter_print,
            seed=seed,
            nonlin=nonlin,
            rescale_transformation=rescale_transformation,
            first_stage_args=first_stage_args,
        )


class RANet(PseudoOutcomeNet):
    """Wrapper for RA-learner using PseudoOutcomeNet"""

    def __init__(
        self,
        first_stage_strategy: str = T_STRATEGY,
        data_split: bool = False,
        cross_fit: bool = False,
        n_cf_folds: int = DEFAULT_CF_FOLDS,
        binary_y: bool = False,
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
        rescale_transformation: bool = False,
        nonlin: str = DEFAULT_NONLIN,
        first_stage_args: Optional[dict] = None,
    ) -> None:
        super().__init__(
            first_stage_strategy=first_stage_strategy,
            data_split=data_split,
            cross_fit=cross_fit,
            n_cf_folds=n_cf_folds,
            transformation=RA_TRANSFORMATION,
            binary_y=binary_y,
            n_layers_out=n_layers_out,
            n_layers_r=n_layers_r,
            n_layers_out_t=n_layers_out_t,
            n_layers_r_t=n_layers_r_t,
            n_units_out=n_units_out,
            n_units_r=n_units_r,
            n_units_out_t=n_units_out_t,
            n_units_r_t=n_units_r_t,
            penalty_l2=penalty_l2,
            penalty_l2_t=penalty_l2_t,
            step_size=step_size,
            step_size_t=step_size_t,
            n_iter=n_iter,
            batch_size=batch_size,
            n_iter_min=n_iter_min,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_print=n_iter_print,
            seed=seed,
            nonlin=nonlin,
            rescale_transformation=rescale_transformation,
            first_stage_args=first_stage_args,
        )


class PWNet(PseudoOutcomeNet):
    """Wrapper for PW-learner using PseudoOutcomeNet"""

    def __init__(
        self,
        first_stage_strategy: str = T_STRATEGY,
        data_split: bool = False,
        cross_fit: bool = False,
        n_cf_folds: int = DEFAULT_CF_FOLDS,
        binary_y: bool = False,
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
        rescale_transformation: bool = False,
        nonlin: str = DEFAULT_NONLIN,
        first_stage_args: Optional[dict] = None,
    ) -> None:
        super().__init__(
            first_stage_strategy=first_stage_strategy,
            data_split=data_split,
            cross_fit=cross_fit,
            n_cf_folds=n_cf_folds,
            transformation=PW_TRANSFORMATION,
            binary_y=binary_y,
            n_layers_out=n_layers_out,
            n_layers_r=n_layers_r,
            n_layers_out_t=n_layers_out_t,
            n_layers_r_t=n_layers_r_t,
            n_units_out=n_units_out,
            n_units_r=n_units_r,
            n_units_out_t=n_units_out_t,
            n_units_r_t=n_units_r_t,
            penalty_l2=penalty_l2,
            penalty_l2_t=penalty_l2_t,
            step_size=step_size,
            step_size_t=step_size_t,
            n_iter=n_iter,
            batch_size=batch_size,
            n_iter_min=n_iter_min,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_print=n_iter_print,
            seed=seed,
            nonlin=nonlin,
            rescale_transformation=rescale_transformation,
            first_stage_args=first_stage_args,
        )


def train_pseudooutcome_net(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    p: Optional[jnp.ndarray] = None,
    first_stage_strategy: str = T_STRATEGY,
    data_split: bool = False,
    cross_fit: bool = False,
    n_cf_folds: int = DEFAULT_CF_FOLDS,
    transformation: str = DR_TRANSFORMATION,
    binary_y: bool = False,
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
    rescale_transformation: bool = False,
    return_val_loss: bool = False,
    nonlin: str = DEFAULT_NONLIN,
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
    first_stage_args: Optional[dict] = None,
) -> Tuple:
    # get shape of data
    n, d = X.shape

    if p is not None:
        p = check_shape_1d_data(p)

    # get transformation function
    transformation_function = _get_transformation_function(transformation)

    # get strategy name
    if first_stage_strategy not in ALL_STRATEGIES:
        raise ValueError(
            "Parameter first stage should be in "
            "catenets.models.pseudo_outcome_nets.ALL_STRATEGIES. "
            "You passed {}".format(first_stage_strategy)
        )

    # split data as wanted
    if p is None or transformation is not PW_TRANSFORMATION:
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

            mu_0, mu_1, pi_hat = _train_and_predict_first_stage(
                X,
                y,
                w,
                fit_mask,
                pred_mask,
                first_stage_strategy=first_stage_strategy,
                binary_y=binary_y,
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
                avg_objective=avg_objective,
                transformation=transformation,
                first_stage_args=first_stage_args,
            )
            if data_split:
                # keep only prediction data
                X, y, w = X[pred_mask, :], y[pred_mask, :], w[pred_mask, :]

                if p is not None:
                    p = p[pred_mask, :]

        else:
            log.debug(f"Training first stage in {n_cf_folds} folds (cross-fitting)")
            # do cross fitting
            mu_0, mu_1, pi_hat = onp.zeros((n, 1)), onp.zeros((n, 1)), onp.zeros((n, 1))
            splitter = StratifiedKFold(
                n_splits=n_cf_folds, shuffle=True, random_state=seed
            )

            fold_count = 1
            for train_idx, test_idx in splitter.split(X, w):

                log.debug(f"Training fold {fold_count}.")
                fold_count = fold_count + 1

                pred_mask = onp.zeros(n, dtype=bool)
                pred_mask[test_idx] = 1
                fit_mask = ~pred_mask

                (
                    mu_0[pred_mask],
                    mu_1[pred_mask],
                    pi_hat[pred_mask],
                ) = _train_and_predict_first_stage(
                    X,
                    y,
                    w,
                    fit_mask,
                    pred_mask,
                    first_stage_strategy=first_stage_strategy,
                    binary_y=binary_y,
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
                    avg_objective=avg_objective,
                    transformation=transformation,
                    first_stage_args=first_stage_args,
                )

    log.debug("Training second stage.")

    if p is not None:
        # use known propensity score
        p = check_shape_1d_data(p)
        pi_hat = p

    # second stage
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    # transform data and fit on transformed data
    if transformation is PW_TRANSFORMATION:
        mu_0 = None
        mu_1 = None

    pseudo_outcome = transformation_function(y=y, w=w, p=pi_hat, mu_0=mu_0, mu_1=mu_1)
    if rescale_transformation:
        scale_factor = onp.std(y) / onp.std(pseudo_outcome)
        if scale_factor > 1:
            scale_factor = 1
        else:
            pseudo_outcome = scale_factor * pseudo_outcome
        params, predict_funs = train_output_net_only(
            X,
            pseudo_outcome,
            binary_y=False,
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
            avg_objective=avg_objective,
        )
        return params, predict_funs, scale_factor
    else:
        return train_output_net_only(
            X,
            pseudo_outcome,
            binary_y=False,
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
            avg_objective=avg_objective,
        )


def _train_and_predict_first_stage(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    fit_mask: jnp.ndarray,
    pred_mask: jnp.ndarray,
    first_stage_strategy: str,
    binary_y: bool = False,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_out: int = DEFAULT_UNITS_OUT,
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
    avg_objective: bool = False,
    transformation: str = DR_TRANSFORMATION,
    first_stage_args: Optional[dict] = None,
) -> Tuple:
    if len(w.shape) > 1:
        w = w.reshape((len(w),))

    if first_stage_args is None:
        first_stage_args = {}

    # split the data
    X_fit, y_fit, w_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]
    X_pred = X[pred_mask, :]

    train_fun: Callable
    predict_fun: Callable

    if first_stage_strategy == T_STRATEGY:
        train_fun, predict_fun = train_tnet, predict_t_net
    elif first_stage_strategy == S_STRATEGY:
        train_fun, predict_fun = train_snet, predict_snet
    elif first_stage_strategy == S1_STRATEGY:
        train_fun, predict_fun = train_snet1, predict_snet1
    elif first_stage_strategy == S2_STRATEGY:
        train_fun, predict_fun = train_snet2, predict_snet2
    elif first_stage_strategy == S3_STRATEGY:
        train_fun, predict_fun = train_snet3, predict_snet3
    elif first_stage_strategy == OFFSET_STRATEGY:
        train_fun, predict_fun = train_offsetnet, predict_offsetnet
    elif first_stage_strategy == FLEX_STRATEGY:
        train_fun, predict_fun = train_flextenet, predict_flextenet
    else:
        raise ValueError(
            "{} is not a valid first stage strategy for a PseudoOutcomeNet".format(
                first_stage_strategy
            )
        )

    log.debug("Training PO estimators")
    trained_params, pred_fun = train_fun(
        X_fit,
        y_fit,
        w_fit,
        binary_y=binary_y,
        n_layers_r=n_layers_r,
        n_units_r=n_units_r,
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
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
        avg_objective=avg_objective,
        **first_stage_args,
    )

    if first_stage_strategy in [S_STRATEGY, S2_STRATEGY, S3_STRATEGY]:
        _, mu_0, mu_1, pi_hat = predict_fun(
            X_pred, trained_params, pred_fun, return_po=True, return_prop=True
        )
    else:
        if transformation is not PW_TRANSFORMATION:
            _, mu_0, mu_1 = predict_fun(
                X_pred, trained_params, pred_fun, return_po=True
            )
        else:
            mu_0, mu_1 = onp.nan, onp.nan

        if transformation is not RA_TRANSFORMATION:
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
                avg_objective=avg_objective,
            )
            pi_hat = predict_fun_prop(params_prop, X_pred)
        else:
            pi_hat = onp.nan

    return mu_0, mu_1, pi_hat
