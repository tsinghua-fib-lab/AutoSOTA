"""
Module implements X-learner from Kuenzel et al (2019) using NNs
"""
# Author: Alicia Curth
from typing import Callable, Optional, Tuple

import jax.numpy as jnp

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_AVG_OBJECTIVE,
    DEFAULT_BATCH_SIZE,
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
from catenets.models.jax.model_utils import check_shape_1d_data, check_X_is_np
from catenets.models.jax.pseudo_outcome_nets import (  # same strategies as other nets
    ALL_STRATEGIES,
    FLEX_STRATEGY,
    OFFSET_STRATEGY,
    S1_STRATEGY,
    S2_STRATEGY,
    S3_STRATEGY,
    S_STRATEGY,
    T_STRATEGY,
    predict_flextenet,
    predict_offsetnet,
    predict_snet,
    predict_snet1,
    predict_snet2,
    predict_snet3,
    predict_t_net,
    train_flextenet,
    train_offsetnet,
    train_snet,
    train_snet1,
    train_snet2,
    train_snet3,
    train_tnet,
)


class XNet(BaseCATENet):
    """
    Class implements X-learner using NNs.

    Parameters
    ----------
    weight_strategy: int, default None
        Which strategy to use to weight the two CATE estimators in the second stage. weight_strategy
        is coded as follows: for tau(x)=g(x)tau_0(x) + (1-g(x))tau_1(x) [eq 9, kuenzel et al (2019)]
        weight_strategy=0 sets g(x)=0, weight_strategy=1 sets g(x)=1,
        weight_strategy=None sets g(x)=pi(x) [propensity score],
         weight_strategy=-1 sets g(x)=(1-pi(x))
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
        weight_strategy: Optional[int] = None,
        first_stage_strategy: str = T_STRATEGY,
        first_stage_args: Optional[dict] = None,
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
        nonlin: str = DEFAULT_NONLIN,
    ):
        # settings
        self.weight_strategy = weight_strategy
        self.first_stage_strategy = first_stage_strategy
        self.first_stage_args = first_stage_args
        self.binary_y = binary_y

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
        return train_x_net

    def _get_predict_function(self) -> Callable:
        # Two step nets do not need this
        return predict_x_net

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
            weight_strategy=self.weight_strategy,
        )


def train_x_net(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    weight_strategy: Optional[int] = None,
    first_stage_strategy: str = T_STRATEGY,
    first_stage_args: Optional[dict] = None,
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
    nonlin: str = DEFAULT_NONLIN,
    return_val_loss: bool = False,
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
) -> Tuple:
    y = check_shape_1d_data(y)
    if len(w.shape) > 1:
        w = w.reshape((len(w),))

    if weight_strategy not in [0, 1, -1, None]:
        # weight_strategy is coded as follows:
        # for tau(x)=g(x)tau_0(x) + (1-g(x))tau_1(x) [eq 9, kuenzel et al (2019)]
        # weight_strategy=0 sets g(x)=0, weight_strategy=1 sets g(x)=1,
        # weight_strategy=None sets g(x)=pi(x) [propensity score],
        # weight_strategy=-1 sets g(x)=(1-pi(x))
        raise ValueError("XNet only implements weight_strategy in [0, 1, -1, None]")

    if first_stage_strategy not in ALL_STRATEGIES:
        raise ValueError(
            "Parameter first stage should be in "
            "catenets.models.twostep_nets.ALL_STRATEGIES. "
            "You passed {}".format(first_stage_strategy)
        )

    # first stage: get estimates of PO regression
    log.debug("Training first stage")

    mu_hat_0, mu_hat_1 = _get_first_stage_pos(
        X,
        y,
        w,
        binary_y=binary_y,
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
        first_stage_strategy=first_stage_strategy,
        first_stage_args=first_stage_args,
    )

    if weight_strategy is None or weight_strategy == -1:
        # also fit propensity estimator
        log.debug("Training propensity net")
        params_prop, predict_fun_prop = train_output_net_only(
            X,
            w,
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

    else:
        params_prop, predict_fun_prop = None, None

    # second stage
    log.debug("Training second stage")
    if not weight_strategy == 0:
        # fit tau_0
        log.debug("Fitting tau_0")
        pseudo_outcome0 = mu_hat_1 - y[w == 0]
        params_tau0, predict_fun_tau0 = train_output_net_only(
            X[w == 0],
            pseudo_outcome0,
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
    else:
        params_tau0, predict_fun_tau0 = None, None

    if not weight_strategy == 1:
        # fit tau_1
        log.debug("Fitting tau_1")
        pseudo_outcome1 = y[w == 1] - mu_hat_0
        params_tau1, predict_fun_tau1 = train_output_net_only(
            X[w == 1],
            pseudo_outcome1,
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

    else:
        params_tau1, predict_fun_tau1 = None, None

    params = params_tau0, params_tau1, params_prop
    predict_funs = predict_fun_tau0, predict_fun_tau1, predict_fun_prop

    return params, predict_funs


def _get_first_stage_pos(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    first_stage_strategy: str = T_STRATEGY,
    first_stage_args: Optional[dict] = None,
    binary_y: bool = False,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_out: int = DEFAULT_UNITS_OUT,
    n_units_r: int = DEFAULT_UNITS_R,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    step_size: float = DEFAULT_STEP_SIZE,
    n_iter: int = DEFAULT_N_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_iter_min: int = DEFAULT_N_ITER_MIN,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    early_stopping: bool = True,
    patience: int = DEFAULT_PATIENCE,
    n_iter_print: int = DEFAULT_N_ITER_PRINT,
    seed: int = DEFAULT_SEED,
    nonlin: str = DEFAULT_NONLIN,
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if first_stage_args is None:
        first_stage_args = {}

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

    trained_params, pred_fun = train_fun(
        X,
        y,
        w,
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
        **first_stage_args
    )

    _, mu_0, mu_1 = predict_fun(X, trained_params, pred_fun, return_po=True)

    return mu_0[w == 1], mu_1[w == 0]


def predict_x_net(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
    weight_strategy: Optional[int] = None,
) -> jnp.ndarray:
    if return_po:
        raise NotImplementedError("TwoStepNets have no Potential outcome predictors.")

    if return_prop:
        raise NotImplementedError("TwoStepNets have no Propensity predictors.")

    params_tau0, params_tau1, params_prop = trained_params
    predict_fun_tau0, predict_fun_tau1, predict_fun_prop = predict_funs

    tau0_pred: jnp.ndarray
    tau1_pred: jnp.ndarray

    if not weight_strategy == 0:
        tau0_pred = predict_fun_tau0(params_tau0, X)
    else:
        tau0_pred = 0

    if not weight_strategy == 1:
        tau1_pred = predict_fun_tau1(params_tau1, X)
    else:
        tau1_pred = 0

    if weight_strategy is None or weight_strategy == -1:
        prop_pred = predict_fun_prop(params_prop, X)

    if weight_strategy is None:
        weight = prop_pred
    elif weight_strategy == -1:
        weight = 1 - prop_pred
    elif weight_strategy == 0:
        weight = 0
    elif weight_strategy == 1:
        weight = 1

    return weight * tau0_pred + (1 - weight) * tau1_pred
