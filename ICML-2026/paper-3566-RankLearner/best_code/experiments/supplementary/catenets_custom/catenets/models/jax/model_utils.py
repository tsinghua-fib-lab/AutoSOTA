"""
Model utils shared across different nets
"""
# Author: Alicia Curth
from typing import Any, Optional

import jax.numpy as jnp
import pandas as pd
from sklearn.model_selection import train_test_split

from catenets.models.constants import DEFAULT_SEED, DEFAULT_VAL_SPLIT

TRAIN_STRING = "training"
VALIDATION_STRING = "validation"


def check_shape_1d_data(y: jnp.ndarray) -> jnp.ndarray:
    # helper func to ensure that output shape won't clash
    # with jax internally
    shape_y = y.shape
    if len(shape_y) == 1:
        # should be shape (n_obs, 1), not (n_obs,)
        return y.reshape((shape_y[0], 1))
    return y


def check_X_is_np(X: pd.DataFrame) -> jnp.ndarray:
    # function to make sure we are using arrays only
    return jnp.asarray(X)


def make_val_split(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: Optional[jnp.ndarray] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
) -> Any:
    if val_split_prop == 0:
        # return original data
        if w is None:
            return X, y, X, y, TRAIN_STRING

        return X, y, w, X, y, w, TRAIN_STRING

    # make actual split
    if w is None:
        X_t, X_val, y_t, y_val = train_test_split(
            X, y, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return X_t, y_t, X_val, y_val, VALIDATION_STRING

    if stratify_w:
        # split to stratify by group
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X,
            y,
            w,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X, y, w, test_size=val_split_prop, random_state=seed, shuffle=True
        )

    return X_t, y_t, w_t, X_val, y_val, w_val, VALIDATION_STRING


def heads_l2_penalty(
    params_0: jnp.ndarray,
    params_1: jnp.ndarray,
    n_layers_out: jnp.ndarray,
    reg_diff: jnp.ndarray,
    penalty_0: jnp.ndarray,
    penalty_1: jnp.ndarray,
) -> jnp.ndarray:
    # Compute l2 penalty for output heads. Either seperately, or regularizing their difference

    # get l2-penalty for first head
    weightsq_0 = penalty_0 * sum(
        [jnp.sum(params_0[i][0] ** 2) for i in range(0, 2 * n_layers_out + 1, 2)]
    )

    # get l2-penalty for second head
    if reg_diff:
        weightsq_1 = penalty_1 * sum(
            [
                jnp.sum((params_1[i][0] - params_0[i][0]) ** 2)
                for i in range(0, 2 * n_layers_out + 1, 2)
            ]
        )
    else:
        weightsq_1 = penalty_1 * sum(
            [jnp.sum(params_1[i][0] ** 2) for i in range(0, 2 * n_layers_out + 1, 2)]
        )
    return weightsq_1 + weightsq_0
