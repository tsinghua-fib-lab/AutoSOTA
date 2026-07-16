"""This file contains functions construction ablation studies for IntiQ."""

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from shapiq import SHAPIQ, KernelSHAPIQ, SVARMIQ


def validate_ablation_kwargs(abalation_kwargs: dict):
    """Validate ablation kwargs."""
    valid_models = ["xgb", "rf", "dt", "lightgbm"]
    valid_approximators = ["shapiq", "kernelshapiq", "svarmiq"]
    valid_sampling_weights = ["default", "leverage"]

    if abalation_kwargs["value_model"] not in valid_models:
        raise ValueError(
            f"Model {abalation_kwargs['value_model']} not recognized. Only {valid_models} are allowed."
        )
    if abalation_kwargs["residual_approximator"] not in valid_approximators:
        raise ValueError(
            f"Residual approximator {abalation_kwargs['residual_approximator']} not recognized. Only {valid_approximators} are allowed."
        )
    if abalation_kwargs["sampling_weights"] not in valid_sampling_weights:
        raise ValueError(
            f"Sampling weights {abalation_kwargs['sampling_weights']} not recognized. Only {valid_sampling_weights} are allowed."
        )


########################################
# Ablation study model configurations #
########################################
def xgb_cross_validation(n_cv: int = 5, random_state: int = 42):
    """Return hyperparameter grid for XGBoost."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [1, 3, 5],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    xgb_model = XGBRegressor(random_state=random_state)
    model = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # Use negative as sklearn expects a score to be maximized
        cv=n_cv,
        n_jobs=-1,
        verbose=1,
    )
    return model


def random_forest_cross_validation(n_cv: int = 5, random_state: int = 42):
    """Return hyperparameter grid for Random Forest."""
    from sklearn.ensemble import RandomForestRegressor

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    rf_model = RandomForestRegressor(random_state=random_state)
    model = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # Use negative as sklearn expects a score to be maximized
        cv=n_cv,
        n_jobs=-1,
        verbose=1,
    )
    return model


def decision_tree_cross_validation(n_cv: int = 5, random_state: int = 42):
    """Return hyperparameter grid for Decision Tree."""
    from sklearn.tree import DecisionTreeRegressor

    param_grid = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    dt_model = DecisionTreeRegressor(random_state=random_state)
    model = GridSearchCV(
        estimator=dt_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # Use negative as sklearn expects a score to be maximized
        cv=n_cv,
        n_jobs=-1,
        verbose=1,
    )
    return model


def lightgbm_cross_validation(n_cv: int = 5, random_state: int = 42):
    """Return hyperparameter grid for LightGBM."""
    from lightgbm import LGBMRegressor

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [1, 3, 5],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    lgbm_model = LGBMRegressor(random_state=random_state)
    model = GridSearchCV(
        estimator=lgbm_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # Use negative as sklearn expects a score to be maximized
        cv=n_cv,
        n_jobs=-1,
        verbose=1,
    )
    return model


def get_cv_ablation_model(model_name: str, n_cv: int = 5, random_state: int = 42):
    """Return cross-validated model based on model name."""
    if model_name == "xgb":
        return xgb_cross_validation(n_cv=n_cv, random_state=random_state)
    elif model_name == "rf":
        return random_forest_cross_validation(n_cv=n_cv, random_state=random_state)
    elif model_name == "dt":
        return decision_tree_cross_validation(n_cv=n_cv, random_state=random_state)
    elif model_name == "lightgbm":
        return lightgbm_cross_validation(n_cv=n_cv, random_state=random_state)
    else:
        raise ValueError(f"Model {model_name} not recognized.")


def get_fixed_ablation_model(
    model_name: str, model_params: dict, random_state: int = 42
):
    """Return fixed hyperparameter model based on model name."""
    if model_name == "xgb":
        return XGBRegressor(
            **model_params,
            random_state=random_state,
        )
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            **model_params,
            random_state=random_state,
        )
    elif model_name == "dt":
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(
            **model_params,
            random_state=random_state,
        )
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            **model_params,
            random_state=random_state,
            verbose=-1,
            n_jobs=1
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")


def get_ablation_model(model_name: str, model_params: dict, random_state: int = 42):
    """Return model based on model name."""
    return get_fixed_ablation_model(
        model_name=model_name, model_params=model_params, random_state=random_state
    )


###############################
# Residual Approximators #####
###############################
def get_ablation_residual_approximator(
    approximator_name: str,
    n_players: int,
    max_order: int,
    index: str,
    pairing_trick: bool,
    replacement: bool,
    sampling_weights: str,
    random_state: int = 40,
):
    """Return residual approximator based on name."""
    if approximator_name == "shapiq":
        return SHAPIQ(
            n=n_players,
            max_order=max_order,
            index=index,
            pairing_trick=pairing_trick,
            replacement=replacement,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )
    elif approximator_name == "kernelshapiq":
        return KernelSHAPIQ(
            n=n_players,
            max_order=max_order,
            index=index,
            pairing_trick=pairing_trick,
            replacement=replacement,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )
    elif approximator_name == "svarmiq" and n_players <= 20:
        return SVARMIQ(
            n=n_players,
            max_order=max_order,
            index=index,
            pairing_trick=pairing_trick,
            replacement=replacement,
            sampling_weights=sampling_weights,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Approximator {approximator_name} not recognized.")


########################################
# Sampling Mechanism ####################
