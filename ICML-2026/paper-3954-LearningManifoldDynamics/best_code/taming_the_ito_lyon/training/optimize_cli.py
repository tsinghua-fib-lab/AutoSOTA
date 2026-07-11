from __future__ import annotations

import argparse

import optuna
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, model_validator

from taming_the_ito_lyon.config import Config, load_toml_config
from taming_the_ito_lyon.training.experiment import experiment


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: PositiveFloat | PositiveInt | None = None
    high: PositiveFloat | PositiveInt | None = None
    step: PositiveInt | None = None
    log: bool = False
    choices: list[str | int | float] | None = None
    as_int: bool = False

    @model_validator(mode="after")
    def validate_spec(self) -> ParamSpec:
        if self.choices is not None:
            if self.low is not None or self.high is not None:
                raise ValueError("choices cannot be combined with low/high")
            if self.step is not None:
                raise ValueError("choices cannot be combined with step")
            return self

        if self.low is None or self.high is None:
            raise ValueError("low and high are required when choices is None")
        if self.as_int:
            if not isinstance(self.low, int) or not isinstance(self.high, int):
                raise ValueError("as_int requires integer low/high")
        return self


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: ParamSpec | None = None
    weight_decay: ParamSpec | None = None
    max_grad_norm: ParamSpec | None = None
    batch_size: ParamSpec | None = None
    epochs: ParamSpec | None = None
    bnrde_hidden_size: ParamSpec | None = None
    bnrde_initial_state_param_dim: ParamSpec | None = None
    bnrde_init_hidden_dim: ParamSpec | None = None
    bnrde_vf_hidden_dim: ParamSpec | None = None
    bnrde_initial_cond_mlp_depth: ParamSpec | None = None
    bnrde_vf_mlp_depth: ParamSpec | None = None
    bnrde_signature_depth: ParamSpec | None = None
    bnrde_signature_window_size: ParamSpec | None = None
    bnrde_rough_solution: ParamSpec | None = None

    extrapolation_scheme: ParamSpec | None = None


SEARCH_SPACE = SearchSpace(
    learning_rate=ParamSpec(low=1e-5, high=1e-3, log=True),
    weight_decay=None,
    batch_size=None,
    epochs=None,
    bnrde_hidden_size=None,
    bnrde_initial_state_param_dim=None,
    bnrde_init_hidden_dim=None,
    bnrde_vf_hidden_dim=None,
    bnrde_initial_cond_mlp_depth=None,
    bnrde_vf_mlp_depth=None,
    bnrde_signature_depth=None,
    bnrde_signature_window_size=None,
    # extrapolation_scheme=ParamSpec(choices=["linear", "hermite", "piecewiseMLP"]),
    extrapolation_scheme=None,
)


PARAM_PATHS: dict[str, list[str]] = {
    "learning_rate": ["experiment_config", "learning_rate"],
    "weight_decay": ["experiment_config", "weight_decay"],
    "max_grad_norm": ["experiment_config", "max_grad_norm"],
    "batch_size": ["experiment_config", "batch_size"],
    "epochs": ["experiment_config", "epochs"],
    "extrapolation_scheme": ["experiment_config", "extrapolation_scheme"],
    "bnrde_hidden_size": ["bnrde_config", "hidden_size"],
    "bnrde_initial_state_param_dim": ["bnrde_config", "initial_state_param_dim"],
    "bnrde_init_hidden_dim": ["bnrde_config", "init_hidden_dim"],
    "bnrde_vf_hidden_dim": ["bnrde_config", "vf_hidden_dim"],
    "bnrde_initial_cond_mlp_depth": ["bnrde_config", "initial_cond_mlp_depth"],
    "bnrde_vf_mlp_depth": ["bnrde_config", "vf_mlp_depth"],
    "bnrde_signature_depth": ["bnrde_config", "signature_depth"],
    "bnrde_signature_window_size": ["bnrde_config", "signature_window_size"],
    "bnrde_rough_solution": ["bnrde_config", "rough_solution"],
}


def _normalize_config_path(path: str) -> str:
    if path.startswith("@"):
        return path[1:]
    return path


def _apply_param(config: Config, path: list[str], value: object) -> None:
    obj: object = config
    for key in path[:-1]:
        obj = getattr(obj, key)
        if obj is None:
            raise ValueError(f"Config path '{'.'.join(path)}' is None")
    setattr(obj, path[-1], value)


def _suggest_value(
    trial: optuna.trial.Trial,
    name: str,
    spec: ParamSpec,
) -> float | int | str | None:
    if spec.choices is not None:
        return trial.suggest_categorical(name, spec.choices)

    assert spec.low is not None and spec.high is not None
    if spec.as_int:
        low = int(spec.low)
        high = int(spec.high)
        if spec.step is None:
            return trial.suggest_int(name, low, high, log=spec.log)
        return trial.suggest_int(name, low, high, step=int(spec.step), log=spec.log)

    low_f = float(spec.low)
    high_f = float(spec.high)
    return trial.suggest_float(name, low_f, high_f, log=spec.log)


def _build_trial_config(
    base_config: Config, trial: optuna.trial.Trial, search_space: SearchSpace
) -> Config:
    config = base_config.model_copy(deep=True)
    base_seed = int(config.experiment_config.seed)
    config.experiment_config.seed = base_seed + int(trial.number)
    for name in search_space.model_fields:
        spec = getattr(search_space, name)
        if spec is None:
            continue
        value = _suggest_value(trial, name, spec)
        _apply_param(config, PARAM_PATHS[name], value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Optuna optimization over training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TOML config file (can be prefixed with '@')",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="taming-the-ito-lyon",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Optuna sampler",
    )
    args = parser.parse_args()

    config_path = _normalize_config_path(args.config)
    base_config = load_toml_config(config_path)

    def objective(trial: optuna.trial.Trial) -> float:
        trial_config = _build_trial_config(base_config, trial, SEARCH_SPACE)
        result = experiment(trial_config, config_path=None, return_metrics=True)
        assert result is not None
        return float(result["min_val_metric"])

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=int(args.n_trials))

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
