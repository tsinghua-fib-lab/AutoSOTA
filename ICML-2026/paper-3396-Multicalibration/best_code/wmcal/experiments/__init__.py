# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Global experiment config state."""

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Type


@dataclass
class ExperimentConfig:
    seed: int

    @property
    def id(self) -> str:
        import hashlib
        import json

        config_dict = asdict(self)
        config_str = json.dumps(config_dict, indent=2, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


ExperimentRunFunction = Callable[[ExperimentConfig], None]


class ExperimentRegistry:
    _configs: Dict[str, Type[ExperimentConfig]] = {}
    _runners: Dict[str, ExperimentRunFunction] = {}

    @classmethod
    def register(
        cls,
        config_class: Type[ExperimentConfig],
        run_function: ExperimentRunFunction,
    ):
        experiment_type = config_class.__name__
        cls._configs[experiment_type] = config_class
        cls._runners[experiment_type] = run_function

    @classmethod
    def get_run_function(cls, experiment_type: str) -> ExperimentRunFunction:
        if experiment_type not in cls._runners:
            raise ValueError(
                f"Experiment type '{experiment_type}' not found. Available types: {list(cls._runners.keys())}"
            )
        return cls._runners[experiment_type]


def register_experiment(
    config_class: Type[ExperimentConfig],
    run_function: ExperimentRunFunction,
):
    ExperimentRegistry.register(config_class, run_function)


def run_experiment(config: ExperimentConfig, redo: bool = False) -> bool:
    """Run an experiment from an ExperimentConfig.

    Args:
        config: Experiment configuration.
        redo: If True, rerun even if already completed.

    Returns:
        True if experiment ran successfully, False if skipped or errored.
    """
    import shutil
    from pathlib import Path

    from ..utils import (
        get_logger,
        set_experiment_config,
        set_global_seed,
    )

    set_global_seed(config.seed)
    set_experiment_config(config)

    log_dir = Path(".logs") / config.id
    done_file = log_dir / "done"

    if done_file.exists() and not redo:
        print(f"Experiment (id: {config.id}) has already been run.")
        print("Use --redo to rerun the experiment.")
        return False

    # Clean previous logs if rerunning
    if redo and log_dir.exists():
        shutil.rmtree(log_dir)

    logger = get_logger(__name__)
    logger.info(f"Experiment ID: {config.id}")
    logger.info("Experiment config:")
    logger.info(config)

    run_function = ExperimentRegistry.get_run_function(type(config).__name__)
    try:
        run_function(config)
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        return False

    logger.log_done()
    return True


from .base import BaseExperimentConfig, run_base_experiment

register_experiment(BaseExperimentConfig, run_base_experiment)
