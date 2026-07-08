from dataclasses import dataclass
from typing import List, Union


@dataclass
class TaskConfig:
    task_type: str
    alpha: Union[float, List[float]]
    data_splits: List[float]
    fc_estimator_mode: str      # single, ensemble_x , enbPi_boostrap_x
    add_config: dict            # Task specific config
    global_norm: bool = None


@dataclass
class TSDataConfig:
    dataset_type: str
    paths: List[str]
    add_config: dict  # Dataset specific params
