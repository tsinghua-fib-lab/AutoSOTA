import copy
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Type
from omegaconf import DictConfig, OmegaConf

SWEEP_TYPE_GRIDVAL = 'grid'
SWEEP_TYPE_SKIP = 'skip'
EXPERIMENT_CONFIG = 'config'

LOGGER = logging.getLogger(__name__)


class Sweeper(ABC):
    """This class is passed the config and it generates multiple configs according to the configuration."""
    def __init__(self, config: DictConfig):
        self.config = config


    @abstractmethod
    def generate_configs(self) -> List[DictConfig]:
        pass

    @staticmethod
    def create_sweeper(config: DictConfig) -> Optional[Type["Sweeper"]]:
        sweep_type = config.sweep.type
        if sweep_type == SWEEP_TYPE_GRIDVAL:
            return SweepGrid(config)
        elif sweep_type == SWEEP_TYPE_SKIP:
            return None
        else:
            raise ValueError(f"Unsupported sweep type: '{sweep_type}'")


class SweepGrid(Sweeper):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        axes = config.sweep.axes
        assert OmegaConf.is_list(axes)

        # generate list of axis generators
        sweep_grid_axes = []
        for ax in axes:
            sweep_grid_axes.append(sweep_axis_grid(ax))

        self.sweep_grid_axes = sweep_grid_axes

    def generate_configs(self) -> List[DictConfig]:
        """Generates a list of DictConfigs specified by the sweep. 

        Returns:
            List[DictConfig]: The DictConfig list with all configs in the sweep.
        """
        # iterate through all possible grid combinations
        grid_sweep_configs = []
        for axis_combination in itertools.product(*self.sweep_grid_axes):
            # contains only the parameters to be updated in the 'main' config:
            grid_point_config = OmegaConf.merge(*axis_combination)
            # create new config with updated params
            tmp_config = copy.deepcopy(self.config)
            tmp_config[EXPERIMENT_CONFIG] = OmegaConf.merge(
                tmp_config[EXPERIMENT_CONFIG], grid_point_config)
            grid_sweep_configs.append(tmp_config)

        LOGGER.info(
            f"Gridsweep with {len(self.sweep_grid_axes)} axes and {len(grid_sweep_configs)} runs generated."
        )
        return grid_sweep_configs


def sweep_axis_grid(axis_config: DictConfig) -> Iterator[DictConfig]:
    """Receives a DictConfig object with keys `parameter` and `vals` and generates parameter->value dicts. 
    For an example see `example_config.yaml`.
    Args:
        axis_config (DictConfig): Sweep config for one axis. 

    Yields:
        Generator[DictConfig]: A generator, that produces single DictConfigs with the parameter as keys and its respective values.
    """
    # check num parameters
    parameters = axis_config.parameter
    if OmegaConf.is_list(parameters):
        num_parameters = len(parameters)
    else:
        num_parameters = 1
        parameters = [parameters]

    vals = axis_config.vals
    assert OmegaConf.is_list(vals)
    num_vals = len(axis_config.vals)

    assert num_vals % num_parameters == 0, f"Number of specified values ({num_vals}) must be divisible by the number of parameters ({num_parameters})"

    # iterate over vars
    for val_index in range(0, num_vals, num_parameters):
        sweep_axis_param = OmegaConf.create()
        for param_index, param in enumerate(parameters):
            OmegaConf.update(sweep_axis_param, param,
                             vals[val_index + param_index])
        yield sweep_axis_param


#! GOAL:
#? How to pass a multiple parameter change, e.g. model / model_args ?
# see: OmegaConf.select() and OmegaConf.update() in https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#utility-functions
