import copy
import logging
import random
import subprocess
import sys
import time
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
from datetime import datetime

from omegaconf import DictConfig, OmegaConf
from sweep.sweep import EXPERIMENT_CONFIG, SweepGrid

from utils.utils import get_config

LOGGER = logging.getLogger(__name__)


class RunHandler(object):
    def __init__(self, config: Union[str, Path, dict, DictConfig], script_path: str):
        self.config = get_config(config)
        self.config_raw = copy.deepcopy(self.config)
        self.script_path = script_path

    def run(self):
        run_config = self.config.run_config
        if run_config.exec_type == 'sequential':
            self._run_sequential()
        elif run_config.exec_type == 'parallel':
            self._run_parallel()
        elif run_config.exec_type == 'only-config':
            self.__run(gpu_ids=self.config.run_config.gpu_ids, runs_per_gpu=self.config.run_config.runs_per_gpu,
                       only_config=True)


    def _run_sequential(self):
        """Run experiments on the (first) gpu_id sequentially."""
        # get gpu_id
        gpu_ids = self.config.run_config.gpu_ids
        if isinstance(gpu_ids, list):
            gpu_id = gpu_ids[0]  # use the first gpu_id
        elif gpu_ids is not None:
            gpu_id = int(gpu_ids)
        else:  # Use CPU
            gpu_id = None
        self.__run(gpu_ids=gpu_id, runs_per_gpu=1)

    def _run_parallel(self):
        """Run experiments in parallel."""
        self.__run(gpu_ids=self.config.run_config.gpu_ids, runs_per_gpu=self.config.run_config.runs_per_gpu)

    def __run(self, runs_per_gpu: int, gpu_ids: Optional[Union[int, List[int]]] = None, only_config=False):
        """Run experiments in separate processes."""
        # get seeds
        seeds = self.config.run_config.seeds
        assert len(seeds) > 0, f"No seeds are given to start runs."
        if hasattr(self.config.run_config, 'init_models'):
            init_models = self.config.run_config.init_models
        else:
            init_models = [None]

        config = copy.deepcopy(self.config)
        sweep_configs = self._extract_sweep(config)

        experiment_configs = []
        # Set Tag and Sweep FP for each config
        sweep_tag = f"sweep_{self.config.config.experiment_data.experiment_name}_{datetime.now().strftime('%m-%d-%H:%M:%S')}"
        for idx, cfg in enumerate(sweep_configs):
            cfg.config.experiment_data.experiment_sweep_fp = f"{sweep_tag}_cfg{idx}"
            cfg.config.experiment_data.experiment_sweep_tag = sweep_tag

        if hasattr(self.config.run_config, 'skip_cfg') and len(self.config.run_config.skip_cfg) > 0:
            skip_list = list(self.config.run_config.skip_cfg)
            LOGGER.warning(f"{len(skip_list)} runconfig-seed combination are skipped!")
        else:
            skip_list = []

        skipped = 0
        for init_model in init_models:
            for seed in seeds:
                for idx, cfg in enumerate(sweep_configs):
                    if [idx, seed] in skip_list:
                        skipped += 1
                        continue
                    current_config = copy.deepcopy(cfg)
                    if init_model is not None:  # Do not overwrite init model when not specified
                        current_config[EXPERIMENT_CONFIG].trainer.trainer_config.init_model = init_model
                    current_config[EXPERIMENT_CONFIG].experiment_data.seed = seed
                    experiment_configs.append(current_config)
        if skipped != len(skip_list):
            raise ValueError("Number of skipped configs specified and actually skipped mismatch! Check run config!")

        if only_config:
            create_slurm_call(experiment_configs, self.script_path, gpu_ids=gpu_ids, runs_per_gpu=runs_per_gpu)
        else:
            schedule_runs(experiment_configs, self.script_path, gpu_ids=gpu_ids, runs_per_gpu=runs_per_gpu)

    def _extract_sweep(self, config: DictConfig) -> List[DictConfig]:
        if OmegaConf.is_missing(config, 'sweep'):
            # no sweep specified
            LOGGER.error("No hyperparameter sweep specified, but experiment started through RunHandler.")
            raise ValueError("No sweep specified")
        else:
            # get sweep
            if config.sweep.type == 'grid':
                grid_sweeper = SweepGrid(config)
                sweep_configs = grid_sweeper.generate_configs()
                return sweep_configs
            else:
                raise ValueError("Unknown or unspecified sweep.type!")
                # ! start from here to add new hyperparameter search methods


## FUNCTIONS:

def update_and_save_config(config: DictConfig, gpu_id: int, count: int, add_seed_to_name=True) -> str:
    """Updates the config-file with seed and gpu_id, saves it in the current working directory and returns its name.

    Args:
        config (DictConfig): The config to be updated and saved.
        seed (int):
        gpu_id (int):

    Returns:
        str: Name of the saved config file.
    """
    # save seed in config
    seed = config[EXPERIMENT_CONFIG].experiment_data.seed
    exp_name = config[EXPERIMENT_CONFIG].experiment_data.experiment_name + (f"-seed{seed}-c{count}" if add_seed_to_name else f"-c{count}")
    config[EXPERIMENT_CONFIG].experiment_data.experiment_name = exp_name  # !< config is modified here
    # set device
    config[EXPERIMENT_CONFIG].experiment_data.gpu_id = gpu_id  # !< config is modified here
    save_name = exp_name + '.yaml'
    OmegaConf.save(config, Path.cwd() / save_name)
    return save_name


def create_slurm_call(experiment_configs: List[DictConfig],
                      script_path: str,
                      runs_per_gpu: int,
                      gpu_ids: Optional[Union[int, List[int]]] = None):

    assert len(experiment_configs) > 0, f"No experiments to schedule given."
    if gpu_ids is None:
        gpu_ids = [-1]
    elif isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]

    logging.info(f"Create Sweep Slum Calls with {len(experiment_configs)} runs on devices '{gpu_ids}'"
                 f" with {runs_per_gpu} parallel runs per device!")

    for idx, config in enumerate(experiment_configs):
        current_config = copy.deepcopy(config)
        config_name = update_and_save_config(config=current_config, gpu_id=gpu_ids[idx % len(gpu_ids)], count=idx,
                                             add_seed_to_name=False)

        # start run via subprocess call
        run_command = f'python {script_path} --config-path {str(Path.cwd())} --config-name {config_name}'
        logging.info(f"\n{run_command}\n")

def schedule_runs(experiment_configs: List[DictConfig],
                  script_path: str,
                  runs_per_gpu: int,
                  gpu_ids: Optional[Union[int, List[int]]] = None):
    """Distribute multiple runs on different gpus of the same machine.

    Example:
    Given: 5 experiment configs, gpu id: 0 1, runs per gpu: 3
    Result: Starts runs on gpu 0 and 1 as long as number runs_per_gpu is not reached
    and there are still runs in the experiment_configs list.

    Args:
        experiment_configs (List[DictConfig]): List of experiment configs to schedule
        script_path (str): Full path to a hydra python script
        runs_per_gpu (int): The max runs per gpu
        gpu_ids (Optional[Union[int, List[int]]], optional): The gpus to schedule runs on. Defaults to None (in this case value taken from config)
    """
    assert len(experiment_configs) > 0, f"No experiments to schedule given."
    # for approximately equal memory usage during hyperparam tuning, randomly shuffle list of processes
    random.shuffle(experiment_configs) #TODO SHUFFLE ONLY BEFORE ADDING SWEEP SO THAT ABORT OF RUN AFTER CERTAIN SEED IS POSSIBLE

    # array to keep track on how many runs are currently running per GPU
    if gpu_ids is None:
        gpu_ids = [-1]
    elif isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]

    logging.info(f"Running Sweep with {len(experiment_configs)} runs on devices '{gpu_ids}'"
                 f" with {runs_per_gpu} parallel runs per device!")

    n_parallel_runs = len(gpu_ids) * runs_per_gpu
    gpu_counter = np.zeros((len(gpu_ids)), dtype=int)

    running_processes = {}
    counter = 0
    while True:

        # * start new runs
        for _ in range(n_parallel_runs - len(running_processes)):

            if counter >= len(experiment_configs):
                break
            # * determine which GPU to use
            node_id = int(np.argmin(gpu_counter))
            gpu_counter[node_id] += 1
            gpu_id = gpu_ids[node_id]

            # * prepare next experiment in list
            current_config = copy.deepcopy(experiment_configs[counter])
            config_name = update_and_save_config(config=current_config, gpu_id=gpu_id, count=counter)

            # start run via subprocess call
            run_command = f'python {script_path} --config-path {str(Path.cwd())} --config-name {config_name}'
            LOGGER.info(f'Starting run {counter + 1}/{len(experiment_configs)}: {run_command}')
            running_processes[(counter + 1, run_command, node_id)] = subprocess.Popen(
                run_command,
                stdout=subprocess.DEVNULL,
                shell=True
            )

            counter += 1
            time.sleep(0.2)

        # check for completed runs
        for key, process in running_processes.items():
            if process.poll() is not None:
                LOGGER.info(f'Finished run {key[0]} ({key[1]})')
                gpu_counter[key[2]] -= 1
                LOGGER.info('Cleaning up...\n')
                try:
                    _ = process.communicate(timeout=5)
                except TimeoutError:
                    LOGGER.warning(f'\nWARNING: PROCESS {key} COULD NOT BE REAPED!\n')
                running_processes[key] = None

        # delete possibly finished runs
        running_processes = {
            key: val
            for key, val in running_processes.items() if val is not None
        }
        time.sleep(2)

        if (len(running_processes) == 0) and (counter >= len(experiment_configs)):
            break

    LOGGER.info('Done')
    sys.stdout.flush()
