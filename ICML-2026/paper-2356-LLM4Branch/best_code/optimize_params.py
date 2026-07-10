"""Standalone script to run parameter optimization for indset benchmark."""
import sys
import os
import logging
import time

os.chdir('/repo')
sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/examples/brancher')

from utils.utils import load_program, get_scip_params
from examples.brancher.evaluator import params_optimizer
from openevolve.config import EvaConfig

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the program
program_path = "./examples/brancher/program/indset/program.py"
score_function, used_features, function_params, bound_params = load_program(program_path)
logger.info(f"Loaded program: {len(used_features)} features, {len(function_params)} params")
logger.info(f"Features: {used_features}")
logger.info(f"Initial params: {function_params}")
logger.info(f"Bounds: {bound_params}")

# Create config
config = EvaConfig.from_yaml(
    './examples/brancher/evaluator_config.yaml',
    'indset',
    '0-7',  # cores list range
    '/tmp/cpu_locks/indset_opt'
)

# Override settings
config._config['train_num'] = 4
config._config['train_seed'] = 666666
config._config['max_workers'] = 8
config._config['target'] = 'time'
config._config['parameter_opt'] = True
config._config['n_calls'] = 50
config._config['n_initial'] = 10

logger.info(f"Config: target={config.target}, n_calls={config.n_calls}, n_initial={config.n_initial}")

# Set up SCIP params for training (180s per instance for training)
scip_params = get_scip_params(time_limit=180)

# Run parameter optimization
train_seeds = list(range(config.train_seed, config.train_seed + config.train_num))
logger.info(f"Training seeds: {train_seeds}")
logger.info("Starting parameter optimization...")

start_time = time.time()
optimized_params = params_optimizer(
    dataset=config.dataset,
    seeds=train_seeds,
    cores_list=config.cores_list,
    lock_dir=config.lock_dir,
    score_function=score_function,
    used_features=used_features,
    function_params=function_params,
    bound_params=bound_params,
    scip_params=scip_params,
    config=config
)
elapsed = time.time() - start_time

logger.info(f"Optimization finished in {elapsed:.1f}s")
logger.info(f"Original params: {[f'{p:.6f}' for p in function_params]}")
logger.info(f"Optimized params: {[f'{p:.6f}' for p in optimized_params]}")

# Print in Python list format for easy copy
print("\n=== OPTIMIZED PARAMS (copy to program.py) ===")
print(f"PARAMS = {[round(p, 10) for p in optimized_params]}")
print("==============================================")
