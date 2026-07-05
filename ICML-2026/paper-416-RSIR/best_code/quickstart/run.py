import os
import wandb
import datetime
from utils import *
import torch
import random
import numpy as np

class RngStateSaver:

    def __init__(self):
        self.states = {}

    def __enter__(self):

        self.states['python'] = random.getstate()
        self.states['numpy'] = np.random.get_state()
        self.states['torch_cpu'] = torch.get_rng_state()
        if torch.cuda.is_available():
            self.states['torch_gpu'] = torch.cuda.get_rng_state_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        random.setstate(self.states['python'])
        np.random.set_state(self.states['numpy'])
        torch.set_rng_state(self.states['torch_cpu'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.states['torch_gpu'])

def run_generation(config: dict):

    log_dir = os.path.join(config['model']['model'], config['data']['dataset'], "generation_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log")
    logger = get_logger(log_path)

    logger.info("Task: Generate sequences only (no training)")
    logger.info(config)

    dataset_list = prepare_datasets(config)
    logger.info(dataset_list[0])

    
    with RngStateSaver():
        model_fixed_path = config.get('theta_r_path')
        model_generator = prepare_model(config, dataset_list)
        model_generator.load_checkpoint(model_fixed_path) 
        model_generator.to(config['train']['device'])

    model_generator.eval()

    model_generator._internal_models = {'fixed': model_generator}
    logger.info("The model's internal 'fixed' reference is set to itself for generation.")

    if config.get('generation', {}).get('enabled', True):
        logger.info("---------- Start generating and saving augmented datasets ----------")
        
        gen_config = config.get('generation', {})
        m = gen_config.get('m', 5)
        k = gen_config.get('k', 20)
        max_len = gen_config.get('max_len', config['data']['max_seq_len'])
        local_prob = gen_config.get('local_prob', 0.7)
        min_len = gen_config.get('min_len', 3)

        logger.info(f"Generate parameters: m={m}, k={k}, max_len={max_len}, local_prob={local_prob}, min_len={min_len}")

        path = os.path.join('dataset', config['data']['dataset'], config['data']['domain_name_list'][0])
        output_trainfile = config.get('output_trainfile', '_1th')
        if not output_trainfile.startswith('_'):
            output_trainfile = '_' + output_trainfile
        save_path = os.path.join(path, f"train{output_trainfile}.pth")
        logger.info(f"Generated dataset will be saved to: {save_path}")

        model_generator.generate_and_save_sequences_parallel(
            save_path=save_path,
            m=m, 
            k=k, 
            max_len=max_len, 
            local_prob=local_prob,
            min_len=min_len
        )

        logger.info("---------- Enhanced dataset generation and saving completed ----------")
    else:
        logger.warning("'generation.enabled' is not set to true in the configuration file. No action was performed.")

def run_recommender(config: dict):
    wandb.init(
        project="KDD2024",
        name=f"{config['model']['model'] + '-' + config['data']['dataset']}",
        config=config,
        mode="disabled",
    )

    log_path = f"{config['model']['model']}/{config['data']['dataset']}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log"
    logger = get_logger(log_path)

    logger.info('PID of this process: {}'.format(os.getpid()))

    dataset_list = prepare_datasets(config)
    logger.info(config)
    logger.info(dataset_list[0])
    
    logger.info("Initializing a new model to be trained from scratch.")
    model_to_train = prepare_model(config, dataset_list)
    
    model_to_train.fit()
    
    model_to_train.evaluate()

    wandb.finish()

def tune(config: dict = None):
    with wandb.init(config=config):

        config = wandb.config
        config = transform_sweep_config_into_config(config)
        log_path = f"{config['model']['model']}/{config['data']['dataset']}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log"
        logger = get_logger(log_path)

        logger.info('PID of this process: {}'.format(os.getpid()))

        dataset_list = prepare_datasets(config)
        logger.info(dataset_list[0])

        model = prepare_model(config, dataset_list)

        model.fit()
        model.evaluate()
