import yaml
from easydict import EasyDict as edict
from utils.print_colors import red, blue, cyan

def get_config(config, seed):
    config_dir = f'./config/{config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed
    return config

def print_cfg(config):
    print(cyan('DATA:') + f' {config.data}')
    print(cyan('MIX:') + f' {config.mix}')
    print(cyan('MODEL:') + f' {config.model}')
    print(cyan('TRAIN:') + f' {config.train}')
    print(cyan('SAMPLER:') + f' {config.sampler}')
    print(cyan('SAMPLE:') + f' {config.sample}')