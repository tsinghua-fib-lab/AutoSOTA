import argparse
import os
import sys
import torch
import yaml
import argparse

# import models
sys.dont_write_bytecode = True


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.config = None
        self.opts = self.initialize()

    def parse_config(self, d, opts=None):
        if opts is None:
            opts = type('new', (object,), d)
        seqs = tuple, list, set, frozenset
        for i, j in d.items():
            if isinstance(j, dict):
                setattr(opts, i, self.parse_config(j))
            elif isinstance(j, seqs):
                setattr(opts, i, type(j)(self.parse_config(sj) if isinstance(sj, dict) else sj for sj in j))
            else:
                setattr(opts, i, j)
        return opts

    def initialize(self):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser = argparse.ArgumentParser()
        # parser.add_argument('--config', type=str, default='../configs/cub_200_2011_classification.yaml', help='Path to the config file.')
        parser.add_argument('--config', type=str, default='../configs/fashion_style_classification.yaml',
                            help='Path to the config file.')
        parser.add_argument('--name', type=str, default='Fashion Style classification',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--output_path', type=str, default='.', help="outputs path")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--eval", action="store_true")
        opts = parser.parse_args()
        self.config = get_config(opts.config)
        opts = self.parse_config(self.config, opts)
        opts.output_path = os.path.join(opts.trainer.output_dir, opts.output_path)
        return opts


def get_config(config_yaml):
    with open(config_yaml, 'r') as stream:
        config = yaml.safe_load(stream)
        # config['gpu_ids'] = parse_gpu_ids(config['gpu_ids'])
        return config


def parse_gpu_ids(str_ids):
    print(str_ids)
    # set gpu ids
    gpu_ids = []
    str_ids = str_ids.split(',')
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    return gpu_ids


if __name__ == '__main__':
    opts = BaseOptions()
    pass
