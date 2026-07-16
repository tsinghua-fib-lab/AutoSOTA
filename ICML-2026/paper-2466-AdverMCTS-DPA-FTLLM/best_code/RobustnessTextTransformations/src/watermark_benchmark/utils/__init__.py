# utils folder

import inspect
import random
import time

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

# from watermark_benchmark.utils.classes import ConfigSpec


def load_config(config_file):
    # Load arguments
    with open(config_file, "r") as infile:
        config = yaml.safe_load(infile)
    return ConfigSpec.from_dict(config)


def setup_randomness(config):
    # Setup randomness
    if config.seed is None:
        seed = int(time.time() * 1000) % 100000
    else:
        seed = config.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_server_args(config):
    # Setup server
    additional_keywords = [
        "gpu_memory_utilization",
        "seed",
        "dtype",
        "trust_remote_code",
    ]
    additional_args = {
        key: config.__dict__[key]
        for key in additional_keywords
        if config.__dict__[key] is not None
    }
    return additional_args


def get_output_file(config):
    # Setup outfile
    return config.output_file


def get_input_file(config):
    # Setup outfile
    return config.input_file


def get_tokenizer(model):
    # Return HF tokenizer
    return AutoTokenizer.from_pretrained(model)


def adapt_arguments(function, kwargs, *non_keyword_args):
    # Return the function with the kwargs if they are included in the signature
    function_signature = inspect.getfullargspec(function)
    if function_signature.varargs is not None:
        raise ValueError(
            f"{function.__name__} function does not accept varargs. Please use kwargs instead."
        )
    if function_signature.varkw is not None:
        filtered_kwargs = kwargs
    else:
        filtered_kwargs = {
            kw: val
            for kw, val in kwargs.items()
            if kw in function_signature.args
        }

    return function(*non_keyword_args, **filtered_kwargs)
