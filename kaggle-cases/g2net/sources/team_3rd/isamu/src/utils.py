import gc
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import yaml
#from coolname import generate_slug

from src.config import COMP_NAME, CONFIG_PATH, OUTPUT_PATH


def prepare_args_test(config_path=CONFIG_PATH, default_config="default_run", test_args=[]):
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        action="store",
        dest="config",
        help="Configuration scheme",
        default=default_config,
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        dest="debug",
        help="Debug mode",
        default=False
    )
    
    parser.set_defaults(logging=True, submit=False)

    args = parser.parse_args(args=test_args)

    # Lookup the config from the YAML file and set args
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        if args.config != default_config:
            print("Using", args.config, "configuration")

        for k, v in cfg[args.config].items():
            setattr(args, k, v)

    return args


def prepare_args(config_path=CONFIG_PATH, default_config="default_run"):
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        action="store",
        dest="config",
        help="Configuration scheme",
        default=default_config,
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        dest="debug",
        help="Debug mode",
        default=False
    )
    
    parser.add_argument(
        "--gpus",
        action="store",
        dest="gpus",
        help="Number of GPUs",
        default=2,
        type=int,
    )

#     parser.add_argument(
#         "--timestamp",
#         action="store",
#         dest="timestamp",
#         help="Timestamp for versioning",
#         default=str(datetime.now().strftime("%Y%m%d-%H%M%S")),
#         type=str,
#     )

#     parser.add_argument(
#         "--fold",
#         action="store",
#         dest="fold",
#         help="Fold number",
#         default=1,
#         type=int,
#     )

#     parser.add_argument(
#         "--seed",
#         action="store",
#         dest="seed",
#         help="Random seed",
#         default=48,
#         type=int,
#     )

#     parser.add_argument(
#         "--slug",
#         action="store",
#         dest="slug",
#         help="Human rememebrable run group",
#         default=generate_slug(3),
#         type=str,
#     )

    parser.add_argument(
        "--logging",
        dest="logging",
        action="store_true",
        help="Flag to log to WandB (on by default)",
    )

    parser.add_argument(
        "--no-logging",
        dest="logging",
        action="store_false",
        help="Flag to prevent logging",
    )

    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        help="Flag to submit on inference",
    )

    parser.set_defaults(logging=True, submit=False)

    args = parser.parse_args()

    # Lookup the config from the YAML file and set args
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        if args.config != default_config:
            print("Using", args.config, "configuration")

        for k, v in cfg[args.config].items():
            setattr(args, k, v)

    return args


def resume_helper(args):
    """
    To resume a run, add this to the YAML/args:

    checkpoint: "20210510-161949"
    wandb_id: 3j79kxq6

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    if hasattr(args, "checkpoint"):
        paths = (
            OUTPUT_PATH / args.checkpoint / args.encoder / f"fold_{args.fold - 1}"
        ).glob("*.*loss.ckpt")
        resume = list(paths)[0]

        if hasattr(args, "wandb_id"):
            run_id = args.wandb_id
        else:
            print("No wandb_id provided. Logging as new run")
            run_id = None
    else:
        resume = None
        run_id = None

    return resume, run_id

