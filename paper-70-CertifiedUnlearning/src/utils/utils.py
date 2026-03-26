import argparse
import logging
import sys

import yaml

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_config(file_path):
    with open(file_path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level (default: 0)",
    )
    return parser.parse_args()
