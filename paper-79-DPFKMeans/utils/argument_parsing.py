import argparse
import sys
import yaml


def add_utils_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--args_config", type=str, default=None)
    return parser


def maybe_inject_arguments_from_config():
    # Check if config is provided in command-line using a temporary parser
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--args_config')
    temp_args, _ = arg_parser.parse_known_args()

    if temp_args.args_config:
        with open(temp_args.args_config) as file:
            config = yaml.safe_load(file)

        # Inject config arguments into sys.argv
        for key, value in config.items():
            # Only add config items if they are not already in sys.argv
            if f"--{key}" not in sys.argv:
                sys.argv.extend([f"--{key}", str(value)])
