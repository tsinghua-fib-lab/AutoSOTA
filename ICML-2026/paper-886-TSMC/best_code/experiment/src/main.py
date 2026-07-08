from __future__ import annotations

import argparse
import time
import os
import sys
import hashlib
import importlib

import yaml

from .types import HasConfigLoader, Experimenter, Identity


class OSEnvironment:
    # Helper variables for global OS configurations
    EXPERIMENT_OUT_PREFIX: str = "EXPERIMENT_OUT_PREFIX"


class Constants:
    DEFAULT_OUTPUT_BRANCH: str = "out"
    DEFAULT_PROJECT_NAME: str = "SMZ"

    ENTRY_POINT: str = "smz-run"

    NEST_SEP: str = "."


def define_console_arguments(parser: argparse.ArgumentParser):
    """Local (IDE collapsable) namespace to define all console arguments."""

    # What implementation to execute; Experiment implementation
    parser.add_argument(
        "-M",
        "--module",
        dest="module",
        nargs=1,
        required=True,
        metavar="relative/path/to/MODULE",
        help="Import path for a Builder implementation for the Experimenter. "
             "If no value is provided, the .env file is checked for a path, "
             "otherwise a ValueError is raised.",
    )
    parser.add_argument(
        "-C",
        "--config",
        dest="config",
        metavar="relative/path/to/CONFIG.yaml",
        nargs="+",
        default=[],
        help="Config `yaml` file for specifying setup parameters.",
    )

    # Where to store output data
    parser.add_argument(  # Experiment identity-token
        "-T",
        "--token",
        dest="token",
        default=Constants.DEFAULT_PROJECT_NAME,
        help="Token/ name to write Artifacts, Plots, Logs etc. to. "
             "This is separate from the output path to separate runs. "
             "Note, this is independent of logging or naming for W&B.",
    )
    parser.add_argument(
        "-O",
        "--out",
        dest="out",
        default=Constants.DEFAULT_OUTPUT_BRANCH,
        help="Main path to write all output to. To distinguish between runs "
             "in sub-folders, use `--token EXPERIMENT_NAME` instead.",
    )

    # Weights and Biases service options
    parser.add_argument(
        "--wandb",
        dest="wandb",
        metavar="PROJECT_NAME",
        default=None,
        const=Constants.DEFAULT_PROJECT_NAME,
        help="Opt for logging with Weights and Biases by specifying the "
             "project using `--wandb PROJECT_NAME`. If `PROJECT_NAME` is not "
             f"specified, it will default to the "
             f"`{Constants.DEFAULT_PROJECT_NAME}` project.",
        nargs="?",
    )
    parser.add_argument(
        "--entity",
        "-E",
        dest="entity",
        default=None,
        nargs="?",
        help="Optionally specify a Weights & Biases entity to write to. "
             "This is only relevant if --wandb is specified.",
    )
    parser.add_argument(
        "--sweep",
        "-S",
        dest="sweep",
        nargs="+",
        metavar="KEY=VALUE",
        default=[],
        help="Optionally pass in sweep data to be merged with config. "
             "This can be a arbitrary sequence of key-value pairs. If the "
             "config to be overriden is arbitrarily nested, pass the key in "
             f"as my{Constants.NEST_SEP}nested{Constants.NEST_SEP}key, which "
             'will be split into: {"my": {"nested": {"key": VALUE}}}.',
    )


def setup_identity(out: str, token: str):
    # Create a name with a readable part + a hash-suffix to prevent clashes
    now = time.time()
    formatted = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))

    h = hashlib.sha3_512()
    h.update(bytes(str(now), "utf-8"))
    suffix = h.hexdigest()[:16]

    name = formatted + "-" + suffix

    prefix = os.environ.get(OSEnvironment.EXPERIMENT_OUT_PREFIX, os.sep)
    out = os.path.join(prefix, out.lstrip(os.sep))

    return Identity(out, token, name, False)


def login_wandb(
        name: str, /, config_data: dict, entity: str | None = None
):
    import wandb

    run_id = Identity()

    return wandb.init(
        project=name, dir=run_id.make_path(), config=config_data, entity=entity
    )


def merge_dicts(
        dict1: dict, dict2: dict,
        /,
        *,
        promote_types: bool = False
) -> dict:
    """Recursively merge two dictionaries, preserving nested structure.

    Overwrites values of dict1 with values of dict2 for matching keys
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict1

    merged = dict1.copy()
    for key, value in dict2.items():
        if (key in merged) and \
                isinstance(merged[key], dict) and \
                isinstance(value, dict):
            merged[key] = merge_dicts(
                merged[key], value, promote_types=promote_types
            )
        else:
            if promote_types:
                value = yaml.safe_load(value)

            merged[key] = value

    return merged


def parse_sweep(entries: list[str]) -> dict:
    # Parse KEY=VALUE pattern into a dictionary
    config = {k: v for item in entries for k, v in [item.split("=")]}

    # Parse MY.NESTED.KEY pattern into nested dictionaries
    result = {}
    for k, v in config.items():
        keys = k.split(Constants.NEST_SEP)

        nest = result
        for subkey in keys[:-1]:
            nest = nest.setdefault(subkey, {})
        nest[keys[-1]] = v

    return result


def get_config(config_files: list[str]) -> dict:
    out = dict()
    for filename in config_files:
        try:
            with open(filename, "r") as f:
                result = yaml.load(f, yaml.loader.SafeLoader)
        except (IOError, FileNotFoundError):
            print(f"Given config file {filename} could not be found!")
            print(f"Program base path: {os.getcwd()}")
            raise

        out = merge_dicts(out, result)

    return out


def main[ExperimentModule: HasConfigLoader]():
    argument_parser = argparse.ArgumentParser(Constants.ENTRY_POINT)
    define_console_arguments(argument_parser)
    argc = argument_parser.parse_args()

    # Modify sys.path to make dynamic modules findable (dependency injection)
    module_path = os.path.join(os.getcwd(), *argc.module)
    path, module_name = os.path.split(module_path.rstrip(os.sep))

    sys.path.append(path)  # Makes module findable.

    # Setup implementation and credentials.
    try:
        my_module: ExperimentModule = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Dynamic Module {module_name} not found in {path}"
        ) from e

    identity = setup_identity(argc.out, argc.token)
    os.makedirs(identity.make_path(), exist_ok=True)

    # Get setup configuration and write a copy to the output-directory.
    config: dict = get_config(argc.config)

    # Override config with sweep specifications if necessary.
    if argc.sweep:
        config = merge_dicts(
            config,
            parse_sweep(argc.sweep),
            promote_types=True,  # CLI is interpreted as string only
        )

    filepath = os.path.join(identity.make_path(), "config.yaml")
    with open(filepath, "w") as output_file:
        yaml.dump(config, output_file)

    # If specified, initialize external debug tools.
    client = None
    if argc.wandb:
        client = login_wandb(argc.wandb, config, entity=argc.entity)

    # Create setup setup.
    experiment: Experimenter = my_module.from_config(config, client)

    # Log-metadata to the console.
    print(f"{__name__}: Setup Complete under ID: {identity.make_path()}")

    experiment.run()


if __name__ == "__main__":
    main()
