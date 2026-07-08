#!/usr/bin/env python
"""Format a Weights and Biases sweep based on config options. """

import argparse
import yaml


def merge_dicts(
        dict1: dict, dict2: dict, /, *,
        promote_types: bool = False
) -> dict:
    """Recursively merge two dictionaries, preserving nested structure.

    Overwrites values of dict1 with values of dict2 for matching keys
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict1

    merged = dict1.copy()
    for key, value in dict2.items():

        if key in merged and isinstance(merged[key], dict) and \
                isinstance(value, dict):
            merged[key] = merge_dicts(
                merged[key], value,
                promote_types=promote_types
            )
        else:

            if promote_types and (key in merged):
                value = type(merged[key])(value)

            merged[key] = value

    return merged


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-S", "--sweep",
        nargs='+', dest="sweep", required=True,
        help="Specify config .yaml files to construct a sweep for."
    )
    args, unknown = parser.parse_known_args()

    grouped, key = dict(), None
    for arg in unknown:
        if arg.startswith("-"):
            key = arg
            grouped[key] = []
        else:
            grouped[key].append(arg)

    return args.sweep, grouped


if __name__ == "__main__":

    sweep_files, console_args = parse_cli()

    sweep = dict()
    for f in sweep_files:
        with open(f, "r") as file:
            sweep = merge_dicts(sweep, yaml.safe_load(file))

    formatted = {
        **sweep,
        "command": [
            "${env}",
            "${program}",
            "-m",
            "experiment.src.main"
        ] + [
            item for k, v in console_args.items() for item in [k] + v
        ] + [
            "-S",
            "${args_no_hyphens}"
        ]
    }

    out_stream = yaml.dump(formatted, default_flow_style=False)

    print(out_stream)
