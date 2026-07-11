import argparse

from taming_the_ito_lyon.config import load_toml_config
from taming_the_ito_lyon.training.experiment import experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment with a config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TOML config file",
    )
    args = parser.parse_args()
    config = load_toml_config(args.config)
    experiment(config, config_path=args.config)


if __name__ == "__main__":
    main()
