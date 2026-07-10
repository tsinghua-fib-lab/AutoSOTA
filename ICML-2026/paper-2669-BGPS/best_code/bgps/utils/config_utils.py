from argparse import ArgumentParser, REMAINDER
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "prepare_config",
    "default_parser",
    "load_config",
    "override_config_by_cli",
    "resolve_config",
]


def prepare_config():
    """All-in-one common workflow."""
    parser = default_parser()
    args = parser.parse_args()
    d_config = load_config(args.config)
    d_config = override_config_by_cli(d_config, args.script_args)
    config = resolve_config(d_config)
    return args, config


def default_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML configuration", required=True)
    parser.add_argument("--debug", action="store_true", help="Debugging flag")
    parser.add_argument("script_args", nargs=REMAINDER, help="Override config by CLI")
    return parser


def load_config(yaml_path: str) -> DictConfig:
    cfg = OmegaConf.load(yaml_path)
    return cfg


def override_config_by_cli(base_cfg: DictConfig, script_args: List[str]) -> DictConfig:
    """Override the read config yaml.

    Note:
        See OmegaConf for details.
        Usage of CLI script_args:
            wandb.mode=disabled
            adam.betas="[0.9, 0.98]"
            resume.ignore_keys="['optimizer', 'scheduler']"
    """
    cli_cfg = OmegaConf.from_dotlist(script_args)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    return cfg


def resolve_config(cfg: DictConfig) -> Dict:
    return OmegaConf.to_container(cfg, resolve=True)
