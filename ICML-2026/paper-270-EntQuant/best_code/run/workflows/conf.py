import importlib
import logging

from hydra_zen import store

from run.hydra_zen import make_config

# Import modules to trigger store registrations
for module in ["entquant", "eval", "model", "quantization", "run", "super_weights"]:
    importlib.import_module(f"run.conf.{module}")
for module in ["build", "evaluation"]:
    importlib.import_module(f"run.workflows.{module}")

logger = logging.getLogger(__name__)

# -------- Workflows --------

store(
    make_config(
        hydra_defaults=[
            {"build": "build_base_model"},
            {"eval": "evaluate_model"},
            "_self_",
        ],
    ),
    group="workflow",
    name="default",
)
store(
    make_config(
        hydra_defaults=[
            {"build": "build_entquant_model"},
            {"eval": "evaluate_model"},
            "_self_",
        ],
    ),
    group="workflow",
    name="entquant",
)
store(
    make_config(
        hydra_defaults=[
            {"build": "build_quantized_model"},
            {"eval": "evaluate_model"},
            "_self_",
        ],
    ),
    group="workflow",
    name="quantization",
)

# -------- Configs --------

store(
    make_config(
        hydra_defaults=[
            {"model": "default"},
            {"tokenizer": "default"},
            {"eval": "none"},
            "_self_",
        ],
        # crucial because otherwise union type hints and similar make hydra fail
        # the actual problem is that omegaconf/hydra treat dataclass configs
        # as a container (also after instantiation)
        hydra_convert="all",
    ),
    group="cfg",
    name="default",
)

store(
    make_config(
        hydra_defaults=[
            {"model": "default"},
            {"tokenizer": "default"},
            {"super_weights": "off"},
            {"entquant": "default"},
            {"entquant/optimizer": "default"},
            {"eval": "none"},
            "_self_",
        ],
        hydra_convert="all",
    ),
    group="cfg",
    name="entquant",
)

store(
    make_config(
        hydra_defaults=[
            {"model": "default"},
            {"tokenizer": "default"},
            {"super_weights": "off"},
            {"quantization": "default"},
            {"quantization/config": "bnb_nf4"},
            {"eval": "none"},
            "_self_",
        ],
        hydra_convert="all",
    ),
    group="cfg",
    name="quantization",
)
