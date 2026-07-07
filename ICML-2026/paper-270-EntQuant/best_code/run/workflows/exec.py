from run import setup_env

setup_env()

import importlib
import logging
import os
from typing import Any

from hydra_zen import store, to_yaml, zen
from transformers import PreTrainedModel

# Import conf to trigger store registrations
importlib.import_module("run.workflows.conf")

logger = logging.getLogger(__name__)


@store(
    name=lambda fn: fn.__name__,
    hydra_defaults=[
        {"workflow": "default"},
        {"cfg": "default"},
        {"run": "default"},
        "_self_",
    ],
    # required for global experiment config
    # https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/412
    # also experiment config need to explicitly override all hydra defaults (with a leading '/')
    zen_dataclass={"kw_only": True},
)
def exec_workflow(zen_cfg: Any, workflow: Any, run: Any, cfg: Any) -> tuple[PreTrainedModel, dict[str, Any]]:
    """Dispatcher that instantiates and runs the selected workflow config."""
    model, results_build = workflow.build()
    results_eval = workflow.eval(model)

    results = {"zen_cfg": zen_cfg, "results_build": results_build, "results_eval": results_eval}
    if run.save_results:
        os.makedirs(run.path, exist_ok=True)
        with open(os.path.join(run.path, "results.yaml"), "w") as f:
            f.write(to_yaml(results))
        logger.info(f"Results saved to {run.path}.")

    return model, results


# late import because experiment requires exec_workflow to be registered
importlib.import_module("run.workflows.experiments")


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(exec_workflow).hydra_main(
        config_name=exec_workflow.__name__,
        config_path=".",
        version_base="1.3",
    )
