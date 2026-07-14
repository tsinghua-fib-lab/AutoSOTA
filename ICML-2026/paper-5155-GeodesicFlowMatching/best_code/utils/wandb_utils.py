import os

import wandb


def initialize_wandb(
    project_name,
    experiment_name,
    tags=None,
    config=None,
    api_key=None,
    entity=None,
):
    key = api_key if api_key is not None else os.environ.get("WANDB_API_KEY")
    if key:
        wandb.login(key=key)

    init_kw = {"project": project_name, "name": experiment_name, "tags": tags or []}
    if entity:
        init_kw["entity"] = entity
    wandb.init(**init_kw)
    if config:
        wandb.config.update(config)


def log_metrics(metrics):
    if wandb.run is not None:
        wandb.log(metrics)
