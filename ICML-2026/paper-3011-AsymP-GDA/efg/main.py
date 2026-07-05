import os

import hydra
from hydra.utils import instantiate
from runner.train import train


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    solver = create_solver(cfg.algorithm)
    param_str = [f"{key}={value}" for key, value in cfg.algorithm.items() if key != "_target_"]
    if param_str != []:
        output_dir = f"logs/{cfg.game}/{cfg.algorithm._target_.split('.')[-1]}/{'_'.join(param_str)}"
    else:
        output_dir = f"logs/{cfg.game}/{cfg.algorithm._target_.split('.')[-1]}"
    os.makedirs(output_dir, exist_ok=True)
    train(
        solver,
        cfg.traverse_type,
        "avg-iterate",
        cfg.iter,
        cfg.print_freq,
        cfg.game,
        output_dir=output_dir,
    )


def create_solver(algo_config):
    return instantiate(algo_config)


if __name__ == "__main__":
    main()
