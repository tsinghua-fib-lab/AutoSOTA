import os
import sys
import torch
import random
import numpy as np
import wandb

import hydra
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_manager import DataManager
from src.runner.ho_sfl_runner import HO_SFLRunner
from src.runner.sfl_runner import SFLRunner
from src.runner.sfl_zo_runner import SFLZORunner
from src.runner.mu_splitfed_runner import MU_SplitFedRunner
from src.runner.fsl_sage_runner import FSLSAGERunner

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _inject_hydra_no_output_defaults():
    argv = sys.argv[1:]
    user_set_hydra = any(a.startswith("hydra.") or a.startswith("hydra/") for a in argv)
    if user_set_hydra:
        return

    sys.argv.extend(
        [
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
            "hydra.job.chdir=false",
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    print(f"[System] Global random seed set to: {seed}")


def _build_runner(cfg, data_manager):
    algo_name = cfg.algo.algo_name
    if algo_name == "HO_SFL":
        return HO_SFLRunner(cfg, data_manager)
    elif algo_name == "SFL":
        return SFLRunner(cfg, data_manager)
    elif algo_name == "SFLZO":
        return SFLZORunner(cfg, data_manager)
    elif algo_name == "MU_SplitFed":
        return MU_SplitFedRunner(cfg, data_manager)
    elif algo_name == "FSL_SAGE":
        return FSLSAGERunner(cfg, data_manager)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")


@hydra.main(
    version_base=None,
    config_path="conf/",
    config_name="base",
)
def main(cfg):
    OmegaConf.resolve(cfg)
    print(cfg)
    print(f"=== Project Configuration Loaded ===")
    print(f"Dataset: {cfg.data.dataset}")
    print(f"Algorithm: {cfg.algo.algo_name}")
    print(f"Device: {cfg.system.device}")
    print(
        f"Total Clients: {cfg.runner.num_clients} | Sampled: {cfg.runner.sampled_clients}"
    )
    print("====================================")

    set_seed(int(cfg.seed))

    if cfg.logging.use_wandb:
        zo_p = cfg.algo.get("zo_p", None)
        zo_mu = cfg.algo.get("zo_mu", None)
        run_name = (
            f"{cfg.algo.algo_name}-{cfg.data.dataset}-{cfg.data.partition.algo}-"
            f"{cfg.model.model_name}"
        )
        if zo_p is not None:
            run_name += f"-p{zo_p}"
        if zo_mu is not None:
            run_name += f"-mu{zo_mu}"

        print(
            f"\n[System] Initializing WandB for project: {cfg.logging.project_name}..."
        )
        wandb.init(
            project=cfg.logging.project_name,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        os.makedirs(cfg.logging.log_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA is not available. Please check your device settings."
        )

    print("\nInitializing Data Manager...")
    data_manager = DataManager(cfg)

    print(f"\nInitializing Runner for {cfg.algo.algo_name}...")
    runner = _build_runner(cfg, data_manager)

    print(f"\nStarting Training Loop...")
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        raise

    if cfg.logging.use_wandb:
        random.seed(None)
        table_data, table_columns = data_manager.get_wandb_table_data()
        wandb_table = wandb.Table(data=table_data, columns=table_columns)
        wandb.log({"Client_Data_Statistics": wandb_table})
        wandb.finish()

    print("\n=== Experiment Finished ===")


if __name__ == "__main__":
    _inject_hydra_no_output_defaults()
    main()
