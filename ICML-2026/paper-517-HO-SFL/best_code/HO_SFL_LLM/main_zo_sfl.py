import sys
import os
import torch
import wandb
import random
import numpy as np

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    ZerothOrderSetting,
    SplitFederatedLearningSetting,
)

from src.runner.sfl_zo_runner import SFLZORunner


class CliSetting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    ZerothOrderSetting,
    SplitFederatedLearningSetting,
):
    pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = CliSetting()
    print("=== Configuration ===")
    print(args)
    print("=====================")

    set_seed(args.seed)

    run_name = (
        f"SFL-ZO-{args.data_setting.dataset.value}-"
        f"p{args.local_epochs}-"
        f"seed{args.seed}"
    )

    if args.log_to_wandb:
        wandb.init(
            project=args.project_name,
            name=run_name,
            config=args.model_dump(mode="json"),
        )

    runner = SFLZORunner(args)

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[Info] Interrupted.")
    except Exception as e:
        print(f"\n[Error] {e}")
        raise
    finally:
        if args.log_to_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
