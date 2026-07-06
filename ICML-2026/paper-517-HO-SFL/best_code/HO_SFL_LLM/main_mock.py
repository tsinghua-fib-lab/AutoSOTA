import sys
import os
import wandb
import numpy as np

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    HO_SplitFederatedLearningSetting,
    SplitFederatedLearningSetting,
    HybridGradientEstimatorSetting,
    ZerothOrderSetting,
)
from src.runner.mock_ho_sfl_runner import MemoryProfilingRunner
from src.runner.mock_sfl_runner import SFLMemoryProfilingRunner
from src.runner.mock_centralized_runner import CentralizedMemoryProfilingRunner
from src.runner.mock_inference_runner import InferenceMemoryProfilingRunner


class CliSetting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    HO_SplitFederatedLearningSetting,
    SplitFederatedLearningSetting,
    HybridGradientEstimatorSetting,
    ZerothOrderSetting,
):
    pass


def main():
    args = CliSetting()
    print("=== Configuration ===")
    print(args)
    print("=====================")
    if args.framework == "HO-SFL":
        runner = MemoryProfilingRunner(args)
    elif args.framework == "SFL":
        runner = SFLMemoryProfilingRunner(args)
    elif args.framework == "Centralized":
        runner = CentralizedMemoryProfilingRunner(args)
    elif args.framework == "Inference":
        runner = InferenceMemoryProfilingRunner(args)
    else:
        raise NotImplementedError(f"Framework {args.framework} not implemented.")
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
