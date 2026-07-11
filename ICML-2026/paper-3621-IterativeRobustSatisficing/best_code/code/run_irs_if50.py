"""Run IRS on CIFAR-10-LT IF=50 with rubric settings."""
import sys
from pathlib import Path

# Ensure the code directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cifar10lt_experiment import (
    CIFAR10LTConfig,
    CIFAR10LTRunner,
    plot_inline,
)

# Rubric settings: IF=50, seed=123, WideResNet-28-10, class-wise IRS, tau=0.1
CONFIG = CIFAR10LTConfig(
    data_dir="/datasets",
    imbalance_factor=50.0,
    seeds=(123,),
    batch_size=512,
    model_arch="wrn28_10",
    use_if_hyperparams=True,
    # Only run IRS for reproduction
    methods=("irs",),
    # IRS specific settings (matches IF=50 preset)
    irs_epochs=100,
    irs_lr=1e-3,
    target_tau=0.2,
    weight_decay=5e-5,
)

if __name__ == "__main__":
    runner = CIFAR10LTRunner(CONFIG)
    runner.run()
