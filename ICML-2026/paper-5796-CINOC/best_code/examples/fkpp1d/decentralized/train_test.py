"""
Centralized Deep Policy Control Training Script for 1D Fisher-KPP Equation:
trains a ControlNet policy to manage agent positions and forcing intensities
to steer the FKPP dynamics towards target states while respecting constraints.
"""
import sys
from pathlib import Path
import os

import jax

# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# jax.config.update("jax_enable_x64", True)

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from train_utils import train

train(net_params_filename="test", plot_filename="test", epochs=500)
