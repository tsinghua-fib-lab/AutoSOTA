import json
import logging
import os
from pathlib import Path
import sys

import git

from cycloreps.utils.utils import load_envs
from rich.logging import RichHandler
from rich.traceback import install

# Enable pretty tracebacks
install(show_locals=False)

from rich.console import Console
from rich.logging import RichHandler
import logging
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

handler = RichHandler(show_level=False, show_time=False, show_path=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[handler],
)

logger = logging.getLogger(__name__)
logger.propagate = False


load_envs()

try:
    PROJECT_ROOT = Path(
        git.Repo(Path.cwd(), search_parent_directories=True).working_dir
    )
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

logger.debug(f"Inferred project root: {PROJECT_ROOT}")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

__all__ = ["PROJECT_ROOT"]


plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "axes.titlesize": 24,  # Larger axes/title fonts
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 20,
        "legend.fontsize": 24,
    }
)
sns.set_context("talk")


FULL_TO_SHORT_NAMES = json.load(
    open(f"{PROJECT_ROOT}/misc/full_to_short_model_name.json")
)

import dotenv

dotenv.load_dotenv(dotenv_path=PROJECT_ROOT / ".env")