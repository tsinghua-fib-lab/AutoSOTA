import json
from pathlib import Path
import subprocess
import sys
from typing import Optional, Union
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import random
import logging

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Union[torch.device, str, int]) -> torch.device:
    if device == "auto":
        device = "cuda"
    elif device == -1:
        device = torch.device("cpu")
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    else:
        device = torch.device(device)
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        LOGGER.warning(f"Device '{str(device)}' is not available! Using cpu now.")
        device = torch.device("cpu")
    return device


def get_git_hash() -> Optional[str]:
    """Try to get the git hash of the current repository.

    Returns
    -------
    Optional[str]
        Git hash if git is installed and the project is cloned via git, None otherwise.
    """
    current_dir = str(Path(__file__).absolute().parent)
    try:
        if subprocess.call(['git', '-C', current_dir, 'branch'],
                           stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL) == 0:
            git_output = subprocess.check_output(
                ['git', '-C', current_dir, 'describe', '--always'])
            return git_output.strip().decode('ascii')
    except OSError:
        pass  # git is probably not installed
    return None

def setup_logging(log_file: str = "output.log"):
    """Initialize logging to `log_file` and stdout.     

    Args:
        log_file (str, optional): Name of the log file. Defaults to "output.log".
    """


    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(handlers=[file_handler, stdout_handler],
                        level=logging.INFO,
                        format='%(asctime)s: %(message)s')

    # Log uncaught exceptions
    def exception_logging(typ, value, traceback):
        LOGGER.exception('Uncaught exception',
                         exc_info=(typ, value, traceback))

    sys.excepthook = exception_logging

    LOGGER.info(f'Logging to {log_file} initialized.')

def setup_exception_logging():
    """Make sure that uncaught exceptions are logged with the logging."""
    # Log uncaught exceptions
    def exception_logging(typ, value, traceback):
        LOGGER.exception('Uncaught exception',
                         exc_info=(typ, value, traceback))

    sys.excepthook = exception_logging

def save_dict_as_json(path: Union[str, Path], filename: str,
                      dictionary: dict) -> None:
    """Save a dictionary as .json file."""
    if isinstance(path, str):
        path = Path(path)
    save_path = path / (filename + ".json")
    with open(save_path, "w") as f:
        json.dump(dictionary, f)


def save_dict_as_yml(path: Union[str, Path], filename: str,
                     dictionary: dict) -> None:
    """Save a dictionary as .yaml file."""
    if isinstance(path, str):
        path = Path(path)
    save_path = path / (filename + ".yaml")
    OmegaConf.save(OmegaConf.create(dictionary), save_path)


def get_config(config: Union[str, Path, dict, DictConfig]) -> DictConfig:
    """Creates a config from different sources."""
    if isinstance(config, (dict, DictConfig)):
        return OmegaConf.create(config)
    else:
        return OmegaConf.load(config)

def convert_dict_to_python_types(d: dict) -> dict:
    """Tries to convert a dictionary and all entries to plane python types."""
    for k,v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.item()
    return d


def access_by_name(obj, attr_path):
    """
    ONLY USE IN EXCEPTIONAL CASES!
    Allows to access attributes or keys of objects by name
    (e.g. X.y.z.[a] can be accsed with (X, "y.z.a"))
    """
    for attr in attr_path.split("."):
        try:
            obj = obj[attr]
        except (TypeError, KeyError):
            obj = getattr(obj, attr)
    return obj