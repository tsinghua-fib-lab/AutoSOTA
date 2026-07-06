import json
import logging
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path

import gin
import pandas as pd

from moretro.inference.calculate_costs import cost_loader
from moretro.inference.heuristic_functions import COST_MAPPING, heuristic_loader
from moretro.utils.base_paths import MODELS_DIR

logger = logging.getLogger(__name__)


@gin.configurable()
def prepare_starting_mols(file_path: str | Path) -> set[str]:
    """
    Load building blocks from file

    Parameters:
        file_path (str): Path to the file containing building blocks

    Returns:
        set[str]: Set of building block SMILES strings
    """
    file_path = MODELS_DIR / Path(file_path)
    if file_path.suffix == ".csv":
        starting_mol = set(pd.read_csv(file_path)["smiles"].tolist())
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            starting_mol = pickle.load(f)
    elif file_path.suffix == ".json":
        with open(file_path, encoding="utf-8") as f:
            starting_mol = set(json.load(f))
    else:
        raise ValueError("Unsupported file format. Use .csv or .pkl or .json")

    logger.info(f"Loaded {len(starting_mol)} building blocks from {file_path}")
    return starting_mol


@gin.configurable()
def prepare_heuristic_fns(heuristics: list[str]) -> Sequence[Callable[[str], float]]:
    for heuristic in heuristics:
        heuristic_loader(heuristic)
    heuristic_fs = []
    for heuristic in heuristics:
        if heuristic in COST_MAPPING:
            heuristic_fs.append(COST_MAPPING[heuristic])
        else:
            logger.error(f"Unknown heuristic: {heuristic}")
            raise ValueError(
                "Please ensure that heuristic is defined for all cost functions"
            )
    return heuristic_fs


@gin.configurable()
def prepare_cost_models(cost_functions: list[str], device: str) -> None:
    for cost in cost_functions:
        cost_loader(cost, device)
