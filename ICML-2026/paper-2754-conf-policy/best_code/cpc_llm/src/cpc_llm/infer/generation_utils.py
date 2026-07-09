from typing import List, Optional
from omegaconf import DictConfig
from ..infrastructure.file_handler import LocalOrS3Client
import json
import logging

logger = logging.getLogger(__name__)


def get_temperatures(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir: str,
    prev_hd: Optional[float],
) -> List[float]:
    # check if previous temperatures file already exists
    temps_fp = f"{model_dir}/temperatures.json"
    if fs.exists(temps_fp):
        logger.info(f"Loading generation temperatures from {temps_fp}.")
        temps = json.load(fs.open(temps_fp))
    else:
        # if not already existing, dynamically compute and write to file
        temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        # temps = [1.0]
        if cfg.temperature_scaling and prev_hd is not None:
            if prev_hd < 0.075:
                temps = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
            elif prev_hd < 0.1:
                temps = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            elif prev_hd < 0.15:
                temps = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
        with fs.open(temps_fp, "w") as f:
            f.write(json.dumps(temps))
    return temps
