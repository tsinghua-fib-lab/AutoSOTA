import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "moretro"
CONFIG_DIR = ROOT_DIR / "configs"
MODELS_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

if not MODELS_DIR.exists() and "pytest" not in sys.modules:
    raise FileNotFoundError(
        f"Model directory {MODELS_DIR} does not exist. "
        "Please first download required models from FigShare."
    )
