import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_ROOT = os.getenv("MODEL_ROOT")
SAE_ROOT = os.getenv("SAE_ROOT")
DATASET_ROOT = os.getenv("DATASET_ROOT")