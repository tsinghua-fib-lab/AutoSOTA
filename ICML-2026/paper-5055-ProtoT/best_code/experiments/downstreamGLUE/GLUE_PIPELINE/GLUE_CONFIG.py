

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List

def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = _path_from_env("DOWNSTREAM_GLUE_ROOT", Path(__file__).resolve().parents[1])
TOKENIZER_PATH = _path_from_env("GLUE_TOKENIZER_PATH", REPO_ROOT / "tok" / "fineweb_bpe_16000.json")
DATA_CACHE = _path_from_env("GLUE_DATA_CACHE", BASE_DIR / "data")
CHECKPOINT_ROOT = _path_from_env("GLUE_CHECKPOINT_ROOT", BASE_DIR / "checkpoints")
OUTPUT_ROOT = _path_from_env("GLUE_OUTPUT_ROOT", BASE_DIR / "outputs")
SUBMISSION_ROOT = _path_from_env("GLUE_SUBMISSION_ROOT", BASE_DIR / "submissions")

MAX_SEQ_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_DELTA = 1e-3

@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_type: str
    checkpoint: Path
    lr: float
    warmup_ratio: float

MODEL_SPECS: Dict[str, ModelSpec] = {
    "protoattn": ModelSpec(
        key="protoattn",
        model_type="PrototypeAttn",
        checkpoint=CHECKPOINT_ROOT / "ProtoAttn_large_FineWeb_scheduler",
        lr=3.5e-5,
        warmup_ratio=0.06,
    ),
    "llama": ModelSpec(
        key="llama",
        model_type="llama",
        checkpoint=CHECKPOINT_ROOT / "LLaMA_large_FineWeb_scheduler",
        lr=5.5e-5,
        warmup_ratio=0.10,
    ),
    "mamba": ModelSpec(
        key="mamba",
        model_type="mamba1",
        checkpoint=CHECKPOINT_ROOT / "mamba1_large_fineweb",
        lr=1.0e-4,
        warmup_ratio=0.10,
    ),
    "deltanet": ModelSpec(
        key="deltanet",
        model_type="deltanet",
        checkpoint=CHECKPOINT_ROOT / "deltanet_large_fineweb",
        lr=7.0e-4,
        warmup_ratio=0.10,
    ),
}

DEFAULT_GLUE_TASKS: List[str] = [
    "cola",
    "sst2",
    "mrpc",
    "qqp",
    "stsb",
    "mnli",
    "qnli",
    "rte",
    "wnli",
]

TASK_TO_TEST_SPLIT: Dict[str, str] = {
    "cola": "test",
    "sst2": "test",
    "mrpc": "test",
    "qqp": "test",
    "stsb": "test",
    "mnli": "test_matched",
    "mnli-mm": "test_mismatched",
    "qnli": "test",
    "rte": "test",
    "wnli": "test",
    "ax": "test",
}

TASK_TO_SUBMISSION_FILE: Dict[str, str] = {
    "cola": "CoLA.tsv",
    "sst2": "SST-2.tsv",
    "mrpc": "MRPC.tsv",
    "qqp": "QQP.tsv",
    "stsb": "STS-B.tsv",
    "mnli": "MNLI-m.tsv",
    "mnli-mm": "MNLI-mm.tsv",
    "qnli": "QNLI.tsv",
    "rte": "RTE.tsv",
    "wnli": "WNLI.tsv",
    "ax": "AX.tsv",
}

LABEL_MAPPINGS: Dict[str, List[str]] = {
    "cola": ["0", "1"],
    "sst2": ["0", "1"],
    "mrpc": ["0", "1"],
    "qqp": ["0", "1"],
    "stsb": [],
    "mnli": ["entailment", "neutral", "contradiction"],
    "mnli-mm": ["entailment", "neutral", "contradiction"],
    "qnli": ["entailment", "not_entailment"],
    "rte": ["entailment", "not_entailment"],
    "wnli": ["0", "1"],
    "ax": ["entailment", "contradiction", "neutral"],
}

DEFAULT_GPU_MAP: Dict[str, str] = {
    "protoattn": "0",
    "llama": "1",
    "mamba": "2",
    "deltanet": "3",
}
