from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import os
from pathlib import Path

# Provenance / filesystem
TIME_FORMAT = "%Y%m%d_%H%M%S"
FILE_STORAGE_ROOT = "file_storage"

# Sketch method identifiers
METHOD_GAUSSIAN_DENSE_CUBLAS = "gaussian_dense_cublas"
METHOD_SJLT_CUSPARSE = "sjlt_cusparse"
METHOD_FLASH_BLOCK_ROW = "flashblockrow"
METHOD_FLASH_SKETCH = "flashsketch"
METHOD_SRHT_FWHT = "srht_fwht"
METHOD_SJLT_GRASS_KERNEL = "sjlt_grass_kernel"
METHOD_GRASS = "GraSS-grass"

# Dataset identifiers
DATASET_SYNTHETIC = "synthetic"
DATASET_SUITESPARSE = "suitesparse"
DATASET_MNIST = "mnist"
DATASET_CIFAR2 = "cifar2"
DATASET_LLM = "llm"

# LLM dataset identifiers
LLM_SOURCE_WEIGHTS = "weights"
LLM_SOURCE_GRADIENTS = "gradients"
LLM_SOURCE_KV_CACHE = "kv_cache"
LLM_MODEL_GPT2 = "gpt2"
LLM_MODEL_DISTILGPT2 = "distilgpt2"
LLM_MODEL_GPT2_MEDIUM = "gpt2-medium"
LLM_MODEL_QWEN2_1P5B = "Qwen/Qwen2-1.5B"
LLM_PARAM_CONCAT_LAYERS = "concat_layers"
LLM_PARAM_CONCAT_LAYERS_FULL = "concat_layers_full"
LLM_PARAM_GPT2_MLP_CFC = "transformer.h.0.mlp.c_fc.weight"
LLM_PARAM_GPT2_MLP_CFC_TEMPLATE = "transformer.h.{layer}.mlp.c_fc.weight"
LLM_PARAM_GPT2_MLP_CPROJ_TEMPLATE = "transformer.h.{layer}.mlp.c_proj.weight"
LLM_PARAM_GPT2_ATTN_CATTN_TEMPLATE = "transformer.h.{layer}.attn.c_attn.weight"
LLM_PARAM_GPT2_ATTN_CPROJ_TEMPLATE = "transformer.h.{layer}.attn.c_proj.weight"
LLM_PARAM_QWEN2_ATTN_Q_TEMPLATE = "model.layers.{layer}.self_attn.q_proj.weight"
LLM_PARAM_QWEN2_ATTN_K_TEMPLATE = "model.layers.{layer}.self_attn.k_proj.weight"
LLM_PARAM_QWEN2_ATTN_V_TEMPLATE = "model.layers.{layer}.self_attn.v_proj.weight"
LLM_PARAM_QWEN2_ATTN_O_TEMPLATE = "model.layers.{layer}.self_attn.o_proj.weight"
LLM_PARAM_QWEN2_MLP_GATE_TEMPLATE = "model.layers.{layer}.mlp.gate_proj.weight"
LLM_PARAM_QWEN2_MLP_UP_TEMPLATE = "model.layers.{layer}.mlp.up_proj.weight"
LLM_PARAM_QWEN2_MLP_DOWN_TEMPLATE = "model.layers.{layer}.mlp.down_proj.weight"
LLM_DEFAULT_SEQ_LEN = 64
LLM_DEFAULT_BATCH_SIZE = 4

# SuiteSparse densification defaults
SUITESPARSE_MAX_DENSE_FRACTION = 0.5

# Synthetic distribution identifiers
DIST_GAUSSIAN = "gaussian"
DIST_RADEMACHER = "rademacher"
DIST_LOW_RANK = "low_rank"

# Dtype identifiers
DTYPE_FP32 = "fp32"

# Timing toggles
ENABLE_CUDA_TIMERS = True

# FWHT backend identifiers
FWHT_BACKEND_FAST = "fast"
FWHT_BACKEND_TORCH = "torch"

# Task identifiers
TASK_SKETCH_SOLVE_LS = "sketch_and_solve_ls"
TASK_RIDGE_REGRESSION = "ridge_regression"
TASK_GRAM_APPROX = "gram_approx"
TASK_OSE_ERROR = "ose_error"
TASK_GRASS_LDS = "grass_lds"

# GraSS pipeline identifiers
MODEL_MLP = "mlp"
MODEL_RESNET9 = "resnet9"
PIPELINE_GRASS = "grass"

# Environment overrides (use sparingly)
ENV_OVERRIDE_PREFIX = "SKETCH_"
ENV_OVERRIDE_K = "SKETCH_K"
ENV_OVERRIDE_S = "SKETCH_S"
ENV_OVERRIDE_DTYPE = "SKETCH_DTYPE"
ENV_OVERRIDE_SEND_SLACK = "SKETCH_SEND_SLACK"
ENV_OVERRIDE_FIGURES_DIR = "SKETCH_FIGURES_DIR"
ENV_CAMERA_READY_PDF_TO_PAPER = "CAMERA_READY_PDF_TO_PAPER"

# Common paths
REPO_ROOT = Path(__file__).resolve().parent
FILE_STORAGE_PATH = REPO_ROOT / FILE_STORAGE_ROOT
EXTERNAL_GRASS_DIR = REPO_ROOT / "external" / "GraSS"

# External GraSS integration env vars
ENV_FLASH_SKETCH_ROOT = "FLASH_SKETCH_ROOT"
ENV_GRASS_DEVICE = "GRASS_DEVICE"
ENV_GRASS_PROJ_TYPE = "GRASS_PROJ_TYPE"
ENV_GRASS_PROJ_DIM = "GRASS_PROJ_DIM"
ENV_GRASS_SEED = "GRASS_SEED"
ENV_GRASS_VAL_RATIO = "GRASS_VAL_RATIO"
ENV_GRASS_BATCH_SIZE = "GRASS_BATCH_SIZE"
ENV_GRASS_PROJ_MAX_BATCH_SIZE = "GRASS_PROJ_MAX_BATCH_SIZE"
ENV_GRASS_SJLT_C = "GRASS_SJLT_C"
ENV_GRASS_FLASH_KAPPA = "GRASS_FLASH_KAPPA"
ENV_GRASS_FLASH_S = "GRASS_FLASH_S"
ENV_GRASS_FLASH_BLOCK_ROWS = "GRASS_FLASH_BLOCK_ROWS"
ENV_GRASS_FLASH_SEED = "GRASS_FLASH_SEED"
ENV_GRASS_FLASH_SKIP_ZEROS = "GRASS_FLASH_SKIP_ZEROS"
ENV_GRASS_MLP_ACTIVATION = "GRASS_MLP_ACTIVATION"
ENV_GRASS_MLP_DROPOUT_RATE = "GRASS_MLP_DROPOUT_RATE"
ENV_TQDM_DISABLE = "TQDM_DISABLE"


def RUN_DIR(run_id):
    """Return the run directory under file_storage for a given run id."""
    return FILE_STORAGE_PATH / "runs" / run_id


def DATASET_CACHE_DIR():
    """Return the dataset cache directory under file_storage."""
    return FILE_STORAGE_PATH / "datasets"


def SUITESPARSE_CACHE_DIR():
    """Return the SuiteSparse cache directory under file_storage."""
    return DATASET_CACHE_DIR() / "suitesparse"


def LLM_CACHE_DIR():
    """Return the LLM dataset cache directory under file_storage."""
    return DATASET_CACHE_DIR() / "llm"


def FIGURES_DIR():
    """Return the figures output directory (default: file_storage/figures)."""
    override = GET_ENV_VAR(ENV_OVERRIDE_FIGURES_DIR)
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path
    return FILE_STORAGE_PATH / "figures"


def FIGURE_MANIFEST_PATH():
    """Return the path to the paper figures manifest file."""
    return FIGURES_DIR() / "manifest.json"


def PAPER_FIGURES_DIR():
    """Return the paper figures directory."""
    return REPO_ROOT / "paper" / "figures"


def PAPER_TABLES_DIR():
    """Return the paper tables directory."""
    return REPO_ROOT / "paper" / "tables"


def FIGURE_SRC_DIR():
    """Return the directory containing figure generation scripts."""
    return REPO_ROOT / "analysis" / "figures_src"


def ENV_FLAG_TRUE(value):
    """Return True if an environment variable string should be treated as truthy."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def GET_ENV_VAR(name, default=None):
    """Get an environment variable with an optional default."""
    return os.environ.get(name, default)
