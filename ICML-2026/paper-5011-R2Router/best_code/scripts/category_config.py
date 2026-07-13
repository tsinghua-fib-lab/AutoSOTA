"""
Shared constants for category-aware R2-Router.

7 categories based on model performance patterns:
  0: Code        — LiveCodeBench
  1: Math        — AIME, MATH, GSM8K, MathQA, AsDiv, MMLUPro_math
  2: Knowledge   — MMLUPro (non-math), ArcMMLU, MMLU_*
  3: NLU         — SuperGLUE-*, Ethics_*, NarrativeQA, SocialiQA
  4: Translation — WMT19-*
  5: Trivia      — OpenTDB_*, QANTA_*
  6: Domain      — PubMedQA, MedMCQA, GeoBench, GeoGraphyData, FinQA,
                   MusicTheoryBench, ChessInstruct*
"""

import os

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(name: str, default: str) -> str:
    """Return an environment-overridable path while preserving local defaults."""
    return os.path.expanduser(os.getenv(name, default))


ROUTERARENA_ROOT = _env_path("R2_ROUTERARENA_ROOT", "/home/ji757406.ucf/RouterArena")
SWEEP_ROOT = _env_path("R2_SWEEP_ROOT", "/orange/qi855292.ucf/ji757406.ucf/budget_sweep")
EMBEDDINGS_PATH = _env_path(
    "R2_EMBEDDINGS_PATH",
    os.path.join(REPO_ROOT, "routerarena_submission", "routerarena_embeddings.pkl"),
)
ROBUSTNESS_EMBEDDINGS_PATH = _env_path(
    "R2_ROBUSTNESS_EMBEDDINGS_PATH",
    os.path.join(REPO_ROOT, "routerarena_submission", "routerarena_robustness_embeddings.pkl"),
)
ROUTER_DATA_PATH = _env_path(
    "R2_ROUTER_DATA_PATH",
    os.path.join(ROUTERARENA_ROOT, "dataset", "router_data.json"),
)
ROUTER_DATA_10_PATH = _env_path(
    "R2_ROUTER_DATA_10_PATH",
    os.path.join(ROUTERARENA_ROOT, "dataset", "router_data_10.json"),
)
ROBUSTNESS_DATA_PATH = _env_path(
    "R2_ROBUSTNESS_DATA_PATH",
    os.path.join(ROUTERARENA_ROOT, "dataset", "router_robustness.json"),
)
MODEL_COST_PATH = _env_path(
    "R2_MODEL_COST_PATH",
    os.path.join(ROUTERARENA_ROOT, "model_cost", "model_cost.json"),
)
TRAINING_DATA_PATH = _env_path(
    "R2_TRAINING_DATA_PATH",
    "/orange/qi855292.ucf/ji757406.ucf/category_router/training_data.pkl",
)
CHECKPOINT_DIR = _env_path(
    "R2_CHECKPOINT_DIR",
    os.path.join(REPO_ROOT, "checkpoints", "category_router"),
)

# ============================================================================
# Categories
# ============================================================================

CATEGORY_NAMES = ["Code", "Math", "Knowledge", "NLU", "Translation", "Trivia", "Domain"]
NUM_CATEGORIES = len(CATEGORY_NAMES)

# Map dataset prefix → category index
# Prefix is extracted from global_index before the last underscore+number
CATEGORY_MAP = {
    # 0: Code
    "LiveCodeBench": 0,

    # 1: Math
    "AIME": 1,
    "MATH": 1,
    "GSM8K": 1,
    "MathQA": 1,
    "AsDiv": 1,

    # 2: Knowledge (MMLUPro non-math handled specially, ArcMMLU, MMLU_*)
    "ArcMMLU": 2,

    # 3: NLU
    "NarrativeQA": 3,
    "SocialiQA": 3,

    # 4: Translation
    # WMT19-* handled by prefix matching

    # 5: Trivia
    # OpenTDB_* and QANTA_* handled by prefix matching

    # 6: Domain Expert
    "PubMedQA": 6,
    "MedMCQA": 6,
    "GeoBench": 6,
    "GeoGraphyData_100k": 6,
    "FinQA": 6,
    "MusicTheoryBench": 6,
    # ChessInstruct* handled by prefix matching
}


def get_dataset_prefix(global_index: str) -> str:
    """Extract dataset prefix from global_index.

    Examples:
        "ArcMMLU_655" -> "ArcMMLU"
        "MMLU_formal_logic_7" -> "MMLU_formal_logic"
        "OpenTDB_General Knowledge_1030" -> "OpenTDB_General Knowledge"
        "WMT19-cs-en_50" -> "WMT19-cs-en"
        "SuperGLUE-CausalReasoning_100" -> "SuperGLUE-CausalReasoning"
        "Ethics_commonsense_5" -> "Ethics_commonsense"
        "MMLUPro_math_123" -> "MMLUPro_math"
        "ChessInstruct500_5" -> "ChessInstruct500"

    Strategy: split on '_', the last part is always the numeric ID.
    Everything before it is the prefix.
    """
    parts = global_index.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    # Fallback: try splitting more aggressively
    # Some indices might have non-numeric suffixes
    return global_index


def get_category(global_index: str) -> int:
    """Get category index (0-6) for a given global_index."""
    prefix = get_dataset_prefix(global_index)

    # Direct match
    if prefix in CATEGORY_MAP:
        return CATEGORY_MAP[prefix]

    # Prefix-based matching
    if prefix.startswith("WMT19"):
        return 4  # Translation
    if prefix.startswith("OpenTDB"):
        return 5  # Trivia
    if prefix.startswith("QANTA"):
        return 5  # Trivia
    if prefix.startswith("SuperGLUE"):
        return 3  # NLU
    if prefix.startswith("Ethics"):
        return 3  # NLU
    if prefix.startswith("ChessInstruct"):
        return 6  # Domain
    if prefix.startswith("MMLU"):
        # MMLU_* and MMLUPro (non-math) → Knowledge
        # MMLUPro_math → Math
        if "MMLUPro" in prefix and "math" in prefix.lower():
            return 1  # Math
        return 2  # Knowledge

    raise ValueError(f"Unknown dataset prefix: '{prefix}' from global_index: '{global_index}'")


# ============================================================================
# Models
# ============================================================================

# Model configs: short_name -> {sweep_dir, cost_key, type}
# cost_key matches keys in model_cost.json
MODELS = {
    "235b": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "235b"),
        "cost_key": "qwen_qwen3-235b-a22b-2507",
        "ra_name": "qwen/qwen3-235b-a22b-2507",
        "type": "vllm",
    },
    "80b": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "80b"),
        "cost_key": "qwen/qwen3-next-80b-a3b-instruct",
        "ra_name": "qwen/qwen3-next-80b-a3b-instruct",
        "type": "vllm",
    },
    "30b": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "30b"),
        "cost_key": "qwen/qwen3-30b-a3b-instruct-2507",
        "ra_name": "qwen/qwen3-30b-a3b-instruct-2507",
        "type": "vllm",
    },
    "coder-next": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "Qwen3-Coder-Next"),
        "cost_key": "Qwen/Qwen3-Coder-Next",
        "ra_name": "Qwen/Qwen3-Coder-Next",
        "type": "vllm",
    },
    "coder-30b": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "coder-30b"),
        "cost_key": "qwen/qwen3-coder-30b-a3b-instruct",
        "ra_name": "qwen/qwen3-coder-30b-a3b-instruct",
        "type": "vllm",
    },
    "ministral-3b": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "ministral-3b"),
        "cost_key": "mistralai/ministral-3-3b-2512",
        "ra_name": "mistralai/ministral-3-3b-2512",
        "type": "vllm",
    },
    "ministral-8b": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "ministral-8b"),
        "cost_key": "mistralai/ministral-3-8b-2512",
        "ra_name": "mistralai/ministral-3-8b-2512",
        "type": "vllm",
    },
    "ministral-14b": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "ministral-14b"),
        "cost_key": "mistralai/ministral-3-14b-2512",
        "ra_name": "mistralai/ministral-3-14b-2512",
        "type": "vllm",
    },
    "gpt4o": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "gpt4o"),
        "cost_key": "gpt-4o",
        "ra_name": "gpt-4o",
        "type": "api",
    },
    "gemini-flash": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "gemini-flash"),
        "cost_key": "gemini-2.5-flash",
        "ra_name": "gemini-2.5-flash",
        "type": "api",
    },
    "haiku": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "haiku"),
        "cost_key": "claude-3-haiku-20240307",
        "ra_name": "claude-3-haiku-20240307",
        "type": "api",
    },
    "gemma-3n-e4b": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "gemma-3n-e4b"),
        "cost_key": "google/gemma-3n-e4b-it",
        "ra_name": "google/gemma-3n-e4b-it",
        "type": "vllm",
    },
    "kimi-k2.5": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "kimi-k2.5"),
        "cost_key": "moonshotai/kimi-k2.5",
        "ra_name": "moonshotai/kimi-k2.5",
        "type": "vllm",
    },
    "glm-5": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "glm-5"),
        "cost_key": "z-ai/glm-5",
        "ra_name": "z-ai/glm-5",
        "type": "api",
    },
    "gpt-5.2": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "gpt-5.2"),
        "cost_key": "gpt-5.2",
        "ra_name": "gpt-5.2",
        "type": "api",
    },
    "gemini-3-pro": {
        "sweep_dir": os.path.join(SWEEP_ROOT, "gemini-3-pro"),
        "cost_key": "gemini-3-pro-preview",
        "ra_name": "google/gemini-3-pro-preview",
        "type": "api",
    },
}

# Budget file names for vLLM models (9 files)
VLLM_BUDGETS = [
    "budget_10", "budget_20", "budget_40", "budget_80",
    "budget_150", "budget_200", "budget_400", "budget_800",
    "concise",
]

# Budget file names for API models (1 file)
API_BUDGETS = ["concise"]

# Budget values for routing (maps budget name -> token limit for prompt)
BUDGET_VALUES = {
    "budget_10": 10,
    "budget_20": 20,
    "budget_40": 40,
    "budget_80": 80,
    "budget_150": 150,
    "budget_200": 200,
    "budget_400": 400,
    "budget_800": 800,
    "concise": "concise",      # Concise prompt mode
}


def get_budgets(model_name: str) -> list:
    """Get budget list for a model."""
    if MODELS[model_name]["type"] == "api":
        return API_BUDGETS
    return VLLM_BUDGETS
