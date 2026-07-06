from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from data.llm import LlmWeightsDatasetConfig


def camera_ready_datasets():
    """Return the GPT2-medium stacked weights dataset for camera-ready plots."""
    return [
        LlmWeightsDatasetConfig(
            name="gpt2_medium_weights_concat_16k_1k",
            model_name=g.LLM_MODEL_GPT2_MEDIUM,
            param_name=g.LLM_PARAM_CONCAT_LAYERS,
            submatrix_rows=16384,
            submatrix_cols=1024,
            seed=12,
        ),
    ]


def camera_ready_datasets_ridge():
    """Return the Qwen2-1.5B stacked weights dataset for ridge regression."""
    return [
        LlmWeightsDatasetConfig(
            name="qwen2_1p5b_weights_full_64k_512",
            model_name=g.LLM_MODEL_QWEN2_1P5B,
            param_name=g.LLM_PARAM_CONCAT_LAYERS_FULL,
            submatrix_rows=65536,
            submatrix_cols=512,
            seed=12,
        ),
    ]
