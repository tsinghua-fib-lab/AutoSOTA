"""Shared test configuration and fixtures."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
    cwd=False,
)

import pytest
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture()
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture()
def sample_inputs(tokenizer):
    """Tokenized sample input on CUDA."""
    text = "The quick brown fox jumps over"
    return tokenizer(text, return_tensors="pt").to("cuda")
