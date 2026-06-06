"""Test whether the default_configs.yaml matches with the pydantic classes in the code."""

from pathlib import Path

from appl.core.config import DEFAULT_CONFIGS
from appl.core.io import load_file


def test_config_match():
    file = Path(__file__).parent.parent / "src" / "appl" / "default_configs.yaml"
    configs = load_file(file)

    assert configs == DEFAULT_CONFIGS.model_dump()
