"""Dry-run training tests for each model config.

These tests execute the training CLI with minimal epochs to verify
the full pipeline works end-to-end.
"""

import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
CONFIG_FILES = list(CONFIGS_DIR.glob("*.toml"))

# Models with working factories
WORKING_MODELS = {"ncde", "nrde"}


def create_dry_run_config(base_config_path: Path, epochs: int = 1) -> str:
    """Create a temporary config with minimal epochs for dry-run testing."""
    with open(base_config_path, "rb") as f:
        config = tomllib.load(f)

    # Override to minimal training
    config["experiment_config"]["epochs"] = epochs
    config["experiment_config"]["batch_size"] = 8
    config["experiment_config"]["early_stopping_patience"] = (
        epochs + 1
    )  # Don't early stop

    # Write as TOML
    lines = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, val in values.items():
            if isinstance(val, str):
                lines.append(f'{key} = "{val}"')
            elif isinstance(val, bool):
                lines.append(f"{key} = {str(val).lower()}")
            elif isinstance(val, float) and val != int(val):
                lines.append(f"{key} = {val}")
            else:
                lines.append(f"{key} = {val}")
        lines.append("")

    return "\n".join(lines)


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_training_dry_run(config_path: Path) -> None:
    """Test that training runs without errors for each config."""
    # Skip models without working factories
    if config_path.stem not in WORKING_MODELS:
        pytest.skip(f"{config_path.stem} factory not implemented")

    # Create temporary config with 5 epochs
    dry_run_toml = create_dry_run_config(config_path, epochs=5)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp:
        tmp.write(dry_run_toml)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["uv", "run", "train", "--config", tmp_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=CONFIGS_DIR.parent,
        )

        # Print output for debugging if it fails
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        assert result.returncode == 0, f"Training failed for {config_path.stem}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
