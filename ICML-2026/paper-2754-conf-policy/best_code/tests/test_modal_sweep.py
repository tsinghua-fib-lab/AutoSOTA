"""Tests for modal_runner sweep job builder."""

import importlib.util
from pathlib import Path

import pytest
import yaml

# Import _build_sweep_jobs from modal_runner.py (a script, not a package).
_spec = importlib.util.spec_from_file_location(
    "modal_runner",
    Path(__file__).parent.parent / "modal_runner.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_build_sweep_jobs = _module._build_sweep_jobs


def _write_sweep_config(tmp_path: Path, config: dict) -> str:
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


class TestBuildSweepJobs:
    def test_cartesian_product(self, tmp_path):
        config = {
            "base_config": "smoke",
            "cache": True,
            "parameters": {
                "initial_seed": [0, 1, 2],
                "conformal_policy_control.alpha": [0.1, 0.4],
            },
        }
        path = _write_sweep_config(tmp_path, config)
        base, cache, fixed, jobs = _build_sweep_jobs(path)

        assert base == "smoke"
        assert cache is True
        assert fixed == []
        assert len(jobs) == 6  # 3 seeds x 2 alphas

    def test_seed_workaround(self, tmp_path):
        """When initial_seed is swept, last_seed should be auto-set to match."""
        config = {
            "base_config": "cpc_llm",
            "parameters": {"initial_seed": [3, 7]},
        }
        path = _write_sweep_config(tmp_path, config)
        _, _, _, jobs = _build_sweep_jobs(path)

        assert len(jobs) == 2
        assert "initial_seed=3" in jobs[0]
        assert "last_seed=3" in jobs[0]
        assert "initial_seed=7" in jobs[1]
        assert "last_seed=7" in jobs[1]

    def test_fixed_overrides(self, tmp_path):
        config = {
            "base_config": "smoke",
            "overrides": ["num_dpo_rounds=3"],
            "parameters": {"initial_seed": [0]},
        }
        path = _write_sweep_config(tmp_path, config)
        _, _, fixed, jobs = _build_sweep_jobs(path)

        assert fixed == ["num_dpo_rounds=3"]
        assert "num_dpo_rounds=3" in jobs[0]
        assert "initial_seed=0" in jobs[0]

    def test_no_parameters(self, tmp_path):
        config = {
            "base_config": "smoke",
            "overrides": ["num_dpo_rounds=1"],
        }
        path = _write_sweep_config(tmp_path, config)
        _, _, _, jobs = _build_sweep_jobs(path)

        assert len(jobs) == 1
        assert jobs[0] == ["num_dpo_rounds=1"]

    def test_missing_base_config(self, tmp_path):
        config = {"parameters": {"initial_seed": [0]}}
        path = _write_sweep_config(tmp_path, config)

        with pytest.raises(ValueError, match="base_config"):
            _build_sweep_jobs(path)

    def test_non_list_parameter(self, tmp_path):
        config = {
            "base_config": "smoke",
            "parameters": {"initial_seed": 0},
        }
        path = _write_sweep_config(tmp_path, config)

        with pytest.raises(ValueError, match="must be a list"):
            _build_sweep_jobs(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _build_sweep_jobs("/nonexistent/sweep.yaml")

    def test_no_seed_workaround_without_initial_seed(self, tmp_path):
        """last_seed should NOT be auto-set when initial_seed is not swept."""
        config = {
            "base_config": "smoke",
            "parameters": {"conformal_policy_control.alpha": [0.1, 0.5]},
        }
        path = _write_sweep_config(tmp_path, config)
        _, _, _, jobs = _build_sweep_jobs(path)

        assert len(jobs) == 2
        for job in jobs:
            assert not any("last_seed" in o for o in job)
