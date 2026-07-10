from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_vision_experiments.py"
    spec = importlib.util.spec_from_file_location("run_vision_experiments", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_merge_case_can_pass_save_merged_checkpoint_path(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner_module()
    commands: list[list[str]] = []

    monkeypatch.setattr(runner, "run_cmd", lambda cmd, *, dry_run: commands.append(list(cmd)))

    args = SimpleNamespace(
        python_bin="python",
        model="vitb",
        device="cpu",
        cache_path=tmp_path / "cache.json",
        merge_config=None,
        run_name="summary",
        alpha_search=False,
        alpha=0.3,
        alpha_min=0.0,
        alpha_max=2.0,
        alpha_step=0.1,
        tasks=None,
        dtype=None,
        batch_size=None,
        num_workers=None,
        seed=None,
        wandb_mode=None,
        save_merged_checkpoints=True,
        save_merged_name="checkpoints/merged.pt",
        merge_extra=None,
        dry_run=True,
        force=True,
    )

    runner.run_merge_case(args, suite="vision8", method="task_arithmetic", results_root=tmp_path / "results")

    assert len(commands) == 1
    cmd = commands[0]
    save_idx = cmd.index("--save-merged")
    assert cmd[save_idx + 1] == str(tmp_path / "results" / "merge" / "vision8" / "task_arithmetic" / "checkpoints" / "merged.pt")


def test_run_merge_case_can_use_custom_merge_config(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner_module()
    commands: list[list[str]] = []
    custom_cfg = tmp_path / "custom_merge.json"
    custom_cfg.write_text('{"suite":"vision8","tuned_ckpts":{}}\n')

    monkeypatch.setattr(runner, "run_cmd", lambda cmd, *, dry_run: commands.append(list(cmd)))

    args = SimpleNamespace(
        python_bin="python",
        model="vitb",
        device="cpu",
        cache_path=tmp_path / "cache.json",
        merge_config=custom_cfg,
        run_name="summary",
        alpha_search=False,
        alpha=0.3,
        alpha_min=0.0,
        alpha_max=2.0,
        alpha_step=0.1,
        tasks=None,
        dtype=None,
        batch_size=None,
        num_workers=None,
        seed=None,
        wandb_mode=None,
        save_merged_checkpoints=False,
        save_merged_name="merged.pt",
        merge_extra=None,
        dry_run=True,
        force=True,
    )

    runner.run_merge_case(args, suite="vision8", method="task_arithmetic", results_root=tmp_path / "results")

    assert len(commands) == 1
    cmd = commands[0]
    config_idx = cmd.index("--config")
    assert cmd[config_idx + 1] == str(custom_cfg)
