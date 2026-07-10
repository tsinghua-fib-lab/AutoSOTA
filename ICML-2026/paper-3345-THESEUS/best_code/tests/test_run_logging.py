from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from merge_and_rebase.cli_args import build_logging_overrides
from merge_and_rebase.run_logging import merge_logging_config, render_config_args, start_run, summary_events_path


def test_build_logging_overrides_parses_cli_tags():
    args = SimpleNamespace(
        use_wandb=True,
        wandb_project="proj",
        wandb_entity="team",
        wandb_tags="alpha,beta",
        wandb_mode="offline",
        local_log_dir="/tmp/logs",
        run_name="demo",
        log_every_n_steps=12,
    )

    out = build_logging_overrides(args)

    assert out == {
        "use_wandb": True,
        "project": "proj",
        "entity": "team",
        "tags": ["alpha", "beta"],
        "mode": "offline",
        "local_log_dir": "/tmp/logs",
        "run_name": "demo",
        "log_every_n_steps": 12,
    }


def test_local_logger_writes_summary_and_events(tmp_path: Path):
    summary_path = tmp_path / "run.json"
    logger = start_run(
        entrypoint="unit.test",
        logging_cfg=merge_logging_config({"local_log_dir": str(tmp_path)}),
        metadata={"hello": "world"},
        summary_path=summary_path,
    )

    logger.log_event("metric", {"foo": 1.0}, step=3, context={"phase": "train"})
    logger.log_summary({"result": 7})
    logger.finish("success")

    payload = json.loads(summary_path.read_text())
    events_path = summary_events_path(summary_path)
    event_lines = [json.loads(line) for line in events_path.read_text().splitlines()]

    assert payload["result"] == 7
    assert payload["run_logging"]["entrypoint"] == "unit.test"
    assert payload["run_logging"]["status"] == "success"
    assert any(item["event_type"] == "metric" for item in event_lines)
    assert any(item["event_type"] == "run_finished" for item in event_lines)


def test_local_logger_skips_wandb_only_events(tmp_path: Path):
    summary_path = tmp_path / "run.json"
    logger = start_run(
        entrypoint="unit.test",
        logging_cfg=merge_logging_config({"local_log_dir": str(tmp_path)}),
        metadata={"hello": "world"},
        summary_path=summary_path,
    )

    logger.log_event("teacher_metric", {"foo": 1.0}, step=3, context={"_wandb_only": True, "phase": "train"})
    logger.finish("success")

    events_path = summary_events_path(summary_path)
    event_lines = [json.loads(line) for line in events_path.read_text().splitlines()]

    assert not any(item["event_type"] == "teacher_metric" for item in event_lines)


def test_start_run_with_wandb_keeps_local_logging(tmp_path: Path, monkeypatch):
    finished: list[int] = []

    class _FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}

        def finish(self, exit_code: int = 0) -> None:
            finished.append(exit_code)

    fake_run = _FakeRun()

    class _FakeWandb:
        def __init__(self) -> None:
            self.logged: list[dict[str, object]] = []

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return fake_run

        def log(self, payload, step=None):
            self.logged.append({"payload": payload, "step": step})

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())

    summary_path = tmp_path / "wandb.json"
    logger = start_run(
        entrypoint="unit.wandb",
        logging_cfg=merge_logging_config({"use_wandb": True, "local_log_dir": str(tmp_path)}),
        metadata={"x": 1},
        summary_path=summary_path,
    )
    logger.log_event("metric", {"foo": 2.0}, step=5)
    logger.log_summary({"done": True})
    logger.finish("success")

    assert summary_path.exists()
    assert json.loads(summary_path.read_text())["done"] is True
    assert finished == [0]


def test_wandb_logger_strips_internal_wandb_only_context(tmp_path: Path, monkeypatch):
    finished: list[int] = []

    class _FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}

        def finish(self, exit_code: int = 0) -> None:
            finished.append(exit_code)

    fake_run = _FakeRun()

    class _FakeWandb:
        def __init__(self) -> None:
            self.logged: list[dict[str, object]] = []

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return fake_run

        def log(self, payload, step=None):
            self.logged.append({"payload": payload, "step": step})

    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    summary_path = tmp_path / "wandb.json"
    logger = start_run(
        entrypoint="unit.wandb",
        logging_cfg=merge_logging_config({"use_wandb": True, "local_log_dir": str(tmp_path)}),
        metadata={"x": 1},
        summary_path=summary_path,
    )
    logger.log_event("teacher_metric", {"foo": 2.0}, step=5, context={"_wandb_only": True, "phase": "train"})
    logger.finish("success")

    payload = fake_wandb.logged[-1]["payload"]
    assert payload["event/type"] == "teacher_metric"
    assert payload["context/phase"] == "train"
    assert "context/_wandb_only" not in payload
    assert finished == [0]


def test_render_config_args_flattens_nested_config():
    rendered = render_config_args(
        {
            "config": "demo.json",
            "logging": {"mode": "offline"},
            "tasks": ["cars", "dtd"],
        },
        title="Run config (unit.test)",
    )

    assert rendered.startswith("Run config (unit.test): ")
    assert "config=demo.json" in rendered
    assert "logging.mode=offline" in rendered
    assert "tasks=cars,dtd" in rendered


def test_start_run_prints_compact_resolved_config(tmp_path: Path, capsys):
    summary_path = tmp_path / "run.json"
    logger = start_run(
        entrypoint="unit.test",
        logging_cfg=merge_logging_config({"local_log_dir": str(tmp_path)}),
        metadata={
            "config_path": "demo.json",
            "summary_path": str(summary_path),
            "resolved_config": {"seed": 7, "tasks": ["a", "b"]},
        },
        summary_path=summary_path,
    )
    logger.finish("success")

    captured = capsys.readouterr()
    assert "Run config (unit.test): " in captured.out
    assert "seed=7" in captured.out
    assert "tasks=a,b" in captured.out
    assert "config=demo.json" in captured.out
    assert "resolved_config." not in captured.out
