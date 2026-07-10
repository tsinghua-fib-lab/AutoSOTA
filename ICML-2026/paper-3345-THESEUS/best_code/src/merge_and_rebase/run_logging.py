from __future__ import annotations

import json
import shlex
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io.utils import atomic_write_json

DEFAULT_LOGGING_CONFIG: dict[str, Any] = {
    "use_wandb": False,
    "project": None,
    "entity": None,
    "tags": [],
    "mode": "online",
    "local_log_dir": None,
    "log_every_n_steps": 50,
    "run_name": None,
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return repr(value)


def _stringify_arg_value(value: Any) -> str | None:
    safe_value = _json_safe(value)
    if safe_value is None:
        return None
    if isinstance(safe_value, bool):
        return "true" if safe_value else "false"
    if isinstance(safe_value, str):
        return safe_value.replace("\n", "\\n")
    if isinstance(safe_value, list):
        if all(not isinstance(item, (dict, list)) for item in safe_value):
            return ",".join(
                "true" if item is True else "false" if item is False else str(item)
                for item in safe_value
            )
        return json.dumps(safe_value, sort_keys=True, separators=(",", ":"))
    return str(safe_value)


def _flatten_config_items(
    value: Any,
    *,
    parent_key: str = "",
) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        items: list[tuple[str, str]] = []
        for key in sorted(value.keys(), key=str):
            current_key = f"{parent_key}.{key}" if parent_key else str(key)
            items.extend(_flatten_config_items(value[key], parent_key=current_key))
        return items
    rendered = _stringify_arg_value(value)
    if rendered is None:
        return []
    return [(parent_key or "value", rendered)]


def render_config_args(
    config: dict[str, Any] | None,
    *,
    title: str | None = None,
) -> str:
    parts = [f"{key}={shlex.quote(value)}" for key, value in _flatten_config_items(config or {})]
    body = " ".join(parts) if parts else "<empty>"
    return f"{title}: {body}" if title else body


def _config_for_display(metadata: dict[str, Any]) -> dict[str, Any]:
    cfg = metadata.get("resolved_config", metadata)
    if not isinstance(cfg, dict):
        cfg = {"value": cfg}
    out = dict(cfg)
    if metadata.get("config_path") is not None and "config" not in out:
        out["config"] = metadata["config_path"]
    if metadata.get("summary_path") is not None and "summary" not in out:
        out["summary"] = metadata["summary_path"]
    return out


def print_config_args(
    config: dict[str, Any] | None,
    *,
    title: str | None = None,
) -> None:
    print(render_config_args(config, title=title))


def merge_logging_config(
    raw_cfg: dict[str, Any] | None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_LOGGING_CONFIG)
    if isinstance(raw_cfg, dict):
        for key, value in raw_cfg.items():
            if key == "tags" and value is not None:
                cfg[key] = [str(v) for v in value]
            elif value is not None:
                cfg[key] = value
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "tags":
                cfg[key] = [str(v) for v in value]
            else:
                cfg[key] = value
    cfg["use_wandb"] = bool(cfg.get("use_wandb", False))
    cfg["tags"] = [str(v) for v in cfg.get("tags", [])]
    cfg["mode"] = str(cfg.get("mode", "online"))
    cfg["log_every_n_steps"] = int(cfg.get("log_every_n_steps", 50))
    return cfg


def default_summary_path(
    *,
    entrypoint: str,
    logging_cfg: dict[str, Any],
    default_parent: str | Path | None = None,
    timestamp: int | None = None,
) -> Path:
    ts = int(time.time()) if timestamp is None else int(timestamp)
    if logging_cfg.get("local_log_dir"):
        parent = Path(str(logging_cfg["local_log_dir"]))
    elif default_parent is not None:
        parent = Path(default_parent)
    else:
        parent = Path("src/.cache/run_logs") / entrypoint.replace(".", "/")
    parent.mkdir(parents=True, exist_ok=True)
    stem = str(logging_cfg.get("run_name") or f"run_{ts}")
    safe_stem = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in stem).strip("._")
    if not safe_stem:
        safe_stem = f"run_{ts}"
    return parent / f"{safe_stem}.json"


def summary_events_path(summary_path: str | Path) -> Path:
    p = Path(summary_path)
    if p.suffix:
        return p.with_name(f"{p.stem}.events.jsonl")
    return p.with_name(f"{p.name}.events.jsonl")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), sort_keys=True))
        f.write("\n")


class RunLogger:
    def log_event(
        self,
        event_type: str,
        metrics: dict[str, Any] | None = None,
        *,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        raise NotImplementedError

    def finish(self, status: str, error: Any = None) -> None:
        raise NotImplementedError


@dataclass
class LocalRunLogger(RunLogger):
    entrypoint: str
    summary_path: Path
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.events_path = summary_events_path(self.summary_path)
        self.created_unix = int(time.time())
        self.finished_unix: int | None = None
        self.status = "running"
        self.error: Any = None
        self.summary_payload: dict[str, Any] | None = None

    def _summary_doc(self) -> dict[str, Any]:
        base = dict(self.summary_payload or {})
        base["run_logging"] = {
            "entrypoint": self.entrypoint,
            "summary_path": str(self.summary_path),
            "events_path": str(self.events_path),
            "status": self.status,
            "error": _json_safe(self.error),
            "created_unix": self.created_unix,
            "finished_unix": self.finished_unix,
            "metadata": _json_safe(self.metadata),
        }
        return _json_safe(base)

    def log_event(
        self,
        event_type: str,
        metrics: dict[str, Any] | None = None,
        *,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        resolved_context = dict(_json_safe(context or {}))
        if bool(resolved_context.pop("_wandb_only", False)):
            return
        payload = {
            "ts": time.time(),
            "event_type": event_type,
            "step": step,
            "metrics": _json_safe(metrics or {}),
            "context": resolved_context,
        }
        _append_jsonl(self.events_path, payload)

    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        self.summary_payload = _json_safe(summary_dict)
        atomic_write_json(str(self.summary_path), self._summary_doc())

    def finish(self, status: str, error: Any = None) -> None:
        self.status = str(status)
        self.finished_unix = int(time.time())
        if error is not None:
            self.error = _json_safe(error)
        atomic_write_json(str(self.summary_path), self._summary_doc())
        self.log_event(
            "run_finished",
            metrics={},
            context={
                "status": self.status,
                "error": self.error,
            },
        )


@dataclass
class WandbRunLogger(RunLogger):
    entrypoint: str
    logging_cfg: dict[str, Any]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=self.logging_cfg.get("project"),
            entity=self.logging_cfg.get("entity"),
            tags=list(self.logging_cfg.get("tags", [])),
            mode=self.logging_cfg.get("mode", "online"),
            name=self.logging_cfg.get("run_name"),
            config=_json_safe(self.metadata),
            reinit=True,
        )

    def log_event(
        self,
        event_type: str,
        metrics: dict[str, Any] | None = None,
        *,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(_json_safe(metrics or {}))
        payload["event/type"] = event_type
        resolved_context = dict(_json_safe(context or {}))
        resolved_context.pop("_wandb_only", None)
        for key, value in resolved_context.items():
            payload[f"context/{key}"] = value
        self._wandb.log(payload, step=step)

    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        self._run.summary.update({"run_summary": _json_safe(summary_dict)})

    def finish(self, status: str, error: Any = None) -> None:
        self._run.summary.update({"status": str(status), "error": _json_safe(error)})
        self._run.finish(exit_code=0 if str(status) == "success" else 1)


@dataclass
class CompositeRunLogger(RunLogger):
    loggers: list[RunLogger]

    def log_event(
        self,
        event_type: str,
        metrics: dict[str, Any] | None = None,
        *,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        for logger in self.loggers:
            logger.log_event(event_type, metrics, step=step, context=context)

    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        for logger in self.loggers:
            logger.log_summary(summary_dict)

    def finish(self, status: str, error: Any = None) -> None:
        for logger in self.loggers:
            logger.finish(status, error=error)


def start_run(
    *,
    entrypoint: str,
    logging_cfg: dict[str, Any],
    metadata: dict[str, Any],
    summary_path: str | Path | None = None,
) -> CompositeRunLogger:
    resolved_summary_path = (
        Path(summary_path)
        if summary_path is not None
        else default_summary_path(entrypoint=entrypoint, logging_cfg=logging_cfg)
    )
    print_config_args(
        _config_for_display(metadata),
        title=f"Run config ({entrypoint})",
    )
    loggers: list[RunLogger] = [
        LocalRunLogger(
            entrypoint=entrypoint,
            summary_path=resolved_summary_path,
            metadata=metadata,
        )
    ]
    if bool(logging_cfg.get("use_wandb", False)):
        loggers.append(
            WandbRunLogger(
                entrypoint=entrypoint,
                logging_cfg=logging_cfg,
                metadata=metadata,
            )
        )
    logger = CompositeRunLogger(loggers=loggers)
    logger.log_event("run_started", context=metadata)
    return logger


def finish_with_error(logger: RunLogger | None, exc: Exception) -> None:
    if logger is None:
        return
    logger.finish(
        "failed",
        error={
            "message": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        },
    )
