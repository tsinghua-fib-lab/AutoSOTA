from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union, Mapping

import yaml


class AttrDict(dict):
    """
    Dict that supports attribute access: obj.key in addition to obj['key'].
    Recursively converts nested mappings.
    """
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "AttrDict":
        def convert(x: Any) -> Any:
            if isinstance(x, Mapping):
                ad = AttrDict()
                for k, v in x.items():
                    ad[k] = convert(v)
                return ad
            if isinstance(x, list):
                return [convert(i) for i in x]
            return x

        return convert(d)


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是 mapping/dict: {path}")
    return data


def resolve_files(config_dir: Union[str, Path], files: Iterable[str]) -> List[Path]:
    config_dir = Path(config_dir)
    resolved: List[Path] = []
    for name in files:
        p = Path(name)
        if not p.suffix:
            candidates = [config_dir / f"{name}.yaml", config_dir / f"{name}.yml"]
        elif p.is_absolute() or p.exists():
            candidates = [p]
        else:
            candidates = [config_dir / name]

        found = next((c for c in candidates if c.exists()), None)
        if not found:
            raise FileNotFoundError(f"找不到配置文件: {name} (searched: {candidates})")
        resolved.append(found)
    return resolved


def load_config(
    config_dir: Union[str, Path] = "config",
    files: Union[None, str, Iterable[str]] = None,
    default_files: Iterable[str] = ("base.yaml",),
    env_key: str = "APP_CONFIG",
) -> AttrDict:
    """
    返回 AttrDict：支持 args.max_turns 访问，同时兼容 args['max_turns']。
    """
    if files is None:
        env_val = os.getenv(env_key, "").strip()
        if env_val:
            files_list = [x.strip() for x in env_val.split(",") if x.strip()]
        else:
            files_list = list(default_files)
    elif isinstance(files, str):
        files_list = [x.strip() for x in files.split(",") if x.strip()]
    else:
        files_list = list(files)

    paths = resolve_files(config_dir, files_list)

    cfg: Dict[str, Any] = {}
    for p in paths:
        cfg = _deep_merge(cfg, load_yaml(p))

    cfg["_meta"] = {
        "config_dir": str(Path(config_dir).resolve()),
        "files": [str(p.resolve()) for p in paths],
    }

    return AttrDict.from_dict(cfg)
