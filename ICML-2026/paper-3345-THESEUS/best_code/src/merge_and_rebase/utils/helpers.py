import json
from pathlib import Path
from typing import Any

import yaml


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() in {".yaml", ".yml"}:
            obj = yaml.safe_load(f)
        else:
            obj = json.load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config file must contain a top-level object. Got: {type(obj)}")
    return obj
