import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelMappingEntry:
    key: str
    alias: str
    plot: bool = True
    selected: bool = True
    run_ids: tuple[str, ...] = ()
    gpqa_run_ids: tuple[str, ...] = ()
    webarena_run_ids: tuple[str, ...] = ()
    release_date: str | None = None


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "t", "true", "y", "yes"}:
        return True
    if text in {"0", "f", "false", "n", "no"}:
        return False
    return default


def _normalize_release_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_run_ids(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, dict):
        iterable: Iterable[Any] = values.keys()
    elif isinstance(values, (list, tuple, set)):
        iterable = values
    else:
        iterable = [values]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in iterable:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _normalize_dict_mapping(raw_mapping: dict[str, Any]) -> dict[str, ModelMappingEntry]:
    normalized: dict[str, ModelMappingEntry] = {}
    for model_key, spec in (raw_mapping or {}).items():
        key = str(model_key).strip()
        if not key:
            continue
        spec = spec or {}
        alias = spec.get("alias") or spec.get("display_name") or key
        alias = str(alias).strip() or key
        release_date = _normalize_release_date(spec.get("release_date"))
        normalized[key] = ModelMappingEntry(
            key=key,
            alias=alias,
            plot=_coerce_bool(spec.get("plot"), True),
            selected=_coerce_bool(spec.get("selected"), True),
            run_ids=_normalize_run_ids(
                spec.get("run_ids") or spec.get("runs") or spec.get("run_names")
            ),
            gpqa_run_ids=_normalize_run_ids(spec.get("gpqa_run_ids")),
            webarena_run_ids=_normalize_run_ids(spec.get("webarena_run_ids")),
            release_date=release_date,
        )
    return normalized


def _load_mapping_from_json(path: Path) -> dict[str, ModelMappingEntry]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        raise ValueError(
            f"Expected a mapping object inside {path}, but found a list instead."
        )
    if not isinstance(payload, dict):
        raise ValueError(f"Model mapping {path} is not a JSON object.")
    return _normalize_dict_mapping(payload)


def _load_mapping_from_csv(path: Path) -> dict[str, ModelMappingEntry]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        temp: dict[str, dict[str, Any]] = {}
        for row in reader:
            model_key = (row.get("model_key") or row.get("run_id") or "").strip()
            if not model_key:
                continue
            display = (
                row.get("display_name")
                or row.get("alias")
                or row.get("model_key")
                or model_key
            )
            alias = str(display).strip() or model_key
            plot_flag = _coerce_bool(row.get("plot"), True)
            run_id = (row.get("run_id") or "").strip()
            spec = temp.setdefault(
                model_key,
                {
                    "alias": alias,
                    "plot": False,
                    "selected": True,
                    "run_ids": [],
                    "release_date": None,
                },
            )
            if spec["alias"] == model_key and alias != model_key:
                spec["alias"] = alias
            spec["plot"] = spec["plot"] or plot_flag
            if run_id and run_id not in spec["run_ids"]:
                spec["run_ids"].append(run_id)
            release_date = _normalize_release_date(
                row.get("release_date") or row.get("date")
            )
            if release_date:
                spec["release_date"] = release_date
        normalized: dict[str, ModelMappingEntry] = {}
        for key, spec in temp.items():
            normalized[key] = ModelMappingEntry(
                key=key,
                alias=spec["alias"] or key,
                plot=_coerce_bool(spec.get("plot"), True),
                selected=_coerce_bool(spec.get("selected"), True),
                run_ids=tuple(spec.get("run_ids") or ()),
                release_date=_normalize_release_date(spec.get("release_date")),
            )
        return normalized


def load_model_mapping(path: Path | str | None) -> dict[str, ModelMappingEntry]:
    """Load model mapping entries from a JSON dictionary or legacy CSV file."""
    if path is None:
        return {}
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file {mapping_path} does not exist.")
    suffix = mapping_path.suffix.lower()
    if suffix == ".json":
        return _load_mapping_from_json(mapping_path)
    if suffix == ".csv":
        return _load_mapping_from_csv(mapping_path)
    raise ValueError(
        f"Unsupported mapping file extension '{mapping_path.suffix}' "
        f"(expected .json or .csv)."
    )


def list_models_from_all_runs(path: Path) -> list[str]:
    """Return the sorted set of models in data/all_runs.jsonl."""
    if not path.exists():
        raise FileNotFoundError(f"all_runs.jsonl not found at {path}")
    models: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get("model")
            if value is None:
                continue
            text = str(value).strip()
            if text:
                models.add(text)
    return sorted(models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the unique models discovered in --all-runs.",
    )
    parser.add_argument(
        "--all-runs",
        type=Path,
        default=Path(__file__).with_name("data") / "all_runs.jsonl",
        help="Path to the all_runs.jsonl file used for listing models.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Optional mapping file to summarize (JSON or CSV).",
    )
    args = parser.parse_args()

    if args.list_models:
        models = list_models_from_all_runs(args.all_runs)
        print(f"Found {len(models)} model(s) in {args.all_runs}:")
        for model in models:
            print(f"  {model}")
        return

    if args.mapping:
        entries = load_model_mapping(args.mapping)
        names = ", ".join(sorted(entries)) or "(none)"
        print(f"Loaded {len(entries)} model(s) from {args.mapping}: {names}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
