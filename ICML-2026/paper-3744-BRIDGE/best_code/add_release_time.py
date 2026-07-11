import argparse
import csv
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
PARAMS_DIR = SCRIPT_DIR / "params"


def _normalize(text: Any) -> str:
    if text is None:
        return ""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def load_release_lookup(mapping_path: Path) -> dict[str, str]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Model mapping not found: {mapping_path}")
    data = json.loads(mapping_path.read_text())
    lookup: dict[str, str] = {}
    for key, spec in data.items():
        if not isinstance(spec, dict):
            continue
        release_date = spec.get("release_date")
        if not release_date:
            continue
        candidates = {key}
        alias = spec.get("alias")
        if alias:
            candidates.add(alias)
        for run_id in spec.get("run_ids") or ():
            if run_id:
                candidates.add(run_id)
        for candidate in candidates:
            normalized = _normalize(candidate)
            if not normalized:
                continue
            lookup[normalized] = release_date
    return lookup


def annotate_rows(
    abilities_path: Path,
    release_lookup: dict[str, str],
    release_column: str,
) -> tuple[list[dict[str, str]], list[str], int]:
    if not abilities_path.exists():
        raise FileNotFoundError(f"Ability CSV not found: {abilities_path}")
    with abilities_path.open(newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = []
        matched = 0
        for row in reader:
            subject_id = row.get("subject_id", "")
            release_date = release_lookup.get(_normalize(subject_id), "")
            if release_date:
                matched += 1
            row[release_column] = release_date
            rows.append(row)
    if release_column not in fieldnames:
        fieldnames.append(release_column)
    return rows, fieldnames, matched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add release time annotations to a pyirt ability CSV."
    )
    parser.add_argument(
        "--abilities-csv",
        type=Path,
        default=PARAMS_DIR / "all_a_pyirt_abilities.csv",
        help="CSV produced by fit_irt.py containing subject_id/ability columns.",
    )
    parser.add_argument(
        "--model-mapping",
        type=Path,
        default=DATA_DIR / "model_run_mapping.json",
        help="Model mapping JSON containing release_date metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the annotated CSV. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--release-column",
        default="release_time",
        help="Name for the release column that will be appended to the CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_lookup = load_release_lookup(args.model_mapping)
    rows, fieldnames, matched = annotate_rows(
        args.abilities_csv, release_lookup, args.release_column
    )
    output_path = args.output or args.abilities_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.verbose:
        total = len(rows)
        print(
            f"Wrote {output_path} with {matched} / {total} rows annotated with release times."
        )


if __name__ == "__main__":
    main()
