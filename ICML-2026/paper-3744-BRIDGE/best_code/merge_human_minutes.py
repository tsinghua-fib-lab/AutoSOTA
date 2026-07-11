import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
PARAMS_DIR = SCRIPT_DIR / "params"


def load_human_minutes(path: Path) -> Dict[str, Optional[float]]:
    mapping: Dict[str, Optional[float]] = {}
    with path.open() as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_id = data.get("task_id")
            if not task_id:
                raise ValueError(f"Missing task_id in {path}:{line_num}")
            mapping[task_id] = data.get("human_minutes")
    return mapping


def read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open(newline="") as handle:
        reader = list(csv.reader(handle))
    if not reader:
        raise ValueError(f"{path} has no rows")
    header = reader[0]
    rows = reader[1:]
    return header, rows


def merge_minutes(
    header: List[str], rows: List[List[str]], minutes: Dict[str, Optional[float]],
    verbose: bool = False,
) -> Tuple[List[str], List[List[str]], int, int]:
    if "human_minutes" in header:
        hm_idx = header.index("human_minutes")
    else:
        header = header[:] + ["human_minutes"]
        hm_idx = len(header) - 1

    matched = 0
    missing = 0
    for row in rows:
        if not row:
            continue
        task_id = row[0]
        hm_value = minutes.get(task_id)
        if len(row) <= hm_idx:
            row.extend([""] * (hm_idx + 1 - len(row)))
        row[hm_idx] = "" if hm_value is None else str(hm_value)
        if hm_value is None:
            missing += 1
            if verbose:
                print(task_id)
        else:
            matched += 1
    return header, rows, matched, missing


def write_rows(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach human_minutes values to the swebench CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PARAMS_DIR / "swebench_selected_plus_all_runs_pyirt.csv",
        help="Path to the CSV to augment (default: %(default)s)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DATA_DIR / "human_minutes_by_task.jsonl",
        help="Path to JSONL file containing {'task_id','human_minutes'} records "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the merged CSV (default: overwrite --csv file in place).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    jsonl_path = args.jsonl
    output_path = args.output or csv_path
    tmp_path = (
        output_path
        if output_path != csv_path
        else csv_path.with_suffix(csv_path.suffix + ".tmp")
    )

    human_minutes = load_human_minutes(jsonl_path)
    header, rows = read_csv_rows(csv_path)
    header, rows, matched, missing = merge_minutes(header, rows, human_minutes, args.verbose)

    write_rows(tmp_path, header, rows)
    if tmp_path != output_path:
        tmp_path.replace(output_path)

    if args.verbose:
        print(
            f"Merged human_minutes for {matched} rows "
            f"(missing {missing}) into {output_path}"
        )


if __name__ == "__main__":
    main()
