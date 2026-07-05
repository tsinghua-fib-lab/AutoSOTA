import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def _iter_json_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List JSON files under a root that fail to parse."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Directory to scan (defaults to current directory).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"Root path is not a directory: {root}", file=sys.stderr)
        return 2

    failed = False
    for path in _iter_json_files(root):
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"{path} ({exc})")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
