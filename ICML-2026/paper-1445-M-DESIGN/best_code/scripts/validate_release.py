"""Validate the public M-DESIGN release tree."""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

PRIVATE_PATTERNS = {
    "OpenAI API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}", re.IGNORECASE),
    "absolute Windows path": re.compile(
        r"\b[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)+[^\\/:*?\"<>|\r\n]*"
    ),
    "absolute home path": re.compile(r"(?<!\w)/(?:home|Users)/[^\s\"'<>]+"),
    "local machine hostname": re.compile(r"\b(?:DESKTOP|LAPTOP)-[A-Za-z0-9]+\b", re.IGNORECASE),
}

SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv"}
SKIP_SUFFIXES = {".db", ".pt", ".pth", ".png", ".pdf", ".zip", ".gz", ".pkl"}
EXPECTED_TASKS = {"node": 11, "link": 11, "graph": 11}
RELEASE_PT_FILES = {"ecc_predictor.pt", "model_graph.pt"}


def scan_private_text(root: Path) -> list[tuple[Path, str]]:
    findings: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if path.is_dir() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                findings.append((path, label))
    return findings


def validate_knowledge_base(root: Path) -> None:
    for task, expected_count in EXPECTED_TASKS.items():
        task_root = root / task
        db_files = sorted(task_root.glob("*/*.db"))
        if len(db_files) != expected_count:
            raise AssertionError(f"{task}: expected {expected_count} DB files, found {len(db_files)}")
        for db_path in db_files:
            with sqlite3.connect(db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM model_records").fetchone()[0]
            if count <= 0:
                raise AssertionError(f"{db_path} has no model records")

    unexpected_pt = [
        path for path in root.rglob("*.pt") if path.name not in RELEASE_PT_FILES
    ]
    if unexpected_pt:
        raise AssertionError("Unexpected .pt artifacts: " + ", ".join(str(path) for path in unexpected_pt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--knowledge-base", default="knowledge_retrieval/knowledge_base")
    args = parser.parse_args()

    root = Path(args.root)
    kb_root = root / args.knowledge_base
    findings = scan_private_text(root)
    if findings:
        for path, label in findings:
            print(f"Private pattern detected ({label}): {path}")
        raise SystemExit(1)
    validate_knowledge_base(kb_root)
    print("Release validation passed.")


if __name__ == "__main__":
    main()
