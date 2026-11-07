"""JSONL records and content hashes shared by generation and evaluation."""

import hashlib
import json


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
