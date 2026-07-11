"""Assemble the released MalTree data files.

Reads the source feature and embedding files, identifies every sample by the
SHA256 of its binary (resolving MD5 identifiers via ``md5_to_sha.json``), and
writes uniform-schema release files to ``--output-dir``:

    embeddings_fused.json       {sha256: {embedding: [...], family: str}}
    embeddings_static.json      {sha256: {embedding: [...], family: str}}
    embeddings_dynamic.json     {sha256: {embedding: [...], family: str}}
    embeddings_image.json       {sha256: {embedding: [...], family: str}}

Every released file is a JSON object keyed by SHA256.

Usage:
    python prepare_release.py --staging <maltree_staging> --output-dir <out>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

try:
    import ijson
except ImportError:  # pragma: no cover
    sys.exit("prepare_release.py requires the 'ijson' package: pip install ijson")

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")

# Source files: (source filename, release filename).
EMBEDDING_FILES = [
    ("extracted_embeddings.json", "embeddings_fused.json"),
    ("embeddings_static_merged_3000.json", "embeddings_static.json"),
    ("embeddings_behavior_merged.json", "embeddings_dynamic.json"),
    ("image_binary_embeddings_merged.json", "embeddings_image.json"),
]


def resolve_sha256(identifier: str, md5_to_sha: dict) -> str | None:
    """Return the SHA256 for an identifier, resolving MD5 aliases."""
    if SHA256_RE.match(identifier):
        return identifier.lower()
    if MD5_RE.match(identifier):
        mapped = md5_to_sha.get(identifier.lower())
        return mapped.lower() if mapped and SHA256_RE.match(mapped) else None
    return None


class StreamWriter:
    """Incrementally writes a JSON object so multi-GB files never sit in RAM."""

    def __init__(self, path: str):
        self._fh = open(path, "w", encoding="utf-8")
        self._fh.write("{")
        self._first = True
        self.count = 0

    def write(self, key: str, value) -> None:
        if not self._first:
            self._fh.write(",")
        self._first = False
        self._fh.write(json.dumps(key))
        self._fh.write(":")
        self._fh.write(json.dumps(value, separators=(",", ":")))
        self.count += 1

    def close(self) -> None:
        self._fh.write("}")
        self._fh.close()


def first_present(value: dict, keys: list[str]):
    """Return the value of the first key present in ``value``, else None."""
    for key in keys:
        if key in value:
            return value[key]
    return None


def prepare_embedding_file(in_path: str, out_path: str, md5_to_sha: dict) -> int:
    """Write an embedding file as {sha256: {embedding, family}}."""
    writer = StreamWriter(out_path)
    seen: set[str] = set()
    with open(in_path, "rb") as fh:
        for key, value in ijson.kvitems(fh, "", use_float=True):
            sha = resolve_sha256(key, md5_to_sha)
            if sha is None or sha in seen or not isinstance(value, dict):
                continue
            embedding = first_present(value, ["embedding", "embeddings"])
            family = first_present(value, ["family", "label", "family_name"])
            if embedding is None:
                continue
            seen.add(sha)
            writer.write(sha, {"embedding": embedding, "family": family})
    writer.close()
    return writer.count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", required=True, help="maltree_staging directory")
    parser.add_argument("--output-dir", required=True, help="release output directory")
    args = parser.parse_args()

    emb_dir = os.path.join(args.staging, "embeddings")
    os.makedirs(args.output_dir, exist_ok=True)

    md5_path = os.path.join(args.staging, "phylo", "drift_analysis", "md5_to_sha.json")
    with open(md5_path, encoding="utf-8") as fh:
        md5_to_sha = {k.lower(): v for k, v in json.load(fh).items()}
    print(f"loaded {len(md5_to_sha)} md5->sha mappings", flush=True)

    for source_name, release_name in EMBEDDING_FILES:
        in_path = os.path.join(emb_dir, source_name)
        out_path = os.path.join(args.output_dir, release_name)
        print(f"writing {release_name} ...", flush=True)
        count = prepare_embedding_file(in_path, out_path, md5_to_sha)
        print(f"  {count} samples", flush=True)

    print("release files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
