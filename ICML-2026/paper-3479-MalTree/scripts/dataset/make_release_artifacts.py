"""Derive the small in-repo artifacts from the canonical manifest.

Produces, from ``data/manifest.csv``:
  * ``data/family_labels.json``  -- {sha256: family}
  * ``data/timestamps.json``     -- {sha256: {first_submission: date}}
  * ``data/example/``            -- a tiny self-consistent subset (a few
    families x a few samples) so the pipeline can be smoke-tested without
    the multi-GB Zenodo download.

The example embeddings are sliced from the staged fused-embedding file.

Usage:
    python make_release_artifacts.py --manifest data/manifest.csv \
        --staged-fused <maltree_staging>/embeddings/extracted_embeddings.json \
        --data-dir data
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

try:
    import ijson
except ImportError:  # pragma: no cover
    sys.exit("make_release_artifacts.py requires 'ijson': pip install ijson")


def read_manifest(path: str) -> list[dict]:
    """Return one row per SHA256 that has a fused embedding.

    The manifest can list the same sample under more than one identifier
    (e.g. an MD5 alias and its SHA256), and those rows may carry slightly
    different family labels. Keep the SHA256-native row so every sample has
    a single, consistent family.
    """
    by_sha: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sha = row["sha256"]
            if not sha or row["has_fused_emb"] != "1":
                continue
            existing = by_sha.get(sha)
            if existing is None or (existing["id_type"] != "sha256"
                                    and row["id_type"] == "sha256"):
                by_sha[sha] = row
    return list(by_sha.values())


def pick_example_shas(rows: list[dict], n_families: int, per_family: int) -> list[str]:
    """Pick families spanning multiple years, then per_family samples from each."""
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    def year(row: dict) -> str:
        return (row["first_submission"] or "")[:4]

    eligible = []
    for family, members in by_family.items():
        years = {year(m) for m in members if year(m)}
        if len(members) >= per_family and len(years) >= 2:
            eligible.append((family, members))
    eligible.sort(key=lambda fm: len(fm[1]), reverse=True)

    chosen: list[str] = []
    for family, members in eligible[:n_families]:
        members = sorted(members, key=lambda m: m["first_submission"])
        step = max(1, len(members) // per_family)
        chosen.extend(m["sha256"] for m in members[::step][:per_family])
    return chosen


def slice_fused(staged_fused: str, wanted: set[str]) -> dict:
    """Stream the staged fused-embedding file and keep only the wanted SHAs."""
    result: dict[str, dict] = {}
    with open(staged_fused, "rb") as fh:
        for key, value in ijson.kvitems(fh, "", use_float=True):
            if key in wanted:
                embedding = value.get("embedding") if isinstance(value, dict) else value
                result[key] = embedding
                if len(result) == len(wanted):
                    break
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--staged-fused", required=True,
                        help="staged fused-embedding JSON (extracted_embeddings.json)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-families", type=int, default=8)
    parser.add_argument("--per-family", type=int, default=12)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    print(f"manifest: {len(rows)} samples with a resolved SHA256 + fused embedding")

    family_labels = {r["sha256"]: r["family"] for r in rows}
    timestamps = {r["sha256"]: {"first_submission": r["first_submission"]} for r in rows}
    with open(os.path.join(args.data_dir, "family_labels.json"), "w", encoding="utf-8") as fh:
        json.dump(family_labels, fh)
    with open(os.path.join(args.data_dir, "timestamps.json"), "w", encoding="utf-8") as fh:
        json.dump(timestamps, fh)
    print(f"wrote family_labels.json ({len(family_labels)}) and "
          f"timestamps.json ({len(timestamps)})")

    example_shas = pick_example_shas(rows, args.n_families, args.per_family)
    print(f"example subset: {len(example_shas)} samples")
    if not example_shas:
        print("error: no eligible families for the example subset", file=sys.stderr)
        return 1

    print("slicing fused embeddings for the example subset ...", flush=True)
    fused = slice_fused(args.staged_fused, set(example_shas))

    example_dir = os.path.join(args.data_dir, "example")
    os.makedirs(example_dir, exist_ok=True)
    ex_embeddings = {sha: {"embedding": vec, "family": family_labels[sha]}
                     for sha, vec in fused.items()}
    ex_families = {sha: family_labels[sha] for sha in fused}
    ex_timestamps = {sha: timestamps[sha] for sha in fused}
    with open(os.path.join(example_dir, "embeddings_fused.json"), "w", encoding="utf-8") as fh:
        json.dump(ex_embeddings, fh, indent=1)
    with open(os.path.join(example_dir, "family_labels.json"), "w", encoding="utf-8") as fh:
        json.dump(ex_families, fh, indent=1)
    with open(os.path.join(example_dir, "timestamps.json"), "w", encoding="utf-8") as fh:
        json.dump(ex_timestamps, fh, indent=1)
    print(f"wrote example subset to {example_dir} "
          f"({len(fused)} samples, {len(set(ex_families.values()))} families)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
