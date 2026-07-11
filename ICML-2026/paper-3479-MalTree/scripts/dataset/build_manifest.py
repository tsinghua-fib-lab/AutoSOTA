"""Build the canonical SHA256-keyed sample manifest.

Streams every released feature/embedding file, normalizes sample identifiers
to SHA256 where possible (via ``md5_to_sha.json``), joins family labels and
VirusTotal first-submission timestamps, and records which modalities each
sample carries. The result, ``data/manifest.csv``, lets anyone cross-check
every released artifact back to a single sample identifier.

Usage:
    python build_manifest.py --staging <maltree_staging> --output manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

try:
    import ijson
except ImportError:  # pragma: no cover
    sys.exit("build_manifest.py requires the 'ijson' package: pip install ijson")

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")

# (modality flag, filename relative to <staging>/embeddings)
EMBEDDING_FILES = [
    ("has_fused_emb", "extracted_embeddings.json"),
    ("has_static_emb", "embeddings_static_merged_3000.json"),
    ("has_dynamic_emb", "embeddings_behavior_merged.json"),
    ("has_image_emb", "image_binary_embeddings_merged.json"),
    ("has_numerical_features", "numerical_with_family.json"),
]


def id_type(identifier: str) -> str:
    """Classify an identifier as sha256, md5, or other."""
    if SHA256_RE.match(identifier):
        return "sha256"
    if MD5_RE.match(identifier):
        return "md5"
    return "other"


def stream_keys(path: str):
    """Yield the top-level object keys of a JSON file without building values."""
    with open(path, "rb") as fh:
        for prefix, event, value in ijson.parse(fh):
            if prefix == "" and event == "map_key":
                yield value


def stream_items(path: str):
    """Yield (key, value) for every entry of a top-level JSON object."""
    with open(path, "rb") as fh:
        yield from ijson.kvitems(fh, "")


def stream_array(path: str):
    """Yield each element of a top-level JSON array."""
    with open(path, "rb") as fh:
        yield from ijson.items(fh, "item")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", required=True,
                        help="path to the maltree_staging directory")
    parser.add_argument("--output", default="manifest.csv",
                        help="output manifest CSV (default: manifest.csv)")
    parser.add_argument("--report", default="manifest_summary.json",
                        help="output JSON summary of coverage statistics")
    args = parser.parse_args()

    emb_dir = os.path.join(args.staging, "embeddings")
    md5_to_sha_path = os.path.join(args.staging, "phylo", "drift_analysis", "md5_to_sha.json")
    times_path = os.path.join(args.staging, "phylo", "times", "merged_times_adapted.json")
    numerical_path = os.path.join(emb_dir, "numerical_with_family.json")
    behavior_path = os.path.join(args.staging, "features", "behavior_features.json")

    # MD5 -> SHA256 resolution map.
    md5_to_sha = {}
    if os.path.exists(md5_to_sha_path):
        with open(md5_to_sha_path, encoding="utf-8") as fh:
            md5_to_sha = {k.lower(): v for k, v in json.load(fh).items()}
    print(f"loaded {len(md5_to_sha)} md5->sha mappings", flush=True)

    # Modality presence: identifier -> set of flags.
    samples: dict[str, dict] = {}
    for flag, filename in EMBEDDING_FILES:
        path = os.path.join(emb_dir, filename)
        if not os.path.exists(path):
            print(f"  warning: missing {filename}", file=sys.stderr)
            continue
        count = 0
        for key in stream_keys(path):
            samples.setdefault(key, {})[flag] = True
            count += 1
        print(f"  {filename}: {count} keys", flush=True)

    # Family labels (numerical_with_family.json: key -> {family_name, vector}).
    family = {}
    if os.path.exists(numerical_path):
        for key, value in stream_items(numerical_path):
            if isinstance(value, dict):
                family[key] = value.get("family_name", "")
    print(f"loaded {len(family)} family labels", flush=True)

    # VirusTotal first-submission timestamps (key -> date string).
    timestamps = {}
    if os.path.exists(times_path):
        for key, value in stream_items(times_path):
            timestamps[key] = value
    print(f"loaded {len(timestamps)} timestamps", flush=True)

    # Raw dynamic-analysis features (behavior_features.json: array of {SHA,...}).
    dynamic_feature_ids = set()
    if os.path.exists(behavior_path):
        for item in stream_array(behavior_path):
            if isinstance(item, dict) and "SHA" in item:
                dynamic_feature_ids.add(item["SHA"])
    print(f"loaded {len(dynamic_feature_ids)} dynamic-feature ids", flush=True)

    # Write the manifest.
    flags = [flag for flag, _ in EMBEDDING_FILES]
    fieldnames = (["id", "id_type", "sha256", "family", "first_submission"]
                  + flags + ["has_dynamic_features"])
    stats = {"total": 0, "id_type": {"sha256": 0, "md5": 0, "other": 0},
             "md5_resolved": 0, "with_family": 0, "with_timestamp": 0,
             "modalities": {f: 0 for f in flags}, "has_dynamic_features": 0}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for identifier in sorted(samples):
            itype = id_type(identifier)
            if itype == "sha256":
                sha256 = identifier
            elif itype == "md5":
                sha256 = md5_to_sha.get(identifier.lower(), "")
                if sha256:
                    stats["md5_resolved"] += 1
            else:
                sha256 = ""

            row = {"id": identifier, "id_type": itype, "sha256": sha256,
                   "family": family.get(identifier, ""),
                   "first_submission": timestamps.get(identifier, "")}
            for flag in flags:
                row[flag] = int(samples[identifier].get(flag, False))
            row["has_dynamic_features"] = int(identifier in dynamic_feature_ids)
            writer.writerow(row)

            stats["total"] += 1
            stats["id_type"][itype] += 1
            if row["family"]:
                stats["with_family"] += 1
            if row["first_submission"]:
                stats["with_timestamp"] += 1
            for flag in flags:
                stats["modalities"][flag] += row[flag]
            stats["has_dynamic_features"] += row["has_dynamic_features"]

    stats["distinct_families"] = len(set(family.values()))
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nmanifest written to {args.output} ({stats['total']} samples)")
    print(f"  id types: {stats['id_type']}")
    print(f"  md5 ids resolved to sha256: {stats['md5_resolved']}")
    print(f"  with family / timestamp: {stats['with_family']} / {stats['with_timestamp']}")
    print(f"  distinct families: {stats['distinct_families']}")
    print(f"  modality coverage: {stats['modalities']}")
    print(f"  raw dynamic features: {stats['has_dynamic_features']}")
    print(f"summary written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
