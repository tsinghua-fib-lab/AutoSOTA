"""Download the released MalTree data from Zenodo and verify checksums.

The raw features, the multi-modal embeddings, and the trained model weights
are far too large for the git repository, so they are archived on Zenodo.
This script fetches them into ``data/`` and verifies every file against
``CHECKSUMS.sha256``.

A small example subset is bundled in the repo under ``data/example/`` so the
pipeline can be smoke-tested without the full download.

Usage:
    python scripts/download_data.py                 # download the full release
    python scripts/download_data.py --record 1234567
    python scripts/download_data.py --example       # just check the bundled subset
    python scripts/download_data.py --verify-only   # re-check already-downloaded files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

# Zenodo record id for the MalTree data release.
ZENODO_RECORD = "20261117"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
EXAMPLE_DIR = os.path.join(DATA_DIR, "example")
CHECKSUMS_FILE = os.path.join(REPO_ROOT, "CHECKSUMS.sha256")


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    """Return the SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checksums() -> dict[str, str]:
    """Parse CHECKSUMS.sha256 into {filename: sha256}."""
    sums: dict[str, str] = {}
    if not os.path.exists(CHECKSUMS_FILE):
        return sums
    with open(CHECKSUMS_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                digest, name = line.split(None, 1)
                sums[os.path.basename(name.strip())] = digest
    return sums


def verify_file(path: str, expected: str) -> bool:
    """Check a file's SHA256 against the expected digest."""
    if not os.path.exists(path):
        print(f"  MISSING: {os.path.basename(path)}")
        return False
    actual = sha256_of(path)
    ok = actual == expected
    print(f"  {'OK      ' if ok else 'MISMATCH'} {os.path.basename(path)}")
    return ok


def download(url: str, dest: str) -> None:
    """Stream a URL to a local file."""
    print(f"  downloading {os.path.basename(dest)} ...", flush=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)


def fetch_zenodo_record(record_id: str) -> list[dict]:
    """Return the file list of a Zenodo record via its public API."""
    api = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api) as response:
        meta = json.load(response)
    return meta.get("files", [])


def cmd_example() -> int:
    """Confirm the bundled example subset is present and self-consistent."""
    if not os.path.isdir(EXAMPLE_DIR):
        print(f"error: example subset not found at {EXAMPLE_DIR}", file=sys.stderr)
        return 1
    files = sorted(f for f in os.listdir(EXAMPLE_DIR) if f.endswith(".json"))
    print(f"example subset ready ({len(files)} files in {EXAMPLE_DIR}):")
    for name in files:
        print(f"  {name}")
    print("\nrun the pipeline on it with:  bash reproduce.sh --example")
    return 0


def cmd_verify() -> int:
    """Verify already-downloaded files against CHECKSUMS.sha256."""
    sums = load_checksums()
    if not sums:
        print(f"error: {CHECKSUMS_FILE} not found or empty", file=sys.stderr)
        return 1
    print(f"verifying {len(sums)} files against CHECKSUMS.sha256 ...")
    all_ok = all(verify_file(os.path.join(DATA_DIR, name), digest)
                 for name, digest in sorted(sums.items()))
    print("all files verified" if all_ok else "verification FAILED")
    return 0 if all_ok else 1


def cmd_download(record_id: str) -> int:
    """Download every file in the Zenodo record and verify it."""
    if record_id.startswith("REPLACE_WITH"):
        print("error: no Zenodo record id set. Pass --record <id> or edit "
              "ZENODO_RECORD in this script once the dataset is uploaded.",
              file=sys.stderr)
        return 1
    os.makedirs(DATA_DIR, exist_ok=True)
    sums = load_checksums()
    print(f"fetching Zenodo record {record_id} ...")
    files = fetch_zenodo_record(record_id)
    if not files:
        print("error: record has no files", file=sys.stderr)
        return 1

    ok = True
    for entry in files:
        name = entry["key"]
        dest = os.path.join(DATA_DIR, name)
        url = entry["links"]["self"]
        expected = sums.get(name)
        if expected and os.path.exists(dest) and sha256_of(dest) == expected:
            print(f"  skip (present) {name}")
            continue
        download(url, dest)
        if expected:
            ok &= verify_file(dest, expected)
        else:
            print(f"  WARNING: {name} not listed in CHECKSUMS.sha256")
    print("\ndownload complete" if ok else "\ndownload finished with checksum errors")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", default=ZENODO_RECORD,
                        help="Zenodo record id (default: the built-in ZENODO_RECORD)")
    parser.add_argument("--example", action="store_true",
                        help="just check the bundled example subset; download nothing")
    parser.add_argument("--verify-only", action="store_true",
                        help="re-verify already-downloaded files against CHECKSUMS.sha256")
    args = parser.parse_args()

    if args.example:
        return cmd_example()
    if args.verify_only:
        return cmd_verify()
    return cmd_download(args.record)


if __name__ == "__main__":
    raise SystemExit(main())
