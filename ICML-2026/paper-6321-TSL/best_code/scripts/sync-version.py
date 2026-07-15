#!/usr/bin/env python3
"""Keep the R package version aligned with the canonical Cargo workspace version.

The single source of truth is `[workspace.package].version` in the root
`Cargo.toml`. The Rust crates inherit it and `tsl-py` reads it through maturin's
dynamic version, so they are always aligned automatically. The R connector
`tsl-r` (extendr package `tslr`), whose version lives in `tsl-r/DESCRIPTION`, has
no such link, so this script propagates the Cargo version into it.

Usage:
    sync-version.py            # write the Cargo version into DESCRIPTION
    sync-version.py --check    # assert DESCRIPTION matches; non-zero on drift

If `DESCRIPTION` does not exist yet (no R package), both modes are a clean no-op.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The R connector (extendr package `tslr`), the R analogue of tsl-py. It depends
# on the core via a git dependency rather than being a Cargo workspace member, so
# its version lives only here and is not reached by `cargo set-version --workspace`.
DESCRIPTION = REPO_ROOT / "tsl-r" / "DESCRIPTION"


def cargo_version() -> str:
    """The workspace version, as resolved by Cargo (handles inheritance)."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    versions = {pkg["version"] for pkg in json.loads(out)["packages"]}
    if len(versions) != 1:
        sys.exit(f"workspace crates disagree on version: {sorted(versions)}")
    return versions.pop()


def description_version(text: str) -> str:
    m = re.search(r"^Version:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        sys.exit(f"no 'Version:' field in {DESCRIPTION}")
    return m.group(1).strip()


def aligned(cargo: str, r: str) -> bool:
    """R may carry a dev suffix (e.g. 1.2.0.9000); compare the release part."""
    return r == cargo or r.startswith(cargo + ".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify instead of write")
    args = ap.parse_args()

    version = cargo_version()

    if not DESCRIPTION.exists():
        print(f"no R package at {DESCRIPTION.relative_to(REPO_ROOT)}; nothing to sync")
        return 0

    text = DESCRIPTION.read_text()
    current = description_version(text)

    if args.check:
        if not aligned(version, current):
            print(
                f"version drift: Cargo={version} but DESCRIPTION={current}\n"
                f"run scripts/sync-version.py to realign",
                file=sys.stderr,
            )
            return 1
        print(f"versions aligned: {version}")
        return 0

    if aligned(version, current):
        print(f"DESCRIPTION already at {current}")
        return 0
    new_text = re.sub(
        r"^(Version:\s*).+$", rf"\g<1>{version}", text, count=1, flags=re.MULTILINE
    )
    DESCRIPTION.write_text(new_text)
    print(f"DESCRIPTION {current} -> {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
