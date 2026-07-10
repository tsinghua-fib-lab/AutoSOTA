#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TSV_FOLDER_URL = "https://drive.google.com/drive/folders/1UEM1Thcz1c7dc1nji1i5uTN53Kf6G3-e"


def _run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)


def _purge_gdown_cache() -> None:
    # gdown caches cookies under ~/.cache/gdown (linux) or similar on other OSes
    cache_dir = Path.home() / ".cache" / "gdown"
    if cache_dir.exists():
        print(f"Purging gdown cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download TSV paper checkpoints from a public Google Drive folder.")
    p.add_argument("--url", type=str, default=TSV_FOLDER_URL, help="Google Drive folder URL")
    p.add_argument("--out", type=str, default="checkpoints/tsv", help="Output directory")
    p.add_argument("--overwrite", action="store_true", help="Delete output dir first")
    p.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = p.parse_args()

    out_dir = Path(args.out)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer CLI because it's stable across gdown versions
    # Flags per gdown README: --folder, --continue, --remaining-ok, --no-cookies  [oai_citation:1‡GitHub](https://github.com/wkentaro/gdown)
    base_cmd = [sys.executable, "-m", "gdown", "--folder", "--continue", "--remaining-ok", args.url, "-O", str(out_dir)]
    if args.quiet:
        base_cmd.insert(3, "--quiet")  # after "-m gdown"

    print(f"Downloading TSV checkpoints folder:\n  {args.url}\ninto:\n  {out_dir.resolve()}\n")

    # Attempt 1: normal
    rc = _run(base_cmd)
    if rc == 0:
        print("\nDone.")
        return

    # If gdown bails due to JSONDecodeError, it's commonly a corrupted cookies cache.
    # Purge and retry.
    print("\nFirst attempt failed. Trying after purging gdown cookie cache...")
    _purge_gdown_cache()
    rc = _run(base_cmd)
    if rc == 0:
        print("\nDone.")
        return

    # Attempt 3: no cookies
    print("\nSecond attempt failed. Retrying with --no-cookies...")
    cmd_no_cookies = base_cmd.copy()
    # Insert --no-cookies right after module name
    cmd_no_cookies.insert(cmd_no_cookies.index("--folder"), "--no-cookies")
    rc = _run(cmd_no_cookies)
    if rc == 0:
        print("\nDone.")
        return

    # If still failing, Drive is likely throttling or requiring real browser cookies.
    print("\nDownload still failing.")
    print("Most likely causes:")
    print("  1) Google Drive rate-limiting / quota for this shared folder")
    print("  2) Drive requiring browser cookies/consent for your IP/session")
    print()
    print("Next step (works when quota/consent pages appear):")
    print("  - Export your browser cookies to a file named cookies.txt")
    print("  - Put it at: ~/.cache/gdown/cookies.txt")
    print("  - Re-run this script.")
    print()
    print("gdown documents this cookies workaround in its FAQ. See README/FAQ. ")
    sys.exit(rc)


if __name__ == "__main__":
    main()
