#!/usr/bin/env python3
"""Download organizer-published artifacts with checksum verification and resume."""
import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "dataset": ("https://osf.io/46dpj/download", "kaggle_dataset.tar.gz", "0b1201faf5035159d78ab8a09c5c0343d10c593840657a577ffceb2c401ac86f"),
    "champion_predictions": ("https://osf.io/hecpw/download", "team_1_submissions.tar.gz", "61030e547a5ac6a45c24f843d5c8fac18aef465d279f45b3b606acf749d146df"),
    "runnerup_checkpoint": ("https://osf.io/3r4zv/download", "team_2_checkpoints.tar", "2f284bc8edc8f2e06a07f248b78183dee52af30e48a7121ac01db18848f9b1b4"),
    "champion_checkpoints_1": ("https://osf.io/89yht/download", "team_1_checkpoints_part_1.tar", "d38bf267d734175e47631d9ec7f982ba74e09b04894beea48f05b9d90fa773d9"),
    "champion_checkpoints_2": ("https://osf.io/46xwf/download", "team_1_checkpoints_part_2.tar", "91e73c16d7b9167464fe2b1e488386610d92468e9c183b4315527224642f7908"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(name):
    url, filename, expected = ASSETS[name]
    dest = ROOT / "cache" / "downloads" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected is None or digest(dest) == expected):
        print(json.dumps({"asset": name, "status": "cached", "bytes": dest.stat().st_size}), flush=True)
        return
    part = dest.with_suffix(dest.suffix + ".partial")
    started = time.time()
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": "bytes=%d-" % offset} if offset else {}
            with requests.get(url, stream=True, headers=headers, timeout=(30, 90)) as r:
                r.raise_for_status()
                resume = r.status_code == 206
                if not resume:
                    offset = 0
                total = offset + int(r.headers.get("Content-Length", 0))
                print(json.dumps({"asset": name, "status": "downloading", "offset": offset, "total": total}), flush=True)
                last = time.time()
                with part.open("ab" if resume else "wb") as f:
                    for chunk in r.iter_content(4 * 1024 * 1024):
                        f.write(chunk)
                        offset += len(chunk)
                        if time.time() - last > 30:
                            print(json.dumps({"asset": name, "bytes": offset, "total": total}), flush=True)
                            last = time.time()
            actual = digest(part)
            if expected and actual != expected:
                raise ValueError("SHA256 mismatch for %s" % name)
            part.replace(dest)
            meta = {"asset": name, "source": url, "bytes": dest.stat().st_size,
                    "sha256": actual, "seconds": round(time.time() - started, 1), "status": "complete"}
            dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(meta, indent=2))
            print(json.dumps(meta), flush=True)
            return
        except Exception as e:
            print(json.dumps({"asset": name, "attempt": attempt + 1, "error": str(e)}), flush=True)
            if attempt == 4:
                raise
            time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", nargs="+", choices=sorted(ASSETS))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, args.assets))
