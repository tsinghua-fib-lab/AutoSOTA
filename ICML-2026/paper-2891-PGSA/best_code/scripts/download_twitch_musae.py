#!/usr/bin/env python3
"""
Download Stanford SNAP ``twitch.zip`` (MUSAE Twitch gamer networks) and arrange files
so ``Utils.pre_data.datasets.prepare_Twitch`` finds them under this repo's ``dataset/Twitch``.

Layout after running (matches ``prepare_Twitch(raw_dir, lang)`` with ``raw_dir=../dataset/Twitch/``):

  Twitch/{lang}/raw/musae_{lang}_edges.csv
  Twitch/{lang}/raw/musae_{lang}_features.json
  Twitch/{lang}/raw/musae_{lang}_target.csv

SNAP uses folder names ``ENGB`` and ``PTBR``; this script maps them to ``EN`` and ``PT``.
DE ships ``musae_DE.json`` (same format as features); it is copied to ``musae_DE_features.json``.

Usage (from repo root, conda env ``general`` or any Python 3):

  python dataset/download_twitch_musae.py

Override destination::

  python dataset/download_twitch_musae.py /path/to/dataset/Twitch
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

SNAP_TWITCH_ZIP = "http://snap.stanford.edu/data/twitch.zip"

# SNAP folder name -> lang code expected by prepare_Twitch / --src_name --tgt_name
_SNAP_TO_LANG = {
    "DE": "DE",
    "ENGB": "EN",
    "ES": "ES",
    "FR": "FR",
    "PTBR": "PT",
    "RU": "RU",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def install_from_extracted(snap_twitch_dir: Path, dest_root: Path) -> None:
    """snap_twitch_dir: extracted ``.../twitch`` with DE/, ENGB/, ..."""
    for snap_name, lang in _SNAP_TO_LANG.items():
        src_dir = snap_twitch_dir / snap_name
        if not src_dir.is_dir():
            continue
        raw_out = dest_root / lang / "raw"
        raw_out.mkdir(parents=True, exist_ok=True)
        snap_upper = snap_name

        def put(suffix: str, dest_name: str | None = None) -> None:
            dest_name = dest_name or suffix
            src = src_dir / f"musae_{snap_upper}_{suffix}"
            dst = raw_out / f"musae_{lang}_{dest_name}"
            if not src.is_file():
                raise FileNotFoundError(f"missing {src}")
            shutil.copy2(src, dst)

        put("edges.csv")
        put("target.csv")
        if snap_name == "DE":
            src_feat = src_dir / "musae_DE.json"
            if not src_feat.is_file():
                raise FileNotFoundError(f"missing {src_feat}")
            shutil.copy2(src_feat, raw_out / "musae_DE_features.json")
        else:
            put("features.json")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Download SNAP Twitch into MUSAE CSV layout for PSAHS.")
    ap.add_argument(
        "dest",
        nargs="?",
        default=str(_repo_root() / "dataset" / "Twitch"),
        help="Output root (default: <repo>/dataset/Twitch)",
    )
    args = ap.parse_args(argv)
    dest_root = Path(args.dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "twitch.zip"
        print(f"Downloading {SNAP_TWITCH_ZIP} ...", flush=True)
        urlretrieve(SNAP_TWITCH_ZIP, zpath)
        extract_dir = Path(td) / "extract"
        extract_dir.mkdir()
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_dir)
        # archive top-level is "twitch/"
        snap_twitch = extract_dir / "twitch"
        if not snap_twitch.is_dir():
            raise RuntimeError(f"unexpected zip layout: {snap_twitch} missing")

        install_from_extracted(snap_twitch, dest_root)

    print(f"Done. Installed under {dest_root}", flush=True)
    print("Use e.g. -d Twitch --src_name EN --tgt_name DE (langs: DE EN ES FR PT RU).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
