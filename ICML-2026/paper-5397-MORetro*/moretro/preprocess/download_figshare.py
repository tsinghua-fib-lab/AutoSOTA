#!/usr/bin/env python3
"""
Script to download files from the MORetro Figshare article.

Usage:
    python download_figshare.py

Downloaded files are written under the repository's ``models/`` directory,
relative to ``Path(__file__).resolve().parents[2]``.

Article: https://figshare.com/articles/software/Data_and_Models_for_MORetro_/30395614
"""

from pathlib import Path

import requests

ARTICLE_ID = "30395614"
FILES_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files"

# Maps each Figshare filename to its subfolder within models/
SUBFOLDER_MAP = {
    "gnn_agent_model.ckpt": "quarc",
    "ffn_reactant_amount_model.ckpt": "quarc",
    "ffn_temperature_model.ckpt": "quarc",
    "ffn_agent_amount_model.ckpt": "quarc",
    "graph2edits.pth": "g2e",
    "config.json": "g2e",
    "atom_lg_vocab.txt": "g2e/vocab",
    "bond_vocab.txt": "g2e/vocab",
    "rollout_model.ckpt": "pdvn",
    "saved_rollout_state_1_2048.ckpt": "pdvn",
    "template_rules_1.dat": "pdvn",
    "model_retro.pt": "template",
    "idx2template_retro.json": "template",
    "std.bin": "objectives",
    "model_toxicity.joblib": "objectives",
    "model_price.pkl": "objectives",
    "model_value.pt": "objectives",
    "model_logp.ckpt": "objectives",
    "origin_dict.csv": "",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def download_figshare_files():
    files = []
    page = 1
    while True:
        response = requests.get(FILES_URL, params={"page": page, "page_size": 100})
        response.raise_for_status()
        page_files = response.json()
        if not page_files:
            break
        files.extend(page_files)
        page += 1

    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files. Starting download...")

    for file_info in files:
        filename = file_info["name"]
        download_url = file_info["download_url"]
        size_mb = file_info.get("size", 0) / (1024 * 1024)

        subfolder = SUBFOLDER_MAP.get(filename, "")
        dest_dir = (
            PROJECT_ROOT / "models" / subfolder
            if subfolder
            else PROJECT_ROOT / "models"
        )
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Downloading {filename} ({size_mb:.1f} MB) -> models/{subfolder or ''}..."
        )
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(dest_dir / filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    print("Download complete!")


def main():
    download_figshare_files()


if __name__ == "__main__":
    main()
