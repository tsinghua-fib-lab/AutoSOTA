#!/usr/bin/env python3
"""Extract an explicit allowlist; organizer test targets are never extracted."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "data_train.csv": "train.csv",
    "data_test_masked.csv": "test.csv",
    "structures.csv": "structures.csv",
    "scalar_coupling_contributions_train.csv": "scalar_coupling_contributions.csv",
    "magnetic_shielding_tensors_train.csv": "magnetic_shielding_tensors.csv",
    "mulliken_charges_train.csv": "mulliken_charges.csv",
    "dipole_moments_train.csv": "dipole_moments.csv",
    "potential_energy_train.csv": "potential_energy.csv",
}


def extract_selected(archive, destination, mapping):
    destination.mkdir(parents=True, exist_ok=True)
    found = []
    with tarfile.open(archive, "r:*") as t:
        for member in t:
            if not member.isfile() or member.name not in mapping:
                continue
            target = destination / mapping[member.name]
            if not target.exists() or target.stat().st_size != member.size:
                with t.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            found.append(member.name)
            print(json.dumps({"file": str(target.relative_to(ROOT)), "bytes": member.size}), flush=True)
    missing = set(mapping) - set(found)
    if missing:
        raise ValueError("Missing archive members: %s" % sorted(missing))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=["dataset", "champion_predictions"])
    args = parser.parse_args()
    if args.asset == "dataset":
        extract_selected(ROOT / "cache/downloads/kaggle_dataset.tar.gz", ROOT / "data/raw", DATA_FILES)
    else:
        config = json.loads((ROOT / "repos/BCAI_kaggle_CHAMPS/config/models.json").read_text())
        mapping = {name + ".csv.bz2": name + ".csv.bz2" for name in config["names"]}
        mapping[config["output_file"] + ".gz"] = "published_final.csv.gz"
        extract_selected(ROOT / "cache/downloads/team_1_submissions.tar.gz", ROOT / "artifacts/champion_predictions", mapping)
