#!/usr/bin/env python3
"""Prepare published checkpoint assets without modifying source repositories."""
import argparse
import json
import shutil
from pathlib import Path

from extract_assets import extract_selected

ROOT = Path(__file__).resolve().parents[1]


def champion():
    target = ROOT / "artifacts/champion_checkpoints"
    for part, letters in ((1, "ABCDEFG"), (2, "HIJKLM")):
        mapping = {"model_%s/model.ckpt" % x: "model_%s/model.ckpt" % x for x in letters}
        for name in mapping.values():
            (target / name).parent.mkdir(parents=True, exist_ok=True)
        extract_selected(ROOT / ("cache/downloads/team_1_checkpoints_part_%d.tar" % part), target, mapping)
    for letter in "ABCDEFGHIJKLM":
        src = ROOT / ("repos/BCAI_kaggle_CHAMPS/models/model_" + letter)
        dest = target / ("model_" + letter)
        for name in ("config", "config_full", "graph_transformer.py"):
            shutil.copy2(src / name, dest / name)
        shutil.copytree(src / "modules", dest / "modules", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))


def runnerup():
    target = ROOT / "artifacts/runnerup_checkpoint"
    archive = ROOT / "cache/downloads/team_2_checkpoints.tar"
    extract_selected(archive, target, {"loss-5.6396.pth": "loss-5.6396.pth"})
    # Historical optimizer pickles need ordinary torch.load. Verify the exact
    # publisher artifact checksum before that one-time conversion.
    from download_assets import ASSETS, digest
    if digest(archive) != ASSETS["runnerup_checkpoint"][2]:
        raise ValueError("Publisher checkpoint checksum mismatch")
    import torch
    state_path = target / "model_state.pt"
    if not state_path.exists():
        published = torch.load(target / "loss-5.6396.pth", map_location="cpu", weights_only=False)
        torch.save({k: v.detach().cpu() for k, v in published["model"].items()}, state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    from atomic_model import build_model
    model = build_model(state)
    record = {"source": "https://osf.io/3r4zv/download", "checkpoint": "loss-5.6396.pth",
              "strict_state_dict_load": True, "state_tensors": len(state),
              "parameters": sum(p.numel() for p in model.parameters()),
              "state_sha256": digest(state_path)}
    (target / "verification.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=("champion", "runnerup", "all"))
    args = parser.parse_args()
    if args.asset in ("champion", "all"):
        champion()
    if args.asset in ("runnerup", "all"):
        runnerup()
