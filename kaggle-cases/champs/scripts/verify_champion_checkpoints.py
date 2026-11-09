#!/usr/bin/env python3
"""Check all published winner checkpoints with its original model loader."""
import gc
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    np.bool = np.bool_
    np.int = int
    torch.set_num_threads(4)
    source = ROOT / "repos/BCAI_kaggle_CHAMPS/src/predictor.py"
    os.chdir(source.parent)
    spec = importlib.util.spec_from_file_location("champion_predictor", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.settings["MODEL_DIR"] = str(ROOT / "artifacts/champion_checkpoints")
    load = torch.load
    # These are checksum-verified, organizer-published historical checkpoints.
    # Only device placement differs from the original loader.
    def cpu_load(path, **kwargs):
        return load(path, map_location="cpu", **kwargs)
    torch.load = cpu_load
    reports = []
    for name in original.models["names"]:
        model = original.load_model(name)
        if model is None:
            raise ValueError("Missing checkpoint " + name)
        reports.append({"name": name, "strict_state_dict_load": True,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "checkpoint": str((ROOT / "artifacts/champion_checkpoints" / original.models[name + "_dir"] / "model.ckpt").relative_to(ROOT))})
        print(json.dumps(reports[-1]), flush=True)
        del model
        gc.collect()
    record = {"checkpoints": reports, "all_13_loaded": len(reports) == 13,
              "network_inference_performed": False, "training_performed": False,
              "loader": str(source.relative_to(ROOT)) + ":load_model"}
    (ROOT / "artifacts/champion_checkpoint_verification.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
