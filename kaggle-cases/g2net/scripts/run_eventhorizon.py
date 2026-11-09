"""Run the original EventHorizon trainer with a packed-data loading adapter.

Architecture, loss, transforms, optimizer, schedule and five-fold splitter are
from upstream configs. Optional epoch/batch overrides are recorded explicitly.
Notifications from the original repository are disabled locally.
"""
import argparse
import copy
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from upstream_models import ROOT, EVENTHORIZON, eventhorizon_config


class PackedSignalCache(dict):
    def __init__(self, directory):
        super().__init__()
        self.arrays = {}
        for split in ["train", "test"]:
            self.arrays[split] = np.load(directory / f"{split}.npy", mmap_mode="r")
            for index in range(len(self.arrays[split])):
                self[f"{split}:{index}"] = (split, index)

    def __getitem__(self, key):
        split, index = super().__getitem__(key)
        return (self.arrays[split][index].astype(np.float64) / 1e20).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Seq03aug3")
    parser.add_argument("--packed", type=Path, default=ROOT / "data/packed")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--experiment-name")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--fixture", action="store_true", help="Software check on generated fixtures; never a competition score")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    # The upstream helper maps a GPU-ID list to an unindexed 'cuda' device in
    # single-device mode. Explicitly select the requested current CUDA device.
    torch.cuda.set_device(args.gpu)
    if not (args.packed / "metadata.json").exists():
        raise FileNotFoundError("Official packed data is unavailable. First accept Kaggle rules and run scripts/download_data.sh.")
    data_metadata = json.loads((args.packed / "metadata.json").read_text())
    if bool(data_metadata.get("fixture", False)) != args.fixture:
        raise ValueError("The fixture flag must match the input data; generated fixtures are never competition results.")
    base = eventhorizon_config(args.config)
    from datasets import G2NetDataset
    import configs
    import utils
    cache = PackedSignalCache(args.packed)

    class PackedDataset(G2NetDataset):
        def __init__(self, *arguments, **kwargs):
            kwargs["cache"] = cache
            super().__init__(*arguments, **kwargs)

    prefix = "fixture" if args.fixture else "reproduction"
    name = args.experiment_name or f"{prefix}_{base.name}"
    if Path(name).name != name or (args.fixture and not name.startswith('fixture')):
        raise ValueError('Invalid experiment name')
    metadata_dir = ROOT / "experiments" / name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        table = pd.read_csv(args.packed / f"{split}.csv", dtype={"id": str})
        table["path"] = [f"{split}:{i}" for i in range(len(table))]
        temporary = metadata_dir / f"{split}.{os.getpid()}.tmp.csv"
        table.to_csv(temporary, index=False)
        temporary.replace(metadata_dir / f"{split}.csv")
    parameters = copy.deepcopy(base.model_params)
    if "pretrained" in parameters:
        parameters["pretrained"] = args.pretrained
    overrides = dict(name=name, train_path=metadata_dir / "train.csv", test_path=metadata_dir / "test.csv",
                     train_cache=None, test_cache=None, dataset=PackedDataset, model_params=parameters)
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    local_config = type("LocalReproduction", (base,), overrides)
    configs.LocalReproduction = local_config
    # The requested work is local training and Kaggle submissions, with no messaging.
    utils.notify_me = lambda *a, **kw: None
    # Parallelize source dataset reads without editing the upstream trainer.
    import torch.utils.data as data_module
    original_loader = data_module.DataLoader
    def parallel_loader(*arguments, **kwargs):
        kwargs['num_workers'] = args.workers
        kwargs['pin_memory'] = args.workers > 0
        if args.workers > 0:
            kwargs['persistent_workers'] = True
        return original_loader(*arguments, **kwargs)
    data_module.DataLoader = parallel_loader
    info = vars(args).copy()
    info.update({"packed": str(args.packed), "result_name": name,
                 "competition_score": None, "trainer": "original EventHorizon train.py",
                 "adaptation": "memory-mapped data source, parallel loading and notification suppression"})
    (metadata_dir / f"launch_fold{args.fold}.json").write_text(json.dumps(info, indent=2))
    os.chdir(ROOT)
    sys.argv = [str(EVENTHORIZON / "train.py"), "--config", "LocalReproduction", "--gpu", str(args.gpu), "--progress_bar"]
    if args.inference:
        sys.argv.append("--inference")
    else:
        sys.argv.extend(["--limit_fold", str(args.fold)])
    if args.tta:
        sys.argv.append("--tta")
    runpy.run_path(str(EVENTHORIZON / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
