import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from latentis.data.processor import AGNews, DBPedia14
from latentis.data import PROJECT_ROOT
from latentis.data.dataset import DatasetView
from latentis.space.vector_source import HDF5Source

from cycloreps import FULL_TO_SHORT_NAMES
from cycloreps.translator.gpa import GeneralizedProcrustesTranslator
from cycloreps.utils.io_utils import load_space
from cycloreps.utils.utils import seed_everything
from scripts.exp_utils import embed_dataset
import omegaconf

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ["DATA_PATH"])
DATASET_NAME = "dbpedia14"
ENCODER_NAMES = [
    "sentence-transformers/gtr-t5-base",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
]
SPLITS = ["train", "test"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DATASET = True
EMBED_BATCH_SIZE = 1024
SEED = 42
GC_TAU_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
GC_LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_dataset_view(dataset_name: str):
    if dataset_name == "dbpedia14":
        return DBPedia14.build().run()["dataset_view"]
    if dataset_name in {"agnews", "ag_news"}:
        return AGNews.build().run()["dataset_view"]
    raise ValueError(f"Unsupported dataset_name={dataset_name}")


def get_embed_feature(dataset_name: str, dataset_view: DatasetView):
    if dataset_name == "dbpedia14":
        return "x", "text"
    if dataset_name in {"agnews", "ag_news"}:
        return "text", "text"
    if dataset_view.features:
        return dataset_view.features[0].name, "text"
    raise ValueError(f"Cannot infer embed feature for dataset_name={dataset_name}")


def build_spaces(
    dataset_name: str, encoder_names: List[str], splits: List[str], data_dir: Path
):
    full_to_short = {k: v for k, v in FULL_TO_SHORT_NAMES.items() if k in encoder_names}

    return {
        split: {
            full_to_short[enc]: HDF5Source.load_from_disk(
                data_dir
                / dataset_name
                / "encodings"
                / enc.replace("/", "-")
                / split
                / "vectors"
            ).as_tensor(device=DEVICE)
            for enc in encoder_names
        }
        for split in splits
    }


def embed_spaces(
    dataset_view: DatasetView,
    encoder_names: List[str],
    splits: List[str],
    data_dir: Path,
    feature_name: str,
    mode: str,
):
    spaces = {split: {} for split in splits}
    full_to_short = {k: v for k, v in FULL_TO_SHORT_NAMES.items() if k in encoder_names}
    for enc in encoder_names:
        for split in splits:
            target_dir = (
                data_dir
                / dataset_view.name
                / "encodings"
                / enc.replace("/", "-")
                / split
            )
            space = embed_dataset(
                data=dataset_view,
                dataset_name=dataset_view.name,
                encoder_name=enc,
                target_dir=target_dir,
                feature_name=feature_name,
                batch_size=EMBED_BATCH_SIZE,
                split=split,
                mode=mode,
                device=DEVICE,
            )
            spaces[split][full_to_short[enc]] = space.as_tensor(device=DEVICE)
    return spaces


@torch.no_grad()
def drift_stats_function(u_hat: torch.Tensor, u_tilde: torch.Tensor):
    cos = torch.nn.functional.cosine_similarity(u_tilde, u_hat, dim=1)
    drift = 1.0 - cos
    return {
        "mean": float(drift.mean()),
        "median": float(drift.median()),
        "p95": float(torch.quantile(drift, 0.95)),
    }

@torch.no_grad()
def geometry_metrics(
    translator,
    spaces: Dict[str, Dict[str, torch.Tensor]],
    split: str
):
    drift_stats = {"mean": [], "median": [], "p95": []}

    for name, x in spaces[split].items():
        # this is equivalent to _to_universe call
        z = translator._zscore(translator._pad(name=name, x=x), name=name)
        # orthogonal transformation
        u_hat = translator._to_universe_impl(z, src=name, use_gc=False)
        # geometry corrected transformation
        u_tilde = translator._to_universe_impl(z, src=name, use_gc=True)

        d = drift_stats_function(u_hat, u_tilde)
        drift_stats["mean"].append(d["mean"])
        drift_stats["median"].append(d["median"])
        drift_stats["p95"].append(d["p95"])

    return {
        "drift": {k: float(torch.tensor(v).mean()) for k, v in drift_stats.items()},
    }


def main():
    logging.basicConfig(level=logging.INFO)
    seed_everything(SEED)

    if EMBED_DATASET:
        dataset_view = load_dataset_view(DATASET_NAME)
        feature_name, mode = get_embed_feature(DATASET_NAME, dataset_view)
        spaces = embed_spaces(
            dataset_view, ENCODER_NAMES, SPLITS, DATA_DIR, feature_name, mode
        )
    else:
        spaces = build_spaces(DATASET_NAME, ENCODER_NAMES, SPLITS, DATA_DIR)

    align_cfg = omegaconf.OmegaConf.load(Path(PROJECT_ROOT / "config" / "alignment.yaml"))
    gpa = GeneralizedProcrustesTranslator(
        device=DEVICE,
        gc_enabled=False,
        max_iter=align_cfg.procrustes.max_iter,
        tol=align_cfg.procrustes.tol,
    )
    gpa.fit(spaces["train"])
    gcpa = GeneralizedProcrustesTranslator(
        device=DEVICE,
        gc_enabled=True,
        max_iter=align_cfg.procrustes.max_iter,
        tol=align_cfg.procrustes.tol,
    )
    gcpa.fit(spaces["train"])

    for split in SPLITS:
        gpa_metrics = geometry_metrics(gpa, spaces, split)
        gcpa_metrics = geometry_metrics(gcpa, spaces, split)

        print(
            f"{split} | GPA  drift mean={gpa_metrics['drift']['mean']} "
            f"median={gpa_metrics['drift']['median']} p95={gpa_metrics['drift']['p95']}"
        )

        print(
            f"{split} | GCPA drift mean={gcpa_metrics['drift']['mean']} "
            f"median={gcpa_metrics['drift']['median']} p95={gcpa_metrics['drift']['p95']}"
        )

    sweep_results = {}
    for tau in GC_TAU_GRID:
        for lam in GC_LAM_GRID:
            gcpa_sweep = GeneralizedProcrustesTranslator(
                device=DEVICE,
                gc_enabled=True,
                gc_tau=tau,
                gc_lam=lam,
                max_iter=align_cfg.procrustes.max_iter,
                tol=align_cfg.procrustes.tol,
            )
            gcpa_sweep.fit(spaces["train"])
            key = f"tau{tau}_lam{lam}"
            sweep_results[key] = {}
            for split in SPLITS:
                sweep_results[key][split] = geometry_metrics(gcpa_sweep, spaces, split)

    out_dir = Path(PROJECT_ROOT) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "11_geometry_gcpa.json"
    with out_path.open("w") as f:
        json.dump(sweep_results, f, indent=2)


if __name__ == "__main__":
    main()
