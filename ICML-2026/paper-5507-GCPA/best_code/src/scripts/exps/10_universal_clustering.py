import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict
from latentis.data.dataset import DatasetView
from latentis.data.processor import DBPedia14, AGNews, IMDB
from latentis.space._base import Space
from latentis.space.vector_source import HDF5Source

from cycloreps import FULL_TO_SHORT_NAMES
from cycloreps.utils.utils import seed_everything
import omegaconf
from latentis.data import PROJECT_ROOT
from scripts.exp_utils import embed_dataset, ensure_normalised, get_translators, load_massive, labels_for_massive, z
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import faiss


logger = logging.getLogger(__name__)


def load_dataset_view(dataset_name: str):
    if dataset_name == "dbpedia14":
        return DBPedia14.build().run()["dataset_view"]
    if dataset_name in {"agnews", "ag_news"}:
        return AGNews.build().run()["dataset_view"]
    if dataset_name in {"IMDB"}:
        return IMDB.build().run()["dataset_view"]
    raise ValueError(f"Unsupported dataset_name={dataset_name}")


def get_embed_feature(dataset_view: DatasetView, dataset_name: str):
    if dataset_name == "dbpedia14":
        return "x"
    if dataset_name in {"agnews", "ag_news"}:
        return "text"
    if dataset_name.startswith("massive_"):
        return "utt"
    # fallback to first feature name if present
    if dataset_view.features:
        return dataset_view.features[0].name
    raise ValueError(f"Cannot infer embed feature for dataset_name={dataset_name}")



def load_space(
    dataset_name: str,
    encoder_name: str,
    split: str,
    data_dir: Path,
    full_to_short_names: Dict[str, str],
):
    logger.info(
        "Loading space for dataset=%s encoder=%s split=%s",
        dataset_name,
        encoder_name,
        split,
    )
    path = (
        data_dir
        / dataset_name
        / "encodings"
        / "sample_complexity"
        / encoder_name.replace("/", "-")
        / split
        / "vectors"
    )
    space = HDF5Source.load_from_disk(path)
    space.encoder_name = full_to_short_names[encoder_name]
    space.dataset_name = dataset_name
    return space


def embed_spaces(
    dataset_views: Dict[str, DatasetView],
    encoder_map: Dict[str, str],
    splits: List[str],
    data_dir: Path,
    full_to_short_names: Dict[str, str],
    embed_feature: str,
    batch_size: int,
    device: str,
):
    spaces = {split: {} for split in splits}
    for key, encoder_name in encoder_map.items():
        ds_view = dataset_views[key]
        dataset_name = ds_view.name
        for split in splits:
            target_dir = (
                data_dir
                / dataset_name
                / "encodings"
                / "sample_complexity"
                / encoder_name.replace("/", "-")
                / split
            )
            space = embed_dataset(
                data=ds_view,
                dataset_name=dataset_name,
                encoder_name=encoder_name,
                target_dir=target_dir,
                feature_name=embed_feature,
                batch_size=batch_size,
                split=split,
                mode="text",
                device=device,
            )
            short_name = full_to_short_names.get(encoder_name, encoder_name)
            space.encoder_name = short_name
            space.dataset_name = dataset_name
            spaces[split][short_name] = space
    return spaces


def build_spaces(
    dataset_name: str,
    encoder_names: List[str],
    splits: List[str],
    data_dir: Path,
    full_to_short_names: Dict[str, str],
):
    return {
        split: {
            full_to_short_names[name]: load_space(
                dataset_name, name, split, data_dir, full_to_short_names
            )
            for name in encoder_names
        }
        for split in splits
    }


def build_space_tensors(spaces, device: str):
    return {
        split: {
            encoder: space.as_tensor(device=device)
            for encoder, space in split_spaces.items()
        }
        for split, split_spaces in spaces.items()
    }


def get_label_column(hf_ds: DatasetDict, split: str = "train"):
    cols = hf_ds[split].column_names
    if "y" in cols:
        return "y"
    if "label" in cols:
        return "label"
    raise ValueError(f"No label column found in {cols}")


def labels_for_space(
    hf_ds: DatasetDict, id_column: str, label_column: str, space, split: str = "train"
):
    ids = [str(i) for i in hf_ds[split][id_column]]
    labels = hf_ds[split][label_column]
    id_to_label = dict(zip(ids, labels))
    space_ids = [str(i) for i in space.keys]
    if len(space_ids) != len(ids):
        logger.warning(
            "ID count mismatch for split=%s: space=%d hf=%d",
            split,
            len(space_ids),
            len(ids),
        )
    if space_ids != ids:
        raise ValueError(
            f"ID order mismatch for split=%s (space.keys vs hf ids).{split}"
        )
    missing = [sid for sid in space_ids if sid not in id_to_label]
    if missing:
        raise ValueError(f"Missing {len(missing)} labels for split={split}")
    return [id_to_label[sid] for sid in space_ids]



def to_universe_map(translator, spaces_train):
    return {
        encoder: translator.to_universe(
            x=space.as_tensor(device=translator.device) if hasattr(space, "as_tensor") else space.to(translator.device),
            src=encoder,
        )
        for encoder, space in spaces_train.items()
    }

def faiss_kmeans(z: np.ndarray, n_clusters: int, seed: int = 42):
    seed_everything(seed)
    kmeans = faiss.Kmeans(
        d=z.shape[1],
        k=n_clusters,
        niter=20,
        nredo=3,
        verbose=False,
        gpu=False,
        seed=seed,
    )
    # use more points for a better centroid estimate
    kmeans.max_points_per_centroid = max(1000, z.shape[0] // max(1, n_clusters))
    kmeans.train(z.astype(np.float32))
    index = faiss.IndexFlatL2(z.shape[1])
    index.add(kmeans.centroids)
    _, labels = index.search(z, 1)
    return labels.ravel()


def cluster_multiple(z: np.ndarray, n_clusters: int, n_seeds: int):
    return [faiss_kmeans(z, n_clusters, seed=seed) for seed in range(n_seeds)]


def eval_clusterings(label_lists: List[np.ndarray], true_labels, tag: str):
    aris = []
    nmis = []

    for i, labels in enumerate(label_lists):
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        print(
            f"{tag} Seed {i} - ARI: {ari:.3f}, NMI: {nmi:.3f}"
        )

        aris.append(ari)
        nmis.append(nmi)

    means = {
        "ARI": float(np.mean(aris)),
        "NMI": float(np.mean(nmis)),
    }
    print(
        f"{tag} Avg - ARI: {means['ARI']:.3f}±{np.std(aris):.3f}, "
        f"NMI: {means['NMI']:.3f}±{np.std(nmis):.3f}"
    )

    return means


def mean_cluster_scores(label_lists: List[np.ndarray], true_labels):
    aris = []
    nmis = []

    for labels in label_lists:
        aris.append(adjusted_rand_score(true_labels, labels))
        nmis.append(normalized_mutual_info_score(true_labels, labels))

    return {
        "ARI": float(np.mean(aris)),
        "NMI": float(np.mean(nmis)),
    }

def main():
    cfg = omegaconf.OmegaConf.load(Path(PROJECT_ROOT / "config" / "10_universal_clustering.yaml"))
    data_dir = Path(cfg.paths.data_dir)
    dataset_name = cfg.dataset.name
    encoder_names = list(cfg.encoders.names)
    splits = list(cfg.dataset.splits)
    eval_split = cfg.dataset.eval_split
    n_clusters_cfg = cfg.dataset.n_clusters
    n_seeds = int(cfg.runtime.n_seeds)
    device = cfg.runtime.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    global_seed = int(cfg.runtime.global_seed)
    embed_dataset_flag = bool(cfg.dataset.embed_dataset)
    embed_feature_cfg = cfg.dataset.embed_feature
    embed_batch_size = int(cfg.dataset.embed_batch_size)
    use_massive = bool(cfg.massive.use_massive)
    massive_dataset_id = cfg.massive.dataset_id
    massive_langs = list(cfg.massive.langs)
    massive_lang_encoders = list(cfg.massive.lang_encoders)

    seed_everything(global_seed)
    if use_massive:
        lang2view = load_massive(
            dataset_id=massive_dataset_id,
            lang_list=massive_langs,
        )
        if not lang2view:
            raise ValueError("No MASSIVE languages available after filtering.")
        if len(massive_lang_encoders) != len(massive_langs):
            raise ValueError(
                "MASSIVE_LANG_ENCODERS must match MASSIVE_LANGS length."
            )
        lang2encoder = dict(zip(massive_langs, massive_lang_encoders))
        encoder_names = list(lang2encoder.values())
        full_to_short_names = {
            k: v for k, v in FULL_TO_SHORT_NAMES.items() if k in encoder_names
        }
        short_names = list(full_to_short_names.values())

        spaces = embed_spaces(
            lang2view,
            lang2encoder,
            splits,
            data_dir,
            full_to_short_names,
            embed_feature="utt",
            batch_size=embed_batch_size,
            device=device,
        )

        ref_lang = next(iter(lang2encoder.keys()))
        ref_view = lang2view[ref_lang]
        hf_ds = ref_view.hf_dataset
        label_column = "intent"
        ref_encoder = lang2encoder[ref_lang]
        ref_key = full_to_short_names.get(ref_encoder, ref_encoder)
        ref_space = spaces[eval_split][ref_key]
        true_labels = labels_for_massive(ref_view, label_column, split=eval_split)
    else:
        dataset_view = load_dataset_view(dataset_name)
        embed_feature = embed_feature_cfg or get_embed_feature(dataset_view, dataset_name)
        hf_ds = dataset_view.hf_dataset
        label_column = get_label_column(hf_ds, split="train")

        full_to_short_names = {
            k: v for k, v in FULL_TO_SHORT_NAMES.items() if k in encoder_names
        }
        short_names = list(full_to_short_names.values())

        if embed_dataset_flag:
            spaces = embed_spaces(
                {name: dataset_view for name in encoder_names},
                {name: name for name in encoder_names},
                splits,
                data_dir,
                full_to_short_names,
                embed_feature,
                batch_size=embed_batch_size,
                device=device,
            )
        else:
            spaces = build_spaces(
                dataset_name,
                encoder_names,
                splits,
                data_dir,
                full_to_short_names,
            )
        ref_key = short_names[0] if short_names[0] in spaces["train"] else next(iter(spaces["train"]))
        ref_space = spaces[eval_split][ref_key]
        true_labels = labels_for_space(
            hf_ds, dataset_view.id_column, label_column, ref_space, split=eval_split
        )

    space_tensors = build_space_tensors(spaces, device=device)


    spaces_train = space_tensors["train"]
    spaces_eval = space_tensors[eval_split]

    n_clusters = n_clusters_cfg or len(set(true_labels))
    if n_clusters <= 1:
        raise ValueError(f"Invalid number of clusters: {n_clusters}")
    if n_clusters > len(true_labels):
        raise ValueError(
            f"n_clusters ({n_clusters}) > number of samples ({len(true_labels)})"
        )

    translator_gpa = get_translators(
        spaces={"train": spaces_train},
        alignment_method="generalised_procrustes",
        device=device,
        gc_enabled=False,
    )
    translator_cycle = get_translators(
        spaces={"train": spaces_train},
        alignment_method="generalised_procrustes",
        device=device,
        gc_enabled=True,
    )
    translator_gcca = get_translators(
        spaces={"train": spaces_train},
        alignment_method="generalised_cca",
        device=device,
    )

    cycle_space_map = to_universe_map(translator_cycle, spaces_eval)
    gpa_space_map = to_universe_map(translator_gpa, spaces_eval)
    gcca_space_map = to_universe_map(translator_gcca, spaces_eval)

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"clustering_{dataset_name}.csv"

    def _cluster_means(x: torch.Tensor, tag: str):
        x_l2 = ensure_normalised(x, name=tag)
        labs = cluster_multiple(x_l2.detach().cpu().numpy(), n_clusters, n_seeds)
        return eval_clusterings(labs, true_labels, tag)

    rows = []
    baseline_by_enc = {}
    baseline_z_by_enc = {}
    for enc, sp in spaces_eval.items():
        x = sp if isinstance(sp, torch.Tensor) else sp.as_tensor()
        baseline_by_enc[enc] = _cluster_means(x, f"Baseline ({enc})")
        train_x = spaces_train[enc]
        mu = train_x.mean(0, keepdim=True)
        sig = train_x.std(0, unbiased=False, keepdim=True) + 1e-8
        x_z = z(x, mu=mu, sig=sig)
        baseline_z_by_enc[enc] = _cluster_means(x_z, f"Baseline+ZScore ({enc})")

    method_maps = {
        "GCPA": cycle_space_map,
        "GPA": gpa_space_map,
        "GCCA": gcca_space_map,
    }

    for enc, base in baseline_by_enc.items():
        rows.append(
            {
                "encoder": enc,
                "method": "Baseline",
                "ARI": base["ARI"],
                "NMI": base["NMI"],
                "delta_ARI": 0.0,
                "delta_NMI": 0.0,
            }
        )
        base_z = baseline_z_by_enc[enc]
        rows.append(
            {
                "encoder": enc,
                "method": "Baseline+ZScore",
                "ARI": base_z["ARI"],
                "NMI": base_z["NMI"],
                "delta_ARI": base_z["ARI"] - base["ARI"],
                "delta_NMI": base_z["NMI"] - base["NMI"],
            }
        )
        for method, space_map in method_maps.items():
            if enc not in space_map:
                continue
            means = _cluster_means(space_map[enc], f"{method} ({enc})")
            rows.append(
                {
                    "encoder": enc,
                    "method": method,
                    "ARI": means["ARI"],
                    "NMI": means["NMI"],
                    "delta_ARI": means["ARI"] - base["ARI"],
                    "delta_NMI": means["NMI"] - base["NMI"],
                }
            )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "encoder",
                "method",
                "ARI",
                "NMI",
                "delta_ARI",
                "delta_NMI",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved clustering metrics to %s", csv_path)


if __name__ == "__main__":
    main()
