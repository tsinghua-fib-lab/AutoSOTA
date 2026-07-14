from latentis.data.dataset import DatasetView
from latentis.data.processor import build_market1501_processor
from latentis.data.dataset import HFDatasetView
import torch
from latentis.data.encoding.encode import EncodeTask
from latentis.data.utils import default_collate
from latentis.data import PROJECT_ROOT
from latentis.nn.encoders import ImageHFEncoder
from latentis.space._base import Space
from datasets import DatasetDict
from latentis.transform.translate.functional import svd_align_state
from collections import Counter, defaultdict
import numpy as np
from latentis.transform.translate.aligner import MatrixAligner, Translator
from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import ZeroPadding
import os
import hydra
import pandas as pd
from pathlib import Path
from cycloreps import FULL_TO_SHORT_NAMES
from cycloreps.utils.utils import seed_everything
from cycloreps.utils.validation_tests import cycle_error, cycle_consistency
from scripts.exp_utils import pca_match, get_translators, z
from scripts.exp_utils import ensure_normalised
import omegaconf
import logging

# Configure the logging system
logging.basicConfig(level=logging.INFO)

def top_cameras(dataset_view, cam_column="camid", pid_column="y", num_cams=4):
    hf_ds = dataset_view.hf_dataset

    # top num_cams cameras from train -- this is 4 here
    cam_counts = Counter(hf_ds["train"][cam_column])
    top_cams = [cam for cam, _ in cam_counts.most_common(num_cams)]
    logging.info(f"Top-{num_cams} cameras (by train freq): {top_cams}")

    #  common pids cams across splits
    pid_to_cams_global = defaultdict(set)
    for split_name, split_ds in hf_ds.items():
        for pid, cam in zip(split_ds[pid_column], split_ds[cam_column]):
            if cam in top_cams:
                pid_to_cams_global[pid].add(cam)
    valid_pids_global = {
        pid for pid, cams in pid_to_cams_global.items() if set(cams) == set(top_cams)
    }
    logging.info(
        f"PIDs present in ALL top-{num_cams} cams (global): {len(valid_pids_global)}"
    )

    # query-gallery PID overlap
    query_pids = set(hf_ds["query"][pid_column])
    gallery_pids = set(hf_ds["gallery"][pid_column])
    valid_eval_pids = query_pids & gallery_pids
    logging.info(
        f"Query PIDs: {len(query_pids)}, Gallery PIDs: {len(gallery_pids)}, Overlap: {len(valid_eval_pids)}"
    )

    # final valid set per split
    valid_pids = {
        "train": valid_pids_global,
        "query": valid_eval_pids,
        "gallery": valid_eval_pids,
    }

    # Count images per (split, cam, pid)
    counts = {
        split_name: {cam: defaultdict(int) for cam in top_cams} for split_name in hf_ds
    }
    for split_name, split_ds in hf_ds.items():
        for pid, cam in zip(split_ds[pid_column], split_ds[cam_column]):
            if pid in valid_pids[split_name] and cam in top_cams:
                counts[split_name][cam][pid] += 1

    # per-camera filtered datasets
    cam_views = {}
    for cam in top_cams:
        cam_splits = {}
        for split_name, split_ds in hf_ds.items():
            allowed = valid_pids[split_name]
            ds_cam = split_ds.filter(
                lambda ex: ex[pid_column] in allowed and ex[cam_column] == cam
            )

            # balance query/gallery by minimum count across cams -- this will help in a fair retrieval experiment
            pid_min_count = {
                pid: min(counts[split_name][c][pid] for c in top_cams)
                for pid in allowed
                if pid in counts[split_name][cam]
            }

            pid_to_idxs = defaultdict(list)
            for idx, pid in enumerate(ds_cam[pid_column]):
                pid_to_idxs[pid].append(idx)

            selected = []
            for pid in sorted(pid_to_idxs):
                selected.extend(sorted(pid_to_idxs[pid])[:pid_min_count[pid]])

            cam_splits[split_name] = ds_cam.select(selected)
            logging.info(
                f"Cam {cam}, Split {split_name}: {len(cam_splits[split_name])} images, {len(set(cam_splits[split_name][pid_column]))} PIDs"
            )

        if "query" in cam_splits and "gallery" in cam_splits:
            q_pids = set(cam_splits["query"][pid_column])
            g_pids = set(cam_splits["gallery"][pid_column])
            overlap = q_pids & g_pids
            if len(overlap) < len(q_pids):  # some queries are dropped
                logging.info(
                    f"Cam {cam}: dropped {len(q_pids - overlap)} queries without gallery match"
                )
            cam_splits["query"] = cam_splits["query"].filter(
                lambda ex: ex[pid_column] in overlap
            )
            cam_splits["gallery"] = cam_splits["gallery"].filter(
                lambda ex: ex[pid_column] in overlap
            )

        cam_views[cam] = HFDatasetView(
            name=f"{dataset_view.metadata['name']}_cam{cam}",
            hf_dataset=DatasetDict(cam_splits),
            id_column=dataset_view.metadata["id_column"],
            features=dataset_view.metadata["features"],
        )

    return cam_views


# retrieval performance computation
def evaluate_rank_map(indices, query_pids, gallery_pids, max_rank=10):
    num_q, _ = indices.shape
    matches = (
        gallery_pids[indices] == query_pids[:, None]
    )  # boolean array to identify matches

    all_cmc = []
    all_AP = []
    valid_queries = 0

    for q_idx in range(num_q):
        hits = matches[q_idx]  # go through the queries
        if not np.any(hits):
            continue  # no valid matches
        valid_queries += 1

        cum_hits = hits.cumsum()
        cmc = (cum_hits > 0).astype(int)
        all_cmc.append(cmc[:max_rank])

        rel_idx = np.flatnonzero(hits)  # ranks where matches occur
        precisions = cum_hits[rel_idx] / (rel_idx + 1)
        AP = precisions.mean()
        all_AP.append(AP)

    assert valid_queries > 0, "Error: all query identities do not appear in gallery"

    return np.mean(all_cmc, axis=0), np.mean(all_AP)

def test_retrieval(
    translator,
    encoder_names,
    cam_views,
    cam_spaces,
    full_to_short_names,
    approach="ORTHOGONAL_UNIVERSE",
):
    results = defaultdict(list)
    for encoder in encoder_names:
        for q_cam, q_view in cam_views.items():
            query_encoder_name = f"{full_to_short_names[encoder]}_{q_cam}"
            # define our src mu and std
            x_src_mu = cam_spaces["train"][query_encoder_name].mean(0, keepdim=True)
            x_src_sig = cam_spaces["train"][query_encoder_name].std(0, unbiased=False, keepdim=True) + 1e-8

            if approach in [
                "ORTHOGONAL_UNIVERSE",
                "GPA_UNIVERSE",
                "GCPA_UNIVERSE",
                "GCCA_UNIVERSE",
            ]:
                query_feats = translator.to_universe(
                    x=cam_spaces["query"][query_encoder_name], src=query_encoder_name
                )
            elif approach in [
                "ORTHOGONAL_TRANSFORMED",
                "GPA_TRANSFORMED",
                "GCPA_TRANSFORMED",
                "GCCA_TRANSFORMED",
                "PW_TRANSFORMED",
                "INITIAL",
            ]:
                query_feats = cam_spaces["query"][query_encoder_name]
                if approach == 'INITIAL':
                    query_feats = z(query_feats, mu=x_src_mu, sig=x_src_sig)
                # otherwise it's anyways going to be processed by the transforms

            query_ds = q_view.hf_dataset["query"]
            query_pids = np.array(query_ds["y"])  # should be the correct pids

            all_gallery_feats = []
            all_gallery_pids = []

            for g_cam, g_view in cam_views.items():
                if g_cam == q_cam:  # skip same cam -- cross-camera design
                    continue
                gallery_encoder_name = f"{full_to_short_names[encoder]}_{g_cam}"
                x_tgt_mu = cam_spaces["train"][gallery_encoder_name].mean(0, keepdim=True)
                x_tgt_sig = cam_spaces["train"][gallery_encoder_name].std(0, unbiased=False, keepdim=True) + 1e-8

                if approach in [
                    "ORTHOGONAL_UNIVERSE",
                    "GPA_UNIVERSE",
                    "GCPA_UNIVERSE",
                    "GCCA_UNIVERSE",
                ]:
                    gallery_feats = translator.to_universe(
                        x=cam_spaces["gallery"][gallery_encoder_name],
                        src=gallery_encoder_name,
                    )
                elif approach in [
                    "ORTHOGONAL_TRANSFORMED",
                    "GPA_TRANSFORMED",
                    "GCPA_TRANSFORMED",
                    "GCCA_TRANSFORMED",
                ]:
                    query_feats = translator.transform(
                        x=cam_spaces["query"][query_encoder_name],
                        src=query_encoder_name,
                        tgt=gallery_encoder_name,
                    )
                    query_feats = z(query_feats, mu=x_tgt_mu, sig=x_tgt_sig)
                    gallery_feats = cam_spaces["gallery"][gallery_encoder_name]
                    gallery_feats = z(gallery_feats, mu=x_tgt_mu, sig=x_tgt_sig)
                elif approach == "PW_TRANSFORMED":
                    translator_ortho = Translator(
                        aligner=MatrixAligner(
                            name="ortho", align_fn_state=svd_align_state
                        ),
                        x_transform=StandardScaling(),
                        y_transform=StandardScaling(),
                        dim_matcher=ZeroPadding(),
                    )

                    translator_ortho.fit(
                        x=cam_spaces["train"][query_encoder_name],
                        y=cam_spaces["train"][gallery_encoder_name],
                    )  # query split should be used to fit
                    query_feats = translator_ortho.transform(
                        x=cam_spaces["query"][query_encoder_name],
                        src=query_encoder_name,
                        tgt=gallery_encoder_name,
                    )["x"]
                    query_feats = z(query_feats, mu=x_tgt_mu, sig=x_tgt_sig)
                    gallery_feats = cam_spaces["gallery"][gallery_encoder_name]
                    gallery_feats = z(gallery_feats, mu=x_tgt_mu, sig=x_tgt_sig)
                else:
                    gallery_feats = cam_spaces["gallery"][gallery_encoder_name]
                    gallery_feats = z(gallery_feats, mu=x_tgt_mu, sig=x_tgt_sig)

                gallery_ds = g_view.hf_dataset["gallery"]
                gallery_pids = np.array(gallery_ds["y"])  # should be the correct pids

                all_gallery_feats.append(gallery_feats)
                all_gallery_pids.extend(gallery_pids)

                # normalise before dot product
                sim_matrix = (
                    ensure_normalised(query_feats) @ ensure_normalised(gallery_feats).T
                )
                # sort based on the descending order of similarities
                indices = sim_matrix.argsort(dim=1, descending=True)

                cmc, mAP = evaluate_rank_map(
                    indices.cpu().numpy(), query_pids, gallery_pids, max_rank=10
                )

                results["entries"].append(
                    {
                        "Model": encoder,
                        "Q": q_cam,
                        "G": g_cam,
                        "Rank-1": cmc[0] * 100,
                        "Rank-5": cmc[4] * 100,
                        "mAP": mAP * 100,
                        "Method": approach,  # keep track of cc_universe / cc_transform etc.
                    }
                )
                logging.info(
                    f"[{full_to_short_names[encoder]} | Q:{q_cam} -> G:{g_cam}] "
                    f"Rank-1={cmc[0]*100:.2f}%, Rank-5={cmc[4]*100:.2f}%, mAP={mAP*100:.2f}%"
                )

            if approach not in [
                "ORTHOGONAL_UNIVERSE",
                "GPA_UNIVERSE",
                "GCPA_UNIVERSE",
                "GCCA_UNIVERSE",
                "INITIAL",
            ]:  # global results can't be reported in pairwise scenarios
                continue

            all_gallery_feats = torch.cat(all_gallery_feats, dim=0)
            all_gallery_pids = np.array(all_gallery_pids)
            # ensure normalisation before dot product
            sim_matrix = (
                ensure_normalised(query_feats) @ ensure_normalised(all_gallery_feats).T
            )
            indices = sim_matrix.argsort(dim=1, descending=True)

            cmc, mAP = evaluate_rank_map(
                indices.cpu().numpy(), query_pids, all_gallery_pids, max_rank=10
            )
            logging.info(
                f"[{full_to_short_names[encoder]} | Q:{q_cam} -> G:ALL] "
                f"Rank-1={cmc[0]*100:.2f}%, Rank-5={cmc[4]*100:.2f}%, mAP={mAP*100:.2f}%"
            )
            
    return results

@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="3_reid.yaml")
def main(cfg: omegaconf.DictConfig):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.experiment.seed)
    data: DatasetView = (
        build_market1501_processor(dataset_path=cfg.experiment.dataset_path)
        .build()
        .run()["dataset_view"]
    )

    cam_views = top_cameras(data)
    encoder_names = cfg.encoders
    splits = cfg.splits

    cam_spaces = {split: {} for split in splits}
    full_to_short_names = {
        k: v for k, v in FULL_TO_SHORT_NAMES.items() if k in encoder_names
    }

    for split in splits:
        for encoder_name in encoder_names:
            encoder = ImageHFEncoder(encoder_name)
            encoder = encoder.eval()
            for cam_id, cam_view in cam_views.items():
                cam_encoder_key = f"{full_to_short_names[encoder_name]}_{cam_id}"
                cam_spaces[split][cam_encoder_key] = {}

                DATA_DIR = Path(cfg.experiment.data_dir)  # to save path
                target_dir = (
                    DATA_DIR
                    / data.name
                    / f"cam_{cam_id}"
                    / encoder_name.replace("/", "-")
                    / split
                )

                if target_dir.exists():
                    logging.info(
                        f"Skipping {cam_encoder_key} for split {split} as target directory already exists: {target_dir}"
                    )
                    cam_spaces[split][cam_encoder_key] = Space.load_from_disk(
                        target_dir
                    )
                    continue

                task = EncodeTask(
                    dataset_view=cam_view,
                    split=split,
                    feature="x",
                    model=encoder,
                    collate_fn=default_collate,
                    encoding_batch_size=32,
                    num_workers=16,
                    pooler=None,
                    save_source_model=False,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    target_path=target_dir,
                    write_every=5,
                )

                task.run()

                cam_spaces[split][cam_encoder_key] = Space.load_from_disk(target_dir)
    
    logging.info(cam_spaces[split].keys())
    cam_spaces = {
        split: {
            encoder: space.as_tensor().to(device)
            for encoder, space in cam_spaces[split].items()
        }
        for split in splits
    }

    dims = set(
        [
            space.shape[1]
            for split_dict in cam_spaces.values()
            for space in split_dict.values()
        ]
    )

    min_dim = min(dims)
    pca_enabled_string = "pca_disabled"
    if len(set(dims)) > 1 and cfg.experiment.pca_enabled:
        cam_spaces = pca_match(cam_spaces, min_dim=min_dim)
        pca_enabled_string = "pca_enabled"

    for encoder in encoder_names:
        for cam_id, view in cam_views.items():
            # check to make sure there are no intersecting pids between train and gallery, and train and query
            assert (
                len(
                    set(view.hf_dataset["gallery"]["y"]).intersection(
                        set(view.hf_dataset["train"]["y"])
                    )
                )
                == 0
            )
            assert (
                len(
                    set(view.hf_dataset["query"]["y"]).intersection(
                        set(view.hf_dataset["train"]["y"])
                    )
                )
                == 0
            )
            # check if all ids are common between gallery and query
            assert len(
                set(view.hf_dataset["gallery"]["y"]).intersection(
                    set(view.hf_dataset["query"]["y"])
                )
            ) == len(set(view.hf_dataset["gallery"]["y"]))

    translator_gpa = get_translators(
        spaces=cam_spaces,
        alignment_method="generalised_procrustes",
        device=device,
        gc_enabled=False,
    )

    translator_gpa_corrected = get_translators(
        spaces=cam_spaces,
        alignment_method="generalised_procrustes",
        device=device,
        gc_enabled=True,
    )

    translator_gcca = get_translators(
        spaces=cam_spaces,
        alignment_method="generalised_cca",
        device=device,
    )

    try:
        # gpa alignment -- cycle consistent
        cycle_method = "gpa"
        cycle_error(translator=translator_gpa, spaces=cam_spaces, splits=splits)
        cycle_consistency(translator=translator_gpa, spaces=cam_spaces, splits=splits)
    except Exception as e:
        logging.info(f"{cycle_method} cycle consistency check failed with error: {e}")

    all_results = []

    methods = [
        "GPA_UNIVERSE",
        "GCPA_UNIVERSE",
        "GCCA_UNIVERSE",
        "PW_TRANSFORMED",
        "INITIAL",
    ]

    
    for method in methods:
        logging.info(f"METHOD: {method}")
        if method in ["GPA_UNIVERSE", "GPA_TRANSFORMED"]:
            results = test_retrieval(
                translator=translator_gpa,
                encoder_names=encoder_names,
                cam_views=cam_views,
                cam_spaces=cam_spaces,
                full_to_short_names=full_to_short_names,
                approach=method,
            )
        elif method in ["GCPA_UNIVERSE", "GCPA_TRANSFORMED"]:
            results = test_retrieval(
                translator=translator_gpa_corrected,
                encoder_names=encoder_names,
                cam_views=cam_views,
                cam_spaces=cam_spaces,
                full_to_short_names=full_to_short_names,
                approach=method,
            )
        elif method in ["GCCA_UNIVERSE", "GCCA_TRANSFORMED"]:
            results = test_retrieval(
                translator=translator_gcca,
                encoder_names=encoder_names,
                cam_views=cam_views,
                cam_spaces=cam_spaces,
                full_to_short_names=full_to_short_names,
                approach=method,
            )
        elif method in ["INITIAL", "PW_TRANSFORMED"]:
            results = test_retrieval(
                translator=None,
                encoder_names=encoder_names,
                cam_views=cam_views,
                cam_spaces=cam_spaces,
                full_to_short_names=full_to_short_names,
                approach=method,
            )
        else:
            raise ValueError("Method not supported!")

        for entry in results["entries"]:
            entry["method"] = method
            all_results.append(entry)

    df_all = pd.DataFrame(all_results)
    out_dir = Path(os.path.join(PROJECT_ROOT, cfg.experiment.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(Path(out_dir) / f"reid_retrieval_{pca_enabled_string}.csv")

if __name__ == "__main__":
    main()
