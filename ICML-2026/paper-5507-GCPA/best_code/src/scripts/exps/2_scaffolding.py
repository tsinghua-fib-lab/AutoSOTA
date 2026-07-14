import omegaconf
import pandas as pd
from pathlib import Path
import torch
from latentis.space._base import Space
from latentis.data.dataset import DatasetView
from typing import Any, Dict, Tuple
import torch.nn.functional as F
from latentis.data import PROJECT_ROOT

import os
import logging
from scripts.exp_utils import pca_match, feat_id_mean, z
from scripts.exp_utils import ensure_normalised, embed_dataset, get_translators
from scripts import dataset_utils
import hydra

from latentis.transform.translate.aligner import MatrixAligner, Translator
from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import ZeroPadding
from latentis.transform.translate.functional import svd_align_state

from cycloreps.utils.validation_tests import cycle_error, cycle_consistency
from cycloreps.utils.utils import seed_everything
from cycloreps.translator.translator import MultiSpaceBase
from cycloreps.translator.identity import IdentityTranslator
from cycloreps import FULL_TO_SHORT_NAMES
import numpy as np

# Configure the logging system
pylogger = logging.getLogger(__name__)


@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="2_scaffolding.yaml")
def main(cfg: omegaconf.DictConfig):

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    seed_everything(cfg.experiment.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = cfg.experiment.dataset
    data: DatasetView = dataset_utils.get_flickr(
        root_path=cfg.experiment.dataset_root, name=dataset_name
    )

    splits = cfg.splits
    run_umap = False
    umap_encoders = []
    for exp_name, exp_cfg in cfg.experiments.items():
        logging.info(f"Running experiment: {exp_name}")
        encoder_list = exp_cfg["encoders"]
        modes_list = exp_cfg["modes"]
        if modes_list[:3] == ['audio', 'text', 'image']:
            run_umap = True

        full_to_short_names = {
            k: v for k, v in FULL_TO_SHORT_NAMES.items() if k in encoder_list
        }
        spaces = {split: {} for split in splits}
        for encoder_name, mode in zip(
            encoder_list, modes_list
        ):  # align encoders with modes
            if mode == 'audio':
                batch_size = 1
            else:
                batch_size = 32
            tr_space = embed_dataset(
                data=data,
                dataset_name=dataset_name,
                encoder_name=encoder_name,
                target_dir = Path(cfg.experiment.data_dir) / mode / encoder_name.replace("/", "-") / "train",
                feature_name=mode,
                batch_size=batch_size,
                split="train",
                num_samples=None,
                seed=cfg.experiment.seed,
                mode=mode,
                device=device,
            )
            val_space = embed_dataset(
                data=data,
                dataset_name=dataset_name,
                encoder_name=encoder_name,
                target_dir = Path(cfg.experiment.data_dir) / mode / encoder_name.replace("/", "-") / "validation",
                feature_name=mode,
                batch_size=batch_size,
                split="validation",
                num_samples=None,
                seed=cfg.experiment.seed,
                mode=mode,
                device=device,
            )
            te_space = embed_dataset(
                data=data,
                dataset_name=dataset_name,
                encoder_name=encoder_name,
                target_dir = Path(cfg.experiment.data_dir) / mode / encoder_name.replace("/", "-") / "test",
                feature_name=mode,
                batch_size=batch_size,
                split="test",
                num_samples=None,
                seed=cfg.experiment.seed,
                mode=mode,
                device=device,
            )
            spaces["train"][f"{full_to_short_names[encoder_name]}_{mode}"] = tr_space
            spaces["validation"][
                f"{full_to_short_names[encoder_name]}_{mode}"
            ] = val_space
            spaces["test"][f"{full_to_short_names[encoder_name]}_{mode}"] = te_space
            if run_umap and len(umap_encoders) < 3:
                umap_encoders += [f'{full_to_short_names[encoder_name]}_{mode}']

        spaces = {
            split: {
                encoder: space.as_tensor().to(device)
                for encoder, space in spaces[split].items()
            }
            for split in splits
        }

        # get the representation dimensions of the spaces
        dims = set(
            [
                space.shape[1]
                for split_dict in spaces.values()
                for space in split_dict.values()
            ]
        )

        # min dim will be used for out_dims in linear + ortho
        min_dim = min(dims)
        pca_flag_string = "pca_disabled"
        if len(set(dims)) > 1 and cfg.experiment.pca_enabled:
            spaces = pca_match(spaces=spaces, min_dim=min_dim)
            pca_flag_string = "pca_enabled"

        # mean of representations
        pooled_ids = {}
        for encoder_name, mode in zip(encoder_list, modes_list):
            key = f"{full_to_short_names[encoder_name]}_{mode}"
            for split in splits:
                feats_train = spaces[split][key]  # per-image train representations
                img_ids = data.hf_dataset[split][
                    "img_id"
                ]  # aligned with train representations
                means, uniq_ids = feat_id_mean(
                    feats_train, img_ids
                )  # collapse to PID means
                spaces[split][key] = means  # overwrite with prototypes
                pooled_ids[(split, key)] = uniq_ids
        
        for split in splits:
            ref_key = next(iter(spaces[split].keys()))
            ref_ids = pooled_ids[(split, ref_key)]
            for k in spaces[split].keys():
                assert list(pooled_ids[(split, k)]) == list(ref_ids), f"ID order mismatch: {split} {k} vs {ref_key}"


        results_init = {
            "INPUT": {},
            "GCCA_UNIVERSE": {},
            "GCCA_TRANSFORMED": {},
            "GPA_UNIVERSE": {},
            "GPA_TRANSFORMED": {},
            "GCPA_UNIVERSE": {},
            "GCPA_TRANSFORMED": {},
            "PW_TRANSFORMED": {},
        }
        results = results_init.copy()
        for method, _ in results_init.items():
            for encoder_name, space in spaces["train"].items():
                results[method][encoder_name] = {}

        translator_gpa = get_translators(
            spaces=spaces,
            alignment_method='generalised_procrustes',
            device=device,
            gc_enabled=False,
        )
        results = do_retrieval(translator=translator_gpa, spaces=spaces, results=results, alignment='generalised_procrustes')

        translator_gpa_corrected = get_translators(
            spaces=spaces,
            alignment_method='generalised_procrustes',
            device=device,
            gc_enabled=True,
        )
        results = do_retrieval(
            translator=translator_gpa_corrected,
            spaces=spaces,
            results=results,
            alignment='generalised_procrustes_corrected',
        )

        translator_gcca = get_translators(spaces=spaces, alignment_method='generalised_cca', device=device)
        results = do_retrieval(translator=translator_gcca, spaces=spaces, results=results, alignment='generalised_cca')

        try:
            # gpa alignment -- cycle consistent
            cycle_method = 'gpa'
            cycle_error(translator=translator_gpa, spaces=spaces, splits=splits)
            cycle_consistency(translator=translator_gpa, spaces=spaces, splits=splits)
        except Exception as e:
            logging.info(f"{cycle_method} cycle consistency check failed with error: {e}")
        
        out_dir = os.path.join(PROJECT_ROOT, cfg.experiment.output_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if run_umap:
            # ensure encoders exist in spaces (avoid missing key errors)
            available = list(spaces["test"].keys())
            audio_candidates = [k for k in available if k.endswith("_audio")]
            text_candidates = [k for k in available if k.endswith("_text")]
            image_candidates = [k for k in available if k.endswith("_image")]
            if not (audio_candidates and text_candidates and image_candidates):
                logging.info("Skipping triplet stats: missing audio/text/image encoders in spaces.")
                continue
            audio_encoder_name = audio_candidates[0]
            text_encoder_name = text_candidates[0]
            image_encoder_name = image_candidates[0]
            stats_by_method = {}
            stats_by_method["GPA"] = compute_triplet_stats(
                spaces,
                image_encoder_name=image_encoder_name,
                text_encoder_name=text_encoder_name,
                audio_encoder_name=audio_encoder_name,
                translator=translator_gpa,
                seed=cfg.experiment.seed,
            )
            stats_by_method["GCPA"] = compute_triplet_stats(
                spaces,
                image_encoder_name=image_encoder_name,
                text_encoder_name=text_encoder_name,
                audio_encoder_name=audio_encoder_name,
                translator=translator_gpa_corrected,
                seed=cfg.experiment.seed,
            )
            stats_by_method["GCCA"] = compute_triplet_stats(
                spaces,
                image_encoder_name=image_encoder_name,
                text_encoder_name=text_encoder_name,
                audio_encoder_name=audio_encoder_name,
                translator=translator_gcca,
                seed=cfg.experiment.seed,
            )
            rows = []
            for method, stats in stats_by_method.items():
                agreement = stats["agreement"]
                pos = stats["pos"]
                rows.append(
                    {
                        "Method": method,
                        "Delta_Pos_Input": pos["delta_pos_in"],
                        "Delta_Pos_Universe": pos["delta_pos_univ"],
                        "Gamma_Input": agreement["gamma_in_mean"],
                        "Gamma_Universe": agreement["gamma_univ_mean"],
                        "Gamma10_Input": agreement["gamma10_in"],
                        "Gamma10_Universe": agreement["gamma10_univ"],
                    }
                )
            pd.DataFrame(rows).to_csv(
                Path(out_dir) / f"{exp_name}_{mode}_{pca_flag_string}_triplet_margin.csv",
                index=False,
            )

        rows = []
        for method, stats in results.items():
            for train_encoder, test_dict in stats.items():
                for test_encoder, metrics in test_dict.items():
                    rows.append(
                        {
                            "Alignment": method,
                            "Encoder FROM": train_encoder,
                            "Encoder TO": test_encoder,
                            "Test Recall": f"{metrics['test_recall']:.4f}",
                            "Mean Cosine": f"{metrics['test_meancos']:.4f}",
                        }
                    )

        df = pd.DataFrame(rows)
        df.to_csv(Path(out_dir) / f"{exp_name}_{mode}_{pca_flag_string}.csv")
        logging.info("\TOPk Results:\n")
        logging.info(df.to_markdown(index=False))
    logging.info("done")


def get_topk(src, tgt):
    sims = ensure_normalised(src) @ ensure_normalised(tgt).T
    recalls = {}
    ks = (1, 5, 10)
    for k in ks:
        correct = 0
        for i in range(sims.size(0)):
            # rank target items by similarity
            topk = torch.topk(sims[i], k=k).indices
            if i in topk:  # assumes aligned indices (caption_i ↔ image_i)
                correct += 1
        recalls[f"R@{k}"] = correct / sims.size(0)

    return recalls

def do_retrieval(
    translator: MultiSpaceBase,
    spaces: Dict[str, Space],
    results: Dict,
    alignment="ortho",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for encoder_name in spaces["test"].keys():
        for other_encoder_name in spaces["test"].keys():
            if encoder_name == other_encoder_name:  # skip
                continue
            try:
                # for computing initial cosine similarities -- needs to be verified
                translator_pairwise = Translator(
                    aligner=IdentityTranslator(),
                    x_transform=StandardScaling(),
                    y_transform=StandardScaling(),
                    dim_matcher=ZeroPadding(),
                )
                translator_pairwise.fit(
                    x=spaces["train"][encoder_name],
                    y=spaces["train"][other_encoder_name],
                )
                # we need to extract the z-score of source and target once and apply it across
                x_src_mu = spaces["train"][encoder_name].mean(0, keepdim=True)
                x_src_sig = spaces["train"][encoder_name].std(0, unbiased=False, keepdim=True) + 1e-8

                x_tgt_mu = spaces["train"][other_encoder_name].mean(0, keepdim=True)
                x_tgt_sig = spaces["train"][other_encoder_name].std(0, unbiased=False, keepdim=True) + 1e-8

                val_src_translated = translator_pairwise.transform(
                    spaces["validation"][encoder_name]
                )["x"]
                # if translated, we will apply a z-score of target
                val_src_translated = z(val_src_translated, mu=x_tgt_mu, sig=x_tgt_sig)


                test_src_translated = translator_pairwise.transform(
                    spaces["test"][encoder_name]
                )["x"]

                # if translated, we will apply a z-score of target
                test_src_translated = z(test_src_translated, mu=x_tgt_mu, sig=x_tgt_sig)

                mean_cos_val = (
                    F.cosine_similarity(val_src_translated, z(spaces["validation"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig)).mean().item()
                )
                mean_cos_test = (
                    F.cosine_similarity(
                        test_src_translated, z(spaces["test"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig)
                    )
                    .mean()
                    .item()
                )

                val_recalls = get_topk(z(spaces["validation"][encoder_name], mu=x_src_mu, sig=x_src_sig),
                                    z(spaces["validation"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig))

                test_recalls = get_topk(z(spaces["test"][encoder_name], mu=x_src_mu, sig=x_src_sig), 
                                    z(spaces["test"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig))

                if other_encoder_name not in results["INPUT"][encoder_name].keys():
                    results["INPUT"][encoder_name][other_encoder_name] = {
                        "val_recall": val_recalls["R@1"],
                        "test_recall": test_recalls["R@1"],
                        "val_recall_5": val_recalls["R@5"],
                        "test_recall_5": test_recalls["R@5"],
                        "val_meancos": mean_cos_val,
                        "test_meancos": mean_cos_test,
                    }

            except Exception as e:
                logging.info(e)
                if other_encoder_name not in results["INPUT"][encoder_name].keys():
                    results["INPUT"][encoder_name][other_encoder_name] = {
                        "val_recall": -1,
                        "test_recall": -1,
                        "val_recall_5": -1,
                        "test_recall_5": -1,
                        "val_meancos": mean_cos_val,
                        "test_meancos": mean_cos_test,
                    }


            # z-score already applied for the universal maps

            # translate both to the universe now
            val_src_translated = translator.to_universe(
                x=spaces["validation"][encoder_name], src=encoder_name
            )
            val_tgt_translated = translator.to_universe(
                x=spaces["validation"][other_encoder_name], src=other_encoder_name
            )
            val_recalls = get_topk(val_src_translated, val_tgt_translated)
            mean_cos_val = (
                F.cosine_similarity(val_src_translated, val_tgt_translated)
                .mean()
                .item()
            )

            test_src_translated = translator.to_universe(
                x=spaces["test"][encoder_name], src=encoder_name
            )
            test_tgt_translated = translator.to_universe(
                x=spaces["test"][other_encoder_name], src=other_encoder_name
            )
            test_recalls = get_topk(test_src_translated, test_tgt_translated)
            mean_cos_test = (
                F.cosine_similarity(test_src_translated, test_tgt_translated)
                .mean()
                .item()
            )
            if alignment == "generalised_cca":
                results["GCCA_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "val_recall": val_recalls["R@1"],
                    "test_recall": test_recalls["R@1"],
                    "val_recall_5": val_recalls["R@5"],
                    "test_recall_5": test_recalls["R@5"],
                    "val_meancos": mean_cos_val,
                    "test_meancos": mean_cos_test,
                }
            elif alignment == "generalised_procrustes":
                results["GPA_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "val_recall": val_recalls["R@1"],
                    "test_recall": test_recalls["R@1"],
                    "val_recall_5": val_recalls["R@5"],
                    "test_recall_5": test_recalls["R@5"],
                    "val_meancos": mean_cos_val,
                    "test_meancos": mean_cos_test,
                }
            elif alignment == "generalised_procrustes_corrected":
                results["GCPA_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "val_recall": val_recalls["R@1"],
                    "test_recall": test_recalls["R@1"],
                    "val_recall_5": val_recalls["R@5"],
                    "test_recall_5": test_recalls["R@5"],
                    "val_meancos": mean_cos_val,
                    "test_meancos": mean_cos_test,
                }
            else:
                raise ValueError("Incorrect alignment method passed!")

            # now transform completely to the other space and do retrieval
            val_src_translated = translator.transform(
                x=spaces["validation"][encoder_name],
                src=encoder_name,
                tgt=other_encoder_name,
            )
            val_src_translated = z(X=val_src_translated, mu=x_tgt_mu, sig=x_tgt_sig)

            test_src_translated = translator.transform(
                x=spaces["test"][encoder_name], src=encoder_name, tgt=other_encoder_name
            )
            test_src_translated = z(X=test_src_translated, mu=x_tgt_mu, sig=x_tgt_sig)

            val_recalls = get_topk(val_src_translated, 
                                   z(X=spaces["validation"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig))
            mean_cos_val = (
                F.cosine_similarity(val_src_translated, z(spaces["validation"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig)).mean().item()
            )

            test_recalls = get_topk(test_src_translated, z(spaces["test"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig))
            mean_cos_test = (
                F.cosine_similarity(test_src_translated, z(spaces["test"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig)).mean().item()
            )

            if alignment == "generalised_cca":
                results["GCCA_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "val_recall": val_recalls["R@1"],
                    "test_recall": test_recalls["R@1"],
                    "val_recall_5": val_recalls["R@5"],
                    "test_recall_5": test_recalls["R@5"],
                    "val_meancos": mean_cos_val,
                    "test_meancos": mean_cos_test,
                }
            elif alignment == "generalised_procrustes":
                results["GPA_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "val_recall": val_recalls["R@1"],
                    "test_recall": test_recalls["R@1"],
                    "val_recall_5": val_recalls["R@5"],
                    "test_recall_5": test_recalls["R@5"],
                    "val_meancos": mean_cos_val,
                    "test_meancos": mean_cos_test,
                }
            elif alignment == "generalised_procrustes_corrected":
                results["GCPA_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "val_recall": val_recalls["R@1"],
                    "test_recall": test_recalls["R@1"],
                    "val_recall_5": val_recalls["R@5"],
                    "test_recall_5": test_recalls["R@5"],
                    "val_meancos": mean_cos_val,
                    "test_meancos": mean_cos_test,
                }
            else:
                raise ValueError("Incorrect alignment method passed!")

            if other_encoder_name not in results["PW_TRANSFORMED"][encoder_name].keys():
                translator_ortho = Translator(
                    aligner=MatrixAligner(name="ortho", align_fn_state=svd_align_state),
                    x_transform=StandardScaling(),
                    y_transform=StandardScaling(),
                    dim_matcher=ZeroPadding(),
                )

                # fit only once -- train space of outer encoder to inner encoder
                translator_ortho.fit(
                    x=spaces["train"][encoder_name],
                    y=spaces["train"][other_encoder_name],
                )
                val_src_translated = translator_ortho.transform(
                    spaces["validation"][encoder_name]
                )["x"]
                # apply z-score of tgt
                val_src_translated = z(X=val_src_translated, mu=x_tgt_mu, sig=x_tgt_sig)
                
                val_recalls = get_topk(
                    val_src_translated, z(spaces["validation"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig)
                )
                mean_cos_val = (
                    F.cosine_similarity(
                        val_src_translated, 
                        z(spaces["validation"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig)
                    )
                    .mean()
                    .item()
                )

                test_src_translated = translator_ortho.transform(
                    spaces["test"][encoder_name]
                )["x"]
                test_src_translated = z(X=test_src_translated, mu=x_tgt_mu, sig=x_tgt_sig)

                test_recalls = get_topk(
                    test_src_translated, 
                    z(spaces["test"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig)
                )
                mean_cos_test = (
                    F.cosine_similarity(
                        test_src_translated, 
                        z(spaces["test"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_sig)
                    )
                    .mean()
                    .item()
                )
                results["PW_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "val_recall": val_recalls["R@1"],
                    "test_recall": test_recalls["R@1"],
                    "val_meancos": mean_cos_val,
                    "test_meancos": mean_cos_test,
                }

    return results

def compute_triplet_stats(
    spaces,
    image_encoder_name,
    text_encoder_name,
    audio_encoder_name,
    translator,
    seed,
):
    image_space = ensure_normalised(translator.to_universe(spaces["test"][image_encoder_name], src=image_encoder_name))
    text_space = ensure_normalised(translator.to_universe(spaces["test"][text_encoder_name], src=text_encoder_name))
    audio_space = ensure_normalised(translator.to_universe(spaces["test"][audio_encoder_name], src=audio_encoder_name))

    # return triplet margin stats for combined reporting
    def _agreement(zI, zT, zA):
        zI = ensure_normalised(zI)
        zT = ensure_normalised(zT)
        zA = ensure_normalised(zA)
        mean_dir = (zI + zT + zA) / 3.0
        return torch.norm(mean_dir, dim=1)

    def _pos_distance(zI, zT, zA):
        zI = ensure_normalised(zI)
        zT = ensure_normalised(zT)
        zA = ensure_normalised(zA)
        pos_it = 1.0 - (zI * zT).sum(dim=1)
        pos_ia = 1.0 - (zI * zA).sum(dim=1)
        pos_ta = 1.0 - (zT * zA).sum(dim=1)
        return ((pos_it + pos_ia + pos_ta) / 3.0)

    gamma_in = _agreement(
        spaces["test"][image_encoder_name],
        spaces["test"][text_encoder_name],
        spaces["test"][audio_encoder_name],
    )
    gamma_univ = _agreement(image_space, text_space, audio_space)
    gamma_in_np = gamma_in.detach().cpu().numpy()
    gamma_univ_np = gamma_univ.detach().cpu().numpy()
    delta_pos_in = _pos_distance(
        spaces["test"][image_encoder_name],
        spaces["test"][text_encoder_name],
        spaces["test"][audio_encoder_name],
    ).mean().item()
    delta_pos_univ = _pos_distance(
        image_space,
        text_space,
        audio_space,
    ).mean().item()

    return {
        "pos": {
            "delta_pos_in": delta_pos_in,
            "delta_pos_univ": delta_pos_univ,
        },
        "agreement": {
            "gamma_in_mean": gamma_in.mean().item(),
            "gamma_univ_mean": gamma_univ.mean().item(),
            "gamma10_in": float(np.quantile(gamma_in_np, 0.90).item()),
            "gamma10_univ": float(np.quantile(gamma_univ_np, 0.90).item())
,
        },
    }

if __name__ == "__main__":
    main()
