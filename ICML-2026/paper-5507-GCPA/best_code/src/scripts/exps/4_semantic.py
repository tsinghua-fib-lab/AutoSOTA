from latentis.data.dataset import DatasetView
from latentis.data.processor import CIFAR10, CIFAR100
from latentis.data.utils import default_collate
from latentis.data import PROJECT_ROOT
from cycloreps import FULL_TO_SHORT_NAMES
from latentis.space._base import Space
import cv2
import numpy as np
from torchvision.utils import save_image
import pandas as pd
import math, torch
from torchvision.utils import save_image
from datasets import DatasetDict
from latentis.data.dataset import HFDatasetView
import time
from cycloreps.utils.utils import seed_everything
from cycloreps.utils.validation_tests import cycle_error, cycle_consistency
from pathlib import Path
import os
import copy
import hydra
import omegaconf
from scripts.exp_utils import get_translators, probing, pairwise_probing
from scripts.exp_utils import embed_dataset, z, train_linear_classifier, test_classifier
from latentis.transform.translate.aligner import MatrixAligner, Translator
from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import ZeroPadding
from latentis.transform.translate.functional import svd_align_state
import hashlib
import logging

# Configure the logging system
logging.basicConfig(level=logging.INFO)


# datasets to work with
datasets_lazy_load = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
}


# so that corruption is reproducible
def _stable_rng(seed, key):
    h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))

# collate function to save images, use only when workers=0 is set in encode task
def overwrite_preview_collate(base_collate, path="image.png", take=16, nrow=None):
    def collate(samples, feature, model, id_column=None):
        # Run the wrapped collate
        b = base_collate(samples, feature, model, id_column)
        assert (
            "proc_out" in b and "pixel_values" in b["proc_out"]
        ), "Expected b['proc_out']['pixel_values'] to exist."

        # Get the tensor actually used by the model
        x = b["proc_out"]["pixel_values"]
        assert isinstance(x, torch.Tensor), "Expected tensor in collate output"

        # Take subset to visualize
        t = min(take, x.size(0))
        cols = nrow or math.ceil(t**0.5)

        # Clamp to valid range [0,1] for visualization
        xx = x[:t].detach().cpu()
        if xx.min() < 0 or xx.max() > 1:  # normalize if required
            xx = xx.clamp(0, 1)

        # Save to disk
        save_image(xx, path, nrow=cols)
        time.sleep(1)  # ensure file write completes

        # Return unmodified batch
        b["proc_out"]["pixel_values"] = x
        return b

    return collate


# different edge detection approaches -- canny, scharr and dog, p represents the probability of edge images in a batch
def edge_collate(
    base_collate,
    enc_key,
    mode="canny",
    auto_k=0.5,
    thicken=1,
    sigma1=1.0,
    sigma2=2.5,
    p=0.5,
    tol=1e-2,
    seed=0
):
    se = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1 + 2 * thicken,) * 2)
        if thicken > 0
        else None
    )

    def collate(samples, feature, model, id_column=None):
        b = base_collate(samples, feature, model, id_column)

        assert (
            "proc_out" in b and "pixel_values" in b["proc_out"]
        ), "Expected b['proc_out']['pixel_values'] to exist."

        x = b["proc_out"]["pixel_values"]  # channels first
        assert x.dtype == torch.float32
        assert x.dtype == torch.float32, "Expected float32 pixel_values."
        mean = torch.tensor(model.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(model.processor.image_std).view(1, 3, 1, 1)

        x_img = x * std + mean
        xmin, xmax = float(x_img.min()), float(x_img.max())
        if not (xmin >= -tol and xmax <= 1.0 + tol):
            raise AssertionError(
                f"Expected processor-normalized inputs; after de-normalization got range [{xmin:.3f},{xmax:.3f}]. "
                "Make sure you pass the output of pre_encode/default_collate (not raw RGB)."
            )

        x_img = x_img.clamp(0, 1)

        xs = (
            (x_img * 255.0)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )  # channels at the end
        ids = [str(s[id_column] if id_column else s["sample_id"]) for s in samples]

        out = []
        for im, sid in zip(xs, ids):
            rng = _stable_rng(seed, f"{enc_key}:{sid}")   # enc_key must differ per encoder
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if mode == "canny":
                v = np.median(g)
                lo = int(max(0, (1 - auto_k) * v))  # lower threshold
                hi = int(min(255, (1 + auto_k) * v))  # higher threshold
                e = cv2.Canny(g, lo, hi, L2gradient=True)
            elif mode == "scharr":
                gx, gy = cv2.Scharr(g, cv2.CV_32F, 1, 0), cv2.Scharr(
                    g, cv2.CV_32F, 0, 1
                )
                e = cv2.normalize(
                    cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
            elif mode == "dog":
                e = cv2.absdiff(
                    cv2.GaussianBlur(g, (0, 0), sigma1),
                    cv2.GaussianBlur(g, (0, 0), sigma2),
                )
                e = cv2.normalize(e, None, 0, 255, cv2.NORM_MINMAX)
            if se is not None:
                e = cv2.dilate(e, se)
            out.append(
                im if rng.random() >= p else np.repeat(e[..., None], 3, axis=-1)
            )

        out = (
            torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).to(x.device)
        )  # channels first again
        out = out.float() / 255.0
        out = (out - mean) / std
        b["proc_out"]["pixel_values"] = out
        return b

    return collate


# gaussian noise collate function -- p represents the probability of noise
def noisy_collate(base_collate, enc_key, blur_sigma=1.6, noise_std=25.0, p=0.5, tol=1e-2, seed=0):
    def collate(samples, feature, model, id_column=None):
        b = base_collate(samples, feature, model, id_column)
        assert (
            "proc_out" in b and "pixel_values" in b["proc_out"]
        ), "Expected b['proc_out']['pixel_values'] to exist."

        x = b["proc_out"]["pixel_values"]
        assert x.dtype == torch.float32, "Expected float32 pixel_values."
        mean = torch.tensor(model.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(model.processor.image_std).view(1, 3, 1, 1)

        # de-normalize to 0–1 RGB
        x_img = x * std + mean
        xmin, xmax = float(x_img.min()), float(x_img.max())
        if not (xmin >= -tol and xmax <= 1.0 + tol):
            raise AssertionError(
                f"Expected processor-normalized inputs; after de-normalization got range [{xmin:.3f},{xmax:.3f}]. "
                "Make sure you pass the output of pre_encode/default_collate (not raw RGB)."
            )

        x_img = x_img.clamp(0, 1)

        xs = (
            (x_img * 255.0)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # channels at the end
        ids = [str(s[id_column] if id_column else s["sample_id"]) for s in samples]
        ys = []
        for I, sid in zip(xs, ids):
            rng = _stable_rng(seed, f"{enc_key}:{sid}")   # enc_key must differ per encoder
            J = cv2.GaussianBlur(I, (0, 0), blur_sigma).astype(np.float32)
            J += rng.normal(0, noise_std, J.shape).astype(np.float32)
            ys.append(
                I if rng.random() >= p else np.clip(J, 0, 255)
            )  # add partial noise to the dataset

        out = (
            torch.from_numpy(np.stack(ys)).permute(0, 3, 1, 2).to(x.device)
        )  # channels first again
        out = out.float() / 255.0
        out = (out - mean) / std
        b["proc_out"]["pixel_values"] = out
        return b

    return collate


# This function is used to extract representations from different models
def get_collate(
    cfg: omegaconf.OmegaConf,
    enc_key: str,
    split: str = "train",
    kind: str = "colour",
    mode: str = "complete",
) -> Space:
    
    assert split in ("train", "validation", "test"), f"Bad split='{split}'"

    # types if input images required
    if kind == "colour":
        # no modification
        use_collate = default_collate
    elif kind == "canny":
        # canny edge detection
        if mode == "corruption":
            if split in ["validation", "test"]:
                use_collate = default_collate
            else:
                use_collate = edge_collate(
                    default_collate,
                    enc_key=enc_key,
                    mode="canny",
                    auto_k=cfg.transforms.canny.auto_k,
                    thicken=cfg.transforms.canny.thicken,
                    p=cfg.transforms.canny.p_corruption,
                    seed=cfg.experiment.seed
                )
        else:
            use_collate = edge_collate(
                default_collate,
                enc_key=enc_key,
                mode="canny",
                auto_k=cfg.transforms.canny.auto_k,
                thicken=cfg.transforms.canny.thicken,
                p=cfg.transforms.canny.p_complete,
                seed=cfg.experiment.seed
            )
    elif kind == "scharr":
        # scharr edges
        if mode == "corruption":
            if split in ["validation", "test"]:
                use_collate = default_collate
            else:
                use_collate = edge_collate(
                    default_collate,
                    enc_key=enc_key,
                    mode="scharr",
                    thicken=cfg.transforms.scharr.thicken,
                    p=cfg.transforms.scharr.p_corruption,
                    seed=cfg.experiment.seed
                )
        else:
            use_collate = edge_collate(
                default_collate,
                enc_key=enc_key,
                mode="scharr",
                thicken=cfg.transforms.scharr.thicken,
                p=cfg.transforms.scharr.p_complete,
                seed=cfg.experiment.seed
            )
    elif kind == "gaussian":
        # gaussian noise added
        if mode == "corruption":
            if split in ["validation", "test"]:
                use_collate = default_collate
            else:
                use_collate = noisy_collate(
                    default_collate,
                    enc_key=enc_key,
                    blur_sigma=cfg.transforms.gaussian.blur_sigma,
                    noise_std=cfg.transforms.gaussian.noise_std,
                    p=cfg.transforms.gaussian.p_corruption,
                    seed=cfg.experiment.seed
                )  # add some corrputed samples
        else:
            use_collate = noisy_collate(
                default_collate,
                enc_key=enc_key,
                blur_sigma=cfg.transforms.gaussian.blur_sigma,
                noise_std=cfg.transforms.gaussian.noise_std,
                p=cfg.transforms.gaussian.p_complete,
                seed=cfg.experiment.seed
            )
    elif kind == "dog":
        # difference of Gaussian
        if mode == "corruption":
            if split in ["validation", "test"]:
                use_collate = default_collate
            else:
                use_collate = edge_collate(
                    default_collate,
                    enc_key=enc_key,
                    mode="dog",
                    sigma1=cfg.transforms.dog.sigma1,
                    sigma2=cfg.transforms.dog.sigma2,
                    p=cfg.transforms.dog.p_corruption,
                    seed=cfg.experiment.seed
                )
        else:
            use_collate = edge_collate(
                default_collate,
                enc_key=enc_key,
                mode="dog",
                sigma1=cfg.transforms.dog.sigma1,
                sigma2=cfg.transforms.dog.sigma2,
                p=cfg.transforms.dog.p_complete,
                seed=cfg.experiment.seed
            )
    else:
        raise ValueError("Can only perform canny or rgb")
    
    return use_collate

def add_model(spaces, translator, encoder_name_to_add, space_to_add, device):
    universe_spaces = {"train": {}, "validation": {}, "test": {}}
    for split, encoders in spaces.items():
        for encoder_name, space in encoders.items():
            universe_spaces[split][encoder_name] = translator.to_universe(
                x=space, src=encoder_name
            )
    
    universe_spaces_added = {"train": {}, "validation": {}, "test": {}}

    # take the mean representative space
    universe_spaces_added["train"]["UNIVERSE"] =  torch.stack(list(universe_spaces["train"].values())).mean(dim=0).to(device)
    universe_spaces_added["validation"]["UNIVERSE"] =  torch.stack(list(universe_spaces["validation"].values())).mean(dim=0).to(device)
    universe_spaces_added["test"]["UNIVERSE"] =  torch.stack(list(universe_spaces["test"].values())).mean(dim=0).to(device)

    # now we'll add another space
    universe_spaces_added["train"][encoder_name_to_add] = space_to_add["train"]
    universe_spaces_added["validation"][encoder_name_to_add] = space_to_add["validation"]
    universe_spaces_added["test"][encoder_name_to_add] = space_to_add["test"]
    # Return the augmented spaces for probing.

    return universe_spaces_added

def prepare_results(results):
    rows = []
    for method, stats in results.items():
        for train_encoder, test_dict in stats.items():
            for test_encoder, metrics in test_dict.items():
                rows.append(
                    {
                        "Alignment": method,
                        "Encoder FROM": train_encoder,
                        "Encoder TO": test_encoder,
                        "Train Accuracy": f"{metrics['train_accuracy']:.4f}",
                        "Val_accuracy": f"{metrics['val_accuracy']:.4f}",
                        "Test Accuracy": f"{metrics['test_accuracy']:.4f}",
                    }
                )
    return rows

@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="4_semanticity.yaml")
def main(cfg: omegaconf.DictConfig):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(cfg.experiment.seed)
    device = "cpu"
    dataset_name = cfg.experiment.dataset

    data: DatasetView = (
            datasets_lazy_load[dataset_name].build().run()["dataset_view"]
    )

    if dataset_name  == "cifar10":
        feature_name = "x" 
    elif dataset_name == "cifar100":
        feature_name = "img"
    else:
        raise ValueError("Dataset not supported!")

    # split train, test into train (45k), val (5k) and test (10k)
    if dataset_name in ["cifar10", "cifar100"]:
        # selecting 5k samples from train to form a validation dataset

        split = data.hf_dataset["train"].train_test_split(
            test_size=cfg.data.validation_size,
            seed=cfg.experiment.seed,
            stratify_by_column=cfg.data.stratify_column,
        )

        ds_new = DatasetDict(
            {
                "train": split["train"],  # 45k
                "validation": split["test"],  # 5k
                "test": data.hf_dataset["test"],  # 10k
            }
        )

        data = HFDatasetView(
            name=dataset_name,
            hf_dataset=ds_new,
            id_column="sample_id",
            features=data.features,
        )
    else:
        raise ValueError("Script not supported for this dataset!")
    
    # define the splits, train and test
    splits = cfg.splits
    experiments = cfg.experiments

    sample_counts = cfg.experiment.sample_counts
    for sample_count in sample_counts:
        for experiment_name, exp_cfg in experiments.items():
            modes = exp_cfg.modes
            kinds = exp_cfg.kinds
            encoder_lists = exp_cfg.encoders
            added_model_increase_exp = exp_cfg.added_model.replace("/", "_")
            for encoders in encoder_lists:
                full_to_short_names = {
                    k: v for k, v in FULL_TO_SHORT_NAMES.items()
                }
                for mode in modes:
                    spaces = {split: {} for split in splits}
                    labels = {split: [] for split in splits}

                    # iterate through encoders
                    for encoder_name in encoders:
                        # iterate through kind of images
                        for kind in kinds:
                            # resnets used for only corrupted images -- uncomment only for a specific experiment
                            if (
                                experiment_name != "standard"
                            ):  # for standard experiment, keep all spaces
                                if "corrupt_encoders" in exp_cfg.keys():
                                    if (
                                        encoder_name in exp_cfg.corrupt_encoders
                                        and kind == "colour"
                                    ):
                                        logging.info(
                                            f"skipping colour for corrupt encoder: {encoder_name}"
                                        )
                                        continue
                                    elif (
                                        encoder_name not in exp_cfg.corrupt_encoders
                                        and kind != "colour"
                                    ):
                                        logging.info(
                                            f"skipping corrupt for standard encoder: {encoder_name}"
                                        )
                                        continue

                            # get the representations
                            tr_space, tr_labels = embed_dataset(
                                data=data,
                                dataset_name=dataset_name,
                                encoder_name=encoder_name,
                                target_dir=Path(cfg.experiment.data_dir) / kind / mode / encoder_name.replace("/", "-") / "train",
                                split="train",
                                feature_name=feature_name,
                                collate_fn=get_collate(cfg, enc_key=encoder_name, split="train", kind=kind, mode=mode),
                                num_samples=None,
                                labels = True,
                                seed=cfg.experiment.seed,
                                mode='image',
                            )
                            val_space, val_labels = embed_dataset(
                                data,
                                dataset_name,
                                encoder_name,
                                target_dir=Path(cfg.experiment.data_dir) / kind / mode / encoder_name.replace("/", "-") / "validation",
                                split="validation",
                                feature_name=feature_name,
                                collate_fn=get_collate(cfg, enc_key=encoder_name, split="validation", kind=kind, mode=mode),
                                num_samples=None,
                                labels = True,
                                seed=cfg.experiment.seed,
                                mode='image',
                            )
                            te_space, te_labels = embed_dataset(
                                data,
                                dataset_name,
                                encoder_name,
                                target_dir=Path(cfg.experiment.data_dir) / kind / mode / encoder_name.replace("/", "-") / "test",
                                split="test",
                                feature_name=feature_name,
                                collate_fn=get_collate(cfg, enc_key=encoder_name, split="test", kind=kind, mode=mode),
                                num_samples=None,
                                labels = True,
                                seed=cfg.experiment.seed,
                                mode='image',
                            )
                            spaces["train"][
                                f"{full_to_short_names[encoder_name]}_{kind}"
                            ] = tr_space
                            spaces["validation"][
                                f"{full_to_short_names[encoder_name]}_{kind}"
                            ] = val_space
                            spaces["test"][
                                f"{full_to_short_names[encoder_name]}_{kind}"
                            ] = te_space

                            # labels should be the same as we use the same seed, so the same shuffling and shuffling is only used if num_samples is explicitly provided
                            if tr_labels is not None:
                                labels["train"] = torch.tensor(tr_labels, dtype=torch.long)
                            if val_labels is not None:
                                labels["validation"] = torch.tensor(
                                    val_labels, dtype=torch.long
                                )
                            if te_labels is not None:
                                labels["test"] = torch.tensor(te_labels, dtype=torch.long)

                    spaces = {
                        split: {
                            encoder: space.as_tensor().to(device)
                            for encoder, space in spaces[split].items()
                        }
                        for split in splits
                    }

                    # define the dictionary for storing the results
                    results_init = {
                        "INPUT": {},
                        "GPA_UNIVERSE": {},
                        "GPA_TRANSFORMED": {},
                        "GCPA_UNIVERSE": {},
                        "GCPA_TRANSFORMED": {},
                        "GCPA_R_UNIVERSE": {},
                        "GCPA_R_TRANSFORMED": {},
                        "GPA_ADD": {},
                        "GCPA_ADD": {},
                        "PW_TRANSFORMED": {},
                    }
                    results = copy.deepcopy(results_init)
                    for method, _ in results_init.items():
                        for encoder_name, space in spaces["train"].items():
                            results[method][encoder_name] = {}
                    

                    if cfg.experiment.stratified:
                        logging.info("Performing stratified selection!")
                        # fit the alignment only on the sample count now
                        n_classes = len(torch.unique(labels["train"]))
                        per_class = sample_count // n_classes

                        embs = torch.cat([
                            torch.where(labels["train"] == c)[0][torch.randperm((labels["train"] == c).sum())[:per_class]]
                                    for c in torch.unique(labels["train"])
                        ])
                    else:
                        logging.info("Performing random selection!")
                        embs = torch.randperm(len(list(spaces["train"].values())[0]))[:sample_count]

                    assert len(embs) == sample_count, f"The number of samples to select {sample_count} should be matching the actually selected: {len(embs)}!"
                    logging.info(f"selected {len(embs)} samples!")

                    align_spaces = copy.deepcopy(spaces)
                    align_spaces["train"] = {enc_name: embeddings[embs] for enc_name, embeddings in align_spaces["train"].items()}

                    translator_gpa = get_translators(
                        spaces=align_spaces,
                        alignment_method="generalised_procrustes",
                        device=device,
                        gc_enabled=False,
                    )  # fit the jp universal space

                    translator_gcpa = get_translators(
                        spaces=align_spaces,
                        alignment_method="generalised_procrustes",
                        device=device,
                        gc_enabled=True,
                    )

                    results = probing(
                        translator=translator_gpa,
                        spaces=spaces,
                        labels=labels,
                        results=results,
                        alignment="generalised_procrustes",
                        subselect_indices=embs,
                        device=device,
                    )

                    results = probing(
                        translator=translator_gcpa,
                        spaces=spaces,
                        labels=labels,
                        results=results,
                        alignment="generalised_procrustes_gc",
                        subselect_indices=embs,
                        gc_rescale=False,
                        device=device,
                    )
                    results = probing(
                        translator=translator_gcpa,
                        spaces=spaces,
                        labels=labels,
                        results=results,
                        alignment="generalised_procrustes_gc_rescale",
                        subselect_indices=embs,
                        gc_rescale=True,
                        device=device,
                    )


                    try:
                        # gpa alignment -- cycle consistent
                        cycle_method = "gpa"
                        cycle_error(translator=translator_gpa, spaces=spaces, splits=splits)
                        cycle_consistency(
                            translator=translator_gpa, spaces=spaces, splits=splits
                        )
                        cycle_method = "gcpa"
                        cycle_error(translator=translator_gcpa, spaces=spaces, splits=splits)
                        cycle_consistency(
                            translator=translator_gcpa, spaces=spaces, splits=splits
                        )

                    except Exception as e:
                        logging.info(
                            f"{cycle_method} cycle consistency check failed with error: {e}"
                        )

                    df = pd.DataFrame(prepare_results(results=results))

                    out_dir = os.path.join(
                        PROJECT_ROOT, cfg.experiment.output_dir
                    )
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    n_spaces = str(len(list(spaces["train"].keys())))
                    logging.info(added_model_increase_exp)
                    df.to_csv(
                        Path(out_dir) / f"{experiment_name}_{mode}_{dataset_name}_{sample_count}_{n_spaces}_{added_model_increase_exp}.csv"
                    )
                    logging.info("\nClassification Results:\n")
                    logging.info(df.to_markdown(index=False))
                    logging.info("done")


                    new_spaces_add = {}
                    # if we're done with rest of the stuff, let's try to add another model
                    if cfg.experiment.add_model:
                        encoder_name_to_add = exp_cfg.added_model
                        add_tr_space, _ = embed_dataset(
                                data=data,
                                dataset_name=dataset_name,
                                encoder_name=encoder_name_to_add,
                                target_dir=Path(cfg.experiment.data_dir) / 'colour' / 'complete' / encoder_name_to_add.replace("/", "-") / "train",
                                split="train",
                                feature_name=feature_name,
                                collate_fn=get_collate(cfg, encoder_name_to_add, split="train", kind=kind, mode=mode),
                                num_samples=None,
                                labels = True,
                                seed=cfg.experiment.seed,
                                mode='image',
                        )
                        
                        add_val_space, _ = embed_dataset(
                            data=data,
                            dataset_name=dataset_name,
                            encoder_name=encoder_name_to_add,
                            target_dir=Path(cfg.experiment.data_dir) / 'colour' / 'complete' / encoder_name_to_add.replace("/", "-") / "validation",
                            split="validation",
                            feature_name=feature_name,
                            collate_fn=get_collate(cfg, enc_key=encoder_name_to_add, split="validation", kind=kind, mode=mode),
                            num_samples=None,
                            labels = True,
                            seed=cfg.experiment.seed,
                            mode='image',
                        )

                        add_te_space, _ = embed_dataset(
                            data=data,
                            dataset_name=dataset_name,
                            encoder_name=encoder_name_to_add,
                            target_dir=Path(cfg.experiment.data_dir) / 'colour' / 'complete' / encoder_name_to_add.replace("/", "-") / "test",
                            split="test",
                            feature_name=feature_name,
                            collate_fn=get_collate(cfg, encoder_name_to_add, split="test", kind=kind, mode=mode),
                            num_samples=None,
                            labels = True,
                            seed=cfg.experiment.seed,
                            mode='image',
                        )

                        new_spaces_add["train"] = add_tr_space.as_tensor().to(device)
                        new_spaces_add["validation"] = add_val_space.as_tensor().to(device)
                        new_spaces_add["test"] = add_te_space.as_tensor().to(device)


                        added_results = copy.deepcopy(results)
                        for method in ["INPUT", "GPA_UNIVERSE", "GPA_TRANSFORMED", "GCPA_UNIVERSE", "GCPA_TRANSFORMED", "GCPA_R_UNIVERSE", "GCPA_R_TRANSFORMED", "GPA_ADD", "GCPA_ADD", "PW_TRANSFORMED"]:
                            for encoder_name in ["UNIVERSE_GPA", "UNIVERSE_GCPA", encoder_name_to_add]:
                                added_results[method][encoder_name] = {}
                                added_results[method][f"{full_to_short_names[encoder_name_to_add]}_colour"] = {}

                        # created spaces with mean universe for gpa
                        new_spaces_added_gpa = add_model(
                                spaces=spaces,
                                translator=translator_gpa,
                                encoder_name_to_add=encoder_name_to_add,
                                space_to_add=new_spaces_add,
                                device=device,
                        )
                        # replace just UNIVERSE with UNIVERSE_GPA
                        for split in new_spaces_added_gpa.keys():
                            new_spaces_added_gpa[split]["UNIVERSE_GPA"] = new_spaces_added_gpa[split].pop("UNIVERSE")

                        # add mean representation and newly added model for GCPA
                        new_spaces_added_gcpa = add_model(
                                spaces=spaces,
                                translator=translator_gcpa,
                                encoder_name_to_add=encoder_name_to_add,
                                space_to_add=new_spaces_add,
                                device=device,
                        )

                        # change UNIVERSE to UNIVERSE_GCPA
                        for split in new_spaces_added_gcpa.keys():
                            new_spaces_added_gcpa[split]["UNIVERSE_GCPA"] = new_spaces_added_gcpa[split].pop("UNIVERSE")

                        # parent dictionary
                        new_spaces_added = {
                            split: {**new_spaces_added_gpa[split], **new_spaces_added_gcpa[split]}
                            for split in new_spaces_added_gpa.keys()
                        }

                        # we need to route the newly added representation to the individual spaces via the mean universe and vice-versa
                        def _add_composed_results(
                            *,
                            method_label: str,
                            translator_base,
                        ):
                            n_classes = len(set(torch.unique(labels["train"]))) if "train" in labels else 0
                            base_encoders = list(spaces["train"].keys())
                            u_train = torch.stack(
                                [
                                    translator_base.to_universe(
                                        x=spaces["train"][enc], src=enc
                                    )
                                    for enc in base_encoders
                                ]
                            ).mean(dim=0)

                            translator_add = Translator(
                                aligner=MatrixAligner(
                                    name="ortho", align_fn_state=svd_align_state
                                ),
                                x_transform=StandardScaling(),
                                y_transform=StandardScaling(),
                                dim_matcher=ZeroPadding(),
                            )
                            if embs is not None:
                                translator_add.fit(
                                    x=new_spaces_add["train"][embs],
                                    y=u_train[embs],
                                )
                            else:
                                translator_add.fit(
                                    x=new_spaces_add["train"],
                                    y=u_train,
                                )

                            new_train_univ = translator_add.transform(
                                new_spaces_add["train"]
                            )["x"]


                            new_train = new_spaces_add["train"]
                            new_val = new_spaces_add["validation"]
                            new_test = new_spaces_add["test"]

                            new_mu = new_train.mean(0, keepdim=True)
                            new_sig = new_train.std(0, unbiased=False, keepdim=True) + 1e-8

                            added_key = f"{full_to_short_names[encoder_name_to_add]}_colour"
                            for enc in base_encoders:
                                base_train = spaces["train"][enc]
                                base_val = spaces["validation"][enc]
                                base_test = spaces["test"][enc]

                                base_mu = base_train.mean(0, keepdim=True)
                                base_sig = base_train.std(0, unbiased=False, keepdim=True) + 1e-8

                                # new -> base via universe composition
                                x_train = translator_base.from_universe(
                                    u=new_train_univ, tgt=enc
                                )
                                classifier, train_acc, val_acc = train_linear_classifier(
                                    z(x_train, mu=base_mu, sig=base_sig),
                                    labels["train"],
                                    z(base_val, mu=base_mu, sig=base_sig),
                                    labels["validation"],
                                    n_classes,
                                    device=device,
                                )
                                test_acc = test_classifier(
                                    classifier,
                                    z(base_test, mu=base_mu, sig=base_sig),
                                    labels["test"],
                                    device=device,
                                )
                                added_results[method_label][added_key][enc] = {
                                    "train_accuracy": train_acc,
                                    "val_accuracy": val_acc,
                                    "test_accuracy": test_acc,
                                }

                                # base -> new via universe composition
                                base_train_univ = translator_base.to_universe(
                                    x=base_train, src=enc
                                )

                                x_train = translator_add.inverse_transform(
                                    x=None, y=base_train_univ
                                )
                                x_val = new_val
                                x_test = new_test

                                classifier, train_acc, val_acc = train_linear_classifier(
                                    z(x_train, mu=new_mu, sig=new_sig),
                                    labels["train"],
                                    z(x_val, mu=new_mu, sig=new_sig),
                                    labels["validation"],
                                    n_classes,
                                    device=device,
                                )
                                test_acc = test_classifier(
                                    classifier,
                                    z(x_test, mu=new_mu, sig=new_sig),
                                    labels["test"],
                                    device=device,
                                )
                                added_results[method_label][enc][added_key] = {
                                    "train_accuracy": train_acc,
                                    "val_accuracy": val_acc,
                                    "test_accuracy": test_acc,
                                }

                        _add_composed_results(
                            method_label="GPA_ADD",
                            translator_base=translator_gpa,
                        )
                        _add_composed_results(
                            method_label="GCPA_ADD",
                            translator_base=translator_gcpa,
                        )

                        pairwise_probing(spaces=new_spaces_added, 
                                results=added_results, labels=labels, device=device)
                        

                        added_df = pd.DataFrame(prepare_results(results=added_results))

                        # lets now add this space to the original and perform probing
                        # Note that this is only allowed for complete and colour mode
                        spaces["train"][
                            f"{full_to_short_names[encoder_name_to_add]}_colour"
                        ] = add_tr_space.as_tensor().to(device)
                        spaces["validation"][
                            f"{full_to_short_names[encoder_name_to_add]}_colour"
                        ] = add_val_space.as_tensor().to(device)
                        spaces["test"][
                            f"{full_to_short_names[encoder_name_to_add]}_colour"
                        ] = add_te_space.as_tensor().to(device)

                        # refitting
                        new_translator_gpa = get_translators(
                                spaces=spaces,
                                alignment_method="generalised_procrustes",
                                device=device,
                                gc_enabled=False,
                        )

                        new_translator_gcpa = get_translators(
                                spaces=spaces,
                                alignment_method="generalised_procrustes",
                                device=device,
                                gc_enabled=True,
                        )

                        # do best case probing
                        probing(
                                translator=new_translator_gpa,
                                spaces=spaces,
                                labels=labels,
                                results=added_results,
                                alignment="generalised_procrustes",
                                device=device,
                        )

                        probing(
                                translator=new_translator_gcpa,
                                spaces=spaces,
                                labels=labels,
                                results=added_results,
                                alignment="generalised_procrustes_gc",
                                device=device,
                        )

                        enc_name_for_file = encoder_name_to_add.replace("/", "_")
                        added_df = pd.DataFrame(prepare_results(results=added_results))
                        added_df.to_csv(
                            Path(out_dir) / f"addition_{experiment_name}_{mode}_{dataset_name}_{sample_count}_{n_spaces}_{enc_name_for_file}.csv"
                        )


if __name__ == "__main__":
    main()
    
