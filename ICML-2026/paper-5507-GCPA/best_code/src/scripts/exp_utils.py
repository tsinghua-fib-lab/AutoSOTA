import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import namedtuple
from datasets import DatasetDict, load_dataset
from latentis.space._base import Space 
from cycloreps.translator.gpa import GeneralizedProcrustesTranslator
from cycloreps.translator.ortho import OrthogonalMultiSpaceTranslator
from cycloreps.translator.linear_ortho import LinearMultiSpaceTranslator
from cycloreps.translator.gcca import GeneralizedCCATranslator
from cycloreps.translator.translator import MultiSpaceBase
from latentis.nn.encoders import ImageHFEncoder, TextHFEncoder, AudioHFEncoder
from latentis.data.dataset import DatasetView, HFDatasetView
from latentis.data.encoding.encode import EncodeTask
from sklearn.decomposition import PCA
from latentis.data import PROJECT_ROOT
from latentis.data.utils import default_collate
from latentis.transform.translate.aligner import MatrixAligner, Translator
from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import ZeroPadding
from latentis.transform.translate.functional import svd_align_state
import omegaconf
from pathlib import Path
import logging

# Configure the logging system
logging.basicConfig(level=logging.INFO)


def z(X, mu, sig):
    logging.info(f"Performing z-score standardisation!")
    return (X - mu) / sig

# used for reid, richermod to obtain mean representations
def feat_id_mean(feats: torch.Tensor, img_ids: List[str]):
    p = np.asarray(img_ids)
    uniq = np.unique(p)
    rows = []
    for pid in uniq:
        idx = torch.from_numpy(np.where(p == pid)[0]).to(feats.device)
        rows.append(feats.index_select(0, idx).mean(dim=0, keepdim=True))
    return torch.cat(rows, dim=0), uniq

# this checks if the features are already normalised or not, for retrieval
def ensure_normalised(x, name="tensor", eps=1e-6):
    # compute norms
    norms = x.norm(dim=1, keepdim=True)
    mean_norm = norms.mean().item()

    # check closeness to 1.0
    if not torch.allclose(norms, torch.ones_like(norms), atol=eps):
        logging.info(f"Normalizing {name}. Mean norm before = {mean_norm:.4f}")
        return torch.nn.functional.normalize(x)
    else:
        return x

def pca_match(spaces, min_dim):

    pca_models = {}
    pca_spaces = {}

    for encoder, space in spaces["train"].items():
        if space.shape[1] > min_dim:
            X = space.detach().cpu().numpy()
            pca = PCA(n_components=min_dim)
            pca.fit(X)
            pca_models[encoder] = pca

    for split, encoders in spaces.items():
        pca_spaces[split] = {}
        for encoder, space in encoders.items():
            if space.shape[1] > min_dim:
                pca = pca_models[encoder]  # must exist, fitted from train
                X = space.detach().cpu().numpy()
                X_pca = pca.transform(X).astype(np.float32)
                X_pca_tensor = torch.tensor(X_pca, device=space.device)
                pca_spaces[split][encoder] = X_pca_tensor
            else:
                # already at min_dim, keep as-is
                pca_spaces[split][encoder] = space

    return pca_spaces

# move any translator to the used device
def move_translator_to_device(translator, device):
    for attr in ["means", "stds", "R_out", "W_out", "T_out", "Q_out"]:
        if getattr(translator, attr, None):
            setattr(
                translator,
                attr,
                {k: v.to(device) for k, v in getattr(translator, attr).items()},
            )

    if getattr(translator, "_corrector", None) is not None:
        translator._corrector = translator._corrector.to(device)

    # Handle functional maps specific attributes
    if (
        hasattr(translator, "functional_maps_aligner")
        and translator.functional_maps_aligner is not None
    ):
        aligner = translator.functional_maps_aligner
        # Move functional maps aligner tensors to device
        for attr_name in [
            "eigenvalues_",
            "eigenvectors_",
            "means_",
            "stds_",
            "_raw_spaces",
        ]:
            attr_dict = getattr(aligner, attr_name, {})
            if attr_dict:
                setattr(
                    aligner, attr_name, {k: v.to(device) for k, v in attr_dict.items()}
                )

        # Move functional maps matrices
        functional_maps = getattr(aligner, "functional_maps_", {})
        if functional_maps:
            aligner.functional_maps_ = {
                k: v.to(device) for k, v in functional_maps.items()
            }

        aligner.device = device

    translator.device = device
    return translator

# universal spaces: orthogonal spaces, reconstruction spaces, and functional maps
def get_translators(
    spaces: Dict[str, Space],
    device: str,
    alignment_method="orthogonal",
    ortho_reg=0.2,
    out_dim=64,
    split="train",
    gc_enabled: bool = True,
    gc_tau: float | None = None,
    gc_lam: float | None = None,
) -> MultiSpaceBase:
    # load config using omegaconf
    cfg = omegaconf.OmegaConf.load(Path(PROJECT_ROOT / "config" / "alignment.yaml"))

    if alignment_method == "orthogonal":
            translator_cyclecons = OrthogonalMultiSpaceTranslator(
                max_iter=cfg.ortho.max_iter,
                tol=cfg.ortho.tol,
                device=device,
            ).to(device)
    elif alignment_method == "generalised_procrustes":
        gp_kwargs = dict(
            max_iter=cfg.procrustes.max_iter,
            tol=cfg.procrustes.tol,
            device=device,
            gc_enabled=gc_enabled,
        )
        for key in [
            "gc_epochs",
            "gc_steps_per_epoch",
            "gc_batch_size",
            "gc_lr",
            "gc_weight_decay",
            "gc_hidden_mult",
            "gc_dropout",
            "gc_val_fraction",
            "gc_tau",
            "gc_lam",
            "gc_val_batches",
        ]:
            if key in cfg.procrustes:
                gp_kwargs[key] = cfg.procrustes[key]
        if gc_tau is not None:
            gp_kwargs["gc_tau"] = gc_tau
        if gc_lam is not None:
            gp_kwargs["gc_lam"] = gc_lam
        translator_cyclecons = GeneralizedProcrustesTranslator(**gp_kwargs).to(device)
    elif alignment_method == "reconstruction":
        translator_cyclecons = LinearMultiSpaceTranslator(
            max_iter=cfg.reconstruction.max_iter,
            tol=cfg.reconstruction.tol,
            device=device,
            out_dim=out_dim,
            ortho_reg=ortho_reg,
            lr=cfg.reconstruction.lr,
        ).to(device)
    elif alignment_method == "generalised_cca":
        translator_cyclecons = GeneralizedCCATranslator(
            device=device,
        ).to(device)
    else:
        raise ValueError(f"{alignment_method} not recognised!")

    translator_cyclecons.fit(spaces[split])
    move_translator_to_device(translator_cyclecons, device=device)

    return translator_cyclecons

def embed_dataset(data: DatasetView, dataset_name: str, encoder_name: str, target_dir: Path, feature_name: str, batch_size=32, collate_fn = default_collate, num_samples: int = None, labels: bool = False, seed:int=42, split: str = 'train', mode='audio', device='cuda') -> Space:
    logging.info(f"Dataset {dataset_name} has labels, will use them for encoding for encoder {encoder_name}.")
    # if we're choosing only a subset of samples, we need to shuffle the dataset
    if num_samples is not None and num_samples < len(data.hf_dataset):
        logging.info(f"Subsampling {num_samples} samples from {dataset_name} for split {split} will occur during encoding. We shuffle preemptively.")
        data = data.shuffle(seed=seed)

    if labels:
        # get the labels, fixed for now
        if dataset_name == "cifar100":
            labels = data.hf_dataset[split]["fine_label"]  # choose correct label key
            logging.info(data.hf_dataset[split].features)
            logging.info(
                f"Dataset {dataset_name} has labels {np.unique(data.hf_dataset[split]['fine_label'])} for split {split}, will use them for encoding."
            )
        elif "massive" in dataset_name.lower():
            labels = data.hf_dataset[split]["intent"]  # choose correct label key
            logging.info(data.hf_dataset[split].features)
            logging.info(
                f"Dataset {dataset_name} has labels {np.unique(data.hf_dataset[split]['intent'])} for split {split}, will use them for encoding."
            )
        else:
            labels = data.hf_dataset[split]["y"]
            logging.info(
                f"Dataset {dataset_name} has labels {np.unique(data.hf_dataset[split]['y'])} for split {split}, will use them for encoding."
            )

    if mode == 'image':
        # define the encoder to be used for extracting the representations
        encoder = ImageHFEncoder(encoder_name)
    elif mode == 'text':
        encoder = TextHFEncoder(encoder_name)
    else:
        #audio
        encoder = AudioHFEncoder(encoder_name)
        assert batch_size == 1, "batch size should be 1 for optimal audio embeddings"

    encoder = encoder.eval()
    encoder = encoder.to(device)

    # load the directory if it doesn't exist
    if target_dir.exists():
        logging.info(f"Skipping {encoder_name} for split {split} as target directory already exists: {target_dir}")
        if labels:
            return Space.load_from_disk(target_dir), labels
        else:
            return Space.load_from_disk(target_dir)
    else:
        logging.info(f"Saving the encodings at {target_dir}!")

    logging.info(data.features)
    assert data.get_feature(feature_name) is not None, (
        "EncodeTask.feature is None. Pass a Feature (e.g. 'image' or 'text') "
        "when constructing the task"
    )
    
    logging.info(f"{feature_name} feature of {dataset_name} will be encoded!")

    task = EncodeTask(
        dataset_view=data,
        split=split,
        feature=feature_name,
        model=encoder,
        collate_fn=collate_fn,
        encoding_batch_size=batch_size, # fix to 1 to ensure no padding and truncation -- optimal embedding
        num_workers=16,
        pooler=None,
        save_source_model=False,
        device=device,
        target_path=target_dir,
        write_every=5,
        # this is intentional to avoid passing num samples
        only_first_N_samples=None,
    )
    task.run()
    # load the representations
    if labels:
        return Space.load_from_disk(target_dir), labels
    else:
        return Space.load_from_disk(target_dir)


def load_massive(
    *,
    dataset_id: str,
    lang_list: Optional[List[str]] = None,
    subset_size: Optional[float] = None,
    seed: int = 42,
    stratify_column: str = "intent",
) -> Dict[str, HFDatasetView]:
    ds = load_dataset(dataset_id, trust_remote_code=True)

    keep = {"id", "utt", "intent", "locale"}
    ds = DatasetDict(
        {
            split: split_ds.remove_columns(
                [c for c in split_ds.column_names if c not in keep]
            )
            for split, split_ds in ds.items()
        }
    )

    ds = DatasetDict(
        {split: d.rename_column("id", "sample_id") for split, d in ds.items()}
    )

    FeatureSpec = namedtuple("FeatureSpec", ["name"])
    features = [FeatureSpec(n) for n in ["utt", "intent", "locale"]]

    available = sorted(set(ds["train"]["locale"]))
    if lang_list is None:
        lang_list = available
    else:
        missing = [l for l in lang_list if l not in available]
        if missing:
            raise ValueError(f"Requested MASSIVE languages not found: {missing}")

    lang2view: Dict[str, HFDatasetView] = {}
    for loc in lang_list:
        per_lang = DatasetDict(
            {
                split: split_ds.filter(lambda ex: ex["locale"] == loc)
                for split, split_ds in ds.items()
            }
        )

        if subset_size is not None and 0 < subset_size < 1:
            per_lang_subset = per_lang["train"].train_test_split(
                test_size=subset_size,
                seed=seed,
                stratify_by_column=stratify_column,
            )
            train_split = per_lang_subset["test"]
        else:
            train_split = per_lang["train"]

        per_lang_new = DatasetDict(
            {
                "train": train_split,
                "validation": per_lang["test"],
                "test": per_lang["test"],
            }
        )

        if any(len(per_lang_new[split]) == 0 for split in per_lang_new):
            continue

        lang2view[loc] = HFDatasetView(
            name=f"massive_{loc}",
            hf_dataset=per_lang_new,
            id_column="sample_id",
            features=features,
        )
    return lang2view


def labels_for_massive(
    dataset_view: HFDatasetView, label_column: str, split: str = "train"
) -> List[int]:
    labels = dataset_view.hf_dataset[split][label_column]
    return [int(lbl) if isinstance(lbl, (int, np.integer)) else lbl for lbl in labels]

def train_linear_classifier(x, y, x_val, y_val, n_classes, epochs=100, patience=10, lr=1e-3, device="cpu"):
    """
    Train a linear classifier on the given data.
    Logs loss and accuracy per epoch.
    Max epochs are 100 by default
    """
    orig_device = device
    if torch.cuda.is_available():
        device = "cuda"
    classifier = torch.nn.Linear(x.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    x = x.to(device).detach()
    y = torch.tensor(y, dtype=torch.long).to(device)
    x_val = x_val.to(device).detach()
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_state = None  # to use the best checkpoint
    noimprove_count = 0

    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        outputs = classifier(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        acc = (preds == y).float().mean().item()

        logging.info(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Accuracy={acc:.4f}")

        # validation checks
        with torch.no_grad():
            val_outputs = classifier(x_val)
        _, val_preds = torch.max(val_outputs, 1)
        val_acc = (val_preds == y_val).float().mean().item()

        logging.info(f"Epoch {epoch+1}: Validation Accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = acc
            best_state = {
                k: v.detach().clone() for k, v in classifier.state_dict().items()
            }  # this will be used
            noimprove_count = 0
        else:
            noimprove_count += 1
            if noimprove_count >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        classifier.load_state_dict(best_state)

    if orig_device != device:
        classifier = classifier.to(orig_device)
    return classifier, best_train_acc, best_val_acc


def test_classifier(
    classifier: torch.nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device="cpu",
) -> float:
    """
    Test the classifier on the test data.
    Returns the accuracy.
    """
    orig_device = device
    if torch.cuda.is_available():
        device = "cuda"
    if orig_device != device:
        classifier = classifier.to(device)
    classifier.eval()
    x_test, y_test = x_test.to(device), y_test.to(device)
    with torch.no_grad():
        outputs = classifier(x_test)
        _, preds = torch.max(outputs, 1)
        acc = (preds == y_test).float().mean().item()

    logging.info(f"Test Accuracy: {acc:.4f}")
    if orig_device != device:
        classifier = classifier.to(orig_device)
    return acc

# probing experiments
def probing(
    translator: MultiSpaceBase,
    spaces: Dict[str, Space],
    labels: Dict[str, Any],
    results: Dict,
    alignment="ortho",
    pca_applied: bool = False,
    subselect_indices = None,
    gc_rescale: bool | None = None,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # number of classes for the task
    n_classes = len(set(torch.unique(labels["train"]))) if "train" in labels else 0

    logging.info(f"The number of classes in this dataset are: {n_classes}")

    if device != "cpu":
        spaces = {
            split: {enc: space.to(device) for enc, space in split_dict.items()}
            for split, split_dict in spaces.items()
        }

    prev_gc_rescale = None
    has_gc_rescale = hasattr(translator, "_gc_rescale_override")
    if gc_rescale is not None and has_gc_rescale:
        prev_gc_rescale = translator._gc_rescale_override
        translator._gc_rescale_override = gc_rescale

    # outer iteration through the encoder spaces
    for encoder_name, space in spaces["train"].items():
        logging.info(f"Training classifier for {encoder_name} on train data")
        x_train = space  # training space
        y_train = labels["train"]  # training labels
        y_val = labels["validation"]  # validation labels
        y_test = labels["test"]  # test labels

        # inner iteration through encoders
        for other_encoder_name in spaces["train"].keys():
            # our fixed mu and sig
            x_src_mu = spaces["train"][encoder_name].mean(0, keepdim=True)
            x_src_sig = spaces["train"][encoder_name].std(0, unbiased=False, keepdim=True) + 1e-8
            x_tgt_mu = spaces["train"][other_encoder_name].mean(0, keepdim=True)
            x_tgt_std = spaces["train"][other_encoder_name].std(0, unbiased=False, keepdim=True) + 1e-8
            

            if other_encoder_name not in results["INPUT"][encoder_name].keys():
                if pca_applied:  # do z-score for a fairer comparison
                    # for complete consistency, apply the z-score used on train
                    classifier, train_acc, val_acc = train_linear_classifier(
                        z(x_train, mu=x_src_mu, sig=x_src_sig),
                        y_train,
                        z(
                            spaces["validation"][other_encoder_name],
                            mu=x_tgt_mu,
                            sig=x_tgt_std,
                        ),
                        y_val,
                        n_classes,
                        device=device,
                    )  # train the classifier
                    test_acc = test_classifier(
                        classifier,
                        z(
                            spaces["test"][other_encoder_name],
                            mu=x_tgt_mu,
                            sig=x_tgt_std,
                        ),
                        y_test,
                        device=device,
                    )  # test
                    # results on input space
                    results["INPUT"][encoder_name][other_encoder_name] = {
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "test_accuracy": test_acc,
                    }
                else:
                    if (
                        encoder_name == other_encoder_name
                    ):  # can only do this for same encoders
                        classifier, train_acc, val_acc = train_linear_classifier(
                            z(x_train, mu=x_src_mu, sig=x_src_sig),
                            y_train,
                            z(
                                spaces["validation"][other_encoder_name],
                                mu=x_tgt_mu,
                                sig=x_tgt_std,
                            ),
                            y_val,
                            n_classes,
                            device=device,
                        )  # train the classifier
                        test_acc = test_classifier(
                            classifier,
                            z(
                                spaces["test"][other_encoder_name],
                                mu=x_tgt_mu,
                                sig=x_tgt_std,
                            ),
                            y_test,
                            device=device,
                        )  # test

                        # results on input space
                        results["INPUT"][encoder_name][other_encoder_name] = {
                            "train_accuracy": train_acc,
                            "val_accuracy": val_acc,
                            "test_accuracy": test_acc,
                        }

            # these spaces are already standardised
            logging.info(f"Translating {encoder_name} to UNIVERSE")
            logging.info(f"Translating {other_encoder_name} to UNIVERSE")

            if gc_rescale is not None and has_gc_rescale:
                x_translated_universe = translator.to_universe(
                    x=spaces["train"][encoder_name], src=encoder_name, gc_rescale=gc_rescale
                )
            else:
                x_translated_universe = translator.to_universe(
                    x=spaces["train"][encoder_name], src=encoder_name
                )
            # validation set always taken from the test space
            if gc_rescale is not None and has_gc_rescale:
                x_val = translator.to_universe(
                    x=spaces["validation"][other_encoder_name], src=other_encoder_name, gc_rescale=gc_rescale
                )
            else:
                x_val = translator.to_universe(
                    x=spaces["validation"][other_encoder_name], src=other_encoder_name
                )
            classifier, train_acc, val_acc = train_linear_classifier(
                x_translated_universe, y_train, x_val, y_val, n_classes, device=device
            )
            if gc_rescale is not None and has_gc_rescale:
                x_test = translator.to_universe(
                    x=spaces["test"][other_encoder_name], src=other_encoder_name, gc_rescale=gc_rescale
                )
            else:
                x_test = translator.to_universe(
                    x=spaces["test"][other_encoder_name], src=other_encoder_name
                )
            test_acc = test_classifier(classifier, x_test, y_test, device=device)
            # universal space results
            if alignment == "ortho":
                results["ORTHO_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "reconstruction":
                results["LINEAR_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,  # this is an indicator for the performance, to be used for grid search
                    "test_accuracy": test_acc,
                }
            elif alignment == "generalised_procrustes":
                results["GPA_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "generalised_procrustes_gc":
                results["GCPA_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "generalised_procrustes_gc_rescale":
                results["GCPA_R_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "generalised_cca":
                results["GCCA_UNIVERSE"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            else:
                raise ValueError("Incorrect alignment method passed!")

            if (
                encoder_name == other_encoder_name
            ):  # pairwise transforms are not suitable to use for mapping onto the same space
                continue

            logging.info(f"Translating {encoder_name} to {other_encoder_name}")

            x_translated = translator.transform(
                x=spaces["train"][encoder_name], src=encoder_name, tgt=other_encoder_name
            )

            # validation set always taken from the test space
            x_val = spaces["validation"][other_encoder_name]
            classifier, train_acc, val_acc = train_linear_classifier(
                z(x_translated, mu=x_tgt_mu, sig=x_tgt_std),
                y_train,
                z(x_val, mu=x_tgt_mu, sig=x_tgt_std),
                y_val,
                n_classes,
                device=device,
            )
            # transform test data
            test_acc = test_classifier(
                classifier, z(spaces["test"][other_encoder_name], mu=x_tgt_mu, sig=x_tgt_std), y_test, device=device
            )
            # pairwise results using the universal space
            if alignment == "ortho":
                results["ORTHO_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "reconstruction":
                results["LINEAR_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "generalised_cca":
                results["GCCA_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "generalised_procrustes":
                results["GPA_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "generalised_procrustes_gc":
                results["GCPA_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            elif alignment == "generalised_procrustes_gc_rescale":
                results["GCPA_R_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }
            else:
                raise ValueError("Incorrect alignment method passed!")

            # now purely pairwise transform between two spaces
            if other_encoder_name not in results["PW_TRANSFORMED"][encoder_name].keys():
                logging.info(
                    f"Pairwise translating {encoder_name} to {other_encoder_name}"
                )
                translator_ortho = Translator(
                    aligner=MatrixAligner(name="ortho", align_fn_state=svd_align_state),
                    x_transform=StandardScaling(),
                    y_transform=StandardScaling(),
                    dim_matcher=ZeroPadding(),
                )

                # fit only once -- train space of outer encoder to inner encoder
                if subselect_indices is not None:
                    translator_ortho.fit(
                        x=spaces["train"][encoder_name][subselect_indices],
                        y=spaces["train"][other_encoder_name][subselect_indices],
                    )
                else:
                    translator_ortho.fit(
                        x=spaces["train"][encoder_name],
                        y=spaces["train"][other_encoder_name],
                    )
                x_translated = translator_ortho.transform(x_train)["x"]

                # validation set always taken from the test space
                x_val = spaces["validation"][other_encoder_name]
                classifier, train_acc, val_acc = train_linear_classifier(
                    z(x_translated, mu=x_tgt_mu, sig=x_tgt_std),
                    y_train,
                    z(x_val, mu=x_tgt_mu, sig=x_tgt_std),
                    y_val,
                    n_classes,
                    device=device,
                )

                x_test = spaces["test"][other_encoder_name]

                test_acc = test_classifier(
                    classifier,
                    z(x_test, mu=x_tgt_mu, sig=x_tgt_std),
                    y_test,
                    device=device,
                )
                results["PW_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }

    if gc_rescale is not None and has_gc_rescale:
        translator._gc_rescale_override = prev_gc_rescale
    return results

def pairwise_probing(
    spaces: Dict[str, Space],
    labels: Dict[str, Any],
    results: Dict,
    pca_applied: bool = False,
    subselect_indices = None,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # number of classes for the task
    n_classes = len(set(torch.unique(labels["train"]))) if "train" in labels else 0

    logging.info(f"The number of classes in this dataset are: {n_classes}")

    # outer iteration through the encoder spaces
    for encoder_name, space in spaces["train"].items():
        logging.info(f"Training classifier for {encoder_name} on train data")
        x_train = space  # training space
        y_train = labels["train"]  # training labels
        y_val = labels["validation"]  # validation labels
        y_test = labels["test"]  # test labels

        # inner iteration through encoders
        for other_encoder_name in spaces["train"].keys():

            # our fixed mean and sig
            x_src_mu = spaces["train"][encoder_name].mean(0, keepdim=True)
            x_src_sig = spaces["train"][encoder_name].std(0, unbiased=False, keepdim=True) + 1e-8
            x_tgt_mu = spaces["train"][other_encoder_name].mean(0, keepdim=True)
            x_tgt_std = spaces["train"][other_encoder_name].std(0, unbiased=False, keepdim=True) + 1e-8
            
            if other_encoder_name not in results["INPUT"][encoder_name].keys():
                
                if pca_applied:  # do z-score for a fairer comparison
                    # for complete consistency, apply the z-score used on train
                    classifier, train_acc, val_acc = train_linear_classifier(
                        z(x_train, mu=x_src_mu, sig=x_src_sig),
                        y_train,
                        z(
                            spaces["validation"][other_encoder_name],
                            mu=x_tgt_mu,
                            sig=x_tgt_std,
                        ),
                        y_val,
                        n_classes,
                        device=device,
                    )  # train the classifier
                    test_acc = test_classifier(
                        classifier,
                        z(
                            spaces["test"][other_encoder_name],
                            mu=x_tgt_mu,
                            sig=x_tgt_std,
                        ),
                        y_test,
                        device=device,
                    )  # test
                    # results on input space
                    results["INPUT"][encoder_name][other_encoder_name] = {
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "test_accuracy": test_acc,
                    }
                else:
                    if (
                        encoder_name == other_encoder_name
                    ):  # can only do this for same encoders
                        classifier, train_acc, val_acc = train_linear_classifier(
                            z(x_train, mu=x_src_mu, sig=x_src_sig),
                            y_train,
                            z(
                                spaces["validation"][other_encoder_name],
                                mu=x_tgt_mu,
                                sig=x_tgt_std,
                            ),
                            y_val,
                            n_classes,
                            device=device,
                        )  # train the classifier
                        test_acc = test_classifier(
                            classifier,
                            z(
                                spaces["test"][other_encoder_name],
                                mu=x_tgt_mu,
                                sig=x_tgt_std,
                            ),
                            y_test,
                            device=device,
                        )  # test

                        # results on input space
                        results["INPUT"][encoder_name][other_encoder_name] = {
                            "train_accuracy": train_acc,
                            "val_accuracy": val_acc,
                            "test_accuracy": test_acc,
                        }


            if (
                encoder_name == other_encoder_name
            ):  # pairwise transforms are not suitable to use for mapping onto the same space
                continue

            # now purely pairwise transform between two spaces
            if other_encoder_name not in results["PW_TRANSFORMED"][encoder_name].keys():
                logging.info(
                    f"Pairwise translating {encoder_name} to {other_encoder_name}"
                )
                translator_ortho = Translator(
                    aligner=MatrixAligner(name="ortho", align_fn_state=svd_align_state),
                    x_transform=StandardScaling(),
                    y_transform=StandardScaling(),
                    dim_matcher=ZeroPadding(),
                )

                # fit only once -- train space of outer encoder to inner encoder
                if subselect_indices is not None:
                    translator_ortho.fit(
                        x=spaces["train"][encoder_name][subselect_indices],
                        y=spaces["train"][other_encoder_name][subselect_indices],
                    )
                else:
                    translator_ortho.fit(
                        x=spaces["train"][encoder_name],
                        y=spaces["train"][other_encoder_name],
                    )
                x_translated = translator_ortho.transform(x_train)["x"]

                # validation set always taken from the test space
                x_val = spaces["validation"][other_encoder_name]
                classifier, train_acc, val_acc = train_linear_classifier(
                    z(x_translated, mu=x_tgt_mu, sig=x_tgt_std),
                    y_train,
                    z(x_val, mu=x_tgt_mu, sig=x_tgt_std),
                    y_val,
                    n_classes,
                    device=device,
                )

                x_test = spaces["test"][other_encoder_name]

                test_acc = test_classifier(
                    classifier,
                    z(x_test, mu=x_tgt_mu, sig=x_tgt_std),
                    y_test,
                    device=device,
                )
                results["PW_TRANSFORMED"][encoder_name][other_encoder_name] = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                }

    return results
