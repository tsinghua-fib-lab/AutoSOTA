import os

from internal.dataset.datasets import (
    Fashion_MNIST,
    RAVDESS,
    BiosBias,
    FACET,
    Credit,
    ACSIncome,
    ACSEducation,
)

_LOADER_KW = {
    "train_batch_size",
    "valid_batch_size",
    "test_batch_size",
    "calib_batch_size",
    "calib_val_batch_size",
    "n_calib",
    "n_test",
    "n_calib_val",  # HPO for conformal
    "n_train",
    "n_val",
    "m",
    "model_checkpoint",
    "save_model_ckpt",
}

DATASET_CLASS_MAP = {
    "ravdess": RAVDESS,
    "bios": BiosBias,
    "facet": FACET,
    "acs-income": ACSIncome,
}


def get_loaders(cfg):
    dataset = cfg["dataset"]
    data_root = cfg["data_root"]
    assert dataset in DATASET_CLASS_MAP.keys(), f"Unknown dataset {dataset}"

    loader_kwargs = get_loader_kwargs(cfg)

    dataset_class = DATASET_CLASS_MAP[dataset]()

    output_dict = dataset_class.get_data(data_root, **loader_kwargs)

    if not hasattr(dataset_class, "group_conformal_category"):
        raise ValueError(
            f"group_conformal_category must be specified in {dataset} dataset class"
        )

    return dataset_class, output_dict


def get_loader_kwargs(cfg):
    # Create subdict of cfg for the keys relevant to dataloading
    loader_kw = _LOADER_KW
    loader_kwargs = {k: cfg[k] for k in cfg.keys() & loader_kw}

    return loader_kwargs
