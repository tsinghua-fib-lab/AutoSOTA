import logging

import hydra
import pytorch_lightning as pl
import torch
from models.structure_models import baseline_models
from omegaconf import OmegaConf
from tqdm import tqdm

from tedbench.utils.linear_clf import train_and_eval_linear

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


def get_cath_data(dataset, tmpdir):
    tmpdir.mkdir(parents=True, exist_ok=True)
    pdb_path_list = []
    labels = []
    for i in tqdm(range(len(dataset)), desc="Processing data"):
        data = dataset[i]
        protein_chain, label = data[-2], data[-1]
        pdb_file = tmpdir / f"protein_{i}.pdb"
        protein_chain = protein_chain.infer_oxygen()
        if not pdb_file.exists():
            protein_chain.to_pdb(pdb_file)
        pdb_path_list.append(pdb_file)
        labels.append(label)
    return pdb_path_list, labels


def print_parameter_count(model):
    """Print a detailed breakdown of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    return total_params, trainable_params


@hydra.main(
    version_base="1.3",
    config_path="./configs",
    config_name="proteinmpnn_linprobe_ted",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)
    device = (
        torch.device(torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = baseline_models[cfg.model.name](device=device)
    print_parameter_count(model.model)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    datamodule.setup("test")
    datamodule_test = hydra.utils.instantiate(cfg.datamodule_test)
    datamodule_test.setup("test")

    train_pdb, train_labels = (
        datamodule.train_dataset.pdb_files_split,
        datamodule.train_dataset.cath_labels,
    )
    val_pdb, val_labels = (
        datamodule.val_dataset.pdb_files_split,
        datamodule.val_dataset.cath_labels,
    )
    test_pdb, test_labels = (
        datamodule.test_dataset.pdb_files_split,
        datamodule.test_dataset.cath_labels,
    )
    tmpdir = datamodule_test.test_dataset.root / "pdb_files"
    ext_test_pdb, ext_test_labels = get_cath_data(datamodule_test.test_dataset, tmpdir)

    X_tr, y_tr = model.encode_structure(
        train_pdb, train_labels, cfg.datamodule.batch_size, cfg.datamodule.num_workers
    )
    X_val, y_val = model.encode_structure(
        val_pdb, val_labels, cfg.datamodule.batch_size, cfg.datamodule.num_workers
    )
    X_te, y_te = model.encode_structure(
        test_pdb, test_labels, cfg.datamodule.batch_size, cfg.datamodule.num_workers
    )
    X_te_ext, y_te_ext = model.encode_structure(
        ext_test_pdb,
        ext_test_labels,
        cfg.datamodule_test.batch_size,
        cfg.datamodule_test.num_workers,
    )

    m = X_tr.mean(0, keepdim=True)
    s = X_tr.std(0, unbiased=False, keepdim=True)
    X_tr = (X_tr - m) / s
    X_val = (X_val - m) / s
    X_te = (X_te - m) / s
    print(X_tr.shape)
    print(X_val.shape)
    print(X_te.shape)
    X_te_ext = (X_te_ext - m) / s
    print(X_te_ext.shape)

    del model

    val_score, test_score = train_and_eval_linear(
        X_tr,
        y_tr,
        X_val,
        y_val,
        [X_te, X_te_ext],
        [y_te, y_te_ext],
        cfg.model.num_classes,
        device=device,
    )

    results = [
        {
            "test_acc": test_score[0],
            "val_acc": val_score,
        }
    ]

    import pandas as pd

    pd.DataFrame(results).to_csv(f"{cfg.logs.path}/results.csv")


if __name__ == "__main__":
    main()
