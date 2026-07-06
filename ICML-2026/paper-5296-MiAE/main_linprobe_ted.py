"""Linear probing evaluation on TEDBench.

Extracts frozen MiAE encoder representations for all train / val / test splits
and the external CATH 4.4 experimental test set, then trains a linear classifier
using L-BFGS with cross-validated regularisation (Appendix B.3).

Config: ``configs/linprobe_ted.yaml``.

Example usage::

    python main_linprobe_ted.py pretrained_model_path=<path/to/ckpt>
    python main_linprobe_ted.py pretrained_model_path=username/miae-b-pretrained

    # Use MiAE-L with sequence input
    python main_linprobe_ted.py pretrained_model_path=<ckpt> model.name=miae_l
"""
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from tedbench.model import MiAEClassifier, MiAE
from tedbench.utils.io import load_from_hf
from tedbench.utils.linear_clf import train_and_eval_linear

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


@torch.no_grad()
def compute_repr(model, data_loader, device="cuda"):
    model.eval()
    X = []
    y = []
    for batch in tqdm(data_loader):
        coords, mask, residue_index, seq_tokens, protein_chain, y_true = (
            batch["coords"],
            batch["mask"],
            batch["residue_index"] - 1,
            batch["seq_ids"],
            batch["protein_chain"],
            batch["label"],
        )
        coords = coords.to(device)
        mask = mask.to(device)
        residue_index = residue_index.to(device)
        seq_tokens = seq_tokens.to(device)
        X_cur = model(
            coords, mask, residue_index, seq_tokens=seq_tokens, repr_only=True
        ).cpu()
        X.append(X_cur)
        y.append(y_true)
    X = torch.concat(X)
    y = torch.concat(y)
    return X, y


def preprocess(X):
    X -= X.mean(dim=-1, keepdim=True)
    X /= X.norm(dim=-1, keepdim=True)
    return X


@hydra.main(version_base="1.3", config_path="./configs", config_name="linprobe_ted")
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    model = MiAEClassifier(cfg)
    p = cfg.pretrained_model_path
    if Path(p).is_dir() or not Path(p).suffix:
        pretrained_model = load_from_hf(p).state_dict()
    else:
        pretrained_model = MiAE.load_from_checkpoint(
            p, map_location=torch.device("cpu"), weights_only=False
        ).state_dict()
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in pretrained_model and pretrained_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del pretrained_model[k]

    msg = model.load_state_dict(pretrained_model, strict=False)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    datamodule.setup("test")

    datamodule_test = hydra.utils.instantiate(cfg.datamodule_test)
    datamodule_test.setup("test")

    device = device = (
        torch.device(torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    X_tr, y_tr = compute_repr(
        model, datamodule.train_dataloader(drop_last=False, shuffle=False)
    )
    X_val, y_val = compute_repr(model, datamodule.val_dataloader())
    X_te, y_te = compute_repr(model, datamodule.test_dataloader())
    X_te_ext, y_te_ext = compute_repr(model, datamodule_test.test_dataloader())

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
