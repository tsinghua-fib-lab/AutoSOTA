import logging

import hydra
import pytorch_lightning as pl
import torch
from esm import FastaBatchedDataset, pretrained
from omegaconf import OmegaConf
from tqdm import tqdm

from tedbench.utils.linear_clf import train_and_eval_linear

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


@torch.no_grad()
def compute_repr(model, data_loader, device):
    X = []
    y = []
    for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader)):
        toks = toks.to(device=device, non_blocking=True)

        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        out = out["representations"][model.num_layers]

        for i, label in enumerate(labels):
            truncate_len = len(strs[i])
            X_cur = out[i, 1 : truncate_len + 1].mean(0, keepdim=True).cpu()
            X.append(X_cur)
            y.append(label)

    X = torch.concat(X)
    y = torch.tensor(y)
    return X, y


def get_dataloader_from_cfg(cfg, alphabet):
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    datamodule.setup("test")
    datamodule_test = hydra.utils.instantiate(cfg.datamodule_test)
    datamodule_test.setup("test")

    dataloader_splits = [
        datamodule.train_dataloader(drop_last=False, shuffle=False),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
        datamodule_test.test_dataloader(),
    ]
    dataloaders = []

    for data_loader in dataloader_splits:
        seqs = []
        labels = []
        for batch in tqdm(data_loader, desc="Processing data"):
            protein_chains = batch["protein_chain"]
            y_true = batch["label"]
            seqs_batch = [protein_chain.sequence for protein_chain in protein_chains]
            seqs += seqs_batch
            labels.append(y_true)
        labels = torch.concat(labels).tolist()

        dataset = FastaBatchedDataset(labels, seqs)
        batches = dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(),
            batch_sampler=batches,
        )
        dataloaders.append(dataloader)
    return dataloaders


@hydra.main(
    version_base="1.3",
    config_path="./configs",
    config_name="esm2_linprobe_ted",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    model, alphabet = pretrained.load_model_and_alphabet(cfg.model.name)
    model.eval()

    train_loader, val_loader, test_loader, ext_test_loader = get_dataloader_from_cfg(
        cfg, alphabet
    )

    device = (
        torch.device(torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    X_tr, y_tr = compute_repr(model, train_loader, device)
    X_val, y_val = compute_repr(model, val_loader, device)
    X_te, y_te = compute_repr(model, test_loader, device)
    X_te_ext, y_te_ext = compute_repr(model, ext_test_loader, device)

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
