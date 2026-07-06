import logging
import os

import hydra
import joblib
import pytorch_lightning as pl
import torch
from biotite.structure.io.pdb import PDBFile
from esm import FastaBatchedDataset
from omegaconf import OmegaConf
from saprot.model.saprot.base import SaprotBaseModel
from saprot.utils.esm_loader import load_esm_saprot
from saprot.utils.foldseek_util import get_struc_seq
from tqdm import tqdm
from transformers import EsmTokenizer

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
        toks = {k: v.to(device) for k, v in toks.items()}

        # out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        out = model.get_hidden_states(toks, reduction="mean")
        # out = out["representations"][model.num_layers]

        # for i, label in enumerate(labels):
        #    truncate_len = len(strs[i])
        #    X_cur = out[i, 1 : truncate_len + 1].mean(0, keepdim=True).cpu()
        #    X.append(X_cur)
        #    y.append(label)
        X += out
        y += labels

    X = torch.stack(X)
    y = torch.tensor(y)
    return X, y


def get_saprot_seq(
    pdb_path,
    foldseek_path="/fs/gpfs41/lv11/fileset01/pool/pool-chen/softwares/micromamba/envs/foldseek/bin/foldseek",
):
    atom_array = PDBFile.read(pdb_path).get_structure(model=1)
    chain_id = atom_array.chain_id[0]
    pid = os.getpid()
    parsed_seqs = get_struc_seq(
        foldseek_path, pdb_path, [chain_id], process_id=pid, plddt_mask=False
    )[chain_id]
    return parsed_seqs[-1]


def batch_converter(raw_batch, tokenizer):
    batch_size = len(raw_batch)
    batch_labels, seq_str_list = zip(*raw_batch)
    tokens = tokenizer(seq_str_list, padding=True, return_tensors="pt")
    return batch_labels, seq_str_list, tokens


def get_cath_data(dataset, tmpdir):
    tmpdir.mkdir(parents=True, exist_ok=True)
    pdb_path_list = []
    labels = []
    for i in tqdm(range(len(dataset)), desc="Processing data"):
        data = dataset[i]
        protein_chain, label = data[-2], data[-1]
        pdb_file = tmpdir / f"protein_{i}.pdb"
        if not pdb_file.exists():
            protein_chain.to_pdb(pdb_file)
        pdb_path_list.append(pdb_file)
        labels.append(label)
    return pdb_path_list, labels


def get_dataloader_from_cfg(cfg, tokenizer):
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    datamodule.setup("test")
    datamodule_test = hydra.utils.instantiate(cfg.datamodule_test)
    datamodule_test.setup("test")
    tmpdir = datamodule_test.test_dataset.root / "pdb_files"
    ext_test_pdb, ext_test_labels = get_cath_data(datamodule_test.test_dataset, tmpdir)

    dataloader_splits = [
        (
            datamodule.train_dataset.pdb_files_split,
            datamodule.train_dataset.cath_labels,
        ),
        (
            datamodule.val_dataset.pdb_files_split,
            datamodule.val_dataset.cath_labels,
        ),
        (
            datamodule.test_dataset.pdb_files_split,
            datamodule.test_dataset.cath_labels,
        ),
        (ext_test_pdb, ext_test_labels),
    ]
    dataloaders = []
    batch_size = cfg.datamodule.batch_size
    num_workers = cfg.datamodule.num_workers

    def collate_fn(raw_batch):
        return batch_converter(raw_batch, tokenizer)

    for data_loader in dataloader_splits:
        seqs = []
        labels = []
        pdb_path_list, labels = data_loader
        for i in tqdm(range(0, len(pdb_path_list), batch_size), desc="Processing data"):
            pdb_path_batch = pdb_path_list[i : min(i + batch_size, len(pdb_path_list))]
            seqs_batch = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(get_saprot_seq)(str(pdb_path), cfg.foldseek_path)
                for pdb_path in pdb_path_batch
            )
            # seqs_batch = []
            # for pdb_path in pdb_path_batch:
            #    seqs_batch.append(get_saprot_seq(str(pdb_path), foldseek_path=cfg.foldseek_path))
            seqs += seqs_batch

        dataset = FastaBatchedDataset(labels, seqs)
        batches = dataset.get_batch_indices(cfg.toks_per_batch, extra_toks_per_seq=1)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_sampler=batches,
        )
        dataloaders.append(dataloader)
    return dataloaders


@hydra.main(
    version_base="1.3",
    config_path="./configs",
    config_name="saprot_linprobe_ted",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    # model, alphabet = load_esm_saprot(cfg.model.path)
    model = SaprotBaseModel(
        task="base", config_path=cfg.model.path, load_pretrained=True
    )
    model.eval()
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.path)

    train_loader, val_loader, test_loader, ext_test_loader = get_dataloader_from_cfg(
        cfg, tokenizer
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
