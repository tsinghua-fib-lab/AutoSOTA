import logging
import os
from pathlib import Path

import hydra
import joblib
import pytorch_lightning as pl
import torch
from biotite.structure.io.pdb import PDBFile
from esm import FastaBatchedDataset
from models.saprot_classifier import SaProtClassifier
from omegaconf import OmegaConf
from saprot.utils.foldseek_util import get_struc_seq
from tqdm import tqdm
from transformers import EsmTokenizer

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


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


def batch_converter(raw_batch, tokenizer, truncation_length=512):
    batch_labels, seq_str_list = zip(*raw_batch)
    tokens = tokenizer(
        seq_str_list,
        padding=True,
        truncation=True,
        max_length=truncation_length,
        return_tensors="pt",
    )
    return torch.tensor(batch_labels), seq_str_list, tokens["input_ids"]


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
    cache_path = Path(cfg.datamodule.root) / "processed_files"
    datamodule_test = hydra.utils.instantiate(cfg.datamodule_test)
    datamodule_test.setup("test")
    tmpdir = datamodule_test.test_dataset.root / "pdb_files"
    ext_test_pdb, ext_test_labels = get_cath_data(datamodule_test.test_dataset, tmpdir)

    def collate_fn(raw_batch):
        return batch_converter(
            raw_batch, tokenizer, truncation_length=cfg.truncation_seq_length
        )

    cache_path_split = cache_path / "saprot_ext_test.pt"
    if cache_path_split.exists():
        seqs = torch.load(cache_path_split, weights_only=False)
    else:
        seqs = []
        for i in tqdm(
            range(0, len(ext_test_pdb), cfg.datamodule.batch_size), desc="Processing data"
        ):
            pdb_path_batch = ext_test_pdb[i : i + cfg.datamodule.batch_size]
            seqs += joblib.Parallel(n_jobs=-1)(
                joblib.delayed(get_saprot_seq)(str(pdb_path), cfg.foldseek_path)
                for pdb_path in pdb_path_batch
            )
        cache_path.mkdir(parents=True, exist_ok=True)
        torch.save(seqs, cache_path_split)

    dataset = FastaBatchedDataset(ext_test_labels, seqs)
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
        pin_memory=True,
    )


def _load_model(cfg) -> SaProtClassifier:
    """Load SaProtClassifier from a local .ckpt or a HuggingFace repo ID."""
    p = cfg.train.ckpt_path
    if Path(p).is_dir() or not Path(p).suffix:
        from huggingface_hub import snapshot_download
        local_dir = Path(p) if Path(p).is_dir() else Path(snapshot_download(repo_id=p))
        model = SaProtClassifier(cfg)
        sd = torch.load(local_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        return model
    return SaProtClassifier.load_from_checkpoint(p, cfg=cfg, weights_only=False)


@hydra.main(
    version_base="1.3",
    config_path="./configs",
    config_name="saprot_finetune_ted",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    model = _load_model(cfg)
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.path)
    test_loader = get_dataloader_from_cfg(cfg, tokenizer)

    logger = [pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs")]
    if cfg.wandb:
        logger.insert(0, pl.loggers.WandbLogger(
            project="TEDBench", config=OmegaConf.to_container(cfg, resolve=True)
        ))

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
