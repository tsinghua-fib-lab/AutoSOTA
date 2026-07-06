import logging

import os
import hydra
import pytorch_lightning as pl
import torch
import joblib
from pathlib import Path
from esm import FastaBatchedDataset
from models.saprot_classifier import SaProtClassifier
from transformers import EsmTokenizer
from saprot.utils.foldseek_util import get_struc_seq
from omegaconf import OmegaConf
from tqdm import tqdm
from biotite.structure.io.pdb import PDBFile


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


def get_saprot_seq(pdb_path, foldseek_path="/fs/gpfs41/lv11/fileset01/pool/pool-chen/softwares/micromamba/envs/foldseek/bin/foldseek"):
    atom_array = PDBFile.read(pdb_path).get_structure(model=1)
    chain_id = atom_array.chain_id[0]
    pid = os.getpid()
    parsed_seqs = get_struc_seq(
        foldseek_path,
        pdb_path,
        [chain_id],
        process_id=pid,
        plddt_mask=False
    )[chain_id]
    return parsed_seqs[-1]


def batch_converter(raw_batch, tokenizer, truncation_length=512):
    batch_size = len(raw_batch)
    batch_labels, seq_str_list = zip(*raw_batch)
    tokens = tokenizer(seq_str_list, padding=True, truncation=True, max_length=truncation_length, return_tensors='pt')
    return torch.tensor(batch_labels), seq_str_list, tokens['input_ids']


def get_dataloader_from_cfg(cfg, tokenizer):
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    datamodule.setup("test")
    cache_path = Path(cfg.datamodule.root) / "processed_files"

    dataloader_splits = [
        ("train", datamodule.train_dataset.pdb_files_split, datamodule.train_dataset.cath_labels,),
        ("val", datamodule.val_dataset.pdb_files_split, datamodule.val_dataset.cath_labels,),
        ("test", datamodule.test_dataset.pdb_files_split, datamodule.test_dataset.cath_labels,)
    ]
    dataloaders = []
    batch_size = cfg.datamodule.batch_size
    num_workers = cfg.datamodule.num_workers
    def collate_fn(raw_batch):
        return batch_converter(raw_batch, tokenizer, truncation_length=cfg.truncation_seq_length)

    for idx, data_loader in enumerate(dataloader_splits):
        #seqs = []
        #labels = []
        split_name, pdb_path_list, labels = data_loader
        cache_path_split = cache_path / f"saprot_{split_name}.pt"
        if cache_path_split.exists():
            seqs = torch.load(cache_path_split, weights_only=False)
        else:
            seqs = []
            for i in tqdm(range(0, len(pdb_path_list), batch_size), desc="Processing data"):
                pdb_path_batch = pdb_path_list[i : min(i + batch_size, len(pdb_path_list))]
                seqs_batch = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(get_saprot_seq)(str(pdb_path), cfg.foldseek_path)
                    for pdb_path in pdb_path_batch
                )
                seqs += seqs_batch
                #labels += all_labels[i : min(i + batch_size, len(pdb_path_list))]
            torch.save(seqs, cache_path_split)

        dataset = FastaBatchedDataset(labels, seqs)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=cfg.batch_size,
            shuffle=idx == 0,
            num_workers=cfg.datamodule.num_workers,
            pin_memory=True,
        )
        dataloaders.append(dataloader)
    return dataloaders


@hydra.main(
    version_base="1.3",
    config_path="./configs",
    config_name="saprot_finetune_ted",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    model = SaProtClassifier(cfg)
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.path)

    train_loader, val_loader, test_loader = get_dataloader_from_cfg(cfg, tokenizer)

    logger = []
    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(
            project="TEDBench", config=OmegaConf.to_container(cfg, resolve=True)
        )
        logger.append(wandb_logger)
    logger.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            monitor="val/acc",
            dirpath=cfg.logs.path,
            filename="model",
            mode="max",
        ),
    ]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    if cfg.train.ckpt_path is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.train.ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(f"{cfg.logs.path}/model-last.ckpt")
    best_ckpt = trainer.checkpoint_callback.best_model_path
    model = SaProtClassifier.load_from_checkpoint(best_ckpt, cfg=cfg, weights_only=False)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
