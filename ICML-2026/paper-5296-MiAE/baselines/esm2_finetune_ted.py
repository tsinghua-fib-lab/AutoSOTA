import logging

import hydra
import pytorch_lightning as pl
import torch
from esm import FastaBatchedDataset
from models.esm2_classifier import ESM2Classifier
from omegaconf import OmegaConf
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


def get_dataloader_from_cfg(cfg, alphabet):
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    datamodule.setup("test")

    dataloader_splits = [
        datamodule.train_dataloader(drop_last=False, shuffle=False),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ]
    dataloaders = []

    for i, data_loader in enumerate(dataloader_splits):
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
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(
                truncation_seq_length=cfg.truncation_seq_length
            ),
            batch_size=cfg.batch_size,
            shuffle=i == 0,
            num_workers=cfg.datamodule.num_workers,
            pin_memory=True,
        )
        dataloaders.append(dataloader)
    return dataloaders


@hydra.main(
    version_base="1.3",
    config_path="./configs",
    config_name="esm2_finetune_ted",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    model = ESM2Classifier(cfg)
    alphabet = model.alphabet

    train_loader, val_loader, test_loader = get_dataloader_from_cfg(cfg, alphabet)

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
    model = ESM2Classifier.load_from_checkpoint(best_ckpt, cfg=cfg, weights_only=False)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
