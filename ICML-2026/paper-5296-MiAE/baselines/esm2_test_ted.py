import logging
from pathlib import Path

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
    datamodule_test = hydra.utils.instantiate(cfg.datamodule_test)
    datamodule_test.setup("test")

    seqs = []
    labels = []
    for batch in tqdm(datamodule_test.test_dataloader(), desc="Processing data"):
        protein_chains = batch["protein_chain"]
        y_true = batch["label"]
        seqs += [protein_chain.sequence for protein_chain in protein_chains]
        labels.append(y_true)
    labels = torch.concat(labels).tolist()

    dataset = FastaBatchedDataset(labels, seqs)
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(
            truncation_seq_length=cfg.truncation_seq_length
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
        pin_memory=True,
    )


def _load_model(cfg) -> ESM2Classifier:
    """Load ESM2Classifier from a local .ckpt or a HuggingFace repo ID."""
    p = cfg.train.ckpt_path
    if Path(p).is_dir() or not Path(p).suffix:
        from huggingface_hub import snapshot_download
        local_dir = Path(p) if Path(p).is_dir() else Path(snapshot_download(repo_id=p))
        model = ESM2Classifier(cfg)
        sd = torch.load(local_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        return model
    return ESM2Classifier.load_from_checkpoint(p, cfg=cfg, weights_only=False)


@hydra.main(
    version_base="1.3",
    config_path="./configs",
    config_name="esm2_finetune_ted",
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    model = _load_model(cfg)
    test_loader = get_dataloader_from_cfg(cfg, model.alphabet)

    logger = [pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs")]
    if cfg.wandb:
        logger.insert(0, pl.loggers.WandbLogger(
            project="TEDBench", config=OmegaConf.to_container(cfg, resolve=True)
        ))

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
