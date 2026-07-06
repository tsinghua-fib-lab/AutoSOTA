"""Test MiAE-S with Test-Time Augmentation."""
import logging
from pathlib import Path
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tedbench.model import MiAE, MiAEClassifier
from tedbench.utils.io import load_from_hf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
OmegaConf.register_new_resolver("eval", eval)
log = logging.getLogger(__name__)


class MiAEClassifierTTA(MiAEClassifier):
    def __init__(self, cfg, tta_noise_stds=[0.1, 0.2, 0.3]):
        super().__init__(cfg)
        self.tta_noise_stds = tta_noise_stds
    
    def test_step(self, batch, batch_idx):
        coords = batch["coords"]
        mask = batch["mask"]
        residue_index = batch["residue_index"] - 1
        seq_tokens = batch["seq_ids"]
        y_true = batch.get("label")
        bs = coords.shape[0]
        
        # Standard forward pass
        y_pred = self(coords, mask, residue_index, seq_tokens=seq_tokens)
        # TTA: average logits over K noisy forward passes
        for noise_std in self.tta_noise_stds:
            noisy = coords + torch.randn_like(coords) * noise_std
            y_pred = y_pred + self(noisy, mask, residue_index, seq_tokens=seq_tokens)
        y_pred = y_pred / (1 + len(self.tta_noise_stds))
        
        loss = self.loss_fn(y_pred, y_true)
        self.metric_fn.update(y_pred, y_true)
        acc = (y_pred.argmax(dim=-1) == y_true).float().mean()
        self.log_dict({"test/loss": loss, "test/acc": acc}, sync_dist=True, batch_size=bs)
        return loss


@hydra.main(version_base="1.3", config_path="/repo/configs", config_name="finetune_ted")
def main(cfg):
    log.info(f"TTA Test with noise stds: [0.1, 0.2, 0.3]")
    pl.seed_everything(cfg.seed, workers=True)
    
    model = MiAEClassifierTTA(cfg)
    p = cfg.pretrained_model_path
    pretrained_model = load_from_hf(p).state_dict()
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in pretrained_model and pretrained_model[k].shape != state_dict[k].shape:
            del pretrained_model[k]
    msg = model.load_state_dict(pretrained_model, strict=False)
    log.info(f"Loaded pretrained model. Missing keys: {msg.missing_keys}")
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    logger = [pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs")]
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=[])
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
