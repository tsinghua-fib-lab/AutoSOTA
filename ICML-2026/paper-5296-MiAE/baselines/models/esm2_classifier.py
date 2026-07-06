import hydra
import pytorch_lightning as pl
import torch
from esm import pretrained
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from tedbench.lr_schedulers import get_layerwise_lr_decay


class ESM2Classifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model, self.alphabet = pretrained.load_model_and_alphabet(cfg.model.name)
        self.head = nn.Linear(self.model.embed_dim, cfg.model.num_classes)
        self.num_layers = self.model.num_layers + 1

        self.instantiate_loss()
        self.save_hyperparameters()

    def instantiate_loss(self):
        self.loss_fn = hydra.utils.call(self.cfg.train.loss)
        self.metric_fn = MetricCollection(
            {
                "balanced_acc": MulticlassAccuracy(
                    num_classes=self.cfg.model.num_classes, average="macro"
                ),
                "macro_f1": MulticlassF1Score(
                    num_classes=self.cfg.model.num_classes, average="macro"
                ),
            }
        )

    def get_layer_id_by_param_name(self, name):
        if name.startswith("model.embed_tokens"):
            return 0
        elif name.startswith("model.layers"):
            return int(name.split(".")[2]) + 1
        else:
            return self.num_layers

    def forward(self, labels, strs, toks):
        out = self.model(
            toks, repr_layers=[self.model.num_layers], return_contacts=False
        )
        out = out["representations"][self.model.num_layers]
        if self.cfg.model.avg_pool:
            out = out[:, 1:]
            mask = toks != self.model.padding_idx
            mask = mask[:, 1:].float().unsqueeze(-1)
            out = out * mask
            out = out.sum(dim=1) / mask.sum(dim=1)
        else:
            out = out[:, 0]
        return self.head(out)

    def shared_step(self, batch, batch_idx, phase="train"):
        labels, strs, toks = batch
        y_true = torch.tensor(labels, device=toks.device, dtype=torch.long)
        bs = toks.shape[0]
        y_pred = self(labels, strs, toks)
        loss = self.loss_fn(y_pred, y_true)
        with torch.no_grad():
            acc = (y_pred.argmax(dim=-1) == y_true).float().mean()
        if phase != "train":
            self.metric_fn.update(y_pred, y_true)
        metrics = {"loss": loss, "acc": acc}
        log_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
        self.log_dict(
            log_metrics, sync_dist=True, batch_size=bs, on_step=False, on_epoch=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase="test")
        return loss

    def on_validation_epoch_end(self):
        metrics = self.metric_fn.compute()
        for key, value in metrics.items():
            self.log(f"val/{key}", value, sync_dist=True)
        self.metric_fn.reset()

    def on_test_epoch_end(self):
        metrics = self.metric_fn.compute()
        for key, value in metrics.items():
            self.log(f"test/{key}", value, sync_dist=True)
        self.metric_fn.reset()

    def configure_optimizers(self):
        if self.cfg.train.llrd < 1:
            optim_groups = get_layerwise_lr_decay(
                self, self.cfg.train.optimizer.lr, self.cfg.train.llrd
            )
        else:
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, optim_groups)
        lr_scheduler = hydra.utils.call(
            self.cfg.train.lr_scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
