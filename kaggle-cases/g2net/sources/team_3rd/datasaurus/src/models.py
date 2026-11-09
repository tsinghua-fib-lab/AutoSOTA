import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torchmetrics
from nnAudio import Spectrogram

from src.augmentation import SpecAugmentation
from src.cwt import CWT
from src.scalers import standard_scaler, standard_scaler_1d
from src.utils import add_weight_decay, mixup_data
from src.resnet1d import ResNet1D
from src.cnn1d import Model1DCNN
from src.losses import rank_loss


cnn1d_models = ["resnet1d", "model1dcnn"]


class GWModel(pl.LightningModule):
    def __init__(
        self,
        encoder: str = "resnet50d",
        lr: float = 0.001,
        weight_decay: float = 0,
        fmin: int = 20,
        fmax: int = 500,
        hop_length: int = 16,
        train_filter: bool = False,
        cwt: bool = False,
        mixup_alpha: float = 0.0,
        window: str = "hann",
        img_size: list = [256, 512],
        bins_per_octave: int = 12 * 2,
        filter_scale: float = 1.0 / 2,
        norm: int = 1,
        warmup: int = 0,
        rn1d_params: dict = {},
        stage2: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.is_vit = encoder.startswith("vit")

        if encoder == "resnet1d":
            self.model = ResNet1D(
                in_channels=3,
                n_classes=1,
                verbose=False,
                **rn1d_params,
            )
        elif encoder == "model1dcnn":
            self.model = Model1DCNN(32)
        else:
            self.model = timm.create_model(
                encoder,
                pretrained=True,
                num_classes=1,
                in_chans=3 if self.is_vit else 4,
            )

            if cwt:
                self.spec_transform = CWT(
                    dj=0.125 / 8, dt=1 / 2048, fmin=fmin, fmax=fmax, hop_length=8
                )
            else:
                self.spec_transform = Spectrogram.CQT1992v2(
                    sr=2048,
                    fmin=fmin,
                    fmax=fmax,
                    hop_length=hop_length,  # img width = sig_length / hop_length
                    trainable=train_filter,  # This is interesting...
                    window=window,
                    norm=norm,
                    # Oversampling freq axis
                    bins_per_octave=bins_per_octave,
                    filter_scale=filter_scale,
                )

        self.spec_augment = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=8,
            freq_drop_width=8,
            freq_stripes_num=4,
        )

        if stage2:
            self.loss_fn = rank_loss
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        self.metric = torchmetrics.AUROC()

    def prepare_image(self, x):
        if self.hparams.cwt:
            spec = self.spec_transform(x)

        else:
            bs = x.shape[0]
            x_reshaped = x.reshape(-1, 4096)
            spec = self.spec_transform(x_reshaped)
            spec = spec.reshape(bs, 3, spec.shape[1], spec.shape[2])

        if self.hparams.img_size:
            spec = nn.functional.interpolate(
                spec, tuple(self.hparams.img_size), mode="bilinear"
            )

        spec = standard_scaler(spec)
        # spec = standard_scaler(spec, imagenet=True)

        if self.is_vit:
            # spec = self.inv_stem(spec)
            pass
        else:
            spec = self.frequency_encoding(spec)

        return spec

    # https://github.com/jfpuget/STFT_Transformer/blob/bb48e4f032736543f3220a773b0a413b6b6db768/stft_transformer_final.py#L235-L241
    def inv_stem(self, x):
        x1 = x.transpose(2, 3).view(x.shape[0], x.shape[1], 24, 24, 16, 16)
        y = torch.zeros(
            x.shape[0], x.shape[1], 384, 384, dtype=x.dtype, device=x.device
        )
        for i in range(24):
            for j in range(24):
                y[:, :, i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16] = x1[:, :, i, j]
        return y

    def channel_permute(self, x):
        mask = torch.randint(0, 2, size=(x.shape[0],)).bool().to(x.device)
        permute = torch.tensor([1, 0, 2]).to(x.device)
        x[mask] = x[:, permute][mask]
        return x

    def forward(self, x):
        if x.ndim != 4 and self.hparams.encoder not in cnn1d_models:
            x = self.prepare_image(x)
        else:
            x = standard_scaler_1d(x)

        return self.model(x)

    def frequency_encoding(self, x):
        device = x.device
        bs, fbins, t = x.shape[0], x.shape[2], x.shape[3]
        freq_encoding = 2 * torch.arange(fbins, device=device) / fbins - 1  # -1 to +1
        freq_encoding = torch.stack([freq_encoding] * t, -1).unsqueeze(0)
        freq_encoding = torch.stack([freq_encoding] * bs)
        return torch.cat([x, freq_encoding], 1)

    def mixed_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)

    def training_step(self, batch, batch_nb):
        data, target = batch
        # img = self.spec_augment(img)

        data = self.channel_permute(data)

        if self.hparams.mixup_alpha > 0:
            # img = self.prepare_image(data)
            (data_mix, target_a, target_b, lam) = mixup_data(
                data, target, self.hparams.mixup_alpha
            )
            logits = self.forward(data_mix)

            if self.is_vit:
                logits = logits[0]

            loss = self.mixed_criterion(logits, target_a, target_b, lam)
        else:
            logits = self.forward(data)

            if self.is_vit:
                logits = logits[0]

            loss = self.loss_fn(logits, target)

        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)

    def validation_step(self, batch, batch_nb):
        data, target = batch
        logits = self.forward(data)
        loss = self.loss_fn(logits, target)
        self.metric.update(logits, target.long())
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log_dict(
            {"loss/valid": loss_val, "auc": self.metric.compute()},
            prog_bar=True,
            sync_dist=True,
        )
        self.metric.reset()

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # skip the first 500 steps
        if self.trainer.global_step < self.hparams.warmup:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        parameters = add_weight_decay(
            self,
            self.hparams.weight_decay,
            skip_list=["bias", "LayerNorm.bias", "LayerNorm.weight"],
        )

        opt = torch.optim.AdamW(
            parameters,
            lr=self.hparams.lr,
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return [opt], [sch]


class GWDenoisingModel(pl.LightningModule):
    def __init__(
        self,
        encoder: str = "resnet50d",
        lr: float = 0.001,
        weight_decay: float = 0,
        fmin: int = 20,
        fmax: int = 500,
        hop_length: int = 16,
        train_filter: bool = False,
        cwt: bool = False,
        mixup_alpha: float = 0.0,
        window: str = "hann",
        img_size: list = [256, 512],
        bins_per_octave: int = 12 * 2,
        filter_scale: float = 1.0 / 2,
        norm: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if cwt:
            self.spec_transform = CWT(
                dj=0.125 / 8, dt=1 / 2048, fmin=fmin, fmax=fmax, hop_length=8
            )
        else:
            self.spec_transform = Spectrogram.CQT1992v2(
                sr=2048,
                fmin=fmin,
                fmax=fmax,
                hop_length=hop_length,  # img width = sig_length / hop_length
                trainable=train_filter,  # This is interesting...
                window=window,
                norm=norm,
                # Oversampling freq axis
                bins_per_octave=bins_per_octave,
                filter_scale=filter_scale,
            )

        # https://github.com/qubvel/segmentation_models.pytorch#auxiliary-classification-output
        # aux_params = dict(
        #     pooling="avg",  # one of 'avg', 'max'
        #     dropout=0.2,  # dropout ratio, default is None
        #     classes=1,  # define number of output labels
        # )
        self.model = smp.Unet(
            encoder_name=encoder,
            classes=3,
            in_channels=3,
            # aux_params=aux_params,
            encoder_weights="imagenet",
        )

        self.spec_augment = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=8,
            freq_drop_width=8,
            freq_stripes_num=4,
        )

        # self.clf_loss_fn = nn.BCEWithLogitsLoss()
        self.recon_loss_fn = nn.MSELoss()
        # self.metric = torchmetrics.AUROC()

    def prepare_image(self, x, freq_encode=True):
        if self.hparams.cwt:
            spec = self.spec_transform(x)

        else:
            bs = x.shape[0]
            x_reshaped = x.reshape(-1, 4096)
            spec = self.spec_transform(x_reshaped)
            spec = spec.reshape(bs, 3, spec.shape[1], spec.shape[2])

        if self.hparams.img_size:
            spec = nn.functional.interpolate(
                spec, tuple(self.hparams.img_size), mode="bilinear"
            )

        spec = standard_scaler(spec)
        # spec = self.frequency_encoding(spec)
        return spec

    def forward(self, x):
        if x.ndim != 4:
            x = self.prepare_image(x)

        return self.model(x)

    def frequency_encoding(self, x):
        device = x.device
        bs, fbins, t = x.shape[0], x.shape[2], x.shape[3]
        freq_encoding = 2 * torch.arange(fbins, device=device) / fbins - 1  # -1 to +1
        freq_encoding = torch.stack([freq_encoding] * t, -1).unsqueeze(0)
        freq_encoding = torch.stack([freq_encoding] * bs)
        return torch.cat([x, freq_encoding], 1)

    def mixed_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)

    def training_step(self, batch, batch_nb):
        data, data_clean, target = batch

        img_clean = self.prepare_image(data_clean)
        img_clean[~target.flatten().bool()] = 0

        # TODO: Mixup
        recon, logits = self.forward(data)
        loss = self.recon_loss_fn(recon, img_clean)

        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        avg_loss1 = torch.stack([x["loss_recon"] for x in training_step_outputs]).mean()
        avg_loss2 = torch.stack([x["loss_clf"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)
        # self.log_dict(
        #     {"loss/train_recon": avg_loss1, "loss/train_clf": avg_loss2},
        #     prog_bar=False,
        #     sync_dist=True,
        # )

    def validation_step(self, batch, batch_nb):
        # data, target = batch
        data, data_clean, target = batch

        img_clean = self.prepare_image(data_clean)
        img_clean[~target.flatten().bool()] = 0

        recon, logits = self.forward(data)
        loss = self.clf_loss_fn(recon, img_clean)
        # self.metric.update(logits, target.long())
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log_dict(
            {"loss/valid": loss_val},
            prog_bar=True,
            sync_dist=True,
        )
        # self.metric.reset()

    def configure_optimizers(self):
        parameters = add_weight_decay(
            self,
            self.hparams.weight_decay,
            skip_list=["bias", "LayerNorm.bias", "LayerNorm.weight"],
        )

        opt = torch.optim.AdamW(
            parameters,
            lr=self.hparams.lr,
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return [opt], [sch]
