import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, List
from dataclasses import dataclass

# SCAM: Self-Correction with Adaptive Mask


class Model(nn.Module):
    def __init__(
        self,
        name: str,
        embedding: nn.Module,
        decoder: nn.Module,
        predictor: nn.Module,
        input_size: int,
        output_size: int,
        num_channels: int,
        if_softmask: bool,
        loss_names: List[str],
        task: Optional[str] = None,
    ):
        super(Model, self).__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.predictor = predictor(
            input_size=input_size, output_size=output_size, num_channels=num_channels
        )
        self.if_softmask = if_softmask

    def init_params(self):
        pass

    def predict(self, x, x_mark, *args, **kwargs):
        return self.predictor(x, x_mark)

    def reconstruct(self, x):
        return self.decoder(self.embedding(x))

    def compute_loss(
        self,
        y: torch.Tensor,
        y_rec: torch.Tensor,
        y_hat: torch.Tensor,
    ):
        loss_rec = F.l1_loss(
            y[:, :, None, :].repeat(1, 1, y_rec.shape[-2], 1), y_rec, reduction="none"
        )
        loss_pred = F.l1_loss(y_hat, y_rec, reduction="none")
        with torch.no_grad():
            mask_y = (y_rec - y[:, :, None, :]) * (y_rec - y_hat) > 0

        
        with torch.no_grad():
            mask_l = loss_pred < loss_rec
        

        loss_l2 = F.mse_loss(
            y_hat, y[:, :, None, :].repeat(1, 1, y_hat.shape[-2], 1), reduction="none"
        )

        loss_tar = loss_l2  # + loss_l1
        
        loss_pred = 2 * mask_y * mask_l * loss_pred + mask_y * F.mse_loss(y_hat, y_rec, reduction="none") # for faster convergence to add mask_y for loss_pred
        loss_rec = 2 * mask_y * ~mask_l * loss_rec
        loss_tar = ~mask_y * loss_tar 

        return loss_pred.mean(), loss_rec.mean(), loss_tar.mean()

    def forward(self, x, y, x_mark, y_mark, *args, **kwargs):
        result = {}

        y_rec = self.reconstruct(y) # [B, C, N, L] N for N condidates
        y_hat = self.predictor(x, x_mark)
            
        result["loss_pred"], result["loss_rec"], result["loss_tar"] = self.compute_loss(
            y, y_rec, y_hat
        )
        result["loss_total"] = (
            result["loss_pred"] + result["loss_rec"] + result["loss_tar"]
        )
        result["y_hat"] = y_hat.mean(dim=-2)

        return result
