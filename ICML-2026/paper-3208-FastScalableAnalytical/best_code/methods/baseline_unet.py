from __future__ import annotations

from typing import Any, Dict, List

import torch

from methods import register_model


from methods import register_baseline_model
from methods.base import BaseDenoiser
from networks.UNet import UNet


@register_baseline_model("baseline_unet")
class BaselineUNet(BaseDenoiser):
    """Wrapper for the baseline U-Net model."""

    def __init__(
        self,
        resolution: int,
        device: str,
        num_steps: int,
        model_path: str,
        dataset_name: str = "cifar10",
        in_channels: int = 3,
        out_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            resolution=resolution,
            device=device,
            num_steps=num_steps,
            dataset_name=dataset_name,
            in_channels=in_channels,
            **kwargs,
        )
        self.model_path = model_path
        
        # Determine U-Net configuration based on resolution/dataset
        self.unet_config = self._get_unet_config(
            dataset_name, resolution, in_channels, out_channels
        )
        
        # T in U-Net config is usually 1000 (training steps). 
        # But here we pass it to the model constructor which uses it for time embeddings.
        # The reference code passes 1000.
        self.model = UNet(
            T=self.unet_config["T"],
            ch=self.unet_config["channel"],
            ch_mult=self.unet_config["channel_mult"],
            attn=self.unet_config["attn"],
            num_res_blocks=self.unet_config["num_res_blocks"],
            dropout=self.unet_config["dropout"],
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.model.to(self.device)
        self._load_weights()
        self.model.eval()


        # total_params = sum(p.numel() for p in self.model.parameters())

    def _get_unet_config(
        self, dataset_name: str, img_size: int, in_channels: int, out_channels: int
    ) -> Dict[str, Any]:
        """Derive U-Net architecture config based on dataset/resolution."""
        
        if img_size == 28:  # MNIST, FashionMNIST
            channel = 64
            channel_mult = [1, 2, 2]  # Only 3 downsamples: 28->14->7->3
        elif img_size == 32:  # CIFAR10, FFHQ
            channel = 128
            channel_mult = [1, 2, 3, 4]  # 32->16->8->4->2
        elif img_size == 64:  # CelebA-HQ, AFHQ
            channel = 128
            channel_mult = [1, 2, 3, 4]  # 64->32->16->8->4
        else:
            # Default fallback or error
             channel = 128
             channel_mult = [1, 2, 3, 4]

        return {
            "T": 1000, # Fixed in reference
            "channel": channel,
            "channel_mult": channel_mult,
            "attn": [],
            "num_res_blocks": 2,
            "dropout": 0.15,
        }

    def _load_weights(self) -> None:
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                 self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                 self.model.load_state_dict(checkpoint)
            else:
                 self.model.load_state_dict(checkpoint.state_dict())
        except Exception as e:
            print(f"Warning: Failed to load weights from {self.model_path}: {e}")
            raise e

    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        B = latents.shape[0]
        
        if isinstance(timestep, (int, float)):
             ts = torch.full((B,), int(timestep), device=self.device, dtype=torch.long)
        elif timestep.ndim == 0:
             ts = timestep.expand(B).long().to(self.device)
        else:
             ts = timestep.long().to(self.device)
        eps = self.model(latents, ts)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[ts.cpu().long()]
        alpha_prod_t = alpha_prod_t.to(self.device).reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        
        pred_x0 = (latents - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()
        
        return pred_x0

    def train(self, dataset: Any) -> None:
        # Pre-trained model, nothing to do.
        pass

