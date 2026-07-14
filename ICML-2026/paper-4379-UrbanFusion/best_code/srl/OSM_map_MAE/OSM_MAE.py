#!/usr/bin/env python3
"""
Description: This implements a masked-autoencoder model based on Hugging
Face's ViT.
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, ViTMAEConfig, ViTMAEForPreTraining


class MaskedAutoencoderViT(nn.Module):
    """
    Wrapper around Hugging Face ViT MAE to match existing MaskedAutoencoderViT
    interface.

    Methods
    -------
    - patchify(imgs): split images into patches
    - unpatchify(x): reconstruct images from patches
    - forward(imgs, return_loss_only=True): compute loss or return (recon_imgs, masked_imgs)
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/vit-mae-base",
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        # Load pretrained MAE model and processor
        config = ViTMAEConfig.from_pretrained(pretrained_model_name)
        config.mask_ratio = mask_ratio
        self.model = ViTMAEForPreTraining.from_pretrained(
            pretrained_model_name, config=config
        )
        self.processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name, use_fast=True
        )
        self.processor.do_rescale = False

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Split images into flattened patches.

        Parameters
        ----------
        imgs : torch.Tensor
            Input tensor of shape (B, 3, H, W), where:
            - B is the batch size
            - 3 is the number of color channels (RGB)
            - H and W are the height and width of the images.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, patch_size * patch_size * 3), where:
            - N is the number of patches per image.
            - Each patch is flattened into a vector of length
            ``patch_size * patch_size * 3``.
        """
        return self.model.patchify(imgs)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images from flattened patches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, patch_size * patch_size * 3), where:
            - B is the batch size
            - N is the number of patches per image
            - Each patch is a flattened vector of length
            ``patch_size * patch_size * 3``.

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (B, 3, H, W), where:
            - 3 is the number of color channels (RGB)
            - H and W are the height and width of the reconstructed images.
        """
        return self.model.unpatchify(x)

    def denormalize(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization applied by the processor, producing images in the [0, 1] range.

        Parameters
        ----------
        imgs : torch.Tensor
            Normalized images of shape (B, C, H, W), where:
            - B is the batch size
            - C is the number of channels
            - H and W are the height and width of the images.

        Returns
        -------
        torch.Tensor
            Images scaled back to the [0, 1] range with shape (B, C, H, W).
        """
        # Retrieve ImageNet normalization parameters
        mean = torch.tensor(
            self.processor.image_mean, device=imgs.device
        ).view(1, -1, 1, 1)
        std = torch.tensor(self.processor.image_std, device=imgs.device).view(
            1, -1, 1, 1
        )
        # Un-normalize: (x * std) + mean -> yields values in [0, 1]
        imgs_denorm = imgs * std + mean
        return imgs_denorm

    def forward(self, imgs: torch.Tensor, return_loss_only: bool = True):
        """
        Forward pass matching the original API.

        Parameters
        ----------
        return_loss_only : bool
            If ``True``, returns only the mean absolute error (MAE) loss.
            If ``False``, returns reconstructed images and masked input images.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            - If ``return_loss_only`` is ``True``: A scalar tensor
            representing the MAE loss.
            - If ``False``: A tuple ``(reconstructed_images,
            masked_input_images)`` where:
                reconstructed_images : torch.Tensor
                    Reconstructed images of shape (B, 3, H, W).
                masked_input_images : torch.Tensor
                    Input images with masking applied, shape (B, 3, H, W).
        """
        inputs = self.processor(images=imgs, return_tensors="pt").to(
            imgs.device
        )
        pixel_values = inputs.pixel_values  # shape [B, C, H, W]

        outputs = self.model(pixel_values=pixel_values, return_dict=True)

        loss = outputs.loss
        if return_loss_only:
            return loss

        # Reconstruct image from model output
        recon_imgs = self.unpatchify(outputs.logits)
        recon_imgs = self.denormalize(recon_imgs).clamp(0, 1)
        transform = T.Compose(
            [
                T.Resize((224, 224)),
            ]
        )
        imgs = transform(imgs)
        mask = outputs.mask.bool()
        patches = self.patchify(imgs)  # normalized input patches
        patches[mask] = 0  # zero out masked patches only
        masked_imgs = self.unpatchify(patches)  # restore image from masked

        return recon_imgs, masked_imgs
