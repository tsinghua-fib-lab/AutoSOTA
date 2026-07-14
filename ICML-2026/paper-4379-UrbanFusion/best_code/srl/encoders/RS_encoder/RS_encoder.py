#!/usr/bin/env python3
"""
Description: Implementation of ViT for Sentinel-2 images.
Same backbone as in SatCLIP (SENTINEL2_ALL_MOCO).
"""

import timm
import torch
import torch.nn as nn
from torchgeo.models import ViTSmall16_Weights


class RSEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        seq_len: int = 1,
        return_encoding: bool = False,
        precomputed_features: bool = False,
    ) -> None:
        """
        A Vision Transformer (ViT) encoder for Sentinel-2 images. The encoder
        uses a pre-trained ViT model with a projection head to create token
        embeddings. The model is based on the ViT architecture and is
        specifically designed for processing Sentinel-2 images.

        Parameters
        ----------
        embed_dim : int, optional
            Output embedding dimensionality, by default 512.
        seq_len : int, optional
            Number of output tokens after projection head, by default 1.
        return_encoding : bool, optional
            Whether to return the encoding of the image model, by default
            False.
        precomputed_features : bool, optional
            Whether to use precomputed features, by default False.
            If True, the model will not compute the image features.
            This is useful for speeding up training when the image features are
            already computed and stored in the dataset, by default False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.return_encoding = return_encoding
        self.precomputed_features = precomputed_features
        weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
        in_chans = weights.meta["in_chans"]
        self.visual = timm.create_model(
            "vit_small_patch16_224", in_chans=in_chans, num_classes=512
        )
        self.visual.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )
        self.visual.requires_grad_(False)
        self.visual.head = nn.Sequential(
            nn.Linear(self.visual.embed_dim, self.embed_dim * self.seq_len),
            nn.LayerNorm(self.embed_dim * self.seq_len),
            nn.GELU(),
        )

    def to(self, device: torch.device) -> "RSEncoder":
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the model to (e.g., 'cpu' or 'cuda').

        Returns
        -------
        RSEncoder
            The model instance moved to the specified device.
        """
        if not self.precomputed_features:
            self.visual.to(device)
        else:
            self.visual.to("cpu")
            self.visual.head.to(device)
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the image through the backbone.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The processed image tensor.
        """
        x = self.visual.forward_features(x)[:, 0]
        return x

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model for creating token embeddings.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.
        return_features : bool, optional
            Whether to return the encoding of the image model, by default
            False. If True, the method returns both the projected embeddings
            and the encoding of the image model.

        Returns
        -------
        torch.Tensor
            Projected embeddings of shape (B, seq_len, embed_dim).
        """
        if self.precomputed_features:
            encoding = x
        else:
            encoding = self.visual.forward_features(x)[:, 0]
        x = self.visual.head(encoding)
        x = x.view(x.shape[0], self.seq_len, -1)
        if self.return_encoding or return_features:
            return x, encoding.detach()
        return x
