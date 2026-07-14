#!/usr/bin/env python3
"""
Description: Implementation of a CLIP-based image encoder for
street view image embeddings (same backbone as GeoCLIP).
"""

import warnings

import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel

warnings.filterwarnings(
    "ignore", category=UserWarning, module="huggingface_hub.*"
)


class SVIEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        seq_len: int = 1,
        return_encoding: bool = False,
        precomputed_features: bool = False,
    ) -> None:
        """
        A CLIP-based image encoder for street view image embeddings.
        The encoder uses the CLIP model to extract image features and then
        projects them into a lower-dimensional space using an MLP.
        The output shape is (B, seq_len, embed_dim), where B is the batch size.

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
        self.CLIP = CLIPModel.from_pretrained("/models/hf/hub/models--openai--clip-vit-large-patch14/snapshots/abc123")
        self.image_processor = AutoProcessor.from_pretrained("/models/hf/hub/models--openai--clip-vit-large-patch14/snapshots/abc123")
        self.head = nn.Sequential(
            nn.Linear(768, self.embed_dim * self.seq_len),
            nn.LayerNorm(self.embed_dim * self.seq_len),
            nn.GELU(),
        )

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the image for the CLIP model.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The preprocessed image tensor.
        """
        x = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        return x

    def to(self, device: torch.device) -> "SVIEncoder":
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the model to (e.g., 'cpu' or 'cuda').

        Returns
        -------
        SVIEncoder
            The model instance moved to the specified device.
        """
        self.head.to(device)
        if not self.precomputed_features:
            self.CLIP.to(device)
        else:
            self.CLIP.to("cpu")
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the image through the CLIP backbone.

        Parameters
        ----------
        x : torch.Tensor
            The input image tensor.

        Returns
        -------
        torch.Tensor
            The processed image tensor.
        """
        x = self.CLIP.get_image_features(pixel_values=x)
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

        Returns
        -------
        torch.Tensor
            Projected embeddings of shape (B, seq_len, embed_dim).
        """
        if self.precomputed_features:
            encoding = x
        else:
            encoding = self.CLIP.get_image_features(pixel_values=x)
        x = self.head(encoding)
        # Reshape to (batch_size, seq_len, embed_dim)
        x = x.view(x.shape[0], self.seq_len, -1)
        if self.return_encoding or return_features:
            return x, encoding.detach()
        return x
