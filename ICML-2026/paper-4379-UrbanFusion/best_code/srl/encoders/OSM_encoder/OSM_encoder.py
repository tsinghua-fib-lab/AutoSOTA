#!/usr/bin/env python3
"""
Description: Implementation of a ViT-MAE encoder for OSM images.
"""

import torch
from torch import nn
from transformers import AutoImageProcessor, ViTMAEConfig, ViTMAEModel


class OSMEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "facebook/vit-mae-base",
        checkpoint_path: str = None,
        embed_dim: int = 512,
        seq_len: int = 1,
        return_encoding: bool = False,
        precomputed_features: bool = False,
    ) -> None:
        """
        A ViT-MAE encoder for OSM images.
        The encoder uses the ViT-MAE model to extract image features and then
        projects them into a lower-dimensional space using an MLP.
        The output shape is (B, seq_len, embed_dim), where B is the batch size.

        Parameters
        ----------
        pretrained_model_name : str, optional
            Name of the ViT-MAE model, by default "facebook/vit-mae-base".
        checkpoint_path : str, optional
            Path to the checkpoint file, by default None.
        embed_dim : int, optional
            Output embedding dimensionality, by default 512.
        seq_len : int, optional
            Number of output tokens after projection head, by default 1.
        return_encoding : bool, optional
            If True, return the encoding before the projection head,
            by default False.
        precomputed_features : bool, optional
            If True, the input is expected to be precomputed features
            of shape (B, representation_dim * 3), by default False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.processor = AutoImageProcessor.from_pretrained(
            "/models/hf/hub/models--facebook--vit-mae-base/snapshots/abc123", use_fast=True
        )
        self.return_encoding = return_encoding
        self.precomputed_features = precomputed_features
        config = ViTMAEConfig.from_pretrained("/models/hf/hub/models--facebook--vit-mae-base/snapshots/abc123")
        config.mask_ratio = 0.0
        self.encoder = ViTMAEModel.from_pretrained(
            "/models/hf/hub/models--facebook--vit-mae-base/snapshots/abc123", config=config
        )
        if checkpoint_path is not None:
            ckpt = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state_dict = ckpt.get("state_dict", ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model.model.vit."):
                    new_key = k[len("model.model.vit.") :]
                elif k.startswith("model.vit."):
                    new_key = k[len("model.") :]
                elif k.startswith("vit."):
                    new_key = k
                else:
                    new_key = k
                if (
                    new_key.startswith("embeddings.")
                    or new_key.startswith("encoder.")
                    or new_key == "layernorm.weight"
                    or new_key == "layernorm.bias"
                ):
                    new_state_dict[new_key] = v
            self.encoder.load_state_dict(new_state_dict, strict=False)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(
                self.encoder.config.hidden_size * 3,
                self.embed_dim * self.seq_len,
            ),
            nn.LayerNorm(self.embed_dim * self.seq_len),
            nn.GELU(),
        )

    def to(self, device: torch.device) -> "OSMEncoder":
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the model to (e.g., 'cpu' or 'cuda').

        Returns
        -------
        OSMEncoder
            The model instance moved to the specified device.
        """
        if not self.precomputed_features:
            self.encoder.to(device)
            self.head.to(device)
        else:
            self.encoder.to("cpu")
            self.head.to(device)
        return self

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
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
        # Reshape to split each 9-channel image into 3 RGB images
        B = imgs.shape[0]
        imgs = imgs.view(B, 3, 3, imgs.shape[2], imgs.shape[3])
        imgs = imgs.view(B * 3, 3, imgs.shape[3], imgs.shape[4])
        inputs = self.processor(
            images=imgs, return_tensors="pt", device=imgs.device
        )
        pixel_values = inputs.pixel_values
        outputs = self.encoder(pixel_values=pixel_values)
        hidden_states = outputs.last_hidden_state
        patch_tokens = hidden_states[:, 1:, :]
        pooled = patch_tokens.mean(dim=1)
        # Fuse pooled outputs for each sample before the head
        pooled = pooled.view(B, 3, -1)  # shape: (B, 3, pooled_dim)
        encoding = pooled.reshape(B, -1)  # shape: (B, 3*pooled_dim)
        return encoding

    def forward(
        self, imgs: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model for creating token embeddings.

        Parameters
        ----------
        imgs : torch.Tensor
            The input image tensor of shape (B, 9, H, W).
            Optionally, it can be a precomputed feature tensor of shape
            (B, representation_dim *3) if `precomputed_features` is set to
            True.
        return_features : bool, optional
            If True, return the encoding before the projection head,
            by default False.

        Returns
        -------
        torch.Tensor
            The processed image tensor of shape (B, seq_len, embed_dim).
        """
        # Reshape to split each 9-channel image into 3 RGB images
        if self.precomputed_features:
            encoding = imgs
        else:
            B = imgs.shape[0]
            imgs = imgs.view(B, 3, 3, imgs.shape[2], imgs.shape[3])
            imgs = imgs.view(B * 3, 3, imgs.shape[3], imgs.shape[4])

            inputs = self.processor(
                images=imgs, return_tensors="pt", device=imgs.device
            )
            pixel_values = inputs.pixel_values
            outputs = self.encoder(pixel_values=pixel_values)
            hidden_states = outputs.last_hidden_state
            patch_tokens = hidden_states[:, 1:, :]
            pooled = patch_tokens.mean(dim=1)
            # Fuse pooled outputs for each sample before the head
            pooled = pooled.view(B, 3, -1)  # shape: (B, 3, pooled_dim)
            encoding = pooled.reshape(B, -1)  # shape: (B, 3*pooled_dim)
        output = self.head(encoding)
        output = output.view(output.shape[0], self.seq_len, -1)
        if self.return_encoding or return_features:
            return output, encoding.detach()
        return output
