#!/usr/bin/env python3
"""
Description: Implementation of a text encoder for encoding text inputs into
representations suitable for multimodal tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextTransformer(nn.Module):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        embed_dim: int = 256,
        head: str = "linear",
        head_hidden_dim: int = 512,
        seq_len: int = 1,
        trainable_layers: int = 0,
        return_encoding: bool = False,
        precomputed_features: bool = False,
    ) -> None:
        """
        A text encoder that uses a text transformer model with optional
        trainable layers and an MLP projection.

        Parameters
        ----------
        model_name : str, optional
            Name of the CLIP model, by default "BAAI/bge-small-en-v1.5".
        embed_dim : int, optional
            Output embedding dimensionality, by default 256.
        head_hiden_dim : int, optional
            Hidden layer size, by default 512.
        seq_len : int, optional
            Number of output tokens after projection head, by default 1.
        trainable_layers : int, optional
            Number of last transformer layers to keep trainable, by default 0.
            Only the projection head is trainable if `trainable_layers=0`.
        return_encoding : bool, optional
            Whether to return the encoding of the text model, by default False.
        precomputed_features : bool, optional
            Whether to use precomputed features, by default False.
            If True, the model will not compute the text features.
            This is useful for speeding up training when the text features are
            already computed and stored in the dataset, by default False.
        """

        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.head = head
        self.head_hidden_dim = head_hidden_dim
        self.seq_len = seq_len
        self.trainable_layers = trainable_layers
        self.return_encoding = return_encoding
        self.precomputed_features = precomputed_features

        # Load pretrained text model
        self.text_model = AutoModel.from_pretrained("/models/hf/hub/models--BAAI--bge-small-en-v1.5/snapshots/abc123")

        # Unfreeze last `trainable_layers` layers
        self._freeze_last_k()

        # Get the original embedding dimension from CLIP
        self.text_model_embed_dim = self.text_model.config.hidden_size

        # Get the projection head
        self._get_head()

    def _freeze_last_k(self) -> None:
        """
        Freezes all parameters in a text model except the last k layers,
        including the pooler.
        """
        k = self.trainable_layers

        # First, freeze everything
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Unfreeze pooler
        if k > 0:
            if hasattr(self.text_model, "pooler"):
                for param in self.text_model.pooler.parameters():
                    param.requires_grad = True

        k = k - 1
        if k > 0:
            # Unfreeze last k layers of the encoder
            encoder_layers = self.text_model.encoder.layer
            for layer in encoder_layers[-k:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def _get_head(self):
        """
        Get the projection head of the model.

        Returns
        -------
        nn.Module
            The projection head.
        """
        if self.head == "mlp":
            self.head_projection = self._get_mlp_head()
        elif self.head == "linear":
            self.head_projection = self._get_linear_head()
        else:
            raise ValueError(
                f"Invalid head type for text CLIP: {self.head}",
                "Choose from ['mlp', 'linear']",
            )  # No bias since position embeddings are added later

    def _get_mlp_head(self):
        """
        Get the MLP projection head.

        Returns
        -------
        nn.Module
            The MLP projection head.
        """
        return nn.Sequential(
            nn.Linear(self.text_model_embed_dim, self.head_hidden_dim),
            nn.LayerNorm(self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.head_hidden_dim,
                self.embed_dim * self.seq_len,
            ),
            nn.LayerNorm(self.embed_dim * self.seq_len),
            nn.GELU(),
        )

    def _get_linear_head(self):
        """
        Get the linear projection head.

        Returns
        -------
        nn.Module
            The linear projection head.
        """
        return nn.Sequential(
            nn.Linear(
                self.text_model_embed_dim,
                self.embed_dim * self.seq_len,
            ),
            nn.LayerNorm(self.embed_dim * self.seq_len),
            nn.GELU(),
        )

    def to(self, device: torch.device) -> "TextTransformer":
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the model to (e.g., 'cpu' or 'cuda').

        Returns
        -------
        TextTransformer
            The model instance moved to the specified device.
        """
        if not self.precomputed_features:
            self.text_model.to(device)
            self.head_projection.to(device)
        else:
            self.text_model.to("cpu")
            self.head_projection.to(device)
        return self

    def forward_features(self, inputs: dict) -> torch.Tensor:
        """
        Process the text through the backbone.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The processed tensor.
        """
        x = self.text_model(**inputs).pooler_output
        return x

    def forward(
        self, inputs: dict, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model for creating text embeddings.

        Parameters
        ----------
        inputs : dict
            Dictionary containing the input text.

        Returns
        -------
        torch.Tensor
            Projected embeddings of shape (B, seq_len, embed_dim).
        """
        # Get text features from CLIP model, average over output tokens
        if self.precomputed_features:
            # Use precomputed features
            encoding = inputs
        else:
            encoding = self.text_model(**inputs).pooler_output

        # Project text features to embedding dimension
        x = self.head_projection(encoding)

        # Reshape to (batch_size, seq_len, embed_dim)
        x = x.view(x.shape[0], self.seq_len, -1)
        if self.return_encoding or return_features:
            return x, encoding.detach()

        return x
