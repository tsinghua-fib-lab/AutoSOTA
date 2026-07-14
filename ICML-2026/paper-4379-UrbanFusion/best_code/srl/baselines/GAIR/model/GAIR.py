#!/usr/bin/env python3
"""
Description: Our own implementation of the GAIR model framework, with using
GeoCLIP location encoder and street-view image encoder, and remote sensing
encoder from SatCLIP.

Code is loosely based on: https://github.com/VicenteVivan/geo-clip
"""
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchgeo.models import ViTSmall16_Weights

from srl.encoders.synthetic_RGB_encoder.rgb_encoder import RGBEncoder

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import file_dir, load_gps_data


class GAIR(nn.Module):
    def __init__(
        self,
        queue_size: int = 4096,
        precomputed_features: bool = False,
        synthetic_experiment: bool = False,
    ) -> None:
        """
        Initializes the GAIR model.

        Parameters
        ----------
        queue_size : int, optional
            Size of the GPS queue. Defaults to 4096.
        precomputed_features : bool, optional
            If True, uses precomputed features of PP2-M dataset (without
            forward pass through the backbones). Defaults to False.
        synthetic_experiment : bool, optional
            If True, uses synthetic RGB encoder for experiments on synthetic
            data for analyzing partial information decomposition (PID).
            Defaults to False.
        """
        super().__init__()
        # Initialize backbones and logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.svi_encoder = ImageEncoder(precomputed_features)
        self.location_encoder = LocationEncoder(
            synthetic_experiment=synthetic_experiment
        )
        weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
        in_chans = weights.meta["in_chans"]
        self.rs_encoder = timm.create_model(
            "vit_small_patch16_224", in_chans=in_chans, num_classes=512
        )
        self.rs_encoder.load_state_dict(
            weights.get_state_dict(progress=True), strict=False
        )

        # Freeze RS encoder, head trainable
        self.rs_encoder.requires_grad_(False)
        self.rs_encoder.head.requires_grad_(True)

        self.precomputed_features = precomputed_features

        # Load GPS gallery from GeoCLIP
        self.gps_gallery = load_gps_data(
            os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv")
        )
        self._initialize_gps_queue(queue_size)

        self.device = "cpu"

        # Alternative encoders for synthetic experiments
        if synthetic_experiment:
            self.svi_encoder = RGBEncoder(dim_hidden=4, dim_out=3)
            self.rs_encoder = RGBEncoder(dim_hidden=4, dim_out=3)
        self.synthetic_experiment = synthetic_experiment

    def to(self, device) -> None:
        """Move model to specified device"""
        self.device = device
        self.svi_encoder.to(device)
        self.rs_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _initialize_gps_queue(self, queue_size: int) -> None:
        """Initialize GPS queue"""
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps: torch.Tensor) -> None:
        """
        Update GPS queue.

        Parameters
        ----------
        gps : torch.Tensor
            GPS tensor of shape (batch_size, 2)
        """
        if self.queue_size == 0:
            return
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        if self.queue_size % gps_batch_size == 0:
            # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and
            # enqueue)
            self.gps_queue[:, gps_ptr : gps_ptr + gps_batch_size] = gps.t()
            # move pointer
            gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size
            self.gps_queue_ptr[0] = gps_ptr
        else:
            return

    def get_gps_queue(self):
        """
        Get the current GPS queue.

        Returns
        -------
        torch.Tensor
            The current GPS queue of shape (2, queue_size)
        """
        return self.gps_queue.t()

    def encode_rs(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode RS image.

        Returns
        -------
        torch.Tensor
            The encoded RS image.
        """
        if self.precomputed_features:
            return self.rs_encoder.head(image)
        else:
            return self.rs_encoder(image)

    def forward(self, svi_image, rs_image, location):
        """
        Forward pass of all modalities

        Parameters
        ----------
        svi_image : torch.Tensor
            Image tensor of shape (n, 3, 224, 224) or precomputed features.
        rs_image : torch.Tensor
            Image tensor of shape (n, 3, 224, 224) or precomputed features.
        location : torch.Tensor
            GPS location tensor of shape (m, 2)

        Returns
        -------
        logits_per_image : torch.Tensor
            Logits per image of shape (n, m)
        """

        # Compute Features
        svi_features = self.svi_encoder(svi_image)
        rs_features = self.encode_rs(rs_image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        svi_features = F.normalize(svi_features, dim=1)
        rs_features = F.normalize(rs_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_SVI_location = logit_scale * (
            svi_features @ location_features.t()
        )
        logits_RS_location = logit_scale * (
            rs_features @ location_features.t()
        )
        logits_RS_SVI = logit_scale * (rs_features @ svi_features.t())

        return logits_SVI_location, logits_RS_location, logits_RS_SVI

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """
        Given an image, predict the top k GPS coordinates

        Parameters
        ----------
        image_path : str
            Path to the image
        top_k : int
            Number of top predictions to return

        Returns
        -------
        top_pred_gps : torch.Tensor
            Top k GPS coordinates of shape (k, 2)
        top_pred_prob : torch.Tensor
            Top k GPS probabilities of shape (k,)
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob
