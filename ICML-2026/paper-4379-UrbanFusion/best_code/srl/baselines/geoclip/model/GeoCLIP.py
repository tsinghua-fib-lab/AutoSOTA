"""
Original code from: https://github.com/VicenteVivan/geo-clip

Original source:
Vivanco, Vicente; Nayak, Gaurav Kumar; Shah, Mubarak.
"GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
NeurIPS 2023. arXiv preprint published September 27, 2023.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from srl.encoders.synthetic_RGB_encoder.rgb_encoder import RGBEncoder

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import file_dir, load_gps_data


class GeoCLIP(nn.Module):
    def __init__(
        self,
        from_pretrained: bool = True,
        queue_size: int = 4096,
        precomputed_features: bool = False,
        synthetic_experiment: bool = False,
    ) -> None:
        """
        GeoCLIP model for image and GPS location encoding

        Parameters
        ----------
        from_pretrained : bool, optional
            If True, load pre-trained weights
        queue_size : int, optional
            Size of the GPS queue for storing coordinates
        precomputed_features : bool, optional
            If True, use precomputed features for the image encoder
        """
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if synthetic_experiment:
            self.image_encoder = RGBEncoder(dim_hidden=4, dim_out=3)
        else:
            self.image_encoder = ImageEncoder(precomputed_features)
        self.location_encoder = LocationEncoder(
            from_pretrained=from_pretrained,
            synthetic_experiment=synthetic_experiment,
        )
        self.precomputed_features = precomputed_features

        self.gps_gallery = load_gps_data(
            os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv")
        )
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = "cpu"
        self.queue_size = queue_size

    def to(self, device: str) -> nn.Module:
        """
        Move model to the specified device

        Parameters
        ----------
        device : str
            Device to move the model to (e.g., 'cpu', 'cuda')
        """
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(
            torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth")
        )
        self.location_encoder.load_state_dict(
            torch.load(f"{self.weights_folder}/location_encoder_weights.pth")
        )
        self.logit_scale = nn.Parameter(
            torch.load(f"{self.weights_folder}/logit_scale_weights.pth")
        )
        print("Weights loaded successfully.", flush=True)

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        if self.queue_size == 0:
            return
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        if self.queue_size % gps_batch_size == 0:
            # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
            self.gps_queue[:, gps_ptr : gps_ptr + gps_batch_size] = gps.t()
            gps_ptr = (
                gps_ptr + gps_batch_size
            ) % self.queue_size  # move pointer
            self.gps_queue_ptr[0] = gps_ptr
        else:
            return

    def get_gps_queue(self):
        return self.gps_queue.t()

    def forward(self, image, location):
        """GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (
            image_features @ location_features.t()
        )

        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
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
