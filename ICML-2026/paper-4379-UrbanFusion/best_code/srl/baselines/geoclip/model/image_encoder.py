"""
Original code from: https://github.com/VicenteVivan/geo-clip

Original source:
Vivanco, Vicente; Nayak, Gaurav Kumar; Shah, Mubarak.
"GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
NeurIPS 2023. arXiv preprint published September 27, 2023.
"""

import warnings

import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel

warnings.filterwarnings(
    "ignore", category=UserWarning, module="huggingface_hub.*"
)


class ImageEncoder(nn.Module):
    def __init__(self, precomputed_features: bool = False):
        super().__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.mlp = nn.Sequential(
            nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512)
        )
        self.precomputed_features = precomputed_features

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        return x

    def forward(self, x):
        if self.precomputed_features:
            x = x
        else:
            x = self.CLIP.get_image_features(pixel_values=x)
        x = self.mlp(x)
        return x
