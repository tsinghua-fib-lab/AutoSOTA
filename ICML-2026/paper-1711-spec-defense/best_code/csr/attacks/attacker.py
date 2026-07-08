"""
CLIP-specific attacker wrapper.

Wraps a CLIP model for adversarial attack usage, providing text feature
precomputation and normalized forward pass for logit computation.
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

from ..config import CLIP_MEAN, CLIP_STD


class CLIPAttacker(nn.Module):
    """
    Wrapper around CLIPModel for adversarial attacks.
    
    Provides:
      - Text feature precomputation
      - Forward pass: images [0,1] -> normalized logits
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def precompute_text_features(self, class_names: list) -> torch.Tensor:
        """
        Precompute normalized text embeddings for all classes.
        
        Returns:
            Tensor of shape [num_classes, embed_dim].
        """
        prompts = [f"a photo of a {c}" for c in class_names]
        all_embeds = []
        batch_size = 100

        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                inputs = self.processor(
                    text=batch_prompts, return_tensors="pt", padding=True
                ).to(self.device)
                embeds = self.model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                all_embeds.append(embeds)

        return torch.cat(all_embeds)

    def forward(self, images_tensor: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits.

        Args:
            images_tensor: [B, 3, H, W] in range [0, 1] (unnormalized).
            text_features: [num_classes, embed_dim] precomputed text embeddings.

        Returns:
            Logits tensor [B, num_classes].
        """
        norm_images = self.normalize(images_tensor)
        image_embeds = self.model.get_image_features(pixel_values=norm_images)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits = torch.matmul(image_embeds, text_features.t()) * logit_scale
        return logits
