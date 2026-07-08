"""
Zero-shot evaluator for CLIP models.

Provides a clean interface for evaluating zero-shot classification accuracy
on both clean and adversarial samples.
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from ..data.dataset import CLIPDataset
from ..data.collate import default_collate_fn


class ZeroShotEvaluator:
    """
    Zero-shot CLIP evaluator for a single model + dataset combination.

    Handles text feature precomputation and batched inference.

    Args:
        model: CLIP model (HuggingFace CLIPModel or CSRDefense wrapper).
        processor: CLIPProcessor for text tokenization.
        device: Computation device.
        batch_size: Inference batch size.
    """

    def __init__(self, model, processor, device: str = "cuda", batch_size: int = 32):
        self.model = model
        self.processor = processor
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def compute_text_features(self, class_names: list) -> torch.Tensor:
        """Precompute normalized text features for all classes using prompt ensembling."""
        # Standard CLIP prompt templates (ensembling improves zero-shot accuracy)
        templates = [
            "a photo of a {}.",
            "a picture of a {}.",
            "a good photo of a {}.",
            "a rendering of a {}.",
            "a cropped photo of a {}.",
            "a bright photo of a {}.",
            "a close-up photo of a {}.",
        ]
        text_batch_size = 100
        all_template_feats = []

        for template in templates:
            prompts = [template.format(cls) for cls in class_names]
            template_feats = []
            for i in range(0, len(prompts), text_batch_size):
                batch = prompts[i : i + text_batch_size]
                text_inputs = self.processor(text=batch, padding=True, return_tensors="pt").to(self.device)
                feats = self.model.get_text_features(**text_inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                template_feats.append(feats)
            all_template_feats.append(torch.cat(template_feats, dim=0))

        # Average text features across templates for stable class representations
        stacked = torch.stack(all_template_feats, dim=0)  # [num_templates, num_classes, dim]
        return stacked.mean(dim=0)  # [num_classes, dim]

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, text_features: torch.Tensor) -> float:
        """
        Evaluate zero-shot accuracy.

        Args:
            dataloader: DataLoader yielding (images, labels, filenames).
            text_features: [num_classes, embed_dim].

        Returns:
            Accuracy as a float in [0, 1].
        """
        self.model.eval()
        correct, total = 0, 0

        for batch_imgs, batch_labels, _ in tqdm(dataloader, leave=False, desc="Eval"):
            if batch_imgs is None:
                continue

            batch_labels = batch_labels.to(self.device)

            # Handle PIL images (no transform applied in dataset)
            if not isinstance(batch_imgs, torch.Tensor):
                # Defense wrappers such as CSRDefense/FastCSRDefense expect pixel values
                # in [0, 1] and apply CLIP normalization internally. Plain CLIPModel
                # still expects already-normalized processor outputs.
                use_raw_pixel_values = hasattr(self.model, "process_images") or hasattr(self.model, "process_images_and_feats")
                processor_kwargs = {"images": batch_imgs, "return_tensors": "pt"}
                if use_raw_pixel_values:
                    processor_kwargs["do_normalize"] = False
                inputs = self.processor(**processor_kwargs).to(self.device)
                image_features = self.model.get_image_features(**inputs)
            else:
                image_features = self.model.get_image_features(pixel_values=batch_imgs.to(self.device))

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            similarity = 100.0 * image_features @ text_features.T
            predictions = similarity.argmax(dim=-1)

            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)

        return correct / total if total > 0 else 0.0
