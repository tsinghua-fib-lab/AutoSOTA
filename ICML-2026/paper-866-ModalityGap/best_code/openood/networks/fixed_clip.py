import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class FixedCLIP(nn.Module):
    """
    Frozen CLIP backbone used purely as a feature extractor.

    All parameters are frozen immediately after loading so that CLIP's weights
    are never modified during evaluation. All adaptation happens in the
    StreamingPrototypeAdapter's visual prototype matrix, not inside CLIP.

    Attributes
    ----------
    preprocess : torchvision transform returned by clip.load(). Pass this to
                 ImglistDataset so images are preprocessed consistently.
    """

    def __init__(self, backbone_name: str, device: str):
        super().__init__()
        self.device = device
        print(f"Loading CLIP backbone: {backbone_name}...")
        self.model, self.preprocess = clip.load(backbone_name, device=device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised image feature vectors, shape [B, D]."""
        return F.normalize(self.model.encode_image(x), dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Return raw (non-normalised) text feature vectors, shape [B, D]."""
        return self.model.encode_text(text)
