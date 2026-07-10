from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ClipBuildConfig:
    hf_id: str = "openai/clip-vit-base-patch32"
    prompt_template: str = "a photo of a {}"
    prompt_templates: list[str] | None = None
    normalize: bool = True
    logit_scale: float = 100.0


class ClipClassifier(nn.Module):
    """
    Two modes:
      1) zero-shot: uses CLIP text encoder to build class prototypes
      2) linear-head: uses a learnable classifier on top of image embeddings

    The module returns logits: [B, num_classes]
    """

    def __init__(
        self,
        clip_model: nn.Module,
        processor,
        *,
        normalize: bool = True,
        logit_scale: float = 100.0,
        classifier: nn.Linear | None = None,
    ):
        super().__init__()
        self.clip = clip_model
        self.processor = processor
        self.normalize = normalize
        self.logit_scale = float(logit_scale)

        self.classifier = classifier  # if provided, use this mode
        self.register_buffer("_zs_text_features", torch.empty(0), persistent=False)

    @staticmethod
    def build(cfg: ClipBuildConfig, device: str = "cpu", dtype: str | None = None) -> ClipClassifier:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            raise ImportError("CLIP support requires: pip install -e '.[xf]'") from e

        clip = CLIPModel.from_pretrained(cfg.hf_id)
        processor = CLIPProcessor.from_pretrained(cfg.hf_id)

        # dtype selection
        if dtype is not None:
            dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            if dtype not in dtype_map:
                raise ValueError(f"Unknown dtype {dtype}, choose from {sorted(dtype_map)}")
            clip = clip.to(dtype=dtype_map[dtype])

        clip = clip.to(device)
        return ClipClassifier(clip, processor, normalize=cfg.normalize, logit_scale=cfg.logit_scale)

    def set_linear_head(self, num_classes: int) -> None:
        # CLIPModel exposes projection dim via config.projection_dim
        dim = int(getattr(self.clip.config, "projection_dim", 512))
        self.classifier = nn.Linear(dim, num_classes, bias=True)

    @torch.no_grad()
    def build_zeroshot_text_features(self, classnames: list[str], cfg: ClipBuildConfig) -> None:
        """
        Precompute text features for zero-shot classification.
        """
        templates = cfg.prompt_templates or [cfg.prompt_template]
        texts = []
        for c in classnames:
            for t in templates:
                texts.append(t.format(c))

        inputs = self.processor(text=texts, images=None, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        # Try the canonical CLIPModel API first
        out = self.clip.get_text_features(**inputs)

        # Some model variants / versions may return a ModelOutput instead of a tensor
        if not isinstance(out, torch.Tensor):
            # Common fields depending on model output type
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                out = out.pooler_output
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                # fallback: mean pool tokens
                out = out.last_hidden_state.mean(dim=1)
            else:
                raise TypeError(f"Unexpected output type from get_text_features: {type(out)}")

        # Normalize
        if self.normalize:
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-12)

        # Average across templates per class
        n_t = len(templates)
        feats = out.reshape(len(classnames), n_t, -1).mean(dim=1)  # [C, D]
        if self.normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

        self._zs_text_features = feats

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W] normalized to CLIP expected distribution.
        Returns logits: [B, C]
        """
        # image embeddings
        # CLIPModel expects pixel_values
        out = self.clip.get_image_features(pixel_values=images)
        if not isinstance(out, torch.Tensor):
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                out = out.pooler_output
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                out = out.last_hidden_state.mean(dim=1)
            else:
                raise TypeError(f"Unexpected output type from get_image_features: {type(out)}")

        if self.normalize:
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-12)

        # mode 1: linear head
        if self.classifier is not None:
            return self.classifier(out)

        # mode 2: zero-shot
        if self._zs_text_features.numel() == 0:
            raise RuntimeError("Zero-shot mode requires build_zeroshot_text_features() first.")
        logits = self.logit_scale * (out @ self._zs_text_features.t())
        return logits

    def load_linear_head_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Supports keys:
          classifier.weight, classifier.bias
        """
        if "classifier.weight" not in state_dict:
            raise KeyError("Missing classifier.weight in provided state_dict.")
        w = state_dict["classifier.weight"]
        b = state_dict.get("classifier.bias", torch.zeros(w.shape[0], device=w.device, dtype=w.dtype))
        head = nn.Linear(w.shape[1], w.shape[0], bias=True)
        head.weight.data.copy_(w)
        head.bias.data.copy_(b)
        self.classifier = head.to(device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)

    @torch.no_grad()
    def top1(self, loader, device: str) -> float:
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.to(dev)
        self.eval()

        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = self(x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return float(correct / max(1, total))
