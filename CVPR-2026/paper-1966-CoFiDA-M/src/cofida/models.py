import timm
import torch
import torch.nn as nn


class TimmBackbone(nn.Module):
    """Wraps a timm EfficientNet model to return pooled features like torchvision."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        feat = self.model.forward_features(x)
        return self.pool(feat).flatten(1)


def _create_efficientnet_b2_backbone():
    """Create EfficientNet-B2 backbone using timm (downloads from HF mirror).
    Returns (backbone, feat_dim).
    """
    model = timm.create_model("efficientnet_b2", pretrained=True)
    feat_dim = model.num_features  # 1408
    backbone = TimmBackbone(model)
    return backbone, feat_dim


class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            shadow_value = self.shadow[key]
            if torch.is_floating_point(value):
                shadow_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(value)

    @torch.no_grad()
    def load_shadow(self, model):
        model.load_state_dict(self.shadow, strict=False)


class MonetProbEmbedder(nn.Module):
    def __init__(self, num_attrs: int, d_attr: int = 32, text_dim: int = 256, gini_pow: float = 1.0):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(2, d_attr) for _ in range(num_attrs)])
        self.proj = nn.Sequential(
            nn.Linear(num_attrs * d_attr, text_dim),
            nn.ReLU(True),
            nn.Linear(text_dim, text_dim),
        )
        self.gini_pow = gini_pow

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(0, 1)
        _, num_attrs = probs.shape
        embeddings = []
        for idx in range(num_attrs):
            prob = probs[:, idx]
            weighted = torch.stack([1.0 - prob, prob], dim=1)
            attr_embed = weighted @ self.embs[idx].weight
            gate = (prob * prob + (1 - prob) * (1 - prob)).pow(self.gini_pow).unsqueeze(1)
            embeddings.append(gate * attr_embed)
        return self.proj(torch.cat(embeddings, dim=1))


class FiLMContext(nn.Module):
    def __init__(self, feat_dim: int = 1408, text_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 2 * feat_dim),
        )

    def forward(self, visual_feat, concept_feat):
        hidden = self.net(concept_feat)
        gamma, beta = torch.chunk(hidden, 2, dim=-1)
        return gamma * visual_feat + beta


class MLPHead(nn.Module):
    def __init__(self, feat_dim: int = 1408, hidden: int = 512, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, visual_feat, adapted_feat):
        return self.mlp(visual_feat + (adapted_feat - visual_feat))


class CoFIDAMonet(nn.Module):
    def __init__(self, num_concepts: int, text_dim: int = 256, num_classes: int = 2, gini_pow: float = 1.0):
        super().__init__()
        efficientnet, self.feat_dim = _create_efficientnet_b2_backbone()
        self.backbone = efficientnet
        self.embed = MonetProbEmbedder(num_attrs=num_concepts, d_attr=32, text_dim=text_dim, gini_pow=gini_pow)
        self.film = FiLMContext(feat_dim=self.feat_dim, text_dim=text_dim, hidden=512)
        self.head = MLPHead(feat_dim=self.feat_dim, hidden=512, num_classes=num_classes)

    def extract(self, images):
        return self.backbone(images)

    def forward_full(self, images, monet_probs):
        visual = self.extract(images)
        concepts = self.embed(monet_probs)
        adapted = self.film(visual, concepts)
        logits = self.head(visual, adapted)
        edit = adapted - visual
        return logits, visual, adapted, edit

    @torch.no_grad()
    def forward_eval(self, images, monet_probs):
        return self.forward_full(images, monet_probs)


class StudentImageOnly(nn.Module):
    def __init__(self, num_classes: int = 2, hidden: int = 512):
        super().__init__()
        efficientnet, self.feat_dim = _create_efficientnet_b2_backbone()
        self.backbone = efficientnet
        self.edit = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden, self.feat_dim),
        )
        self.head = MLPHead(feat_dim=self.feat_dim, hidden=hidden, num_classes=num_classes)

    def extract(self, images):
        return self.backbone(images)

    def forward(self, images):
        visual = self.extract(images)
        adapted = visual + self.edit(visual)
        logits = self.head(visual, adapted)
        return logits, adapted

    @torch.no_grad()
    def logits(self, images):
        logits, _ = self.forward(images)
        return logits
