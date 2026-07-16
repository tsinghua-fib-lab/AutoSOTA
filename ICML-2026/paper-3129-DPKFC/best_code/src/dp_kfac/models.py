from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, ...] = (512, 256),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.layers(x)


class TinyViT(nn.Module):
    """Small Vision Transformer for spectrum analysis (2 blocks, 4 heads)."""

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.patch_size = patch_size

        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.LayerNorm(embed_dim))
            blocks.append(nn.MultiheadAttention(embed_dim, num_heads, batch_first=True))
            blocks.append(nn.LayerNorm(embed_dim))
            blocks.append(nn.Linear(embed_dim, embed_dim))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, C * p * p)
        x = self.patch_embed(x) + self.pos_embed
        for i in range(0, len(self.blocks), 4):
            ln1 = self.blocks[i]
            attn = self.blocks[i + 1]
            ln2 = self.blocks[i + 2]
            ffn = self.blocks[i + 3]
            h = ln1(x)
            h, _ = attn(h, h, h)
            x = x + h
            x = x + ffn(ln2(x))
        x = self.norm(x).mean(dim=1)
        return self.head(x)


class SimpleCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        img_size: int = 28,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        fc_size = img_size // 4
        self.fc1 = nn.Linear(32 * fc_size * fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CrossViTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        img_size: int = 240,
        model_name: str = "crossvit_tiny_240",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        for param in self.backbone.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)


class CrossViTLoRAClassifier(nn.Module):
    """CrossViT with LoRA adapters on attention layers + trainable classifier.

    Instead of only training the linear head, we also inject low-rank adapters
    into the backbone's attention projections. The base weights remain frozen.
    """

    def __init__(
        self,
        num_classes: int = 100,
        img_size: int = 240,
        model_name: str = "crossvit_tiny_240",
        pretrained: bool = True,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_targets: Tuple[str, ...] = ("qkv", "proj"),
    ) -> None:
        super().__init__()
        import timm
        from .lora import apply_lora, count_lora_params

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Apply LoRA to attention layers in the backbone
        self.backbone, num_replaced = apply_lora(
            self.backbone,
            target_modules=list(lora_targets),
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]

        self.classifier = nn.Linear(self.feature_dim, num_classes)

        trainable, total = count_lora_params(self)
        # classifier params are also trainable
        trainable += sum(p.numel() for p in self.classifier.parameters())
        self._lora_info = {
            "num_replaced": num_replaced,
            "trainable": trainable,
            "total": total + sum(p.numel() for p in self.classifier.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone forward WITH LoRA (no torch.no_grad — LoRA params need grad)
        features = self.backbone(x)
        return self.classifier(features)


class ConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        img_size: int = 224,
        model_name: str = "convnext_femto.d1_in1k",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        for param in self.backbone.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            features = self.backbone(x)
        return self.classifier(features.float())


class RoBERTaClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "roberta-base",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)


class BERTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "bert-base-uncased",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)


class DistilBERTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "distilbert-base-uncased",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler_output; use [CLS] token
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# Alias for IMDB logistic regression experiments
LogisticRegression = LinearClassifier


def create_model(
    model_type: str,
    num_classes: int = 10,
    img_size: int = 28,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    model_type = model_type.lower()

    if model_type == "mlp":
        input_dim = in_channels * img_size * img_size
        return MLP(input_dim=input_dim, num_classes=num_classes)

    elif model_type == "cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes, img_size=img_size)

    elif model_type == "crossvit":
        return CrossViTClassifier(num_classes=num_classes, img_size=240, pretrained=pretrained)

    elif model_type == "crossvit_lora":
        return CrossViTLoRAClassifier(num_classes=num_classes, img_size=240, pretrained=pretrained)

    elif model_type == "convnext":
        return ConvNeXtClassifier(num_classes=num_classes, img_size=224, pretrained=pretrained)

    elif model_type == "roberta":
        return RoBERTaClassifier(num_classes=num_classes)

    elif model_type == "bert":
        return BERTClassifier(num_classes=num_classes)

    elif model_type == "distilbert":
        return DistilBERTClassifier(num_classes=num_classes)

    elif model_type == "logreg":
        input_dim = in_channels * img_size * img_size if in_channels > 0 else img_size
        return LogisticRegression(input_dim=input_dim, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_img_size(model_type: str, default_img_size: int = 28) -> int:
    model_type = model_type.lower()
    if model_type in ("crossvit", "crossvit_lora"):
        return 240
    elif model_type == "convnext":
        return 224
    return default_img_size


def freeze_backbone(model: nn.Module) -> None:
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False


def get_trainable_params(model: nn.Module) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def create_backbone(
    model_type: str,
    img_size: int = 224,
) -> Tuple[nn.Module, int]:
    model_type = model_type.lower()

    if model_type == "convnext":
        import timm
        model_name = "convnext_tiny.fb_in22k_ft_in1k"
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            feature_dim = backbone(dummy).shape[1]
        return backbone, feature_dim

    elif model_type == "crossvit":
        import timm
        model_name = "crossvit_tiny_240"
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            feature_dim = backbone(dummy).shape[1]
        return backbone, feature_dim

    else:
        raise ValueError(f"No backbone available for model type: {model_type}")


def uses_pretrained_backbone(model_type: str) -> bool:
    return model_type.lower() in ["convnext", "crossvit"]
