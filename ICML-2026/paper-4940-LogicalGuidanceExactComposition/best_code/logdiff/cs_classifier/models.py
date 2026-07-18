from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import math

class TimeEmbedding(nn.Module):
    """
    Creates a sinusoidal positional embedding for the timestep t, 
    followed by a small MLP to generate the final time features.
    """
    def __init__(self, time_channels: int = 128, embed_dim: int = 256):
        super().__init__()
        # MLP to process the sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(time_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is assumed to be a tensor of shape [B] containing integer timesteps
        
        # 1. Sinusoidal positional encoding setup
        half_dim = self.mlp[0].in_features // 2
        
        # Prepare frequency multipliers (e.g., 1/(10000^(2i/d)))
        embeddings = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        
        # 2. Apply encoding
        t = t.float().unsqueeze(-1) # Shape [B] -> [B, 1]
        embeddings = t * embeddings.unsqueeze(0) # Shape [B, half_dim]
        
        # 3. Concatenate sin and cos for the full embedding
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # Shape [B, time_channels]
        
        # 4. Pass through MLP
        return self.mlp(embeddings) # Shape [B, embed_dim]


# --- Model Architecture: Shared Features, Independent Heads with Time Input ---
class MultiheadClassifier(nn.Module):
    """
    A generalized multi-task model that takes a noisy image (x) and a timestep (t), 
    combines image features and time features, and produces N sets of logits, 
    where N = len(num_classes_per_label).
    """
    def __init__(self, base_model: nn.Module, num_classes_per_label: List[int], time_embed_dim: int = 256):
        super().__init__()
        assert len(num_classes_per_label) >= 1, "Must define at least one classification task."
        
        self.model = base_model
        dim_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        self.time_embed = TimeEmbedding(time_channels=128, embed_dim=time_embed_dim)
        combined_dim = dim_features + time_embed_dim
        
        self.classifier_heads = nn.ModuleList([
            nn.Linear(combined_dim, num_classes)
            for num_classes in num_classes_per_label
        ])

        self.num_classes_per_label = num_classes_per_label
        self.time_embed_dim = time_embed_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        img_features = self.model(x)
        time_features = self.time_embed(t)

        B_img = img_features.shape[0]
        B_time = time_features.shape[0]

        if B_time != B_img:
            if B_time == 1:
                time_features = time_features.repeat(B_img, 1)
            else:
                raise ValueError(
                    f"Batch dimension mismatch: Image features batch size {B_img} "
                    f"does not match time features batch size {B_time}."
                )
            
        combined_features = torch.cat([img_features, time_features], dim=1)
        
        logits = [
            head(combined_features)
            for head in self.classifier_heads
        ]
        return tuple(logits)
    
class MultiheadLatentClassifier(MultiheadClassifier):
    """
    A latent-space version of the MultiheadClassifier.
    It adapts the base_model (usually ResNet) to accept latent channels (e.g., 16)
    and removes downsampling layers that are too aggressive for small latent inputs.
    """
    def __init__(
        self, 
        base_model: nn.Module, 
        num_classes_per_label: List[int], 
        time_embed_dim: int = 256,
        in_channels: int = 16
    ):
        # 1. Initialize the parent MultiheadClassifier
        # This sets up self.model (the backbone), self.time_embed, and self.classifier_heads
        super().__init__(
            base_model=base_model, 
            num_classes_per_label=num_classes_per_label, 
            time_embed_dim=time_embed_dim
        )
        
        # 2. Modify the backbone for Latent Space
        self._modify_first_layer(in_channels)

    def _modify_first_layer(self, in_channels):
        # Access the backbone via self.model (as defined in your MultiheadClassifier)
        if hasattr(self.model, 'conv1'):
            old_layer = self.model.conv1
            
            # A. Swap the input convolution
            # Create a new layer with the correct in_channels (e.g. 16 instead of 3)
            self.model.conv1 = nn.Conv2d(
                in_channels,
                old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias
            )
            
            # Initialize weights for the new layer
            nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # B. Remove MaxPool (Critical for 16x16 inputs)
            # ResNet normally downsamples 4x in the first two layers. 
            # For 16x16 input, this destroys too much info. We replace it with Identity.
            if hasattr(self.model, 'maxpool'):
                self.model.maxpool = nn.Identity()


class MultiLabelClassifier(nn.Module):
    def __init__(self,base_model:nn.Module,num_classes_per_label:list[int]):
        super().__init__()
        self.model = base_model
        dim_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.classifier_heads = nn.ModuleList([nn.Linear(dim_features,num_class) for num_class in num_classes_per_label])
        self.num_classes_per_label = num_classes_per_label
        
    def forward(self,x):
        x = self.model(x)
        return [head(x) for head in self.classifier_heads]
    

class VAEClassifer(nn.Module):
    def __init__(self,num_classes_per_label:list[int],input_dim:int=(16,8,8)):
        super().__init__()

        input_dim = math.prod(input_dim)
        hidden_dim = input_dim//2
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim),        
            nn.GELU(),
        ])
        self.classifier_heads = nn.ModuleList([nn.Linear(hidden_dim,num_class) for num_class in num_classes_per_label])
        self.num_classes_per_label = num_classes_per_label
    def forward(self,x):
        x = x.flatten(1)
        x = self.mlp(x)
        return [head(x) for head in self.classifier_heads]
    
    
class ClipClassifer(nn.Module):
    def __init__(self,num_classes_per_label:list[int]):
        super().__init__()
        input_dim = 768
        hidden_dim = 512
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim),        
            nn.GELU(),
        ])
        self.classifier_heads = nn.ModuleList([nn.Linear(hidden_dim,num_class) for num_class in num_classes_per_label])
        self.num_classes_per_label = num_classes_per_label

    def forward(self,x):
        x = self.mlp(x)
        return [head(x) for head in self.classifier_heads]