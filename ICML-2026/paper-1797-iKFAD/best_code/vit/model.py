import timm
import torch.nn as nn
import torch

def get_model(device):
    """
    Returns a NanoViT for CIFAR10.
    
    Configuration:
    - Input: 32x32 images
    - Patch Size: 4 (Results in 8x8 = 64 tokens, sufficient for 'Nano' benchmarking)
    - Embed Dim: 192
    - Depth: 9 layers
    - Heads: 3
    """
    model = timm.create_model(
        'vit_tiny_patch16_224',       # Base config to override
        pretrained=False,
        img_size=32,                  # Override: CIFAR Size
        patch_size=4,                 # Override: Smaller patches for small images
        num_classes=10,               # Override: CIFAR10 classes
        embed_dim=192,
        depth=9,
        num_heads=3,
        drop_rate=0.1
    )
    return model.to(device)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device)
    
    total, trainable = count_parameters(model)
    
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    
    # Optional: Breakdown by size (MB) assuming float32 (4 bytes)
    print(f"Model Size: {total * 4 / 1024**2:.2f} MB")