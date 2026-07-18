"""
Model utilities for loading and caching ImageNet classification weights.

This module provides functions for loading pre-trained ImageNet classification head weights
with caching support to avoid re-downloading models.
"""

import os
import torch
import torchvision.models as models
from typing import Tuple, Optional, List

# Import model lists from config
from config.config import TORCHVISION_MODEL_NAMES, HUGGINGFACE_MODEL_NAMES, IMAGENET21K_HEAD_MODELS, IMAGENET21K_2EXTRA_HEAD_MODELS, IMAGENET1K_HEAD_MODELS, INATURALIST_HEAD_MODELS, WEIGHTS_DIR


def get_all_huggingface_models() -> List[str]:
    """
    Return a list of all supported HuggingFace model names.
    Note: These are ImageNet-1K classification models (1000 classes).
    """
    return list(HUGGINGFACE_MODEL_NAMES)


def load_weight_only_from_cache(model_name: str, weights_dir: Optional[str] = None) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Load only the weight matrix from cached model weights, ignoring bias.
    
    Args:
        model_name (str): Name/ID of the model
        weights_dir (str): Directory where weights are cached
        
    Returns:
        Optional[Tuple[torch.Tensor, int]]: (weight, num_classes) if cached file exists, None otherwise
    """
    weights_dir = weights_dir or WEIGHTS_DIR
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    model_weights_dir = os.path.join(weights_dir, safe_model_name)
    
    # Check for appropriate cache file based on model type
    if model_name in IMAGENET21K_HEAD_MODELS:
        weights_path = os.path.join(model_weights_dir, "imagenet21k.pth")
    else:
        weights_path = os.path.join(model_weights_dir, "imagenet1k.pth")
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        weight = checkpoint['weight']
        num_classes = checkpoint['num_classes']
        return weight, num_classes
    
    return None


def _get_final_linear_layer_with_setter(model, model_name):
    """
    Get the final linear layer from specific TIMM model architectures and return
    both the layer reference and a function to replace it.
    Currently only supports the specific TIMM models listed below.
    
    Args:
        model: PyTorch model with a classification head
        model_name: Name of the model (required - must be one of the supported TIMM models)
        
    Returns:
        tuple: (layer, setter_function) where setter_function takes a new layer as argument
        
    Raises:
        ValueError: If the model_name is not one of the supported TIMM models
    """
    # Only handle specific TIMM models with exact name matching
    if not model_name or not model_name.startswith('timm/'):
        raise ValueError(f"This function only supports specific TIMM models. Got: {model_name}")
    
    timm_model_name = model_name[5:]  # Remove 'timm/' prefix
    
    # Only handle these specific models with exact name matching
    supported_timm_models = [
        'beit_base_patch16_224.in22k_ft_in22k',
        'caformer_s18.sail_in22k',
        'convformer_s18.sail_in22k',
        'convnext_base.fb_in22k',
        'eva02_base_patch14_448.mim_in22k_ft_in22k',
        'swin_base_patch4_window12_384.ms_in22k',
        'tiny_vit_21m_224.dist_in22k',
        'vit_base_patch32_224_in21k',
        'vit_base_patch16_224_in21k',
        'resnetv2_101x1_bit.goog_in21k',
        'resnetv2_50x1_bit.goog_in21k',
        'beit_base_patch16_224.in22k_ft_in22k_in1k',
        'caformer_s18.sail_in22k_ft_in1k',
        'convformer_s18.sail_in22k_ft_in1k',
        'convnext_base.fb_in22k_ft_in1k',
        'eva02_base_patch14_448.mim_in22k_ft_in1k',
        'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
        'tiny_vit_21m_224.dist_in22k_ft_in1k',
        'resnetv2_50x1_bit.goog_in21k_ft_in1k',
        'resnetv2_101x1_bit.goog_in21k_ft_in1k',
        'vit_base_patch32_224.augreg_in21k_ft_in1k',
        'vit_base_patch16_224.augreg_in21k_ft_in1k',
        'convnext_large_mlp.laion2b_ft_augreg_inat21',
        'vit_large_patch14_clip_336.laion2b_ft_augreg_inat21'
    ]
    
    if timm_model_name not in supported_timm_models:
        raise ValueError(f"Unsupported TIMM model: {timm_model_name}. "
                        f"Supported models: {supported_timm_models}")
    
    # Use exact name matching instead of substrings
    if timm_model_name in ['eva02_base_patch14_448.mim_in22k_ft_in22k',
                            'eva02_base_patch14_448.mim_in22k_ft_in1k']:
        # Return entire head for eva02 models
        print(f"   Getting entire head for {timm_model_name}")
        return model.head, lambda new_layer: setattr(model, 'head', new_layer)
    elif timm_model_name in ['vit_large_patch14_clip_336.laion2b_ft_augreg_inat21']:
        # Return entire head for vit_large_patch14_clip models
        print(f"   Getting entire head for {timm_model_name}")
        return model.head, lambda new_layer: setattr(model, 'head', new_layer)
    elif timm_model_name in ['beit_base_patch16_224.in22k_ft_in22k',
                                'beit_base_patch16_224.in22k_ft_in22k_in1k']:
        # Return entire head for beit models
        print(f"   Getting entire head for {timm_model_name}")
        return model.head, lambda new_layer: setattr(model, 'head', new_layer)
    elif timm_model_name in ['convformer_s18.sail_in22k',
                                'convformer_s18.sail_in22k_ft_in1k']:
        # Return fc2 layer for convformer models
        print(f"   Getting head.fc.fc2 for {timm_model_name}")
        return model.head.fc.fc2, lambda new_layer: setattr(model.head.fc, 'fc2', new_layer)
    elif timm_model_name in ['caformer_s18.sail_in22k',
                                'caformer_s18.sail_in22k_ft_in1k']:
        # Return fc2 layer for caformer models
        print(f"   Getting head.fc.fc2 for {timm_model_name}")
        return model.head.fc.fc2, lambda new_layer: setattr(model.head.fc, 'fc2', new_layer)
    elif timm_model_name in ['swin_base_patch4_window12_384.ms_in22k',
                                'swin_base_patch4_window12_384.ms_in22k_ft_in1k']:
        # Return fc layer for swin models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc, lambda new_layer: setattr(model.head, 'fc', new_layer)
    elif timm_model_name in ['tiny_vit_21m_224.dist_in22k',
                                'tiny_vit_21m_224.dist_in22k_ft_in1k']:
        # Return fc layer for tiny_vit models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc, lambda new_layer: setattr(model.head, 'fc', new_layer)
    elif timm_model_name in ['convnext_base.fb_in22k',
                                'convnext_base.fb_in22k_ft_in1k', 
                                'convnext_large_mlp.laion2b_ft_augreg_inat21']:
        # Return fc layer for convnext models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc, lambda new_layer: setattr(model.head, 'fc', new_layer)
    elif timm_model_name in ['vit_base_patch32_224_in21k',
                                'vit_base_patch32_224.augreg_in21k_ft_in1k',]:
        # Return fc layer for ViT models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head, lambda new_layer: setattr(model, 'head', new_layer)
    elif timm_model_name in ['vit_base_patch16_224_in21k',
                                'vit_base_patch16_224.augreg_in21k_ft_in1k']:
        # Return fc layer for ViT models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head, lambda new_layer: setattr(model, 'head', new_layer)
    elif timm_model_name in ['resnetv2_101x1_bit.goog_in21k',
                                'resnetv2_101x1_bit.goog_in21k_ft_in1k']:
        # Return fc layer for ResNetv2 models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc, lambda new_layer: setattr(model.head, 'fc', new_layer)
    elif timm_model_name in ['resnetv2_50x1_bit.goog_in21k',
                                'resnetv2_50x1_bit.goog_in21k_ft_in1k']:
        # Return fc layer for ResNetv2 models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc, lambda new_layer: setattr(model.head, 'fc', new_layer)
    else:
        # This should never happen due to the check above, but just in case
        raise ValueError(f"Unhandled TIMM model: {timm_model_name}")


def _get_final_linear_layer(model, model_name):
    """
    Get the final linear layer from specific TIMM model architectures.
    Currently only supports the specific TIMM models listed below.
    
    Args:
        model: PyTorch model with a classification head
        model_name: Name of the model (required - must be one of the supported TIMM models)
        
    Returns:
        torch.nn.Linear: The final linear layer
        
    Raises:
        ValueError: If the model_name is not one of the supported TIMM models
    """
    # Only handle specific TIMM models with exact name matching
    if not model_name or not model_name.startswith('timm/'):
        raise ValueError(f"This function only supports specific TIMM models. Got: {model_name}")
    
    timm_model_name = model_name[5:]  # Remove 'timm/' prefix
    
    # Only handle these specific models with exact name matching
    supported_timm_models = [
        'beit_base_patch16_224.in22k_ft_in22k',
        'caformer_s18.sail_in22k',
        'convformer_s18.sail_in22k',
        'convnext_base.fb_in22k',
        'eva02_base_patch14_448.mim_in22k_ft_in22k',
        'swin_base_patch4_window12_384.ms_in22k',
        'tiny_vit_21m_224.dist_in22k',
        'vit_base_patch32_224_in21k',
        'vit_base_patch16_224_in21k',
        'resnetv2_101x1_bit.goog_in21k',
        'resnetv2_50x1_bit.goog_in21k',
        'beit_base_patch16_224.in22k_ft_in22k_in1k',
        'caformer_s18.sail_in22k_ft_in1k',
        'convformer_s18.sail_in22k_ft_in1k',
        'convnext_base.fb_in22k_ft_in1k',
        'eva02_base_patch14_448.mim_in22k_ft_in1k',
        'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
        'tiny_vit_21m_224.dist_in22k_ft_in1k',
        'resnetv2_50x1_bit.goog_in21k_ft_in1k',
        'resnetv2_101x1_bit.goog_in21k_ft_in1k',
        'vit_base_patch32_224.augreg_in21k_ft_in1k',
        'vit_base_patch16_224.augreg_in21k_ft_in1k',
        'convnext_large_mlp.laion2b_ft_augreg_inat21',
        'vit_large_patch14_clip_336.laion2b_ft_augreg_inat21'
    ]
    
    if timm_model_name not in supported_timm_models:
        raise ValueError(f"Unsupported TIMM model: {timm_model_name}. "
                        f"Supported models: {supported_timm_models}")
    
    # Use exact name matching instead of substrings
    if timm_model_name in ['eva02_base_patch14_448.mim_in22k_ft_in22k',
                           'eva02_base_patch14_448.mim_in22k_ft_in1k']:
        # Return entire head for eva02 models
        print(f"   Getting entire head for {timm_model_name}")
        return model.head
    elif timm_model_name in ['beit_base_patch16_224.in22k_ft_in22k',
                             'beit_base_patch16_224.in22k_ft_in22k_in1k']:
        # Return entire head for beit models
        print(f"   Getting entire head for {timm_model_name}")
        return model.head
    elif timm_model_name in ['convformer_s18.sail_in22k',
                             'convformer_s18.sail_in22k_ft_in1k']:
        # Return fc2 layer for convformer models
        print(f"   Getting head.fc.fc2 for {timm_model_name}")
        return model.head.fc.fc2
    elif timm_model_name in ['caformer_s18.sail_in22k',
                             'caformer_s18.sail_in22k_ft_in1k']:
        # Return fc2 layer for caformer models
        print(f"   Getting head.fc.fc2 for {timm_model_name}")
        return model.head.fc.fc2
    elif timm_model_name in ['swin_base_patch4_window12_384.ms_in22k',
                             'swin_base_patch4_window12_384.ms_in22k_ft_in1k']:
        # Return fc layer for swin models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc
    elif timm_model_name in ['tiny_vit_21m_224.dist_in22k',
                             'tiny_vit_21m_224.dist_in22k_ft_in1k']:
        # Return fc layer for tiny_vit models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc
    elif timm_model_name in ['vit_large_patch14_clip_336.laion2b_ft_augreg_inat21']:
        return model.head
    elif timm_model_name in ['convnext_base.fb_in22k',
                             'convnext_base.fb_in22k_ft_in1k',
                             'convnext_large_mlp.laion2b_ft_augreg_inat21']:
        # Return fc layer for convnext models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc
    elif timm_model_name in ['vit_base_patch32_224_in21k',
                             'vit_base_patch32_224.augreg_in21k_ft_in1k']:
        # Return fc layer for ViT models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head
    elif timm_model_name in ['vit_base_patch16_224_in21k',
                             'vit_base_patch32_224.augreg_in21k_ft_in1k']:
        # Return fc layer for ViT models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head
    elif timm_model_name in ['resnetv2_101x1_bit.goog_in21k',
                             'resnetv2_101x1_bit.goog_in21k_ft_in1k']:
        # Return fc layer for ResNetv2 models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc
    elif timm_model_name in ['resnetv2_50x1_bit.goog_in21k',
                             'resnetv2_50x1_bit.goog_in21k_ft_in1k']:
        # Return fc layer for ResNetv2 models
        print(f"   Getting head.fc for {timm_model_name}")
        return model.head.fc
    else:
        # This should never happen due to the check above, but just in case
        raise ValueError(f"Unhandled TIMM model: {timm_model_name}")


def cleanup_model_cache():
    """Clean up model from GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def zero_out_bias_in_last_layer(model, model_name=None):
    """
    Zero out the bias terms in the last linear layer of the model using _get_final_linear_layer.
    
    Args:
        model: The model to modify
        model_name: Name of the model (required for TIMM models)
        
    Returns:
        Modified model with bias zeroed out in last layer
    """
    try:
        last_linear = _get_final_linear_layer(model, model_name)
        
        if last_linear.bias is not None:
            print(f"  Found last linear layer with bias shape: {last_linear.bias.shape}")
            print(f"  Original bias stats: min={last_linear.bias.min().item():.4f}, max={last_linear.bias.max().item():.4f}, mean={last_linear.bias.mean().item():.4f}")
            
            # Zero out the bias
            with torch.no_grad():
                last_linear.bias.zero_()
            
            print(f"  Bias successfully zeroed out")
        else:
            print("  Last linear layer has no bias term")
            
    except Exception as e:
        print(f"  Error getting final linear layer: {e}")
        raise
    
    return model


def normalize_weights_in_last_layer(model, model_name=None):
    """
    Normalize the weight matrix in the last linear layer using L2 norm.
    This function should be called after bias has been zeroed out.
    
    Args:
        model: The model to modify
        model_name: Name of the model (required for TIMM models)
        
    Returns:
        Modified model with normalized weights in last layer
    """
    try:
        last_linear = _get_final_linear_layer(model, model_name)
        
        if last_linear.weight is not None:
            print(f"  Found last linear layer with weight shape: {last_linear.weight.shape}")
            
            # Get original weight statistics
            original_weight = last_linear.weight.data
            original_norm = torch.norm(original_weight, dim=1, keepdim=True)  # L2 norm per row
            print(f"  Original weight norms - min: {original_norm.min().item():.4f}, max: {original_norm.max().item():.4f}, mean: {original_norm.mean().item():.4f}")
            
            # Normalize weights using L2 norm (normalize each row/class)
            with torch.no_grad():
                # Compute L2 norm for each row (each class)
                weight_norms = torch.norm(original_weight, dim=1, keepdim=True)
                # Add small epsilon to avoid division by zero
                eps = 1e-8
                normalized_weights = original_weight / (weight_norms + eps)
                
                # Update the weights
                last_linear.weight.data = normalized_weights
            
            # Verify normalization
            new_norms = torch.norm(last_linear.weight.data, dim=1, keepdim=True)
            print(f"  Normalized weight norms - min: {new_norms.min().item():.4f}, max: {new_norms.max().item():.4f}, mean: {new_norms.mean().item():.4f}")
            print(f"  Weights successfully normalized")
        else:
            print("  Last linear layer has no weight matrix")
            
    except Exception as e:
        print(f"  Error getting final linear layer: {e}")
        raise
    
    return model


def copy_model_state(model):
    """
    Create a deep copy of the model state to avoid modifying the original model.
    
    Args:
        model: The model to copy
        
    Returns:
        Deep copy of the model
    """
    import copy
    return copy.deepcopy(model)


def load_pretrained_model(model_name: str):
    """
    Load a pre-trained model using the same logic as get_backbone function.
    This centralizes the model loading logic for consistency between scripts.
    
    NOTE: This function is primarily intended for weight extraction from models
    with ImageNet classification heads. For CLIP and DINOv2 models, use get_backbone()
    instead as they don't have ImageNet classification heads.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        torch.nn.Module: The loaded pre-trained model
        
    Raises:
        ValueError: If attempting to load CLIP or DINOv2 models for weight extraction
    """
    if model_name.startswith('clip'):
        # CLIP models - use torch.hub like in get_backbone
        clip_model_name_map = {
            "clip_vitb32": "ViT_B_32",
            "clip_vitb16": "ViT_B_16",
            "clip_vitl14": "ViT_L_14",
            "clip_vitl14_336px": "ViT_L_14_336px"
        }
        
        assert model_name in clip_model_name_map, f"Unsupported CLIP model name: {model_name}"
        
        # Load the CLIP model using torch.hub (same as get_backbone)
        model, _ = torch.hub.load('openai/CLIP', clip_model_name_map[model_name])
        model = model.float()
        model.eval()
        return model
        
    elif model_name.startswith('dinov2'):
        # DINOv2 models - use torch.hub like in get_backbone
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval()
        return model
        
    elif model_name.startswith('convnext'):
        # ConvNext models - load with classification head for weight extraction
        from transformers import ConvNextForImageClassification
        
        convnext_model_name_map = {
            "convnext_base": "facebook/convnext-base-224-22k",
            "convnext_large": "facebook/convnext-large-224-22k", 
            "convnext_tiny": "facebook/convnext-tiny-224",
            "convnext_small": "facebook/convnext-small-224"
        }
        
        assert model_name in convnext_model_name_map, f"Unsupported ConvNext model name: {model_name}"
        
        # Load ConvNext model with classification head (consistent with torchvision approach)
        model = ConvNextForImageClassification.from_pretrained(convnext_model_name_map[model_name])
        model.eval()
        return model
        
    elif model_name in TORCHVISION_MODEL_NAMES:
        # Torchvision models
        model_fn = getattr(models, model_name)
        model = model_fn(weights='DEFAULT')  # Use 'DEFAULT' for latest weights
        model.eval()
        return model
        
    elif model_name.startswith('timm/'):
        # TIMM models - use timm library for proper loading
        import timm
        # Remove 'timm/' prefix for timm.create_model
        timm_model_name = model_name[5:]  # Remove 'timm/' prefix
        print(f"Loading TIMM model: {timm_model_name}")
        try:
            model = timm.create_model(timm_model_name, pretrained=True)
        except Exception as e:
            print(f"Warning: Could not load TIMM model {timm_model_name} directly: {e}. Trying with hf_hub:timm/{timm_model_name}")
            model = timm.create_model(f"hf_hub:timm/{timm_model_name}", pretrained=True)
        model.eval()
        return model
        
    else:
        # HuggingFace models - use specific model classes for proper classification heads
        if model_name.startswith('google/vit'):
            from transformers import ViTForImageClassification
            print(f"***** Loading HuggingFace ViT model: {model_name} *****")
            model = ViTForImageClassification.from_pretrained(model_name)
            model.eval()
            return model
        elif model_name.startswith('microsoft/swin'):
            from transformers import SwinForImageClassification
            model = SwinForImageClassification.from_pretrained(model_name)
            model.eval()
            return model
        else:
            # Fallback to AutoModel for other HuggingFace models
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(model_name)
            model.eval()
            return model


def load_image_model_weights(model_name: str, device: str = "cpu", weights_dir: Optional[str] = None) -> Tuple[torch.Tensor, int]:
    """
    Load the ImageNet-trained classification head weights for a given image model.
    First checks for cached weights, downloads and caches if not found.
    
    This function is optimized for the CCA alignment matrices script and returns
    only weight and num_classes (no bias). However, both weight and bias are saved
    to the cache file for completeness.

    Args:
        model_name (str): Name/ID of the model. For torchvision models,
                          use the function name (e.g. 'resnet50').
                          For Hugging Face models, use the repo id
                          (e.g. 'google/vit-base-patch16-224').
        device (str): Device to load the model on (default: "cpu").
        weights_dir (str): Directory to cache weights (default: "weights").

    Returns:
        weight (torch.Tensor): (num_classes, feature_dim) weight matrix.
        num_classes (int): Number of classes.
        
    Note:
        Both weight and bias are saved to the cache file, but only weight is returned.
        Use load_head_weights() if you need both weight and bias.
    """
    # Create safe filename for the model
    weights_dir = weights_dir or WEIGHTS_DIR
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    model_weights_dir = os.path.join(weights_dir, safe_model_name)
    
    # Special handling for ImageNet-21K pretrained models
    if model_name in IMAGENET21K_HEAD_MODELS:
        weights_path = os.path.join(model_weights_dir, "imagenet21k.pth")
    elif model_name in IMAGENET21K_2EXTRA_HEAD_MODELS:
        # These models have 2 extra heads, but we still use the same cache file
        weights_path = os.path.join(model_weights_dir, "imagenet21k_2extra.pth")
    elif model_name in IMAGENET1K_HEAD_MODELS:
        weights_path = os.path.join(model_weights_dir, "imagenet1k.pth")
    elif model_name in INATURALIST_HEAD_MODELS:
        weights_path = os.path.join(model_weights_dir, "inaturalist.pth")
    else:
        raise ValueError(f"Model {model_name} not recognized as having an ImageNet or iNaturalist classification head.")
    
    # Check if cached weights exist
    if os.path.exists(weights_path):
        print(f"Loading cached weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        
        # Check if the cache file has the new format with bias
        if 'bias' not in checkpoint:
            print(f"⚠️  Cache file is missing bias. Re-downloading model to get complete weights...")
            os.remove(weights_path)
        else:
            weight = checkpoint['weight']
            # Note: bias is also available in checkpoint['bias'] but not returned by this function
            num_classes = checkpoint['num_classes']
            return weight, num_classes
    
    print(f"Cached weights not found. Downloading model {model_name}...")
    
    # Download and extract weights using centralized loading function
    model = load_pretrained_model(model_name)

    head_layer = _get_final_linear_layer(model, model_name)
    weight = head_layer.weight.detach().cpu()
    weight = weight.squeeze(-1).squeeze(-1)
    bias = head_layer.bias.detach().cpu()  # Also save bias for completeness
    num_classes = weight.shape[0]
    
    # Cache the weights (save both weight and bias)
    os.makedirs(model_weights_dir, exist_ok=True)
    checkpoint = {
        'weight': weight,
        'bias': bias,  # Save bias even though we only return weight
        'num_classes': num_classes,
        'model_name': model_name
    }
    torch.save(checkpoint, weights_path)
    print(f"Cached weights saved to {weights_path}")
    
    # Clean up model from memory
    del model
    cleanup_model_cache()
    
    return weight, num_classes


def load_head_weights(model_name: str, device: str = "cpu", weights_dir: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load the ImageNet-trained classification head weights for a given model.
    First checks for cached weights, downloads and caches if not found.
    
    This function is optimized for the similarity matrices script and returns
    weight, bias, and num_classes.

    Args:
        model_name (str): Name/ID of the model. For torchvision models,
                          use the function name (e.g. 'resnet50').
                          For Hugging Face models, use the repo id
                          (e.g. 'google/vit-base-patch16-224').
        device (str): Device to load the model on (default: "cpu").
        weights_dir (str): Directory to cache weights (default: "weights").

    Returns:
        weight (torch.Tensor): (num_classes, feature_dim) weight matrix.
        bias   (torch.Tensor): (num_classes,) bias vector.
        num_classes (int): Number of classes (1000 for all ImageNet models).
    """
    # Create safe filename for the model
    weights_dir = weights_dir or WEIGHTS_DIR
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    model_weights_dir = os.path.join(weights_dir, safe_model_name)
    
    # Special handling for ImageNet-21K pretrained models
    if model_name in IMAGENET21K_HEAD_MODELS:
        weights_path = os.path.join(model_weights_dir, "imagenet21k.pth")
    elif model_name in IMAGENET21K_2EXTRA_HEAD_MODELS:
        weights_path = os.path.join(model_weights_dir, "imagenet21k_2extra.pth")
    elif model_name in IMAGENET1K_HEAD_MODELS:
        weights_path = os.path.join(model_weights_dir, "imagenet1k.pth")
    elif model_name in INATURALIST_HEAD_MODELS:
        weights_path = os.path.join(model_weights_dir, "inaturalist.pth")
    else:
        raise ValueError(f"Model {model_name} not recognized as having an ImageNet or iNaturalist classification head.")    
    # Check if cached weights exist
    if os.path.exists(weights_path):
        print(f"Loading cached weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        weight = checkpoint['weight']
        bias = checkpoint['bias']
        num_classes = checkpoint['num_classes']
        return weight, bias, num_classes
    
    print(f"Cached weights not found. Downloading model {model_name}...")
    
    # Download and extract weights using centralized loading function
    model = load_pretrained_model(model_name)

    head_layer = _get_final_linear_layer(model, model_name)
    weight = head_layer.weight.detach().cpu()
    bias = head_layer.bias.detach().cpu()

    num_classes = weight.shape[0]
    
    # Validate expected number of classes
    if model_name in IMAGENET21K_HEAD_MODELS:
        # ImageNet-21K models have 21841 classification classes
        if num_classes != 21841:
            print(f"⚠️  Warning: Expected 21841 classes for ImageNet-21K model {model_name}, got {num_classes}")
    elif model_name in IMAGENET1K_HEAD_MODELS:
        # All other models should have 1000 classes
        if num_classes != 1000:
            raise ValueError(f"Expected 1000 classes for ImageNet model {model_name}, got {num_classes}. "
                            f"All models should have ImageNet-1K classification heads with 1000 classes.")
    elif model_name in INATURALIST_HEAD_MODELS:
        # iNaturalist models have 1010 classes
        if num_classes != 10000:
            print(f"⚠️  Warning: Expected 10000 classes for iNaturalist model {model_name}, got {num_classes}")
    
    # Cache the weights
    os.makedirs(model_weights_dir, exist_ok=True)
    checkpoint = {
        'weight': weight,
        'bias': bias,
        'num_classes': num_classes,
        'model_name': model_name
    }
    torch.save(checkpoint, weights_path)
    print(f"Cached weights saved to {weights_path}")
    
    # Clean up model from memory
    del model
    cleanup_model_cache()
    
    return weight, bias, num_classes


def turn_final_layer_to_id(model, model_name):
    """
    Replace the final linear layer of a model with Identity layer.
    This is useful for feature extraction when we want to remove the classification head.
    
    Args:
        model: PyTorch model with a classification head
        model_name: Name of the model (required for TIMM models)
        
    Returns:
        Modified model with final linear layer replaced by Identity
        
    Raises:
        ValueError: If the model doesn't have an ImageNet classification head
    """
    try:
        # Get both the layer reference and the setter function
        last_linear, set_layer = _get_final_linear_layer_with_setter(model, model_name)

        if last_linear is None:
            raise ValueError(f"Model {model_name} does not have a final linear layer to replace.")
        
        print(f"  Replacing final linear layer: {last_linear}")
        print(f"Model name: {isinstance(last_linear, torch.nn.modules.conv.Conv2d)}")
        if not isinstance(last_linear, torch.nn.Linear) and not (
            isinstance(last_linear, torch.nn.modules.conv.Conv2d) and
            model_name.lower() in ['timm/resnetv2_50x1_bit.goog_in21k', 'timm/resnetv2_101x1_bit.goog_in21k',
                                   'timm/resnetv2_50x1_bit.goog_in21k_ft_in1k', 'timm/resnetv2_101x1_bit.goog_in21k_ft_in1k']):
                raise ValueError(f"Expected final layer to be Linear, got {type(last_linear)} for model {model_name}")
        
        # Replace the final linear layer with Identity using the setter function
        identity_layer = torch.nn.Identity()
        set_layer(identity_layer)

        print(f"  Successfully replaced with Identity layer")

    except Exception as e:
        print(f"  Error replacing final linear layer: {e}")
        raise
    
    return model


def detect_model_type(model, model_name):
    """
    Detect if a model has ImageNet-1k (1000 classes) or ImageNet-21k (21843 classes) output.
    
    Args:
        model: The PyTorch model to analyze
        model_name: Name of the model (required for TIMM models)
        
    Returns:
        tuple: (model_type, out_features) where model_type is 'imagenet1k', 'imagenet21k', or 'unknown'
    """
    try:
        final_layer = _get_final_linear_layer(model, model_name)
        out_features = final_layer.out_features
        
        if out_features == 1000:
            return 'imagenet1k', out_features
        elif out_features == 21841:
            return 'imagenet21k', out_features
        elif out_features == 21843:
            # Special case for models with 2 extra heads (e.g. CLIP)
            print(f"  Detected model with 2 extra heads: {model_name} (21843 output features)")
            return 'imagenet21k_2extra', out_features
        else:
            print(f"  Warning: Unexpected number of output features: {out_features}")
            return 'unknown', out_features
            
    except Exception as e:
        print(f"  Warning: Could not detect model type for {model_name}: {e}")
        return 'unknown', None


def convert_21k_to_1k_model(model, model_name, device='cpu'):
    """
    Convert an ImageNet-21k model to ImageNet-1k by replacing the final layer.
    Uses the extract_and_convert_imagenet21k_to_1k function from extract_imagenet1k_layer.
    
    Args:
        model: The PyTorch model with ImageNet-21k head
        model_name: Name of the model (required for TIMM models)
        device: Device to create the new layer on
        
    Returns:
        PyTorch model with ImageNet-1k head (1000 classes)
    """
    try:
        # Import the conversion function
        from utils.extract_imagenet1k_layer import extract_and_convert_imagenet21k_to_1k
        
        print(f"  Converting ImageNet-21k model to ImageNet-1k...")
        
        # Use the complete conversion pipeline 
        imagenet1k_layer, metadata = extract_and_convert_imagenet21k_to_1k(model_name, device)
        
        # Replace the final layer in the model using the setter function
        final_layer, set_layer = _get_final_linear_layer_with_setter(model, model_name)
        set_layer(imagenet1k_layer)
        
        print(f"  ✅ Successfully converted to ImageNet-1k:")
        print(f"     Available classes: {metadata['available_classes']}/1000")
        print(f"     Missing classes: {metadata['missing_classes']}")
        
        return model, metadata
        
    except ImportError as e:
        raise ImportError(f"Could not import conversion function: {e}. Make sure extract_imagenet1k_layer.py is available.")
    except Exception as e:
        raise RuntimeError(f"Failed to convert model from ImageNet-21k to ImageNet-1k: {e}")


def prepare_model_for_imagenet1k(model, model_name, device='cpu'):
    """
    Prepare a model for ImageNet-1k evaluation by converting from ImageNet-21k if needed.
    
    Args:
        model: The PyTorch model to prepare
        model_name: Name of the model (required for TIMM models)
        device: Device to use for computation
        
    Returns:
        tuple: (prepared_model, model_type, conversion_applied)
            - prepared_model: Model ready for ImageNet-1k evaluation
            - model_type: 'imagenet1k', 'imagenet21k', or 'unknown'
            - conversion_applied: True if conversion was applied, False otherwise
    """
    model_type, out_features = detect_model_type(model, model_name)
    
    print(f"  Detected model type: {model_type} ({out_features} output features)")
    
    if model_type == 'imagenet1k':
        print(f"  ✅ Model already has ImageNet-1k head, no conversion needed")
        return model, model_type, False
        
    elif model_type == 'imagenet21k':
        print(f"  🔄 Converting ImageNet-21k model to ImageNet-1k...")
        converted_model, metadata = convert_21k_to_1k_model(model, model_name, device)
        return converted_model, 'imagenet1k_converted', True
    elif model_type == 'imagenet21k_2extra':
        print(f"  🔄 Converting ImageNet-21k model with 2 extra heads to ImageNet-1k...")
        converted_model, metadata = convert_21k_to_1k_model(model, model_name, device)
        return converted_model, 'imagenet1k_converted', True
    else:
        print(f"  ⚠️  Unknown model type with {out_features} output features")
        print(f"     Proceeding without conversion (may fail evaluation)")
        return model, model_type, False
